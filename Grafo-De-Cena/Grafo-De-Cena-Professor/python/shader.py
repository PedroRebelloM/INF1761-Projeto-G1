from __future__ import annotations

from typing import TYPE_CHECKING, Any

import graphics_math as gm
import wgpu

import shaderutl as sutl
import wgslreflect
from uniformbuffer import StorageArray, UniformBlock

if TYPE_CHECKING:
  from state import State
  from light import Light
  from material import Material
  from textureset import TextureSet

class Shader:
  """Wraps a compiled WGSL module: vertex input layout, lighting
  (light/space), and bind groups discovered via reflection. Owns
  everything related to shading - materials, "global" uniforms, and
  textures/samplers - each created and validated once, eagerly, at
  setup (see add_material/commit_global/add_texture_set below), never
  during graph traversal. Rasterizer state lives in Pipeline, not here.

  Four different strategies for the four conventional groups:
  - "matrix": a growable StorageArray, one row appended per distinct
    value, indexed per-draw via instance_index (see commit_matrix).
  - "material": one persistent UniformBlock per registered Material
    instance (see add_material/bind_material/unbind_material) - a
    Material must be added to every shader it's used under; using one
    that wasn't raises, and a shader with no "material" group at all is
    always a no-op regardless.
  - "global": one persistent UniformBlock, built here in __init__, fed
    by set_value/get_value plus camera position (computed directly from
    State - "the camera is always known") and the shader's own `light`
    (see commit_global).
  - textures/samplers: one persistent GPUBindGroup per registered
    TextureSet (see add_texture_set/bind_texture_set/unbind_texture_set) -
    every texture/sampler a shader needs lives under one @group.
  Any other app-specific global (there's no remaining "rare one-off
  group" case - everything that isn't matrix/material/textures belongs
  in "global", set via set_value)."""

  def __init__ (self, device: wgpu.GPUDevice, wgsl_path: str,
                light: Light | None = None, space: str = "camera", max_instances: int = 1024) -> None:
    """Compiles the WGSL module at `wgsl_path`, reflects it into one bind
    group layout per declared group, builds the "matrix" StorageArray
    (if declared) and the "global" UniformBlock (if declared), discovers
    the "material" and texture/sampler groups (if any), and initializes
    the vertex buffer registry. Stores `light`/`space` for commit_global.
    `max_instances` bounds how many distinct matrices can be committed
    in one frame."""
    self.device = device
    self.light = light
    self.space = space

    code = sutl.readfile(wgsl_path)
    self.module: wgpu.GPUShaderModule = device.create_shader_module(code=code)
    self._reflection = wgslreflect.ShaderReflection(wgsl_path)

    self._layouts: dict[int, wgpu.GPUBindGroupLayout] = {}
    for g in self._reflection.group_indices():
      self._layouts[g] = device.create_bind_group_layout(entries=self._reflection.layout_entries(g))

    self._matrix_array: StorageArray | None = None
    try:
      matrix_group = self._reflection.uniform_var_group("matrix")
    except KeyError:
      matrix_group = None
    if matrix_group is not None:
      fields = self._reflection.storage_array_fields(matrix_group)
      self._matrix_array = StorageArray(device, fields, self._layouts[matrix_group], matrix_group, max_instances)
    self._frame_state: State | None = None
    self._matrix_tick = -1
    self._matrix_row = -1

    # --- "material" group: discovered here, buffers built lazily per
    # Material instance via add_material (never here) ---
    try:
      self._material_group: int | None = self._reflection.uniform_var_group("material")
    except KeyError:
      self._material_group = None
    self._materials: dict[Material, UniformBlock] = {}
    self._material_revisions: dict[Material, int] = {}

    # --- "global" group: built eagerly, once, right here - never rebuilt
    # per Pipeline.load (see commit_global) ---
    try:
      global_group = self._reflection.uniform_var_group("global")
    except KeyError:
      global_group = None
    self._global_block: UniformBlock | None = None
    if global_group is not None:
      fields = self._reflection.uniform_fields(global_group)
      self._global_block = UniformBlock(device, fields, self._layouts[global_group], global_group)
    self._values: dict[str, Any] = {}

    # --- texture/sampler group: every texture/sampler this shader needs
    # lives under one @group - discovered as the one declared group with
    # neither uniform_fields nor storage_array_fields ---
    texture_groups = [
      g for g in self._reflection.group_indices()
      if self._reflection.uniform_fields(g) is None and self._reflection.storage_array_fields(g) is None
    ]
    if len(texture_groups) > 1:
      raise RuntimeError(
        f"shader declares more than one texture/sampler-only group {texture_groups} - "
        "merge them into a single @group"
      )
    self._texture_group: int | None = texture_groups[0] if texture_groups else None
    self._texture_sets: dict[TextureSet, wgpu.GPUBindGroup] = {}

    # vertex buffers registered via set_vertex_buffers - see get_vertex_buffer_layout
    self._vertex_buffers: list[dict] = []

  # --- "matrix" storage array: one row per distinct value, indexed per-draw ---

  def commit_matrix (self, st: State) -> int:
    """Called once per Shape.draw(): resolves the current "vertex"/
    "normal"/"projection" values to a StorageArray row (packing a fresh
    row only when the "matrix" group's version has changed since the
    last commit this frame - see State.get_matrix_version), and returns
    that row's index. Pass it as `first_instance` to draw()/
    draw_indexed() - vs_main's @builtin(instance_index) reads it back."""
    if self._matrix_array is None:
      raise RuntimeError("commit_matrix called on a shader with no \"matrix\" storage array declared")
    if st is not self._frame_state:
      self._frame_state = st
      self._matrix_array.reset()
      self._matrix_tick = -1
      self._matrix_row = -1
    st.register_matrix_shader(self)
    latest = st.get_matrix_version()
    if latest != self._matrix_tick:
      values = {}
      for name, _ in self._matrix_array.schema:
        v = st.get_derived(name)
        if v is not None:
          values[name] = v
      self._matrix_row = self._matrix_array.append(values)
      self._matrix_tick = latest
    return self._matrix_row

  def flush_matrices (self) -> None:
    """Uploads this pass's matrix rows in one transfer. Called by
    Renderer/Algorithm after the traversal, before submit, for every
    shader that committed a matrix (see State.register_matrix_shader)."""
    if self._matrix_array is not None:
      self._matrix_array.flush()

  def get_matrix_bind_group (self) -> wgpu.GPUBindGroup:
    """Returns the "matrix" StorageArray's bind group, for
    Pipeline.activate to (re)bind unconditionally every time."""
    if self._matrix_array is None:
      raise RuntimeError("get_matrix_bind_group called on a shader with no \"matrix\" storage array declared")
    return self._matrix_array.bind_group

  def get_matrix_group_index (self) -> int:
    """Returns the @group index this shader's "matrix" StorageArray is
    bound at, for Pipeline.activate to bind it at the right slot."""
    if self._matrix_array is None:
      raise RuntimeError("get_matrix_group_index called on a shader with no \"matrix\" storage array declared")
    return self._matrix_array.group_index

  def declares_matrix_field (self, name: str) -> bool:
    """Returns whether this shader's "matrix" StorageArray schema
    declares a field named `name` (e.g. "normal") - used by
    State.load_matrices/unload_matrices to skip computing and pushing
    a value this shader's own commit_matrix would never read anyway
    (a depth-only or 2D shader's Matrix struct is often just
    "projection"). False if this shader has no "matrix" group at all."""
    if self._matrix_array is None:
      return False
    return any(n == name for n, _ in self._matrix_array.schema)

  # --- "material" group: one persistent UniformBlock per registered Material ---

  def add_material (self, mat: Material) -> None:
    """Registers `mat` with this shader: builds its persistent
    UniformBlock and writes its initial fields, right here (never
    during traversal). Raises if this shader has no "material" group.
    A generic Material may carry extra values; write_fields maps only the
    values declared by this shader. Must be called once, at setup, before
    `mat` is ever used under this shader."""
    if self._material_group is None:
      raise RuntimeError("this shader has no \"material\" group")
    fields = self._reflection.uniform_fields(self._material_group)
    block = UniformBlock(self.device, fields, self._layouts[self._material_group], self._material_group)
    block.begin()
    mat.write_fields(block)
    block.end()
    self._materials[mat] = block
    self._material_revisions[mat] = mat.get_revision()

  def bind_material (self, st: State, mat: Material) -> None:
    """Binds `mat`'s persistent bind group, rewriting it first if this
    shader's own copy is behind `mat`'s current revision - tracked per
    shader (not a shared flag on `mat`), so one shader rewriting its
    copy doesn't hide the update from another shader the same instance
    is also registered with. No-op if this shader has no "material"
    group at all; raises if it does but `mat` was never add_material()'d
    here."""
    if self._material_group is None:
      return
    block = self._materials.get(mat)
    if block is None:
      raise RuntimeError("Material was used under a shader it was never add_material()'d to")
    if self._material_revisions.get(mat) != mat.get_revision():
      block.begin()
      mat.write_fields(block)
      block.end()
      self._material_revisions[mat] = mat.get_revision()
    st.push_bind_group(block.group_index, block.bind_group)

  def unbind_material (self, st: State) -> None:
    """Restores whatever material bind group was active before the
    matching bind_material call. No-op if this shader has no "material"
    group."""
    if self._material_group is None:
      return
    st.pop_bind_group(self._material_group)

  # --- "global" group: one persistent UniformBlock, fed by set_value + camera + light ---

  def set_value (self, name: str, value: Any) -> None:
    """Sets the "global" group field `name` to `value` - the entry point
    for clip plane, fog, or any other app-global (once at setup, or
    whenever it changes). Raises the next commit_global() if `name`
    isn't a field this shader's "global" struct declares - unlike
    camera/light fields, the app deliberately targeted this shader, so
    an unrecognized name is treated as a real mistake."""
    self._values[name] = value

  def get_value (self, name: str, default: Any = None) -> Any:
    return self._values.get(name, default)

  def _compute_projection_matrix (self, st: State) -> gm.Mat4:
    """The matrix taking this shader's lighting space to NDC - the
    scene's half of the old per-instance MVP. In camera space the view
    transform is already folded into "vertex", so only the projection
    remains; in world space it is projection * view."""
    if self.get_lighting_space() == "camera":
      return st.get_proj_matrix()
    return st.get_view_proj_matrix()

  def _compute_camera_position (self, st: State) -> gm.Vec4:
    """Camera position in this shader's lighting space - in camera
    space the camera is always at the origin. Uses State's cached,
    lazily-inverted view matrix rather than recomputing it here."""
    camera_position = gm.vec4(0, 0, 0, 1)
    if self.get_lighting_space() == "world":
      mat = st.get_inverse_view_matrix()
      camera_position = mat @ camera_position
    return camera_position

  def commit_global (self, st: State) -> None:
    """Rewrites and pushes this shader's "global" block, once per
    Pipeline.load: camera_position (computed directly from st.get_camera()
    - the camera is always known, so nothing needs to push it) and this
    shader's own `light` contribution are attempted but skipped if this
    particular shader's "global" struct doesn't declare them (not every
    shader does); every app-set `set_value` is written unconditionally
    (raises if undeclared - see set_value). Always rewrites, no
    dirty-check - camera position realistically changes most frames
    anyway. No-op if this shader has no "global" group. Pushed via
    State's bind-group stack (like Material/TextureSet), restored by
    unbind_global on the matching Pipeline.unload."""
    block = self._global_block
    if block is None:
      return
    block.begin()
    if block.has_field("projection"):
      block.set("projection", self._compute_projection_matrix(st))
    if block.has_field("camera_position"):
      block.set("camera_position", self._compute_camera_position(st))
    if self.light is not None:
      self.light.write_fields(block, st, self.get_lighting_space())
    for name, value in self._values.items():
      block.set(name, value)
    block.end()
    st.push_bind_group(block.group_index, block.bind_group)

  def unbind_global (self, st: State) -> None:
    """Restores whatever global bind group was active before the
    matching commit_global call. No-op if this shader has no "global"
    group. Pairs with commit_global, called from Pipeline.unload."""
    if self._global_block is None:
      return
    st.pop_bind_group(self._global_block.group_index)

  # --- textures/samplers: one persistent GPUBindGroup per registered TextureSet ---

  def _resolve_binding (self, varname: str) -> tuple[int, int]:
    """Returns (group, binding) for the texture or sampler variable
    `varname`; raises KeyError if this shader declares neither."""
    try:
      return self._reflection.texture_binding(varname)
    except KeyError:
      pass
    return self._reflection.sampler_binding(varname)

  def add_texture_set (self, ts: TextureSet) -> None:
    """Registers `ts` with this shader: validates it covers this
    shader's texture/sampler group exactly (every declared binding,
    each exactly once - a GPUBindGroup can't be created otherwise) and
    builds its persistent bind group, right here (never during
    traversal). Raises if this shader has no texture/sampler group, if
    an item's varname isn't declared by this shader, or if the set is
    incomplete/has a duplicate binding. Retains every item that
    supports it (e.g. Texture1D), so it can refuse to be replaced out
    from under this bind group - see Texture1D.retain/set_data."""
    if self._texture_group is None:
      raise RuntimeError("this shader has no texture/sampler group")
    entries: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item in ts.items:
      try:
        group, binding = self._resolve_binding(item.varname)
      except KeyError:
        raise ValueError(f"'{item.varname}' is not declared by this shader") from None
      if group != self._texture_group:
        raise ValueError(f"'{item.varname}' does not belong to this shader's texture group")
      if binding in seen:
        raise ValueError(f"binding {binding} supplied more than once")
      seen.add(binding)
      entries.append({"binding": binding, "resource": item.resource})
    expected = len(self._reflection.layout_entries(self._texture_group))
    if len(entries) != expected:
      raise ValueError(f"texture set covers {len(entries)} binding(s), shader declares {expected}")
    self._texture_sets[ts] = self.device.create_bind_group(
      layout=self._layouts[self._texture_group], entries=entries,
    )
    for item in ts.items:
      retain = getattr(item, "retain", None)
      if retain is not None:
        retain()

  def bind_texture_set (self, st: State, ts: TextureSet) -> None:
    """Binds `ts`'s persistent bind group. No-op if this shader has no
    texture/sampler group at all; raises if it does but `ts` was never
    add_texture_set()'d here."""
    if self._texture_group is None:
      return
    bind_group = self._texture_sets.get(ts)
    if bind_group is None:
      raise RuntimeError("TextureSet was used under a shader it was never add_texture_set()'d to")
    st.push_bind_group(self._texture_group, bind_group)

  def unbind_texture_set (self, st: State) -> None:
    """Restores whatever texture bind group was active before the
    matching bind_texture_set call. No-op if this shader has no
    texture/sampler group."""
    if self._texture_group is None:
      return
    st.pop_bind_group(self._texture_group)

  # --- used by State, to resolve a pushed field name to its group ---

  def get_module (self) -> wgpu.GPUShaderModule:
    """Returns the compiled GPUShaderModule, for Pipeline's vertex and
    fragment stages."""
    return self.module

  def get_bind_group_layouts (self) -> list[wgpu.GPUBindGroupLayout]:
    """Layouts in group order (0,1,2,...) - required by create_pipeline_layout."""
    return [self._layouts[g] for g in self._reflection.group_indices()]

  def get_lighting_space (self) -> str:
    """Returns the coordinate space ("world" or "camera") this shader
    expects the light's position in; Light.write_fields uses this to
    decide how to transform light_position before writing it."""
    return self.space

  # --- vertex input contract: the app declares how its buffers are packed ---

  def set_vertex_buffers (self, buffers: list[dict]) -> None:
    """Registers this shader's vertex buffer layout, in wgpu's
    `vertex.buffers` shape; any attribute may give `var_name` instead of
    `shader_location`, resolved to that input's declared @location.
    Raises ValueError if a `var_name` isn't a declared vertex input."""
    resolved: list[dict] = []
    for buf in buffers:
      attributes = []
      for attr in buf["attributes"]:
        attr = dict(attr)
        if "var_name" in attr:
          name = attr.pop("var_name")
          try:
            attr["shader_location"] = self._reflection.vertex_location(name)
          except KeyError:
            raise ValueError(f"set_vertex_buffers references '{name}', but this shader "
                              f"doesn't declare a vertex input with that name") from None
        attributes.append(attr)
      resolved.append({**buf, "attributes": attributes})
    self._vertex_buffers = resolved

  def get_vertex_buffer_layout (self) -> list[dict]:
    """Returns the registered vertex.buffers list, after checking every
    @location this shader declares is covered by exactly one attribute.
    Raises ValueError on a mismatch."""
    covered: dict[int, None] = {}
    for buf in self._vertex_buffers:
      for attr in buf["attributes"]:
        location = attr["shader_location"]
        if location in covered:
          raise ValueError(f"Vertex buffer location {location} registered more than once via set_vertex_buffers")
        covered[location] = None
    missing = self._reflection.vertex_locations() - covered.keys()
    if missing:
      raise ValueError(f"Shader declares vertex input location(s) {sorted(missing)} "
                        f"with no matching set_vertex_buffers attribute")
    return self._vertex_buffers
