from __future__ import annotations

from typing import TYPE_CHECKING, Any

import graphics_math as gm
import wgpu

if TYPE_CHECKING:
  from camera import Camera
  from shader import Shader
  from pipeline import Pipeline

class State:
  """Per-frame render state threaded through Scene.render/Node.render:
  the active camera/device/render_pass, the pipeline stack, a per-group
  bind-group stack (see push_bind_group/pop_bind_group - used by
  Material/TextureSet), and a flat named-value stack per "matrix"-group
  field name (fed by Transform/SkyBox, read by Shader.commit_matrix -
  see push_matrix/pop_matrix/get_matrix). "material"/"global"/
  textures don't go through this value-stack at all - each Shader owns
  its own persistent per-Material buffers, its own "global" buffer, and
  its own per-TextureSet bind groups.
  The "matrix" field name is a stack too, but accumulates via
  push_matrix instead of a plain push; vertex/normal/projection are
  derived from it once per Node, via load_matrices/unload_matrices.

  The camera's view/projection matrices are computed once here, not
  per-Node: one State is built fresh per render pass (see Renderer.render/
  Algorithm.render_pass), and the camera is logically constant for its
  whole duration - recomputing per draw was redundant work, not a
  per-draw necessity."""

  def __init__ (self, camera: Camera, device: wgpu.GPUDevice, render_pass: wgpu.GPURenderPassEncoder,
                canvas_size: tuple[int, int]) -> None:
    """Starts the pipeline/bind-group/value stacks empty and snapshots
    the camera's view/projection matrices for this whole render pass.
    camera_position isn't seeded here - it's computed directly by
    Shader.commit_global, called via Pipeline.load, since it needs a
    real Shader active to know the lighting space to compute it in."""
    self.camera = camera
    self.device = device
    self.render_pass = render_pass
    self.canvas_size = canvas_size
    self.pipeline: list[Pipeline] = []
    self._active_pipeline: Pipeline | None = None
    self.bind_group_stacks: dict[int, list[wgpu.GPUBindGroup]] = {}
    # a unica coisa herdada que nao e bind group: a matriz acumulada.
    # Pilha dedicada, sem nome - era um dict[str, list] generico que na
    # pratica so guardava "matrix".
    self._matrix_stack: list[gm.Mat4] = []
    self._matrix_version = 0
    # vertex/normal nao sao estado de travessia: sao derivados da matriz
    # acumulada a cada no que desenha, lidos por Shader.commit_matrix no
    # mesmo no, e descartados. Valores simples, portanto - nao pilhas.
    self.derived: dict[str, gm.Mat4] = {}

    self._view: gm.Mat4 = camera.get_view_matrix()
    self._proj: gm.Mat4 = camera.get_proj_matrix(canvas_size)
    self._view_proj: gm.Mat4 = self._proj @ self._view
    self._matrix_shaders: list[Any] = []   # shaders que commitaram matriz nesta passada
    self._inverse_view: gm.Mat4 | None = None  # lazy - only Shader.commit_global needs it, and only for world-space lighting

  def push_pipeline (self, pip: Pipeline) -> None:
    """Pushes `pip` onto the pipeline stack, making it what get_pipeline
    returns until it's popped. Must be paired with pop_pipeline."""
    self.pipeline.append(pip)

  def pop_pipeline (self) -> None:
    """Pops the innermost Pipeline off the stack, restoring whatever was
    active before. Pairs with push_pipeline."""
    self.pipeline.pop()

  def get_pipeline (self) -> Pipeline:
    """Returns the innermost currently-loaded Pipeline (top of the stack);
    raises RuntimeError if no Node in the current path has set one."""
    if not self.pipeline:
      raise RuntimeError("Pipeline not defined")
    return self.pipeline[-1]

  def get_shader (self) -> Shader:
    """Returns the shader of the currently active Pipeline."""
    return self.get_pipeline().get_shader()

  def get_active_pipeline (self) -> Pipeline | None:
    """Returns whichever Pipeline last actually bound its GPU pipeline
    on the render pass (see set_active_pipeline) - None if none has
    yet. Used by Pipeline.activate to skip a redundant re-bind when
    consecutive drawing Nodes share the same Pipeline."""
    return self._active_pipeline

  def set_active_pipeline (self, pip: Pipeline) -> None:
    """Records `pip` as the Pipeline whose GPU pipeline/matrix bind
    group are currently actually bound on the render pass. Called by
    Pipeline.activate right after it does that binding."""
    self._active_pipeline = pip

  def push_bind_group (self, group_index: int, bind_group: wgpu.GPUBindGroup) -> None:
    """Binds `bind_group` at `group_index` immediately and remembers it,
    so pop_bind_group can restore whatever was bound before - used by
    Material subclasses, each instance owning a persistent bind group,
    so a child Node without its own Material correctly inherits the
    ancestor's, and a later sibling still does too after the child's
    subtree pops back off. Must be paired with pop_bind_group."""
    stack = self.bind_group_stacks.setdefault(group_index, [])
    stack.append(bind_group)
    self.render_pass.set_bind_group(group_index, bind_group)

  def pop_bind_group (self, group_index: int) -> None:
    """Pops `group_index`'s bind-group stack and re-binds whatever's now
    on top (the ancestor's), if anything remains. Pairs with
    push_bind_group."""
    stack = self.bind_group_stacks[group_index]
    stack.pop()
    if stack:
      self.render_pass.set_bind_group(group_index, stack[-1])

  def register_matrix_shader (self, shader: Any) -> None:
    """Records that `shader` appended a matrix row in this pass, so
    Renderer/Algorithm know whose array to flush before submit. Called
    by Shader.commit_matrix; cheap, and the list is tiny (one entry per
    shader actually used)."""
    if shader not in self._matrix_shaders:
      self._matrix_shaders.append(shader)

  def get_matrix_shaders (self) -> list[Any]:
    """The shaders that committed a matrix row in this pass."""
    return self._matrix_shaders

  def push_matrix (self, mat: gm.Mat4) -> None:
    """Pushes the current top composed with `mat` - matrix accumulation,
    where each Transform composes onto its ancestors' matrix rather than
    overriding it. Identity is the implicit bottom of the stack. Pairs
    with pop_matrix."""
    self._matrix_stack.append(self.get_matrix() @ mat)
    self._matrix_version += 1

  def push_matrix_absolute (self, mat: gm.Mat4) -> None:
    """Pushes `mat` *without* composing, discarding what the ancestors
    accumulated - for the rare transform that positions itself in the
    scene rather than relative to its parent (see SkyBoxTransform).
    Pairs with pop_matrix."""
    self._matrix_stack.append(mat)
    self._matrix_version += 1

  def pop_matrix (self) -> None:
    """Pops the matrix stack, restoring the ancestor's. Pairs with
    push_matrix or push_matrix_absolute; raises if unbalanced, which is
    a bug worth failing on."""
    self._matrix_stack.pop()
    self._matrix_version += 1

  def get_matrix (self) -> gm.Mat4:
    """The accumulated matrix of the path down to here - identity if no
    Transform pushed anything yet."""
    return self._matrix_stack[-1] if self._matrix_stack else gm.mat4(1.0)

  def get_matrix_version (self) -> int:
    """A counter bumped by every push/pop, so Shader.commit_matrix can
    tell whether the accumulated matrix changed since the row it last
    handed out. Monotonic within a pass, so it can never false-match."""
    return self._matrix_version

  def get_camera (self) -> Camera:
    return self.camera

  def get_view_matrix (self) -> gm.Mat4:
    """Returns this render pass's camera view matrix, computed once in
    __init__ - see the class docstring for why per-Node recomputation
    was unnecessary."""
    return self._view

  def get_proj_matrix (self) -> gm.Mat4:
    """Returns this render pass's camera projection matrix, computed
    once in __init__."""
    return self._proj

  def get_view_proj_matrix (self) -> gm.Mat4:
    """Returns the precomputed proj * view matrix, once per render
    pass."""
    return self._view_proj

  def get_inverse_view_matrix (self) -> gm.Mat4:
    """Returns the inverse view matrix, computed on first use and
    cached - only Shader.commit_global's world-space camera_position
    needs it, so it's not worth computing unconditionally in
    __init__."""
    if self._inverse_view is None:
      self._inverse_view = gm.inverse(self._view)
    return self._inverse_view

  def load_matrices (self) -> None:
    """Computes vertex/normal from the current "matrix" stack and this
    render pass's cached view matrix (respecting the active shader's
    lighting space) and stores them in `derived`. Called once per Node
    just before its Shapes are drawn, paired with unload_matrices right
    after: they are freshly derived per Node and never inherited by
    descendants, so they are plain values, not stacks - only "matrix"
    needs stack discipline.

    "projection" is deliberately NOT here: it takes the lighting space
    to NDC and is the same for the whole pass, so it belongs to the
    scene, not to a node - Shader.commit_global writes it into the
    "global" block once per Pipeline.load.

    Each field is skipped if the active shader's "matrix" struct doesn't
    declare it - commit_matrix would never read it anyway, so computing
    (and, for normal, inverting) it would be pure waste."""
    shd = self.get_shader()
    mat = self.get_matrix()
    needs_vertex = shd.declares_matrix_field("vertex")
    needs_normal = shd.declares_matrix_field("normal")
    if needs_vertex or needs_normal:
      mv = mat      # to global space
      if shd.get_lighting_space() == "camera":
        mv = self.get_view_matrix() @ mv  # to camera space
      if needs_vertex:
        self.derived["vertex"] = mv
      if needs_normal:
        # a rank-deficient transform (e.g. the planar-shadow projection,
        # which flattens geometry onto a plane) has no inverse; the pass
        # that uses one never lights anything, so identity is a safe
        # stand-in and beats raising in the middle of a frame.
        try:
          self.derived["normal"] = gm.transpose(gm.inverse(mv))
        except Exception:
          self.derived["normal"] = gm.mat4(1.0)

  def unload_matrices (self) -> None:
    """Discards what load_matrices derived. Pairs with it. Nothing is
    restored: no descendant ever inherits these - the next Node that
    draws derives its own from its own accumulated "matrix"."""
    self.derived.clear()

  def get_derived (self, name: str) -> gm.Mat4 | None:
    """Returns the matrix load_matrices derived for `name` in the Node
    being drawn, or None if this shader doesn't declare it. Read by
    Shader.commit_matrix."""
    return self.derived.get(name)
