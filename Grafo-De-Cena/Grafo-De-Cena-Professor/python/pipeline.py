from __future__ import annotations

from typing import TYPE_CHECKING, Any

import wgpu

if TYPE_CHECKING:
  from state import State
  from shader import Shader

# sentinel distinguishing "depth_stencil not passed" (use the 3D default
# below) from an explicit None (no depth-stencil attachment at all)
_UNSET: Any = object()

_DEFAULT_PRIMITIVE = {"topology": "triangle-list", "front_face": "ccw", "cull_mode": "none"}
_DEFAULT_MULTISAMPLE = {"count": 1, "mask": 0xFFFFFFFF, "alpha_to_coverage_enabled": False}
_DEFAULT_DEPTH_STENCIL = {
  "format": "depth24plus", "depth_write_enabled": True, "depth_compare": wgpu.CompareFunction.less,
  "depth_bias": 0, "depth_bias_slope_scale": 0, "depth_bias_clamp": 0,
}

class Pipeline:
  """Rasterizer/target configuration: primitive, depth_stencil, and
  multisample state, plus the color target's format/blend/write_mask.
  Bound to exactly one Shader at construction time, which supplies the
  vertex buffer and bind group layouts. Builds the immutable
  GPURenderPipeline once, eagerly."""

  def __init__ (self, shader: Shader, target_format: str | None,
                primitive: dict | None = None,
                depth_stencil: dict | None = _UNSET,
                multisample: dict | None = None,
                blend: dict | None = None,
                write_mask: int = wgpu.ColorWrite.ALL,
                stencil_reference: int | None = None) -> None:
    """Builds the immutable GPURenderPipeline for `shader`: merges
    primitive/multisample/depth_stencil overrides over the defaults, and
    derives the fragment stage from `target_format` (None for a
    depth-only pass). Raises ValueError if both target_format and
    depth_stencil are absent."""
    self.shader = shader
    self.device = shader.device
    self.target_format = target_format
    self.stencil_reference = stencil_reference

    self.primitive = {**_DEFAULT_PRIMITIVE, **(primitive or {})}
    self.multisample = {**_DEFAULT_MULTISAMPLE, **(multisample or {})}
    if depth_stencil is _UNSET:
      self.depth_stencil = dict(_DEFAULT_DEPTH_STENCIL)
    elif depth_stencil is None:
      self.depth_stencil = None
    else:
      self.depth_stencil = {**_DEFAULT_DEPTH_STENCIL, **depth_stencil}

    # target_format=None: a depth-only pipeline (e.g. a shadow-map generation
    # pass) - no color target, no fragment stage at all. Needs some
    # attachment to render into, so a depth-stencil one is required here.
    if target_format is None and self.depth_stencil is None:
      raise ValueError("Pipeline needs target_format or depth_stencil (or both) - neither given")
    fragment = None
    if target_format is not None:
      fragment = {"module": shader.get_module(), "entry_point": "fs_main",
                  "targets": [{"format": target_format, "blend": blend, "write_mask": write_mask}]}

    vertex_buffers = shader.get_vertex_buffer_layout()
    pipeline_layout = self.device.create_pipeline_layout(bind_group_layouts=shader.get_bind_group_layouts())

    self._gpu_pipeline = self.device.create_render_pipeline(
      layout=pipeline_layout,
      vertex={"module": shader.get_module(), "entry_point": "vs_main", "buffers": vertex_buffers},
      primitive=self.primitive,
      depth_stencil=self.depth_stencil,
      multisample=self.multisample,
      fragment=fragment,
    )

  def get_shader (self) -> Shader:
    return self.shader

  def activate (self, st: State) -> None:
    """Binds this Pipeline's GPURenderPipeline and its Shader's "matrix"
    storage array on the render pass. Called from load (on the way in)
    and from the unload of a nested Pipeline (on the way out), since the
    render pass has no stack of its own: popping State's pipeline stack
    doesn't restore what the encoder actually has bound. Skips the actual
    rebind if this Pipeline is already the one last bound (e.g. a child
    Node re-setting the Pipeline it already inherited) - safe because
    set_pipeline is only ever called from here, so State's record of
    what's bound can't go stale behind its back."""
    if st.get_active_pipeline() is self:
      return
    st.render_pass.set_pipeline(self._gpu_pipeline)
    st.render_pass.set_bind_group(self.shader.get_matrix_group_index(), self.shader.get_matrix_bind_group())
    st.set_active_pipeline(self)

  def load (self, st: State) -> None:
    """Pushes this Pipeline onto State's pipeline stack, binds it on the
    render pass (see activate), delegates to the shader to rewrite and
    push its "global" block (camera position, light, and any app-set
    globals - see Shader.commit_global), and applies stencil_reference
    (0 if unset - an explicit default, not "leave whatever a
    previously-active Pipeline set"). Pair with unload() around a Node's
    subtree. activate comes before commit_global so bind groups are
    always set after the pipeline they belong to, never before."""
    st.push_pipeline(self)
    self.activate(st)
    self.shader.commit_global(st)
    st.render_pass.set_stencil_reference(self.stencil_reference if self.stencil_reference is not None else 0)

  def unload (self, st: State) -> None:
    """Pops this Pipeline off State's stack and restores the enclosing
    one on the render pass (activate again - the encoder keeps no stack),
    restores whatever global bind group was active before this Pipeline's
    load (see Shader.unbind_global), and re-applies the enclosing
    Pipeline's stencil_reference (0 if it doesn't set one, same
    explicit-default reasoning as load). Pair with load()."""
    st.pop_pipeline()
    top = st.pipeline[-1] if st.pipeline else None
    if top is not None:
      top.activate(st)
    self.shader.unbind_global(st)
    if top is not None:
      st.render_pass.set_stencil_reference(top.stencil_reference if top.stencil_reference is not None else 0)
