from __future__ import annotations

from typing import TYPE_CHECKING

import wgpu

if TYPE_CHECKING:
  from camera import Camera
  from scene import Scene

class Renderer:
  """Owns the per-frame wgpu mechanics for a single render pass: the
  encoder/render_pass/submit dance and the depth-texture lifecycle. The
  depth texture (when depth_test=True) is created lazily and resized to
  match `target_texture`, so no separate Resize() call is needed."""

  def __init__ (self, device: wgpu.GPUDevice,
                depth_test: bool = False,
                clear_value: tuple[float, float, float, float] = (0, 0, 0, 1),
                depth_format: str = "depth24plus") -> None:
    """Stores the depth/clear configuration; the depth texture itself is
    created lazily on the first render call."""
    self.device = device
    self.depth_test = depth_test
    self.clear_value = clear_value
    self.depth_format = depth_format
    self._depth_texture: wgpu.GPUTexture | None = None
    self._depth_view: wgpu.GPUTextureView | None = None

  def _ensure_depth_texture (self, size: tuple[int, int]) -> None:
    if not self.depth_test:
      return
    if self._depth_texture is not None and (self._depth_texture.width, self._depth_texture.height) == size:
      return
    if self._depth_texture is not None:
      self._depth_texture.destroy()
    self._depth_texture = self.device.create_texture(
      size=(size[0], size[1], 1), format=self.depth_format, usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
    )
    self._depth_view = self._depth_texture.create_view()

  def render (self, target_texture: wgpu.GPUTexture, scene: Scene, camera: Camera) -> None:
    """Encodes and submits one render pass into `target_texture` (clearing
    it, and the depth buffer if depth_test is set), rendering `scene`
    through `camera`. Resizes the depth texture as needed to match
    `target_texture` - call once per frame, no separate resize wiring
    needed."""
    size = (target_texture.width, target_texture.height)
    self._ensure_depth_texture(size)

    encoder = self.device.create_command_encoder()

    depth_stencil_attachment = None
    if self.depth_test:
      depth_stencil_attachment = {
        "view": self._depth_view, "depth_clear_value": 1.0,
        "depth_load_op": "clear", "depth_store_op": "store",
      }

    render_pass = encoder.begin_render_pass(
      color_attachments=[{
        "view": target_texture.create_view(), "clear_value": self.clear_value,
        "load_op": "clear", "store_op": "store",
      }],
      depth_stencil_attachment=depth_stencil_attachment,
    )
    from state import State
    st = State(camera, self.device, render_pass, size)
    scene.render(st)
    # a travessia so montou as linhas na CPU; um write_buffer por shader,
    # aqui, antes do submit (write_buffer esta na fila, o draw no encoder)
    for shd in st.get_matrix_shaders():
      shd.flush_matrices()
    render_pass.end()
    self.device.queue.submit([encoder.finish()])
