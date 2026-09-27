from __future__ import annotations

# Em WebGPU não existe um objeto-handle de framebuffer (FBO) — um render pass
# referencia GPUTextureViews diretamente. Esta classe é só um par nomeado de
# descritores já prontos, na mesma forma esperada por
# encoder.begin_render_pass(color_attachments=..., depth_stencil_attachment=...)
# - quem monta os dicts (view, clear_value/load_op, etc.) é sempre quem chama,
# igual a Shader.set_vertex_buffers - nada é inferido aqui.
class Framebuffer:
  """Named pair of ready-made attachment descriptors, in the exact shape
  encoder.begin_render_pass(color_attachments=..., depth_stencil_attachment=...)
  expects. Building the individual dicts (view, clear_value/load_op, etc.)
  is always the caller's job - nothing is inferred here (see Algorithm)."""

  def __init__ (self, color_attachments: list[dict], depth_stencil_attachment: dict | None = None) -> None:
    self.color_attachments = color_attachments
    self.depth_stencil_attachment = depth_stencil_attachment
