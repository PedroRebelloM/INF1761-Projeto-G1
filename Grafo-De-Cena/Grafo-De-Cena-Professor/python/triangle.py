from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import wgpu
from shape import Shape

if TYPE_CHECKING:
  from state import State

class Triangle (Shape):
  """Single 2D triangle with vertices (-1,0), (1,0), (0,1). One vertex
  buffer bound at both slot 0 (position) and slot 1 (texcoord)."""

  def __init__ (self, device: wgpu.GPUDevice) -> None:
    coord = [[-1, 0], [1, 0], [0, 1]]
    bcoord = np.array(coord, dtype='float32')
    self.vbo: wgpu.GPUBuffer = device.create_buffer_with_data(data=bcoord, usage=wgpu.BufferUsage.VERTEX)

  def draw (self, st: State) -> None:
    """Binds the buffer to both slots and issues an unindexed draw(3)."""
    first_instance = st.get_shader().commit_matrix(st)
    st.render_pass.set_vertex_buffer(0, self.vbo)  # posição
    st.render_pass.set_vertex_buffer(1, self.vbo)  # mesmos dados servem de texcoord
    st.render_pass.draw(3, 1, 0, first_instance)
