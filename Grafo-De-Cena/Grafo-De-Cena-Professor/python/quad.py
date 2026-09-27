from __future__ import annotations

from typing import TYPE_CHECKING

import wgpu
from shape import *
from grid import *



if TYPE_CHECKING:
  from state import State

class Quad (Shape):
  """Flat grid spanning [0,1]x[0,1], subdivided into nx by ny cells via
  Grid. Coord/texcoord buffers bound at slots 0/1 - two buffers, and not
  the same one twice, because t runs opposite to y (origin at the image's
  top-left corner)."""

  def __init__ (self, device: wgpu.GPUDevice, nx: int = 1, ny: int = 1) -> None:
    grid = Grid(nx, ny)
    self.nind: int = grid.index_count()
    self.coord_vbo: wgpu.GPUBuffer = device.create_buffer_with_data(data=grid.get_coords(), usage=wgpu.BufferUsage.VERTEX)
    self.texcoord_vbo: wgpu.GPUBuffer = device.create_buffer_with_data(data=grid.get_texcoords(), usage=wgpu.BufferUsage.VERTEX)
    self.ibo: wgpu.GPUBuffer = device.create_buffer_with_data(data=grid.get_indices(), usage=wgpu.BufferUsage.INDEX)

  def draw (self, st: State) -> None:
    first_instance = st.get_shader().commit_matrix(st)
    st.render_pass.set_vertex_buffer(0, self.coord_vbo)
    st.render_pass.set_vertex_buffer(1, self.texcoord_vbo)
    st.render_pass.set_index_buffer(self.ibo, wgpu.IndexFormat.uint32)
    st.render_pass.draw_indexed(self.nind, 1, 0, 0, first_instance)
