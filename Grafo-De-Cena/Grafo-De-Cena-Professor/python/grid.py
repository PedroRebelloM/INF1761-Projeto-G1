from __future__ import annotations

import numpy as np

class Grid:
  """Generic indexed nx-by-ny grid mesh generator, shared by Quad and
  Sphere. nx/ny are cell counts, not vertex counts - there are
  (nx+1)x(ny+1) vertices. Carries two parallel arrays: coords, with y
  growing up (the geometry), and texcoords, with t growing down (the
  WebGPU convention, origin at the image's top-left corner)."""

  def __init__ (self, nx: int, ny: int) -> None:
    """Builds the vertex coordinate and index arrays as plain numpy
    arrays (no GPU buffers - callers upload those). Raises if nx/ny
    isn't positive - both are cell counts used as a divisor below, so
    zero would raise a bare ZeroDivisionError instead of a clear one."""
    if nx <= 0 or ny <= 0:
      raise ValueError(f"Grid needs nx > 0 and ny > 0, got nx={nx}, ny={ny}")
    self.nx: int = nx
    self.ny: int = ny
    # fill coordinates and texture coordinates
    self.coords: np.ndarray = np.empty(2*self.vertex_count(), dtype = 'float32')
    self.texcoords: np.ndarray = np.empty(2*self.vertex_count(), dtype = 'float32')
    dx = 1 / nx
    dy = 1 / ny
    nc = 0
    for j in range(0,ny+1):
      for i in range(0,nx+1):
        self.coords[nc+0] = i*dx
        self.coords[nc+1] = j*dy
        self.texcoords[nc+0] = i*dx
        self.texcoords[nc+1] = 1 - j*dy   # t cresce para baixo
        nc += 2

    # fill indices
    def findex (i: int, j: int, nx: int) -> int:
      """Flattens grid coordinate (i, j) into a row-major vertex index:
      j*(nx+1) + i."""
      return j*(nx+1) + i

    self.indices: np.ndarray = np.empty(self.index_count(), dtype = 'uint32')
    ni = 0
    for j in range(0,ny):
      for i in range(0,nx):
        self.indices[ni+0] = findex(i,j,nx)
        self.indices[ni+1] = findex(i+1,j,nx)
        self.indices[ni+2] = findex(i+1,j+1,nx)
        self.indices[ni+3] = findex(i,j,nx)
        self.indices[ni+4] = findex(i+1,j+1,nx)
        self.indices[ni+5] = findex(i,j+1,nx)
        ni += 6

  def get_nx (self) -> int:
    return self.nx

  def get_ny (self) -> int:
    return self.ny

  def vertex_count (self) -> int:
    """Total vertex count: (nx+1)*(ny+1)."""
    return (self.nx+1)*(self.ny+1)

  def get_coords (self) -> np.ndarray:
    """Flat float32 array of interleaved (x,y) coords in [0,1]x[0,1],
    length 2*vertex_count()."""
    return self.coords

  def get_texcoords (self) -> np.ndarray:
    """Flat float32 array of interleaved (s,t) coords in [0,1]x[0,1],
    length 2*vertex_count(). Same order as get_coords(), but with t
    mirrored: t = 1 - y, so t = 0 is the top of the image."""
    return self.texcoords

  def index_count (self) -> int:
    """Total index count: 6*nx*ny (2 triangles per cell)."""
    return 6*self.nx*self.ny

  def get_indices (self) -> np.ndarray:
    """Flat uint32 triangle-list index array, length index_count()."""
    return self.indices
