from __future__ import annotations

import graphics_math as gm
from camera import *

class Camera2D (Camera):
  """Orthographic 2D camera defined by a world-space view window
  (xmin/xmax/ymin/ymax). View matrix is always identity."""

  def __init__(self, xmin: float = -1, xmax: float = 1, ymin: float = -1, ymax: float = 1) -> None:
    """Store the requested view window; defaults to the symmetric [-1,1]
    unit square."""
    self.xmin = xmin
    self.xmax = xmax
    self.ymin = ymin
    self.ymax = ymax

  def get_proj_matrix (self, canvas_size: tuple[int, int]) -> gm.Mat4:
    """Orthographic projection from the view window, expanding the shorter
    axis to match the canvas aspect ratio. Recomputed every frame - depends
    on canvas_size."""
    # Expands the shorter axis of the requested window to match the canvas
    # aspect ratio, so content isn't stretched non-uniformly.
    w, h = canvas_size
    if w <= 0 or h <= 0:
      raise ValueError(f"get_proj_matrix needs a canvas_size with both dimensions > 0, got {canvas_size}")
    dx = self.xmax - self.xmin
    dy = self.ymax - self.ymin
    if w/h > dx/dy:
      xc = (self.xmin + self.xmax) / 2
      xmin = xc - dx/2 * w/h
      xmax = xc + dx/2 * w/h 
      ymin = self.ymin
      ymax = self.ymax
    else:
      yc = (self.ymin + self.ymax) / 2
      ymin = yc - dy/2 * h/w
      ymax = yc + dy/2 * h/w
      xmin = self.xmin
      xmax = self.xmax
    # WebGPU's NDC z range is [0,1], not OpenGL's [-1,1] that plain
    # Use WebGPU's zero-to-one depth range; see Camera3D.get_proj_matrix.
    return gm.ortho(xmin,xmax,ymin,ymax,-1,1)

  def get_view_matrix (self) -> gm.Mat4:
    """Always identity - 2D scenes are already expressed in the view
    window's coordinates."""
    return gm.mat4(1.0)
