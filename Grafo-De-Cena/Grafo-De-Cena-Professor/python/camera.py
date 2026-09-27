from __future__ import annotations

from typing import TYPE_CHECKING

import graphics_math as gm

if TYPE_CHECKING:
  from state import State

class Camera:
  """Base interface for cameras: projection matrix, view matrix, and uniform
  upload. Defaults return identity; Camera2D/Camera3D override them."""

  def get_proj_matrix (self, canvas_size: tuple[int, int]) -> gm.Mat4:
    """Projection matrix mapping camera space to clip space, for the given
    canvas size (width, height in pixels). Recompute every frame - depends
    on canvas_size."""
    return gm.mat4(1)

  def get_view_matrix (self) -> gm.Mat4:
    """View matrix mapping world space into camera (eye) space. Recomputed
    every frame since subclasses may derive it from interactive state."""
    return gm.mat4(1)

  def load (self, st: State) -> None:
    """Pushes this camera's per-frame uniforms (e.g. camera_position) onto
    State's per-field value stacks, in the active shader's lighting space.
    Called via Pipeline.load whenever a Node's own Pipeline is loaded, so
    a real Shader is always active by the time this runs. Base default
    pushes nothing; pair any override with a matching unload."""
    pass

  def unload (self, st: State) -> None:
    """Pops whatever load pushed. Pairs with load; base default is a
    no-op, matching load's no-op default."""
    pass