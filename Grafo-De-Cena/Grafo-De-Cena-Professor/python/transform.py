from __future__ import annotations

from typing import TYPE_CHECKING

import graphics_math as gm

if TYPE_CHECKING:
  from state import State

class Transform:
  """Accumulates an affine transform as a single 4x4 matrix;
  translate/scale/rotate/mult_matrix each right-multiply the current
  matrix. On load, composes this matrix onto the state's matrix stack;
  unload pops it back off."""

  def __init__ (self) -> None:
    """Starts with the identity matrix - no transform applied."""
    self.mat: gm.Mat4 = gm.mat4(1.0)

  def load_identity (self) -> None:
    """Resets the accumulated matrix to identity, discarding any prior
    translate/scale/rotate/mult_matrix calls."""
    self.mat = gm.mat4(1.0)

  def mult_matrix (self, mat: gm.Mat4) -> None:
    """Right-multiplies the accumulated matrix by `mat`, so `mat` takes
    effect before whatever was already accumulated."""
    self.mat = self.mat @ mat

  def translate (self, x: float, y: float, z: float) -> None:
    """Right-multiplies the accumulated matrix by a translation of
    (x, y, z), applied before whatever was already accumulated."""
    self.mat = gm.translate(self.mat,gm.vec3(x,y,z))

  def scale (self, x: float, y: float, z: float) -> None:
    """Right-multiplies the accumulated matrix by a scale of (x, y, z),
    applied before whatever was already accumulated."""
    self.mat = gm.scale(self.mat,gm.vec3(x,y,z))

  def rotate (self, angle: float, x: float, y: float, z: float) -> None:
    """Rotate by `angle` degrees around axis (x,y,z)."""
    self.mat = gm.rotate(self.mat,gm.radians(angle),gm.vec3(x,y,z))

  def get_matrix (self) -> gm.Mat4:
    return self.mat

  def load (self, st: State) -> None:
    """Pushes this transform's matrix onto State's matrix stack,
    composed with the enclosing current matrix. Pair with unload."""
    st.push_matrix(self.mat)

  def unload (self, st: State) -> None:
    """Pops the matrix pushed by load, restoring the enclosing current
    matrix."""
    st.pop_matrix()
