from __future__ import annotations

from typing import Any

import graphics_math as gm

from light import Light


class PointLight(Light):
    """Point light whose position is translated by coordinate mappings."""

    def __init__ (self, x: float, y: float, z: float,
                  space: str = "world", **values: Any) -> None:
      Light.__init__(self, space, **values)
      self.set_position(x, y, z)

    def set_position (self, x: float, y: float, z: float) -> None:
      self.set("light_position", gm.vec4(x,y,z,1.0))

    def map (self, matrix: gm.Mat4) -> dict[str, Any]:
      position = self.get("light_position")
      return {"light_position": matrix @ position}
