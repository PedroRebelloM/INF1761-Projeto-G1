from __future__ import annotations

import graphics_math as gm
from material import Material

class ColorMaterial(Material):
    """Flat, unlit color appearance: an RGB color plus opacity, written
    into the shader's "material" group (e.g. ColorBlock's "color"/
    "opacity" fields)."""

    def __init__ (self, r: float, g: float, b: float, opacity: float = 1) -> None:
      Material.__init__(self, color=gm.vec3(r,g,b), opacity=opacity)

    def set_color (self, r: float, g: float, b: float) -> None:
      """Replaces the stored RGB color."""
      self.set("color", gm.vec3(r,g,b))

    def set_opacity (self, opacity: float) -> None:
      """Replaces the stored opacity."""
      self.set("opacity", opacity)
