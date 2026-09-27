from __future__ import annotations

import graphics_math as gm
from material import Material

class PhongMaterial(Material):
    """Blinn-Phong surface appearance: a diffuse base color, a specular
    color, shininess exponent, and opacity. The ambient term isn't a
    material field - shaders source it straight from the "global" group's
    light (see shaders/app).

    Fields are private and only ever change through the setters below,
    each of which calls _mark_dirty() - this is what makes the revision
    counter (see Material) a reliable signal that every Shader this
    instance is registered with can trust, instead of a convention a
    direct `material.base_color[0] = ...`-style mutation could silently
    step around."""

    def __init__ (self, r: float, g: float, b: float, opacity: float = 1.0) -> None:
      Material.__init__(
        self,
        base_color=gm.vec3(r,g,b),
        opacity=opacity,
        specular_color=gm.vec3(1,1,1),
        shininess=32.0,
      )

    def set_base_color (self, r: float, g: float, b: float) -> None:
      self.set("base_color", gm.vec3(r,g,b))

    def set_specular (self, r: float, g: float, b: float) -> None:
      self.set("specular_color", gm.vec3(r,g,b))

    def set_shininess (self, shi: float) -> None:
      self.set("shininess", shi)

    def set_opacity (self, opacity: float) -> None:
      """Sets the material's opacity - only has a visible effect with a
      shader that actually reads material.opacity in its fragment stage
      (most don't - shaders/app/gloss/ilum_frag/quad.wgsl is the one
      that does) *and* a Pipeline configured for blending (see
      main_reflection.py's reflector for a working example). Otherwise
      this value is written to the buffer but has no effect."""
      self.set("opacity", opacity)
