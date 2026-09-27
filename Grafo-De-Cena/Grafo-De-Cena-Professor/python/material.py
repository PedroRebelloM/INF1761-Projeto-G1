from __future__ import annotations

from typing import TYPE_CHECKING, Any

from appearance import Appearance

if TYPE_CHECKING:
  from state import State
  from uniformbuffer import UniformBlock

class Material (Appearance):
    """Generic named-value "material" appearance plus a revision counter.
    It can be used directly for an arbitrary shader material block, or as
    the base of a semantic convenience class such as PhongMaterial.

    Material has no Shader/GPU knowledge. Each Shader owns the actual
    UniformBlock *and* tracks the last revision it uploaded (see
    Shader.add_material/bind_material) - two different shaders each get
    their own, independently validated, written, and revision-tracked,
    so the same Material instance can be used under more than one
    shader safely, and one shader rewriting its copy doesn't hide the
    update from another (a shared dirty flag would - whichever shader's
    bind_material ran first would consume it and clear_dirty() would
    leave every other shader's copy stale forever). Subclasses normally
    provide only semantic constructors and setters; the generic
    write_fields implementation handles the GPU-facing values. Values not
    declared by a particular shader are ignored, allowing one material to
    carry a superset used by several shaders."""

    def __init__ (self, **values: Any) -> None:
      self._values: dict[str, Any] = dict(values)
      self._revision: int = 0

    def write_fields (self, block: UniformBlock) -> None:
      """Writes the intersection of this material's named values and the
      shader's reflected material fields."""
      for name, value in self._values.items():
        if block.has_field(name):
          block.set(name, value)

    def set (self, name: str, value: Any) -> None:
      """Sets an arbitrary material value and invalidates every shader's
      cached GPU copy of this material."""
      self._values[name] = value
      self._mark_dirty()

    def get (self, name: str, default: Any = None) -> Any:
      """Returns a named material value, or `default` when absent."""
      return self._values.get(name, default)

    def get_values (self) -> dict[str, Any]:
      """Returns a shallow copy so callers cannot replace dictionary
      entries without incrementing the material revision via set()."""
      return self._values.copy()

    def _mark_dirty (self) -> None:
      """Called by every setter - bumps the revision, so every shader
      this instance is registered with will rewrite its own block
      before its next bind (see Shader.bind_material)."""
      self._revision += 1

    def get_revision (self) -> int:
      return self._revision

    def load (self, st: State) -> None:
      """Delegates to the active shader - see Shader.bind_material."""
      st.get_shader().bind_material(st, self)

    def unload (self, st: State) -> None:
      """Delegates to the active shader - see Shader.unbind_material."""
      st.get_shader().unbind_material(st)
