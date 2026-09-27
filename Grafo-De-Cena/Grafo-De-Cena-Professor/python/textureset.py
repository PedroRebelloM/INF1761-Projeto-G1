from __future__ import annotations

from typing import TYPE_CHECKING, Any

from appearance import Appearance

if TYPE_CHECKING:
  from state import State

class TextureSet (Appearance):
  """A named bundle of already-constructed Texture/Sampler/TexCube/
  TexDepth resources, freely combined - the only thing a Node's
  `apps` holds for texturing. Must be
  registered with a shader via Shader.add_texture_set before use -
  that's where the bundle is validated against the shader's declared
  texture/sampler group and its persistent bind group is built."""

  def __init__ (self, items: list[Any]) -> None:
    """`items` is a list of Texture/Sampler/TexCube/TexDepth objects,
    each already built - this class doesn't create GPU resources
    itself, it just bundles references to them."""
    self.items = items

  def load (self, st: State) -> None:
    """Delegates to the active shader - see Shader.bind_texture_set."""
    st.get_shader().bind_texture_set(st, self)

  def unload (self, st: State) -> None:
    """Delegates to the active shader - see Shader.unbind_texture_set."""
    st.get_shader().unbind_texture_set(st)
