from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from state import State

class Appearance:
  """Base class for state a Node loads/unloads around its draws
  (Material subclasses, TextureSet). Subclasses implement load(st);
  unload defaults to a no-op."""

  def load (self, st: State) -> None:
    """Subclasses override to load their state into the GPU pipeline (bind
    pipeline, bind group, set uniforms, etc.) before a Node's draw. Called
    once per frame for each Node that uses this Appearance."""
    raise NotImplementedError()

  def unload (self, st: State) -> None:
    """No-op by default; subclasses override only if they need explicit
    teardown on unbind."""
    pass