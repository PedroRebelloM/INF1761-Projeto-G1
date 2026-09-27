from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from state import State

class Shape:
  """Base class for drawable geometry attached to a Node. Subclasses
  implement draw(st) to issue their draw call(s)."""

  def draw (self, st: State) -> None:
    """Subclasses override to issue their draw call(s)."""
    raise NotImplementedError()
