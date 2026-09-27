from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from node import Node
  from engine import Engine
  from state import State

class Scene:
  """Top-level container: a root Node plus the Engines (simulation/logic
  updated once per frame, e.g. animation or physics) that drive it."""

  def __init__ (self, root: Node) -> None:
    self.root = root
    self.engines: list[Engine] = []

  def get_root (self) -> Node:
    return self.root

  def add_engine (self, engine: Engine) -> None:
    self.engines.append(engine)

  def update (self, dt: float) -> None:
    """Advances every registered Engine by `dt` seconds. Call once per
    frame, before render."""
    for e in self.engines:
      e.update(dt)

  def render (self, state: State) -> None:
    """Walks the scene graph from the root, issuing draw calls through
    `state`. Call once per render pass/camera, after update."""
    self.root.render(state)
