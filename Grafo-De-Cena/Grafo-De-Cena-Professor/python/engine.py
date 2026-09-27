from __future__ import annotations

class Engine:
  """Base class for per-frame animation/update logic, registered via
  Scene.add_engine. Subclasses override update to mutate scene state over
  time; this base no-ops."""

  def update (self, dt: float) -> None:
    """Called once per frame by Scene.update with elapsed time `dt` in
    seconds. No-op here; subclasses override to advance their own state."""
    pass