from __future__ import annotations

from typing import Any

import wgpu

class Sampler:
  """Sampling rule (addressing/filter), independent of any Texture class.
  A pure resource-holder - no load/unload of its own, see TextureSet
  (which can freely combine any Texture/Sampler pair or larger set, in
  any order)."""

  def __init__ (self, device: wgpu.GPUDevice, varname: str,
                address_mode_u: str = "repeat", address_mode_v: str = "repeat",
                mag_filter: str = "linear", min_filter: str = "linear",
                mipmap_filter: str = "linear", compare: str | None = None) -> None:
    """Creates a regular filtering sampler, or a comparison sampler (usable
    with texture_depth types, e.g. shadow map PCF) when `compare` is given."""
    self.varname = varname
    kwargs: dict[str, Any] = dict(
      address_mode_u=address_mode_u, address_mode_v=address_mode_v,
      mag_filter=mag_filter, min_filter=min_filter, mipmap_filter=mipmap_filter,
    )
    if compare is not None:
      kwargs["compare"] = compare
    self.sampler: wgpu.GPUSampler = device.create_sampler(**kwargs)

  @property
  def resource (self) -> wgpu.GPUSampler:
    """The GPU resource TextureSet/Shader.add_texture_set binds - this
    sampler itself."""
    return self.sampler
