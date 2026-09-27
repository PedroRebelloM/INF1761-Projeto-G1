from __future__ import annotations

import struct
from typing import TYPE_CHECKING, Any

import graphics_math as gm
import numpy as np
import wgpu

if TYPE_CHECKING:
  from state import State

# (align, size) in bytes, per WGSL layout rules for the "uniform"/"storage"
# address spaces (WebGPU's equivalent of GLSL's std140/std430): each field is
# placed at the next offset that's a multiple of its own type's alignment
# (not just its size - e.g. vec3 is 12 bytes but aligns to 16, leaving a
# 4-byte hole after it whenever something follows it), fields are laid out in
# declaration order with no reordering/repacking, and the overall element
# size is rounded up to a multiple of 16 bytes (a conservative choice that's
# valid for both address spaces, even though storage doesn't strictly
# require it).
_ALIGN_SIZE = {
  "i32": (4, 4),
  "f32": (4, 4),
  "vec2": (8, 8),
  "vec3": (16, 12),
  "vec4": (16, 16),
  "mat4x4": (16, 64),
}

# A schema entry's type is either one of the primitive names above, or -
# for a field whose WGSL type is itself a struct (see wgslreflect's
# _normalize_type) - a nested schema: the same list-of-(name, type) shape,
# one level down.
WgslType = Any
Schema = list[tuple[str, WgslType]]

_COMPONENT_COUNT = {
  "vec2": 2,
  "vec3": 3,
  "vec4": 4,
  "mat4x4": 16,
}

_ZERO = {
  "i32": 0, "f32": 0.0, "vec2": gm.vec2(0), "vec3": gm.vec3(0),
  "vec4": gm.vec4(0), "mat4x4": gm.mat4(0),
}

def _align (offset: int, alignment: int) -> int:
  return (offset + alignment - 1) // alignment * alignment

def _pack (value: Any, wgsl_type: str) -> bytes:
  if wgsl_type == "i32":
    return struct.pack("<i", int(value))
  if wgsl_type == "f32":
    return struct.pack("<f", float(value))
  if wgsl_type == "mat4x4":
    return gm.mat4_bytes(value)
  if wgsl_type in _COMPONENT_COUNT:
    n = _COMPONENT_COUNT[wgsl_type]
    vector = np.asarray(value, dtype="<f4")
    if vector.shape != (n,):
      raise ValueError(f"{wgsl_type} value must have shape ({n},), got {vector.shape}")
    return vector.tobytes()
  raise TypeError("Unsupported wgsl_type in uniformbuffer: " + wgsl_type)

def _type_align_size (wgsl_type: WgslType) -> tuple[int, int]:
  """(align, size) for one field's type: a direct table lookup for a
  primitive, or - recursively, for a nested struct schema - align is the
  max alignment among its own members and size is that alignment rounded
  up from the unrounded end of its last member (WGSL's own struct
  layout rule; distinct from the 16-byte rounding _compute_layout applies
  to the outermost buffer/stride, which nothing but the top level gets)."""
  if isinstance(wgsl_type, list):
    _, _, natural_size, align = _layout_schema(wgsl_type)
    return align, _align(natural_size, align)
  return _ALIGN_SIZE[wgsl_type]

def _layout_schema (schema: Schema) -> tuple[dict[str, int], dict[str, str], int, int]:
  """Lays out `schema`'s own fields starting at offset 0, recursing into
  any nested struct field so the returned offsets/types are flat, with a
  dotted name ("light.color") for every leaf reached through a nested
  struct. Returns (offsets, types, natural_size, align): natural_size is
  the unrounded offset just past the last member (rounding it to this
  struct's own alignment, or to 16 for the outermost buffer, is the
  caller's job - see _type_align_size/_compute_layout); align is the max
  alignment among schema's own members, per WGSL's struct-alignment rule."""
  offsets: dict[str, int] = {}
  types: dict[str, str] = {}
  offset = 0
  align = 1
  for name, wgsl_type in schema:
    field_align, size = _type_align_size(wgsl_type)
    align = max(align, field_align)
    offset = _align(offset, field_align)
    if isinstance(wgsl_type, list):
      sub_offsets, sub_types, _, _ = _layout_schema(wgsl_type)
      for sub_name, sub_offset in sub_offsets.items():
        offsets[f"{name}.{sub_name}"] = offset + sub_offset
        types[f"{name}.{sub_name}"] = sub_types[sub_name]
    else:
      offsets[name] = offset
      types[name] = wgsl_type
    offset += size
  return offsets, types, offset, align

def _compute_layout (schema: Schema) -> tuple[dict[str, int], dict[str, str], int]:
  """Computes every leaf field's byte offset and WGSL type (flat, with a
  dotted name for one nested inside a struct field) per the alignment
  rules above, and the overall (16-byte-rounded) element size - shared by
  StorageArray (one element's stride) and UniformBlock (the whole
  buffer's size)."""
  offsets, types, natural_size, _ = _layout_schema(schema)
  return offsets, types, _align(natural_size, 16)

def _pack_row (values: dict[str, Any], types: dict[str, str], offsets: dict[str, int], stride: int) -> bytes:
  """Packs `values` (missing fields default to a type-appropriate zero)
  into one `stride`-sized row, fields placed at their precomputed
  offsets. `types` is the flat, dotted-leaf mapping _compute_layout
  returns - never a nested schema, so every lookup here is a primitive
  type."""
  row = bytearray(stride)
  for name, wgsl_type in types.items():
    data = _pack(values.get(name, _ZERO[wgsl_type]), wgsl_type)
    off = offsets[name]
    row[off:off + len(data)] = data
  return bytes(row)


class StorageArray:
  """A growable `var<storage, read> name: array<StructType>` buffer -
  used for the "matrix" group, where every draw call needs its own row
  (many transforms per frame, appended rather than overwritten so
  `write_buffer` timing relative to a single per-frame `submit()`
  doesn't matter - see Shader.commit_matrix). Pre-allocated to
  `max_rows`; `append` raises if that capacity is exceeded rather than
  silently corrupting or growing unbounded.

  `append` only packs into a CPU-side staging bytearray; the GPU sees
  the rows on `flush`, in a single `write_buffer` of everything the
  pass appended. One transfer per pass instead of one per row - the
  row index is handed out immediately, so deferring the upload costs
  the caller nothing (see Renderer.render, which flushes before
  submit)."""

  def __init__ (self, device: wgpu.GPUDevice, schema: Schema,
                layout: wgpu.GPUBindGroupLayout, group_index: int, max_rows: int, binding: int = 0) -> None:
    self.device = device
    self.schema = schema
    self.group_index = group_index
    self.max_rows = max_rows
    self.offsets, self._types, self.stride = _compute_layout(schema)
    self.count = 0
    self._staging = bytearray(self.stride * max_rows)
    self.buffer = device.create_buffer(
      size=self.stride * max_rows, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
    )
    self.bind_group = device.create_bind_group(
      layout=layout,
      entries=[{"binding": binding, "resource": {"buffer": self.buffer, "offset": 0, "size": self.stride * max_rows}}],
    )

  def append (self, values: dict[str, Any]) -> int:
    """Packs `values` into the next free row of the CPU staging buffer,
    returning that row's index (the value to pass as `first_instance`).
    Nothing reaches the GPU until `flush`."""
    if self.count >= self.max_rows:
      raise RuntimeError(
        f"StorageArray for group {self.group_index} exceeded its {self.max_rows}-row capacity "
        "(see Shader's max_instances) - raise it or draw fewer distinct instances per frame"
      )
    row = self.count
    off = row * self.stride
    self._staging[off:off + self.stride] = _pack_row(values, self._types, self.offsets, self.stride)
    self.count += 1
    return row

  def flush (self) -> None:
    """Uploads every row appended since the last `reset` in one
    `write_buffer`. Must run before the pass's `submit` - anywhere
    before it works, since write_buffer is on the queue timeline.
    No-op when the pass appended nothing."""
    if self.count == 0:
      return
    self.device.queue.write_buffer(self.buffer, 0, bytes(self._staging[:self.count * self.stride]))

  def reset (self) -> None:
    """Starts a fresh frame: every previously-appended row becomes
    unreachable (nothing will reference its index again) and capacity
    is reclaimed from row 0."""
    self.count = 0


class UniformBlock:
  """A single-instance `var<uniform> name: StructType` buffer + bind
  group, built once and reused for its owner's whole lifetime -
  `Shader` builds one per registered Material (see `add_material`) and
  one for its own "global" group (see `commit_global`), both eagerly, so
  `create_buffer`/`create_bind_group` never happen during traversal.

  Written via `begin`/`set`/`end`: `set(name, value)` raises on a name
  this block's schema doesn't declare - callers that want to skip an
  optional field (not every shader declares every global/light field)
  check `has_field` first, explicitly, rather than `set` silently
  tolerating it. A field nested inside a struct-typed field is named
  with a dot ("light.color") - see _layout_schema."""

  def __init__ (self, device: wgpu.GPUDevice, schema: Schema,
                layout: wgpu.GPUBindGroupLayout, group_index: int, binding: int = 0) -> None:
    self.device = device
    self.schema = schema
    self.group_index = group_index
    self.offsets, self._types, self.size = _compute_layout(schema)
    self._pending: bytearray | None = None
    self.buffer = device.create_buffer(
      size=self.size, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
    )
    self.bind_group = device.create_bind_group(
      layout=layout,
      entries=[{"binding": binding, "resource": {"buffer": self.buffer, "offset": 0, "size": self.size}}],
    )

  def has_field (self, name: str) -> bool:
    """Returns whether this block's schema declares field `name`."""
    return name in self.offsets

  def begin (self) -> None:
    """Starts a fresh write pass: a zeroed staging row, so any field
    never `set` this pass stays zero. Pairs with `end`."""
    self._pending = bytearray(self.size)

  def set (self, name: str, value: Any) -> None:
    """Packs `value` into the field `name` of the row started by
    `begin`. Raises if `name` isn't declared in this block's schema -
    callers that want to tolerate an absent field must check
    `has_field` first."""
    if name not in self.offsets:
      raise ValueError(f"'{name}' is not a field of this shader's group")
    data = _pack(value, self._types[name])
    off = self.offsets[name]
    assert self._pending is not None
    self._pending[off:off + len(data)] = data

  def end (self) -> None:
    """Writes the row accumulated since `begin` to the GPU buffer in
    one call. Pairs with `begin`."""
    assert self._pending is not None
    self.device.queue.write_buffer(self.buffer, 0, bytes(self._pending))
    self._pending = None
