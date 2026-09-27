from __future__ import annotations

import re

import wgpu

import shaderutl as sutl

# subset of WGSL types this project supports (see uniformbuffer._ALIGN_SIZE)
_NORMALIZABLE = {"i32", "f32", "vec2", "vec3", "vec4", "mat4x4"}

_STRUCT_RE = re.compile(r'struct\s+(\w+)\s*\{([^}]*)\}', re.DOTALL)

# WGSL comments, stripped before any parsing: a trailing "// ..." on a struct
# field line would otherwise be glued onto the *next* field's name (fields are
# split on ","), silently making has_field() miss it - and a miss means the
# field is never written, so it reads as zero on the GPU.
_COMMENT_RE = re.compile(r'//[^\n]*|/\*.*?\*/', re.DOTALL)


def _strip_comments (code: str) -> str:
  """Removes line and block comments, keeping newlines so that anything
  reported by line number still lines up."""
  def _keep_newlines (m: re.Match) -> str:
    return "\n" * m.group(0).count("\n")
  return _COMMENT_RE.sub(_keep_newlines, code)
_ATTR_PREFIX_RE = re.compile(r'@\w+(?:\([^)]*\))?\s*')
# group(3) captures whatever's inside var<...> (e.g. "uniform" or
# "storage, read") - None for a bare `var` (textures/samplers).
_BINDING_RE = re.compile(
  r'@group\(\s*(\d+)\s*\)\s*@binding\(\s*(\d+)\s*\)\s*var(?:<([^>]*)>)?\s+(\w+)\s*:\s*([^;]+);'
)
_STORAGE_ARRAY_RE = re.compile(r'array<\s*(\w+)\s*>')
_VERTEX_MAIN_START_RE = re.compile(r'fn\s+vs_main\s*\(')
_VERTEX_PARAM_RE = re.compile(r'@location\(\s*(\d+)\s*\)\s*(\w+)\s*:\s*([^,]+)')

_TEXTURE_VIEW_DIM = {
  "texture_1d": "1d",
  "texture_2d": "2d",
  "texture_2d_array": "2d-array",
  "texture_cube": "cube",
  "texture_cube_array": "cube-array",
  "texture_3d": "3d",
  "texture_depth_2d": "2d",
  "texture_depth_2d_array": "2d-array",
  "texture_depth_cube": "cube",
  "texture_depth_cube_array": "cube-array",
}


def _extract_vs_main_params (code: str) -> str | None:
  """Returns vs_main's parameter list text, or None if there's no
  vs_main. Tracks paren depth manually since the params themselves
  contain parens (`@location(0)`)."""
  m = _VERTEX_MAIN_START_RE.search(code)
  if not m:
    return None
  depth = 1
  i = m.end()
  start = i
  while i < len(code) and depth > 0:
    if code[i] == "(":
      depth += 1
    elif code[i] == ")":
      depth -= 1
    i += 1
  return code[start:i - 1]


def _normalize_type (wgsl_type: str, structs: dict[str, list[tuple[str, str]]], stack: tuple[str, ...] = ()) -> str | list[tuple[str, object]]:
  """Resolves one raw field type to either a primitive name (one of
  _NORMALIZABLE) or, for a reference to another struct declared in this
  same file, a nested schema - a list of (name, resolved_type) pairs,
  each resolved the same way, so a struct nested inside a struct still
  bottoms out in primitives. `stack` is the chain of struct names being
  resolved, to raise instead of recursing forever on a cyclic (and
  therefore un-instantiable) struct reference."""
  base = wgsl_type.split("<")[0].strip()
  if base in _NORMALIZABLE:
    return base
  if base in structs:
    if base in stack:
      raise ValueError(f"Recursive struct definition: {' -> '.join(stack + (base,))}")
    return [(fname, _normalize_type(ftype, structs, stack + (base,))) for fname, ftype in structs[base]]
  raise ValueError(
    f"Unsupported WGSL type in uniform block: '{wgsl_type}' "
    f"(supported: {sorted(_NORMALIZABLE)}, or another struct declared in this file)"
  )


def _parse_structs (code: str) -> dict[str, list[tuple[str, str]]]:
  structs: dict[str, list[tuple[str, str]]] = {}
  for m in _STRUCT_RE.finditer(code):
    name = m.group(1)
    fields: list[tuple[str, str]] = []
    for part in m.group(2).split(","):
      part = _ATTR_PREFIX_RE.sub("", part).strip()
      if not part:
        continue
      fname, _, ftype = part.partition(":")
      fname = fname.strip()
      ftype = ftype.strip()
      if fname and ftype:
        fields.append((fname, ftype))
    structs[name] = fields
  return structs


def _texture_entry (binding: int, wgsl_type: str) -> dict:
  base = wgsl_type.split("<")[0].strip()
  sample_type = "depth" if base.startswith("texture_depth") else "float"
  view_dimension = _TEXTURE_VIEW_DIM.get(base, "2d")
  return {
    "binding": binding, "visibility": wgpu.ShaderStage.FRAGMENT,
    "texture": {"sample_type": sample_type, "view_dimension": view_dimension},
  }


def _sampler_entry (binding: int, wgsl_type: str) -> dict:
  kind = "comparison" if wgsl_type == "sampler_comparison" else "filtering"
  return {"binding": binding, "visibility": wgpu.ShaderStage.FRAGMENT, "sampler": {"type": kind}}


def _uniform_entry (binding: int) -> dict:
  return {
    "binding": binding,
    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
    "buffer": {"type": wgpu.BufferBindingType.uniform},
  }


def _storage_array_entry (binding: int) -> dict:
  return {
    "binding": binding,
    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
  }


class ShaderReflection:
  """Parses one WGSL shader file (regex-based, not a full parser) into a
  structured dict, and exposes query methods for bind group layout
  entries, uniform field locations, and vertex input locations.
  Missing-name queries raise KeyError; some callers catch it for
  silent-skip semantics, others let it surface as a validation error."""

  def __init__ (self, filename: str) -> None:
    """Reads and parses the WGSL source at `filename` once, caching the
    result for the query methods below."""
    self.filename = filename
    self._data = self._parse(_strip_comments(sutl.readfile(filename)))

  def _parse (self, code: str) -> dict:
    structs = _parse_structs(code)
    groups: dict[int, dict] = {}
    fields: dict[str, int] = {}
    textures: dict[str, tuple[int, int]] = {}
    samplers: dict[str, tuple[int, int]] = {}
    uniform_vars: dict[str, int] = {}

    def group (index: int) -> dict:
      """Returns the accumulator dict for bind group `index`, creating it
      empty on first reference."""
      return groups.setdefault(index, {"entries": [], "uniform_fields": None, "storage_array_fields": None})

    def register_fields (idx: int, varname: str, raw_fields: list[tuple[str, str]]) -> list[tuple[str, object]]:
      normalized = [(fname, _normalize_type(ftype, structs)) for fname, ftype in raw_fields]
      uniform_vars[varname] = idx
      for fname, _ in normalized:
        if fname in fields and fields[fname] != idx:
          raise ValueError(
            f"Field '{fname}' declared in more than one group ({fields[fname]} and {idx}) - "
            "field names must be unique across a shader's uniform blocks"
          )
        fields[fname] = idx
      return normalized

    for m in _BINDING_RE.finditer(code):
      idx = int(m.group(1))
      binding = int(m.group(2))
      addr_space = (m.group(3) or "").strip()
      varname = m.group(4)
      type_expr = m.group(5).strip()
      g = group(idx)

      if type_expr.startswith("texture"):
        g["entries"].append(_texture_entry(binding, type_expr))
        textures[varname] = (idx, binding)
        continue

      if type_expr in ("sampler", "sampler_comparison"):
        g["entries"].append(_sampler_entry(binding, type_expr))
        samplers[varname] = (idx, binding)
        continue

      array_match = _STORAGE_ARRAY_RE.fullmatch(type_expr)
      if addr_space.startswith("storage") and array_match:
        # var<storage, read> varname: array<StructType>
        struct_name = array_match.group(1)
        raw_fields = structs.get(struct_name)
        if raw_fields is None:
          raise ValueError(f"Storage array '{varname}' (group {idx}) references undeclared struct: '{struct_name}'")
        g["entries"].append(_storage_array_entry(binding))
        g["storage_array_fields"] = register_fields(idx, varname, raw_fields)
        continue

      # var<uniform> varname: StructType
      raw_fields = structs.get(type_expr)
      if raw_fields is None:
        raise ValueError(f"Uniform '{varname}' (group {idx}) references undeclared struct: '{type_expr}'")
      g["entries"].append(_uniform_entry(binding))
      g["uniform_fields"] = register_fields(idx, varname, raw_fields)

    indices = sorted(groups)
    if indices and indices != list(range(len(indices))):
      raise ValueError(
        f"Bind group indices must be contiguous starting at 0; found: {indices}. "
        "WebGPU doesn't allow 'skipping' a group index."
      )

    # name -> location; format/size aren't reflected at all - WebGPU lets an
    # application's buffer supply a smaller vector format than vs_main
    # declares, so only the application (via Shader.set_vertex_buffers)
    # knows the actual packing, never reflection.
    vertex_inputs: dict[str, int] = {}
    vs_main_params = _extract_vs_main_params(code)
    if vs_main_params is not None:
      for pm in _VERTEX_PARAM_RE.finditer(vs_main_params):
        vertex_inputs[pm.group(2)] = int(pm.group(1))

    group_var_names = {idx: varname for varname, idx in uniform_vars.items()}

    return {"groups": groups, "fields": fields, "textures": textures,
            "samplers": samplers, "vertex_inputs": vertex_inputs,
            "uniform_vars": uniform_vars, "group_var_names": group_var_names}

  # --- bind group queries, used by Shader to build GPUBindGroupLayouts ---

  def group_indices (self) -> list[int]:
    """Every declared @group index, sorted ascending; contiguous from
    0."""
    return sorted(self._data["groups"])

  def layout_entries (self, group: int) -> list[dict]:
    """Returns the GPUBindGroupLayoutEntry dicts for `group`, in WGSL
    declaration order."""
    return self._data["groups"][group]["entries"]

  def uniform_fields (self, group: int) -> list[tuple[str, object]]:
    """Returns (field_name, wgsl_type) pairs for the uniform struct bound
    in `group`, in declaration order. wgsl_type is a primitive type name,
    or - for a field whose type is itself a struct declared in this file -
    a nested list of the same shape (see _normalize_type); uniformbuffer's
    layout code is what flattens that into dotted leaf fields."""
    return self._data["groups"][group]["uniform_fields"]

  def storage_array_fields (self, group: int) -> list[tuple[str, object]]:
    """Returns (field_name, wgsl_type) pairs for the element struct of
    the `var<storage, read> name: array<StructType>` bound in `group`,
    in declaration order - None if `group` isn't a storage array. See
    uniform_fields for what wgsl_type can be."""
    return self._data["groups"][group]["storage_array_fields"]

  def uniform_var_group (self, varname: str) -> int:
    """Returns the bind group index of the uniform variable literally
    named `varname` (e.g. "material", "global", "matrix" - the variable
    name, not its struct type); raises KeyError if undeclared."""
    return self._data["uniform_vars"][varname]

  def group_var_name (self, group: int) -> str:
    """Returns the uniform variable name bound in `group` - the inverse
    of uniform_var_group; raises KeyError if `group` has no uniform var
    (e.g. a texture/sampler-only group)."""
    return self._data["group_var_names"][group]

  # --- named-field queries ---

  def field_group (self, name: str) -> int:
    """Returns the bind group index of the uniform field `name`; raises
    KeyError if undeclared. `name` is the top-level field name only - a
    struct-typed field nested inside it (e.g. "color" inside a "light"
    field) isn't separately registered here, only "light" is."""
    return self._data["fields"][name]

  def texture_binding (self, name: str) -> tuple[int, int]:
    """Returns (group, binding) for the texture variable `name`."""
    return self._data["textures"][name]

  def sampler_binding (self, name: str) -> tuple[int, int]:
    """Returns (group, binding) for the sampler variable `name`."""
    return self._data["samplers"][name]

  # --- vertex input queries, used by Shader to resolve set_vertex_buffers' var_name ---

  def vertex_location (self, name: str) -> int:
    """Returns the @location vs_main declares for input `name`; raises
    KeyError if not found."""
    return self._data["vertex_inputs"][name]

  def vertex_locations (self) -> set[int]:
    """Every @location this shader's vs_main declares, for Shader to
    verify coverage."""
    return set(self._data["vertex_inputs"].values())
