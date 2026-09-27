struct Matrix {
  vertex: mat4x4<f32>,
  normal: mat4x4<f32>,
}
@group(1) @binding(0) var<storage, read> matrix: array<Matrix>;

struct Global {
  projection: mat4x4<f32>,   // espaco de iluminacao -> NDC
  light_ambient: vec3<f32>,
  light_diffuse: vec3<f32>,
  light_specular: vec3<f32>,
  light_position: vec4<f32>,
  camera_position: vec4<f32>,  // posição da câmera no espaço de iluminação (não usada nesta fórmula)
}
@group(0) @binding(0) var<uniform> global: Global;

struct Material {
  base_color: vec3<f32>,
  opacity: f32,

  specular_color: vec3<f32>,
  shininess: f32,
}
@group(2) @binding(0) var<uniform> material: Material;

@group(3) @binding(0) var decal_texture: texture_2d<f32>;
@group(3) @binding(1) var decal_sampler: sampler;

struct VertexOutput {
  @builtin(position) clip_position: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) texcoord: vec2<f32>,
}

// Quad plano no eixo XY (ver quad.py) — normal constante, sem tangente/textura
// por-vértice além do texcoord (posição reaproveitada como texcoord).
@vertex
fn vs_main (@builtin(instance_index) instance_index: u32, @location(0) coord: vec2<f32>, @location(1) texcoord: vec2<f32>) -> VertexOutput {
  let m = matrix[instance_index];
  var out: VertexOutput;
  let normal = vec3<f32>(0.0, 0.0, 1.0);
  let pos4 = vec4<f32>(coord, 0.0, 1.0);
  let veye = (m.vertex * pos4).xyz;
  var light: vec3<f32>;
  if (global.light_position.w == 0.0) {
    light = normalize(global.light_position.xyz);
  } else {
    light = normalize(global.light_position.xyz - veye);
  }
  let neye = normalize((m.normal * vec4<f32>(normal, 0.0)).xyz);
  let ndotl = dot(neye, light);
  var color = material.base_color * global.light_ambient.rgb + material.base_color * global.light_diffuse.rgb * max(0.0, ndotl);
  if (ndotl > 0.0) {
    let refl = normalize(reflect(-light, neye));
    color += material.specular_color * global.light_specular.rgb * pow(max(0.0, dot(refl, normalize(-veye))), material.shininess);
  }
  out.color = vec4<f32>(color, 1.0);
  out.texcoord = texcoord;
  out.clip_position = global.projection * (m.vertex * pos4);
  return out;
}

@fragment
fn fs_main (in: VertexOutput) -> @location(0) vec4<f32> {
  return in.color * textureSample(decal_texture, decal_sampler, in.texcoord);
}
