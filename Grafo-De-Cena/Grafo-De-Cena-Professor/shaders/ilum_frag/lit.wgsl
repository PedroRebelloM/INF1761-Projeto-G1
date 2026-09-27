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

struct VertexOutput {
  @builtin(position) clip_position: vec4<f32>,
  @location(0) normal: vec3<f32>,
  @location(1) light: vec3<f32>,
  @location(2) view: vec3<f32>,
}

// Iluminacao por fragmento (Phong shading): vs_main so transforma
// posicao/normal e entrega normal/luz/vista interpoladas; toda a formula
// de Blinn-Phong (ambiente+difusa+especular) e calculada em fs_main, por
// pixel - ao contrario de ilum_vert/lit.wgsl, que calcula a cor inteira
// por vertice (Gouraud shading) e so interpola a cor final.
@vertex
fn vs_main (@builtin(instance_index) instance_index: u32, @location(0) coord: vec3<f32>, @location(1) normal: vec3<f32>) -> VertexOutput {
  let m = matrix[instance_index];
  var out: VertexOutput;
  let veye = (m.vertex * vec4<f32>(coord, 1.0)).xyz;
  var light: vec3<f32>;
  if (global.light_position.w == 0.0) {
    light = normalize(global.light_position.xyz);
  } else {
    light = normalize(global.light_position.xyz - veye);
  }
  out.normal = (m.normal * vec4<f32>(normal, 0.0)).xyz;
  out.light = light;
  out.view = normalize(-veye);
  out.clip_position = global.projection * (m.vertex * vec4<f32>(coord, 1.0));
  return out;
}

@fragment
fn fs_main (in: VertexOutput) -> @location(0) vec4<f32> {
  // renormaliza: a interpolacao entre vertices desnormaliza normal/luz/vista
  let normal = normalize(in.normal);
  let light = normalize(in.light);
  let view = normalize(in.view);

  var color = material.base_color * global.light_ambient.rgb;
  let ndotl = dot(normal, light);
  if (ndotl > 0.0) {
    color += material.base_color * global.light_diffuse.rgb * ndotl;
    let refl = normalize(reflect(-light, normal));
    color += material.specular_color * global.light_specular.rgb * pow(max(0.0, dot(refl, view)), material.shininess);
  }
  return vec4<f32>(color, 1.0);
}
