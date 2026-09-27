struct Matrix {
  vertex: mat4x4<f32>,       // objeto -> espaco global (2D nao ilumina)
}
@group(0) @binding(0) var<storage, read> matrix: array<Matrix>;

struct Global {
  projection: mat4x4<f32>,   // espaco de iluminacao -> NDC
}
@group(2) @binding(0) var<uniform> global: Global;


struct ColorBlock {
  color: vec3<f32>,
  opacity: f32,
}
@group(1) @binding(0) var<uniform> material: ColorBlock;

@vertex
fn vs_main (@builtin(instance_index) instance_index: u32, @location(0) pos: vec2<f32>) -> @builtin(position) vec4<f32> {
  return global.projection * (matrix[instance_index].vertex * vec4<f32>(pos, 0.0, 1.0));
}

@fragment
fn fs_main () -> @location(0) vec4<f32> {
  return vec4<f32>(material.color, material.opacity);
}
