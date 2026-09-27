// Binding 0: buffer de armazenamento de floats (equivalente ao antigo imageBuffer R32F)
@group(0) @binding(0) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(4)
fn main (@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  data[idx] = data[idx] + 1.0;
}
