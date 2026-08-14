//! Microsoft `MatMulNBits` Q4 kernel used by LFM2.5 Audio.
//!
//! The exported weights are `[N, K/block_size, block_size/2]` uint8 with the
//! low nibble first and one f32 scale per `(N, K/block_size)` block. No explicit
//! zero-point input means the ONNX Runtime contrib-op default of 8.

pub const BINDINGS: u32 = 4;
pub const PUSH_BYTES: u32 = 12;
pub const WORKGROUP_SIZE: u32 = 256;

pub const MATMUL_NBITS_Q4: &str = r#"
struct Params { m: u32, k: u32, n: u32 }
var<immediate> pc: Params;

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> packed_w: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

fn packed_byte(byte_index: u32) -> u32 {
    let word = packed_w[byte_index >> 2u];
    return (word >> ((byte_index & 3u) * 8u)) & 0xffu;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let output_index = gid.x;
    let output_count = pc.m * pc.n;
    if (output_index >= output_count) { return; }

    let row = output_index / pc.n;
    let col = output_index - row * pc.n;
    let blocks = pc.k / 32u;
    var sum = 0.0;
    for (var kk = 0u; kk < pc.k; kk = kk + 1u) {
        let byte_index = col * (pc.k / 2u) + (kk / 2u);
        let byte = packed_byte(byte_index);
        let q = select(byte & 0x0fu, byte >> 4u, (kk & 1u) != 0u);
        let scale = scales[col * blocks + kk / 32u];
        sum = sum + a[row * pc.k + kk] * (f32(q) - 8.0) * scale;
    }
    y[output_index] = sum;
}
"#;

#[cfg(test)]
mod tests {
    #[test]
    fn source_compiles() {
        vk_compute::compile_wgsl(super::MATMUL_NBITS_Q4)
            .expect("MatMulNBits Q4 shader must compile to SPIR-V");
    }
}
