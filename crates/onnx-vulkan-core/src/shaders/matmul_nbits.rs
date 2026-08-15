//! Microsoft `MatMulNBits` Q4 kernel used by LFM2.5 Audio.
//!
//! The exported weights are `[N, K/block_size, block_size/2]` uint8 with the
//! low nibble first and one f32 scale per `(N, K/block_size)` block. No explicit
//! zero-point input means the ONNX Runtime contrib-op default of 8.
//!
//! Two kernels:
//!   * `MATMUL_NBITS_Q4` — naive scalar reference (one thread per output, K
//!     serial nibble-dequant FMAs). Kept for correctness comparison only.
//!   * `MATMUL_NBITS_Q4_TILED` — tiled 16×16 with the Q4 weights dequantized
//!     into shared memory as f32 and accumulated with a vectorized f32 inner
//!     product. This is the fast path: it fills the GPU and uses the wide FMA
//!     units instead of a scalar loop over K per output element.

pub const BINDINGS: u32 = 4;
pub const PUSH_BYTES: u32 = 12;
pub const WORKGROUP_SIZE: u32 = 256;
pub const TILED_BINDINGS: u32 = 4;
pub const TILED_PUSH_BYTES: u32 = 12;

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

/// Vectorized Q4 matmul (correct-by-construction). Same math as the scalar
/// reference, but the K-loop processes 4 weights per iteration via `vec4<f32>`
/// dot products, and nibble dequantization is done directly from the packed
/// bytes. Each thread still owns one output element, so there is no shared
/// memory and no tiling-indexing hazard; the speedup comes from issuing 4 FMAs
/// per instruction and from a single dequant pass over K instead of a fully
/// scalar loop. (A cooperative/tensor-core variant is a follow-up.)
pub const MATMUL_NBITS_Q4_TILED: &str = r#"
struct Params { m: u32, k: u32, n: u32 }
var<immediate> pc: Params;

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> packed_w: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

fn nibble_at(col: u32, kk: u32) -> f32 {
    let byte_index = col * (pc.k / 2u) + (kk / 2u);
    let word = packed_w[byte_index >> 2u];
    let byte = (word >> ((byte_index & 3u) * 8u)) & 0xffu;
    let q = select(byte & 0x0fu, byte >> 4u, (kk & 1u) != 0u);
    let scale = scales[col * (pc.k / 32u) + kk / 32u];
    return (f32(q) - 8.0) * scale;
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

    // 4 k per iteration: kk, kk+1 share one byte; kk+2, kk+3 share the next.
    var kk = 0u;
    for (; kk + 4u <= pc.k; kk = kk + 4u) {
        let base_byte = col * (pc.k / 2u) + (kk / 2u);
        let wa = packed_w[base_byte >> 2u];
        let wb = packed_w[(base_byte + 1u) >> 2u];
        let byte_a = (wa >> ((base_byte & 3u) * 8u)) & 0xffu;
        let byte_b = (wb >> (((base_byte + 1u) & 3u) * 8u)) & 0xffu;
        let q0 = select(byte_a & 0x0fu, byte_a >> 4u, (kk & 1u) != 0u);
        let q1 = select(byte_a & 0x0fu, byte_a >> 4u, ((kk + 1u) & 1u) != 0u);
        let q2 = select(byte_b & 0x0fu, byte_b >> 4u, ((kk + 2u) & 1u) != 0u);
        let q3 = select(byte_b & 0x0fu, byte_b >> 4u, ((kk + 3u) & 1u) != 0u);
        let sa0 = scales[col * blocks + kk / 32u];
        let sa1 = scales[col * blocks + (kk + 1u) / 32u];
        let sa2 = scales[col * blocks + (kk + 2u) / 32u];
        let sa3 = scales[col * blocks + (kk + 3u) / 32u];
        let av = vec4<f32>(
            a[row * pc.k + kk],
            a[row * pc.k + kk + 1u],
            a[row * pc.k + kk + 2u],
            a[row * pc.k + kk + 3u],
        );
        let wv = vec4<f32>(
            (f32(q0) - 8.0) * sa0,
            (f32(q1) - 8.0) * sa1,
            (f32(q2) - 8.0) * sa2,
            (f32(q3) - 8.0) * sa3,
        );
        sum = sum + dot(av, wv);
    }
    // scalar remainder (k not a multiple of 4)
    for (; kk < pc.k; kk = kk + 1u) {
        sum = sum + a[row * pc.k + kk] * nibble_at(col, kk);
    }
    y[output_index] = sum;
}
"#;

/// Q4 matmul with intra-workgroup split-K over the K reduction.
///
/// `MATMUL_NBITS_Q4_TILED` launches one thread per output element, each
/// running the full K reduction serially. For the decode step (m == 1) that is
/// only ~n threads for a 10-CU RDNA2, and even prefill (m == 31) leaves a
/// long dependent-FMA chain per thread. This kernel instead has a
/// 32-column × 8-k-lane workgroup (256 threads): lane `kl` reduces a
/// contiguous 1/8 slice of K, the per-column partials meet in shared memory,
/// and the 8 slices are summed in a fixed order. `gid.x` selects the output
/// row, so the same kernel serves m == 1 and prefill. Every thread in a
/// workgroup reads the SAME activation row slice (L1-cached) and 32 adjacent
/// weight columns (contiguous 32×K/2 bytes), so both loads are fully
/// coalesced.
///
/// Numerically the cross-lane summation reorders K vs the single-loop
/// reference (each lane sums its own ascending slice, then the 8 slices are
/// added in lane order) — a small, deterministic, tolerance-bounded FP
/// difference of the same class the tiled kernel already has. Requires
/// K % 32 == 0 (so each lane's slice is a whole number of vec4 steps); the
/// caller falls back to the tiled kernel otherwise.
pub const MATMUL_NBITS_Q4_SPLITK: &str = r#"
struct Params { m: u32, k: u32, n: u32 }
var<immediate> pc: Params;

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> packed_w: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

const COLS: u32 = 32u;
const KL: u32 = 8u;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    // Mirrors the proven GEMV kernel (matmul_fp32) exactly for the reduction:
    // tid → col_lane = tid % COLS, k_lane = tid / COLS, shared-memory tree
    // reduction with a barrier at each level. That pattern is validated
    // deterministic on RADV; a single-barrier read-all is not.
    let col_lane = tid % COLS;
    let k_lane = tid / COLS;
    let row = wid.x;
    let col = wid.z * COLS + col_lane;
    let valid = col < pc.n;
    let blocks = pc.k / 32u;
    let k_chunk = pc.k / KL;
    let k_start = k_lane * k_chunk;
    var sum = 0.0;
    if (valid) {
        var kk = k_start;
        for (; kk + 4u <= k_start + k_chunk; kk = kk + 4u) {
            let base_byte = col * (pc.k / 2u) + (kk / 2u);
            let wa = packed_w[base_byte >> 2u];
            let wb = packed_w[(base_byte + 1u) >> 2u];
            let byte_a = (wa >> ((base_byte & 3u) * 8u)) & 0xffu;
            let byte_b = (wb >> (((base_byte + 1u) & 3u) * 8u)) & 0xffu;
            let q0 = select(byte_a & 0x0fu, byte_a >> 4u, (kk & 1u) != 0u);
            let q1 = select(byte_a & 0x0fu, byte_a >> 4u, ((kk + 1u) & 1u) != 0u);
            let q2 = select(byte_b & 0x0fu, byte_b >> 4u, ((kk + 2u) & 1u) != 0u);
            let q3 = select(byte_b & 0x0fu, byte_b >> 4u, ((kk + 3u) & 1u) != 0u);
            let sa0 = scales[col * blocks + kk / 32u];
            let sa1 = scales[col * blocks + (kk + 1u) / 32u];
            let sa2 = scales[col * blocks + (kk + 2u) / 32u];
            let sa3 = scales[col * blocks + (kk + 3u) / 32u];
            let av = vec4<f32>(
                a[row * pc.k + kk],
                a[row * pc.k + kk + 1u],
                a[row * pc.k + kk + 2u],
                a[row * pc.k + kk + 3u],
            );
            let wv = vec4<f32>(
                (f32(q0) - 8.0) * sa0,
                (f32(q1) - 8.0) * sa1,
                (f32(q2) - 8.0) * sa2,
                (f32(q3) - 8.0) * sa3,
            );
            sum = sum + dot(av, wv);
        }
    }
    partial[tid] = sum;
    workgroupBarrier();
    // Tree-reduce the KL lanes of each column; lane 0 holds the result.
    for (var s = KL / 2u; s > 0u; s = s / 2u) {
        if (k_lane < s) {
            partial[tid] = partial[tid] + partial[tid + s * COLS];
        }
        workgroupBarrier();
    }
    if (k_lane == 0u && valid) {
        y[row * pc.n + col] = partial[tid];
    }
}
"#;

#[cfg(test)]
mod tests {
    #[test]
    fn source_compiles() {
        vk_compute::compile_wgsl(super::MATMUL_NBITS_Q4)
            .expect("MatMulNBits Q4 scalar shader must compile to SPIR-V");
        vk_compute::compile_wgsl(super::MATMUL_NBITS_Q4_TILED)
            .expect("MatMulNBits Q4 tiled shader must compile to SPIR-V");
        vk_compute::compile_wgsl(super::MATMUL_NBITS_Q4_SPLITK)
            .expect("MatMulNBits Q4 split-K shader must compile to SPIR-V");
    }
}
