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

/// Q4 matmul with inter-workgroup split-K for the skinny-n / large-k decode
/// GEMV (e.g. m=1 n=2048 k=8192).
///
/// The intra-workgroup split-K kernel above launches `ceil(n/32)` workgroups;
/// for small `n` that is far too few to fill Vangogh's 20 CUs, so each one
/// carries a long dependent K chain and the op is latency-bound (n=2048,
/// k=8192 measured at 331 ms/op — ~12000x off the memory peak). This kernel
/// adds a `k_chunks` grid dimension: workgroup `(row, chunk, coltile)` reduces
/// only its K-slice (the same 32-col × 8-k-lane intra-workgroup tree-reduce as
/// before, but over `k / k_chunks / 8` vec4 steps instead of `k / 8`), then
/// writes the per-column partial to a scratch buffer. A second pass
/// (`MATMUL_NBITS_Q4_SPLITK_REDUCE`) sums the `k_chunks` partials in fixed
/// order. Both passes are deterministic: the intra-workgroup reduce reuses the
/// validated tree-reduce, and the cross-chunk sum is a fixed-order sequential
/// add (chunk 0, 1, 2, ...), so the result is reproducible run-to-run.
///
/// `k_chunks` must divide `k / 32` (so every chunk's 8 k-lanes each get a whole
/// number of vec4 steps); the host guarantees that. Push is `{m,k,n,k_chunks}`.
pub const MATMUL_NBITS_Q4_SPLITK_DEEP: &str = r#"
struct Params { m: u32, k: u32, n: u32, k_chunks: u32 }
var<immediate> pc: Params;

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> packed_w: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> partial: array<f32>;

const COLS: u32 = 32u;
const KL: u32 = 8u;

var<workgroup> lds: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let col_lane = tid % COLS;
    let k_lane = tid / COLS;
    let row = wid.x;
    let chunk = wid.y;
    let col = wid.z * COLS + col_lane;
    let valid = col < pc.n;
    let blocks = pc.k / 32u;
    let slice = pc.k / pc.k_chunks;      // K extent of this chunk
    let lane_span = slice / KL;          // K extent of this k-lane
    let k_start = chunk * slice + k_lane * lane_span;
    var sum = 0.0;
    if (valid) {
        var kk = k_start;
        let k_end = k_start + lane_span;
        for (; kk + 4u <= k_end; kk = kk + 4u) {
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
    lds[tid] = sum;
    workgroupBarrier();
    // Intra-chunk tree-reduce across the 8 k-lanes (validated deterministic on
    // RADV); lane 0 of each column holds the chunk's partial.
    for (var s = KL / 2u; s > 0u; s = s / 2u) {
        if (k_lane < s) {
            lds[tid] = lds[tid] + lds[tid + s * COLS];
        }
        workgroupBarrier();
    }
    if (k_lane == 0u && valid) {
        partial[(row * pc.k_chunks + chunk) * pc.n + col] = lds[tid];
    }
}
"#;

/// Second pass of inter-workgroup split-K: sums the `k_chunks` per-column
/// partials (fixed chunk order) into the output. One thread per output element.
/// Push is `{m,k,n,k_chunks}` (only m, n, k_chunks are read).
pub const MATMUL_NBITS_Q4_SPLITK_REDUCE: &str = r#"
struct Params { m: u32, k: u32, n: u32, k_chunks: u32 }
var<immediate> pc: Params;

@group(0) @binding(0) var<storage, read> partial: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = pc.m * pc.n;
    if (idx >= total) { return; }
    let row = idx / pc.n;
    let col = idx - row * pc.n;
    var sum = 0.0;
    for (var c = 0u; c < pc.k_chunks; c++) {
        sum = sum + partial[(row * pc.k_chunks + c) * pc.n + col];
    }
    y[idx] = sum;
}
"#;

/// Push-constant size for the inter-workgroup split-K pair: {m,k,n,k_chunks}.
pub const SPLITK_DEEP_PUSH_BYTES: u32 = 16;
/// Bindings for the deep (pass-1) kernel: a, packed_w, scales, partial.
pub const SPLITK_DEEP_BINDINGS: u32 = 4;
/// Bindings for the reduce (pass-2) kernel: partial, y.
pub const SPLITK_REDUCE_BINDINGS: u32 = 2;

/// Small-M tiled Q4 GEMM for prefill (m=2..~144).
///
/// The split-K kernel is one-thread-per-output-column with an 8-way intra-K
/// reduction; for m>1 every row independently re-reads and re-unpacks the same
/// Q4 weights, which dominates (m=31, n=8192, k=2048 measured at 3390 ms/op).
/// This kernel stages a `BM=4` row × `BK=32` K-block tile of dequantized
/// weights in shared memory ONCE per k-block and does `BM` dot-products
/// (one per row) against it — the Q4 unpack (shift/mask/dequant-scale) is
/// amortized across 4 rows, and weights are read only once per
/// (rowtile, coltile) instead of once per row. Accumulation is in registers
/// across all K-blocks; the output is written exactly once at the end (no
/// global read-modify-write).
///
/// Layout: 256-thread workgroup (32 col-lanes × 8 staging threads). Grid:
/// (ceil(m/BM), 1, ceil(n/BN)) — each workgroup covers one BM×BN tile and loops
/// over all K-blocks. LDS: B tile [BK][BN]=1024 f32 + A tile [BM][BK]=128 f32.
///
/// Requires `k % 32 == 0` (K advances in whole scale-blocks) and m*4 to be the
/// padded row extent (rows >= m are masked). Push is `{m,k,n}`.
pub const MATMUL_NBITS_Q4_TILED_SMALL_M: &str = r#"
struct Params { m: u32, k: u32, n: u32 }
var<immediate> pc: Params;

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> packed_w: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

const BM: u32 = 4u;
const BK: u32 = 32u;
const BN: u32 = 32u;

var<workgroup> b_lds: array<f32, 1024>;
var<workgroup> a_lds: array<f32, 128>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let col_lane = tid % BN;
    let stage = tid / BN;        // 0..7
    let row0 = wid.x * BM;       // first output row of this tile
    let col0 = wid.z * BN;       // first output col of this tile
    let nblocks = pc.k / BK;

    var acc0 = 0.0;
    var acc1 = 0.0;
    var acc2 = 0.0;
    var acc3 = 0.0;

    for (var kb = 0u; kb < nblocks; kb = kb + 1u) {
        // --- Stage A: BM=4 rows x BK=32 cols = 128 f32; 16 per thread.
        // Thread `stage` handles k-cols [stage*4, stage*4+4) for all 4 rows.
        for (var r = 0u; r < BM; r = r + 1u) {
            let grow = row0 + r;
            var v0 = 0.0; var v1 = 0.0; var v2 = 0.0; var v3 = 0.0;
            if (grow < pc.m) {
                let base = grow * pc.k + kb * BK;
                v0 = a[base + stage * 4u + 0u];
                v1 = a[base + stage * 4u + 1u];
                v2 = a[base + stage * 4u + 2u];
                v3 = a[base + stage * 4u + 3u];
            }
            a_lds[r * BK + stage * 4u + 0u] = v0;
            a_lds[r * BK + stage * 4u + 1u] = v1;
            a_lds[r * BK + stage * 4u + 2u] = v2;
            a_lds[r * BK + stage * 4u + 3u] = v3;
        }
        // --- Stage B (dequant): BK=32 K-rows x BN=32 cols = 1024 f32;
        // 128 per thread... no, 4 rows x 32 cols = 128 per thread? 256 threads
        // x 4 = 1024. Thread `stage` handles K-rows [stage*4, stage*4+4) for
        // all 32 cols. Each B element = one Q4 nibble from packed_w + one scale.
        for (var rr = 0u; rr < 4u; rr = rr + 1u) {
            let brow = kb * BK + stage * 4u + rr;  // global K index
            for (var c = 0u; c < BN; c = c + 1u) {
                let col = col0 + c;
                if (col >= pc.n) {
                    b_lds[(stage * 4u + rr) * BN + c] = 0.0;
                    continue;
                }
                let byte_index = col * (pc.k / 2u) + (brow / 2u);
                let word = packed_w[byte_index >> 2u];
                let byte = (word >> ((byte_index & 3u) * 8u)) & 0xffu;
                let q = select(byte & 0x0fu, byte >> 4u, (brow & 1u) != 0u);
                let scale = scales[col * (pc.k / 32u) + kb];
                b_lds[(stage * 4u + rr) * BN + c] = (f32(q) - 8.0) * scale;
            }
        }
        workgroupBarrier();
        // --- Inner product: BM=4 dot-products, one per row, 32 elements each.
        // a_lds[row][kk] broadcast across threads; b_lds[kk][col_lane] is
        // each thread's own column. 4 rows x 32 k = 128 FMA per thread.
        for (var kk = 0u; kk < BK; kk = kk + 1u) {
            let bcol = b_lds[kk * BN + col_lane];
            acc0 = fma(a_lds[0u * BK + kk], bcol, acc0);
            acc1 = fma(a_lds[1u * BK + kk], bcol, acc1);
            acc2 = fma(a_lds[2u * BK + kk], bcol, acc2);
            acc3 = fma(a_lds[3u * BK + kk], bcol, acc3);
        }
        workgroupBarrier();
    }

    // --- Write the 4-row x 1-col micro-tile once (no global RMW).
    let gcol = col0 + col_lane;
    if (gcol >= pc.n) { return; }
    if (row0 + 0u < pc.m) { y[(row0 + 0u) * pc.n + gcol] = acc0; }
    if (row0 + 1u < pc.m) { y[(row0 + 1u) * pc.n + gcol] = acc1; }
    if (row0 + 2u < pc.m) { y[(row0 + 2u) * pc.n + gcol] = acc2; }
    if (row0 + 3u < pc.m) { y[(row0 + 3u) * pc.n + gcol] = acc3; }
}
"#;

/// Bindings for the small-M tiled kernel: a, packed_w, scales, y.
pub const TILED_SMALL_M_BINDINGS: u32 = 4;
/// Push-constant size for the small-M tiled kernel: {m,k,n} (12 bytes).
pub const TILED_SMALL_M_PUSH_BYTES: u32 = 12;
/// BM (rows per tile) for the small-M tiled kernel.
pub const TILED_SMALL_M_BM: u32 = 4;

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
        vk_compute::compile_wgsl(super::MATMUL_NBITS_Q4_SPLITK_DEEP)
            .expect("MatMulNBits Q4 split-K-deep (pass 1) must compile to SPIR-V");
        vk_compute::compile_wgsl(super::MATMUL_NBITS_Q4_SPLITK_REDUCE)
            .expect("MatMulNBits Q4 split-K-deep reduce (pass 2) must compile to SPIR-V");
        vk_compute::compile_wgsl(super::MATMUL_NBITS_Q4_TILED_SMALL_M)
            .expect("MatMulNBits Q4 small-M tiled shader must compile to SPIR-V");
    }
}
