//! End-to-end test on GPU (or lavapipe): upload → dispatch → readback → CPU comparison.

use vk_compute::{VkContext, compile_wgsl};

fn as_bytes<T: Copy>(data: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), std::mem::size_of_val(data)) }
}

fn from_bytes<T: Copy>(data: &[u8]) -> Vec<T> {
    assert_eq!(data.len() % std::mem::size_of::<T>(), 0);
    let n = data.len() / std::mem::size_of::<T>();
    let mut out = Vec::with_capacity(n);
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr().cast::<T>(), out.as_mut_ptr(), n);
        out.set_len(n);
    }
    out
}

const ADD_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Push { n: u32 }
var<immediate> pc: Push;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < pc.n) {
        out[gid.x] = a[gid.x] + b[gid.x];
    }
}
"#;

#[test]
fn add_f32() {
    let ctx = VkContext::new().expect("contesto Vulkan");
    let n = 4099usize; // not a multiple of the workgroup: verifies the bounds check

    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
    let b: Vec<f32> = (0..n).map(|i| 100.0 - i as f32).collect();
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();

    let spirv = compile_wgsl(ADD_WGSL).expect("WGSL compilation");
    let pipeline = ctx.create_pipeline(&spirv, 3, 4).unwrap();

    let size = (n * 4) as u64;
    let buf_a = ctx.create_storage_buffer(size).unwrap();
    let buf_b = ctx.create_storage_buffer(size).unwrap();
    let buf_out = ctx.create_storage_buffer(size).unwrap();
    ctx.upload(&buf_a, as_bytes(&a)).unwrap();
    ctx.upload(&buf_b, as_bytes(&b)).unwrap();

    let groups = [(n as u32).div_ceil(256), 1, 1];
    ctx.dispatch(
        &pipeline,
        &[&buf_a, &buf_b, &buf_out],
        &(n as u32).to_le_bytes(),
        groups,
    )
    .unwrap();

    let result: Vec<f32> = from_bytes(&ctx.download(&buf_out).unwrap());
    assert_eq!(result, expected);

    ctx.destroy_buffer(buf_a);
    ctx.destroy_buffer(buf_b);
    ctx.destroy_buffer(buf_out);
    ctx.destroy_pipeline(pipeline);
}

/// ONNX-style MatMulInteger: dynamic A u8 [M,K], constant B u8 [K,N],
/// scalar zero points, i32 [M,N] output. B packed by column:
/// b_packed[col][i] = 4 bytes along K.
const MATMUL_U8_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<u32>;   // [M, K/4] packed
@group(0) @binding(1) var<storage, read> b: array<u32>;   // [N, K/4] packed (B trasposta)
@group(0) @binding(2) var<storage, read_write> out: array<i32>; // [M, N]

struct Push { m: u32, k: u32, n: u32, a_zp: i32, b_zp: i32 }
var<immediate> pc: Push;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if (col >= pc.n || row >= pc.m) {
        return;
    }
    let k4 = pc.k / 4u;
    var acc: i32 = 0;
    for (var i = 0u; i < k4; i = i + 1u) {
        let av = vec4<i32>(unpack4xU8(a[row * k4 + i])) - vec4<i32>(pc.a_zp);
        let bv = vec4<i32>(unpack4xU8(b[col * k4 + i])) - vec4<i32>(pc.b_zp);
        acc = acc + dot(av, bv);
    }
    out[row * pc.n + col] = acc;
}
"#;

#[test]
fn matmul_integer_u8() {
    let ctx = VkContext::new().expect("contesto Vulkan");
    let (m, k, n) = (17usize, 64usize, 23usize); // K is a multiple of 4
    let a_zp: u8 = 121;
    let b_zp: u8 = 130;

    // deterministic pseudo-random data
    let a: Vec<u8> = (0..m * k).map(|i| ((i * 37 + 11) % 251) as u8).collect();
    let b: Vec<u8> = (0..k * n).map(|i| ((i * 53 + 7) % 249) as u8).collect();

    // CPU reference
    let mut expected = vec![0i32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0i32;
            for i in 0..k {
                acc +=
                    (a[row * k + i] as i32 - a_zp as i32) * (b[i * n + col] as i32 - b_zp as i32);
            }
            expected[row * n + col] = acc;
        }
    }

    // packing: A row-major along K; B transposed [N][K] then packed along K
    let a_packed: Vec<u32> = a
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut b_t = vec![0u8; n * k];
    for i in 0..k {
        for col in 0..n {
            b_t[col * k + i] = b[i * n + col];
        }
    }
    let b_packed: Vec<u32> = b_t
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let spirv = compile_wgsl(MATMUL_U8_WGSL).expect("WGSL compilation");
    let pipeline = ctx.create_pipeline(&spirv, 3, 20).unwrap();

    let buf_a = ctx
        .create_storage_buffer((a_packed.len() * 4) as u64)
        .unwrap();
    let buf_b = ctx
        .create_storage_buffer((b_packed.len() * 4) as u64)
        .unwrap();
    let buf_out = ctx.create_storage_buffer((m * n * 4) as u64).unwrap();
    ctx.upload(&buf_a, as_bytes(&a_packed)).unwrap();
    ctx.upload(&buf_b, as_bytes(&b_packed)).unwrap();

    let mut push = Vec::with_capacity(20);
    for v in [m as u32, k as u32, n as u32] {
        push.extend_from_slice(&v.to_le_bytes());
    }
    for v in [a_zp as i32, b_zp as i32] {
        push.extend_from_slice(&v.to_le_bytes());
    }

    let groups = [(n as u32).div_ceil(16), (m as u32).div_ceil(16), 1];
    ctx.dispatch(&pipeline, &[&buf_a, &buf_b, &buf_out], &push, groups)
        .unwrap();

    let result: Vec<i32> = from_bytes(&ctx.download(&buf_out).unwrap());
    assert_eq!(result, expected);

    ctx.destroy_buffer(buf_a);
    ctx.destroy_buffer(buf_b);
    ctx.destroy_buffer(buf_out);
    ctx.destroy_pipeline(pipeline);
}

/// Variant with `dot4U8Packed` + zero-point correction:
/// Σ(aᵢ−az)(bᵢ−bz) = Σaᵢbᵢ − az·Σbᵢ − bz·Σaᵢ + 4·az·bz  (per block of 4)
/// naga polyfills the builtin if the SPIR-V target < 1.6, so the path
/// stays portable; on drivers with VK_KHR_shader_integer_dot_product it
/// becomes a native instruction.
const MATMUL_U8_DOT4_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<i32>;

struct Push { m: u32, k: u32, n: u32, a_zp: i32, b_zp: i32 }
var<immediate> pc: Push;

fn sum4_u8(v: u32) -> i32 {
    let u = unpack4xU8(v);
    return i32(u.x + u.y + u.z + u.w);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if (col >= pc.n || row >= pc.m) {
        return;
    }
    let k4 = pc.k / 4u;
    var acc: i32 = 0;
    for (var i = 0u; i < k4; i = i + 1u) {
        let av = a[row * k4 + i];
        let bv = b[col * k4 + i];
        acc = acc + i32(dot4U8Packed(av, bv))
            - pc.a_zp * sum4_u8(bv)
            - pc.b_zp * sum4_u8(av)
            + 4 * pc.a_zp * pc.b_zp;
    }
    out[row * pc.n + col] = acc;
}
"#;

#[test]
fn matmul_integer_u8_dot4() {
    let ctx = VkContext::new().expect("contesto Vulkan");
    let (m, k, n) = (17usize, 64usize, 23usize);
    let a_zp: u8 = 121;
    let b_zp: u8 = 130;

    let a: Vec<u8> = (0..m * k).map(|i| ((i * 37 + 11) % 251) as u8).collect();
    let b: Vec<u8> = (0..k * n).map(|i| ((i * 53 + 7) % 249) as u8).collect();

    let mut expected = vec![0i32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0i32;
            for i in 0..k {
                acc +=
                    (a[row * k + i] as i32 - a_zp as i32) * (b[i * n + col] as i32 - b_zp as i32);
            }
            expected[row * n + col] = acc;
        }
    }

    let a_packed: Vec<u32> = a
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut b_t = vec![0u8; n * k];
    for i in 0..k {
        for col in 0..n {
            b_t[col * k + i] = b[i * n + col];
        }
    }
    let b_packed: Vec<u32> = b_t
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let spirv = compile_wgsl(MATMUL_U8_DOT4_WGSL).expect("WGSL dot4 compilation");
    let pipeline = ctx.create_pipeline(&spirv, 3, 20).unwrap();

    let buf_a = ctx
        .create_storage_buffer((a_packed.len() * 4) as u64)
        .unwrap();
    let buf_b = ctx
        .create_storage_buffer((b_packed.len() * 4) as u64)
        .unwrap();
    let buf_out = ctx.create_storage_buffer((m * n * 4) as u64).unwrap();
    ctx.upload(&buf_a, as_bytes(&a_packed)).unwrap();
    ctx.upload(&buf_b, as_bytes(&b_packed)).unwrap();

    let mut push = Vec::with_capacity(20);
    for v in [m as u32, k as u32, n as u32] {
        push.extend_from_slice(&v.to_le_bytes());
    }
    for v in [a_zp as i32, b_zp as i32] {
        push.extend_from_slice(&v.to_le_bytes());
    }

    let groups = [(n as u32).div_ceil(16), (m as u32).div_ceil(16), 1];
    ctx.dispatch(&pipeline, &[&buf_a, &buf_b, &buf_out], &push, groups)
        .unwrap();

    let result: Vec<i32> = from_bytes(&ctx.download(&buf_out).unwrap());
    assert_eq!(result, expected);

    ctx.destroy_buffer(buf_a);
    ctx.destroy_buffer(buf_b);
    ctx.destroy_buffer(buf_out);
    ctx.destroy_pipeline(pipeline);
}

#[test]
fn enumerate_devices() {
    let devices = VkContext::enumerate_devices().expect("enumerazione device");
    assert!(!devices.is_empty(), "no Vulkan device found");
    for (name, vendor, ty) in &devices {
        eprintln!("device: {name} (vendor 0x{vendor:04x}, {ty:?})");
    }
}

// --- MatMulNBits Q4: scalar vs tiled must agree (isolates tiled-kernel bugs) ---

const NBITS_SCALAR: &str = r#"
struct Params { m: u32, k: u32, n: u32 }
var<immediate> pc: Params;
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> packed_w: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
fn pbyte(bi: u32) -> u32 { let w = packed_w[bi >> 2u]; return (w >> ((bi & 3u) * 8u)) & 0xffu; }
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let oi = gid.x;
    if (oi >= pc.m * pc.n) { return; }
    let row = oi / pc.n; let col = oi - row * pc.n;
    let blocks = pc.k / 32u; var sum = 0.0;
    for (var kk = 0u; kk < pc.k; kk = kk + 1u) {
        let bi = col * (pc.k / 2u) + (kk / 2u);
        let byte = pbyte(bi);
        let q = select(byte & 0x0fu, byte >> 4u, (kk & 1u) != 0u);
        let scale = scales[col * blocks + kk / 32u];
        sum = sum + a[row * pc.k + kk] * (f32(q) - 8.0) * scale;
    }
    y[oi] = sum;
}
"#;

const NBITS_TILED: &str = r#"
struct Params { m: u32, k: u32, n: u32 }
var<immediate> pc: Params;
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> packed_w: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
fn nibble_at(col: u32, kk: u32) -> f32 {
    let bi = col * (pc.k / 2u) + (kk / 2u);
    let word = packed_w[bi >> 2u];
    let byte = (word >> ((bi & 3u) * 8u)) & 0xffu;
    let q = select(byte & 0x0fu, byte >> 4u, (kk & 1u) != 0u);
    return (f32(q) - 8.0) * scales[col * (pc.k / 32u) + kk / 32u];
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let oi = gid.x;
    if (oi >= pc.m * pc.n) { return; }
    let row = oi / pc.n; let col = oi - row * pc.n;
    let blocks = pc.k / 32u;
    var sum = 0.0;
    var kk = 0u;
    for (; kk + 4u <= pc.k; kk = kk + 4u) {
        let base = col * (pc.k / 2u) + (kk / 2u);
        let wa = packed_w[base >> 2u];
        let wb = packed_w[(base + 1u) >> 2u];
        let ba = (wa >> ((base & 3u) * 8u)) & 0xffu;
        let bb = (wb >> (((base + 1u) & 3u) * 8u)) & 0xffu;
        let q0 = select(ba & 0x0fu, ba >> 4u, (kk & 1u) != 0u);
        let q1 = select(ba & 0x0fu, ba >> 4u, ((kk + 1u) & 1u) != 0u);
        let q2 = select(bb & 0x0fu, bb >> 4u, ((kk + 2u) & 1u) != 0u);
        let q3 = select(bb & 0x0fu, bb >> 4u, ((kk + 3u) & 1u) != 0u);
        let sa0 = scales[col * blocks + kk / 32u];
        let sa1 = scales[col * blocks + (kk + 1u) / 32u];
        let sa2 = scales[col * blocks + (kk + 2u) / 32u];
        let sa3 = scales[col * blocks + (kk + 3u) / 32u];
        let av = vec4<f32>(a[row * pc.k + kk], a[row * pc.k + kk + 1u],
                           a[row * pc.k + kk + 2u], a[row * pc.k + kk + 3u]);
        let wv = vec4<f32>((f32(q0) - 8.0) * sa0, (f32(q1) - 8.0) * sa1,
                            (f32(q2) - 8.0) * sa2, (f32(q3) - 8.0) * sa3);
        sum = sum + dot(av, wv);
    }
    for (; kk < pc.k; kk = kk + 1u) {
        sum = sum + a[row * pc.k + kk] * nibble_at(col, kk);
    }
    y[oi] = sum;
}
"#;

fn run_nbits(
    ctx: &VkContext,
    wgsl: &str,
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    w: &[u8],
    scales: &[f32],
) -> Vec<f32> {
    let scalar = compile_wgsl(wgsl).expect("compile");
    let pipeline = ctx.create_pipeline(&scalar, 4, 12).unwrap();
    // w: [N, K/2] u8 packed (low nibble first). scales: [N, K/32].
    let w_u32: Vec<u32> = w
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let buf_a = ctx.create_storage_buffer((a.len() * 4) as u64).unwrap();
    let buf_w = ctx.create_storage_buffer((w_u32.len() * 4) as u64).unwrap();
    let buf_s = ctx.create_storage_buffer((scales.len() * 4) as u64).unwrap();
    let buf_o = ctx.create_storage_buffer((m * n * 4) as u64).unwrap();
    ctx.upload(&buf_a, as_bytes(a)).unwrap();
    ctx.upload(&buf_w, as_bytes(&w_u32)).unwrap();
    ctx.upload(&buf_s, as_bytes(scales)).unwrap();
    let mut push = Vec::with_capacity(12);
    for v in [m as u32, k as u32, n as u32] {
        push.extend_from_slice(&v.to_le_bytes());
    }
    // Both kernels are @workgroup_size(256), one thread per output element.
    let groups = [((m * n) as u32).div_ceil(256), 1, 1];
    ctx.dispatch(&pipeline, &[&buf_a, &buf_w, &buf_s, &buf_o], &push, groups)
        .unwrap();
    let out: Vec<f32> = from_bytes(&ctx.download(&buf_o).unwrap());
    ctx.destroy_buffer(buf_a);
    ctx.destroy_buffer(buf_w);
    ctx.destroy_buffer(buf_s);
    ctx.destroy_buffer(buf_o);
    ctx.destroy_pipeline(pipeline);
    out
}

#[test]
fn matmul_nbits_q4_scalar_vs_tiled() {
    // Cover shapes that exercise the 32-element scale-block boundary (K>=32,
    // including K straddling the boundary mid vec4-group) and non-multiples of
    // 4 (scalar remainder path).
    let shapes: &[(usize, usize, usize)] = &[
        (4, 16, 96),
        (1, 32, 8),     // K exactly one scale block
        (1, 34, 8),     // K straddles block, not multiple of 4
        (2, 64, 48),
        (1, 300, 16),   // K straddles block mid vec4-group (kk=29 -> q3 in next block)
        (3, 2048, 512), // depthformer-class shape
    ];
    // Single context: the host's VkContext create/drop crashes when repeated,
    // so reuse one for all shapes and print a clear per-shape result.
    let ctx = VkContext::new().expect("Vulkan context");
    for &(m, k, n) in shapes {
        let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.13).sin()).collect();
        let w: Vec<u8> = (0..n * k / 2)
            .map(|i| ((i * 31 + 5) & 0xff) as u8)
            .collect();
        let scales: Vec<f32> = (0..n * k / 32)
            .map(|i| 0.02 + (i % 7) as f32 * 0.01)
            .collect();
        let expected = run_nbits(&ctx, NBITS_SCALAR, m, k, n, &a, &w, &scales);
        let tiled = run_nbits(&ctx, NBITS_TILED, m, k, n, &a, &w, &scales);
        assert_eq!(tiled.len(), expected.len(), "len mismatch m={m} k={k} n={n}");
        let mut max_diff = 0.0f32;
        for (i, (e, t)) in expected.iter().zip(&tiled).enumerate() {
            let d = (e - t).abs();
            if d > max_diff {
                max_diff = d;
                if d > 1e-2 {
                    eprintln!(
                        "mismatch m={m} k={k} n={n} at {i}: expected {e} tiled {t} (diff {d})"
                    );
                }
            }
        }
        assert!(
            max_diff < 1e-2,
            "tiled kernel diverges m={m} k={k} n={n}: max_diff={max_diff}"
        );
        eprintln!("PASS m={m} k={k} n={n} max_diff={max_diff}");
    }
}
