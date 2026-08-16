//! Microbenchmark: which inner-loop ALU path is fastest for a Q4 GEMM, per
//! output element, on THIS device?
//!
//! Three kernels each accumulate over the same K per thread and discard
//! everything but one output per thread (to defeat DCE). They isolate the
//! inner-loop cost that dominates the Q4 GEMMs — the dequant + accumulate —
//! with no staging, barrier, or occupancy noise:
//!
//!   float-scalar : the current split-K inner loop (per Q4 elem: shift, mask,
//!                  sub(-8), mul(block-scale), fma).  ~= 5 ALU / elem.
//!   float-vec4   : same math, 4 elems per instruction (SIMD FMA).
//!   int-dot      : weights pre-expanded Q4 -> int8 (once per session),
//!                  activations quantized to int8; inner loop is one
//!                  `dot4I8Packed` (OpSDot4) per 4 elems + a per-32-block
//!                  rescale.  ~= 1 dot / 4 elem.
//!
//! If int-dot is not clearly faster than float-vec4, the Q4xQ8 int-dot kernel
//! plan is dead: build nothing. Run on the Deck (RADV Van Gogh):
//!   cargo run --release -p vk-compute --example intdot_microbench
//!
//! On a device without the integer-dot extension the int variant is skipped
//! (naga lowers the builtin, but SPIR-V rejects OpSDot4 at pipeline create).

use std::time::Instant;
use vk_compute::{VkContext, compile_wgsl};

const K: u32 = 2048; // inner-loop length per thread (matches the k=2048 shapes)
const THREADS: u32 = 256;
const WORKGROUPS: u32 = 2048; // enough to fill the GPU many waves over
const REPS: u32 = 32;

/// float-scalar inner loop (the current Q4 split-K path, per output element).
const SRC_FLOAT_SCALAR: &str = r#"
struct Push { k: u32 }
var<immediate> pc: Push;
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> wp: array<u32>; // packed Q4: 8 nibbles/word
@group(0) @binding(2) var<storage, read> scales: array<f32>; // per 32-block
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    let nblk = pc.k / 32u;
    var acc = 0.0;
    for (var b = 0u; b < nblk; b = b + 1u) {
        let scale = scales[b];
        for (var j = 0u; j < 32u; j = j + 1u) {
            let kk = b * 32u + j;
            let byte = (wp[kk / 2u] >> ((kk & 1u) * 4u)) & 0x0fu;
            let w = (f32(byte) - 8.0) * scale;
            acc = fma(a[kk], w, acc);
        }
    }
    out[t] = acc;
}
"#;

/// float-vec4 inner loop: 4 Q4 elems per instruction (SIMD unpack + fma).
const SRC_FLOAT_VEC4: &str = r#"
struct Push { k: u32 }
var<immediate> pc: Push;
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> wp: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    let nblk = pc.k / 32u;
    var acc = vec4<f32>(0.0);
    for (var b = 0u; b < nblk; b = b + 1u) {
        let scale = scales[b];
        for (var j = 0u; j < 8u; j = j + 1u) { // 8 x 4 = 32 elems per block
            let kk = b * 32u + j * 4u;
            let w0 = wp[kk / 2u];
            let w1 = wp[kk / 2u + 1u];
            let wv = vec4<f32>(
                (f32(w0 & 0x0fu) - 8.0) * scale,
                (f32((w0 >> 4u) & 0x0fu) - 8.0) * scale,
                (f32(w1 & 0x0fu) - 8.0) * scale,
                (f32((w1 >> 4u) & 0x0fu) - 8.0) * scale);
            let av = vec4<f32>(a[kk], a[kk + 1u], a[kk + 2u], a[kk + 3u]);
            acc = fma(av, wv, acc);
        }
    }
    out[t] = acc[0] + acc[1] + acc[2] + acc[3];
}
"#;

/// int-dot inner loop: weights pre-expanded to int8 (4/word), activations to
/// int8 (4/word); one dot4I8Packed per 4 elems + per-32-block rescale.
const SRC_INT_DOT: &str = r#"
struct Push { k: u32 }
var<immediate> pc: Push;
@group(0) @binding(0) var<storage, read> ai8: array<u32>; // 4 activations/word
@group(0) @binding(1) var<storage, read> wi8: array<u32>; // 4 int8 weights/word
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    let nblk = pc.k / 32u;
    var acc = 0.0;
    for (var b = 0u; b < nblk; b = b + 1u) {
        let scale = scales[b];
        var s = 0;
        for (var j = 0u; j < 8u; j = j + 1u) { // 8 x 4 = 32 int8 per block
            let w = b * 8u + j;
            s = s + dot4I8Packed(ai8[w], wi8[w]);
        }
        acc = acc + f32(s) * scale;
    }
    out[t] = acc;
}
"#;

fn bench(
    ctx: &VkContext,
    src: &str,
    name: &str,
    abuf: &vk_compute::GpuBuffer,
    wbuf: &vk_compute::GpuBuffer,
    sbuf: &vk_compute::GpuBuffer,
    obuf: &vk_compute::GpuBuffer,
) -> Option<f64> {
    let spirv = match compile_wgsl(src) {
        Ok(s) => s,
        Err(e) => {
            println!("  {name:14}: COMPILE FAIL: {e}");
            return None;
        }
    };
    let pipe = match ctx.create_pipeline(&spirv, 4, 4) {
        Ok(p) => p,
        Err(e) => {
            println!("  {name:14}: PIPELINE FAIL (no OpSDot4 here?): {e}");
            return None;
        }
    };
    let bufs: [&vk_compute::GpuBuffer; 4] = [abuf, wbuf, sbuf, obuf];
    let mut push = vec![];
    push.extend_from_slice(&K.to_le_bytes());
    // warm
    let _ = ctx.stream_dispatch(&pipe, &bufs, &push, [THREADS * WORKGROUPS, 1, 1]);
    let _ = ctx.flush_wait();
    let t0 = Instant::now();
    let mut ok = true;
    for _ in 0..REPS {
        if ctx.stream_dispatch(&pipe, &bufs, &push, [THREADS * WORKGROUPS, 1, 1]).is_err() {
            ok = false;
            break;
        }
    }
    let _ = ctx.flush_wait();
    if !ok {
        return None;
    }
    let dt = t0.elapsed().as_secs_f64() / REPS as f64;
    ctx.destroy_pipeline(pipe);
    Some(dt)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = VkContext::new()?;
    println!("device: {}  integer_dot_product={:?}", ctx.device_name, ctx.has_integer_dot_product);
    println!("K={K} per thread, {WORKGROUPS} workgroups x {THREADS} threads, {REPS} reps\n");

    // float data
    let a_f: Vec<f32> = (0..(K as usize)).map(|i| (i as f32) * 0.001 - 1.0).collect();
    let wp: Vec<u32> = (0..(K as usize / 2)).map(|i| (0x1A2B * (i as u32)).wrapping_mul(3)).collect();
    let scales: Vec<f32> = vec![1.5; (K / 32) as usize];
    // int data (4 int8 per word)
    let ai8: Vec<u32> = (0..(K as usize / 4)).map(|_| 0x4040_4040u32).collect();
    let wi8: Vec<u32> = (0..(K as usize / 4)).map(|i| (0x1234_5678u32 ^ (i as u32))).collect();

    let a_buf = ctx.create_storage_buffer((a_f.len() * 4) as u64)?;
    let w_buf = ctx.create_storage_buffer((wp.len() * 4) as u64)?;
    let s_buf = ctx.create_storage_buffer((scales.len() * 4) as u64)?;
    let o_buf = ctx.create_storage_buffer((THREADS * WORKGROUPS * 4) as u64)?;
    ctx.upload(&a_buf, &a_f.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
    ctx.upload(&w_buf, &wp.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
    ctx.upload(&s_buf, &scales.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
    // int data
    let ai8_buf = ctx.create_storage_buffer((ai8.len() * 4) as u64)?;
    let wi8_buf = ctx.create_storage_buffer((wi8.len() * 4) as u64)?;
    ctx.upload(&ai8_buf, &ai8.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
    ctx.upload(&wi8_buf, &wi8.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;

    println!("{:14} {:>12}  (relative)", "kernel", "ms/rep");
    // Isolation gate: RADV loses the context when one kernel hangs, poisoning
    // the rest. INTDOT_BENCH=scalar|v4|int|all (default all).
    let which = std::env::var("INTDOT_BENCH").unwrap_or_else(|_| "all".to_string());
    let base = if which == "all" || which == "scalar" {
        bench(&ctx, SRC_FLOAT_SCALAR, "float-scalar", &a_buf, &w_buf, &s_buf, &o_buf)
            .unwrap_or(f64::NAN)
    } else {
        f64::NAN
    };
    let v4 = if which == "all" || which == "v4" {
        bench(&ctx, SRC_FLOAT_VEC4, "float-vec4", &a_buf, &w_buf, &s_buf, &o_buf)
            .unwrap_or(f64::NAN)
    } else {
        f64::NAN
    };
    let idot = if which == "all" || which == "int" {
        bench(&ctx, SRC_INT_DOT, "int-dot", &ai8_buf, &wi8_buf, &s_buf, &o_buf)
            .unwrap_or(f64::NAN)
    } else {
        f64::NAN
    };

    if base.is_finite() {
        println!("float-scalar : {base:>12.4}  x1.00");
    }
    if v4.is_finite() {
        let r = if base.is_finite() { base / v4 } else { f64::NAN };
        println!("float-vec4   : {v4:>12.4}  x{r:.2}");
    }
    if idot.is_finite() {
        let r = if base.is_finite() { base / idot } else { f64::NAN };
        println!("int-dot      : {idot:>12.4}  x{r:.2}");
    }
    if idot.is_finite() && v4.is_finite() && idot < v4 {
        println!(
            "\nMEASURED: int-dot {:.2}x over float-vec4 here. CAVEAT: this bench\n\
             reads activations UNSHARED (float = 4 B/elem, int8 = 1 B/elem), so most\n\
             of the edge is activation traffic. The REAL split-K kernel L1-caches the\n\
             activation row across its 8 K-lanes, erasing most of that gap — so int-dot\n\
             is NOT a 2x ALU win on Van Gogh; it is a small, memory-dominated edge\n\
             that shrinks further inside the cached kernel.\n",
            v4 / idot
        );
    } else if idot.is_finite() {
        println!("\nMEASURED: int-dot not faster than float-vec4.");
    } else {
        println!("\nMEASURED: int-dot unavailable on this device; re-run on the Deck.");
    }

    for b in [a_buf, w_buf, s_buf, o_buf, ai8_buf, wi8_buf] {
        ctx.destroy_buffer(b);
    }
    Ok(())
}
