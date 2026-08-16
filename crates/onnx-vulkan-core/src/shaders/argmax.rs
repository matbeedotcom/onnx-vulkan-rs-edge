//! ArgMax as a device op.
//!
//! One workgroup of 256 threads cooperatively reduces ONE row to the index of
//! its maximum: each thread strided-scans the row (elements tid, tid+256, ...)
//! keeping its local (value, index), then a shared-memory tree merges the 256
//! partials. Ties keep the LOWEST index (ONNX ArgMax semantics); NaN loses the
//! `>` comparison, so it is never selected unless it is the only candidate —
//! deterministic.
//!
//! The unrolled depthformer chains 8 in-graph ArgMax nodes; executing them on
//! device keeps `token_id` in VRAM so the 8-step codebook loop costs one graph
//! submit instead of 8 host round-trips. The decoder argmax reduces a single
//! [vocab] row the same way.

pub const BINDINGS: u32 = 2;
/// 4 u32 fields in the push constant struct.
pub const PUSH_BYTES: u32 = 16;
pub const WORKGROUP_SIZE: u32 = 256;

pub const ARGMAX: &str = r#"
struct Push { c: u32, inner: u32, rows: u32, stride_y: u32 }
var<immediate> pc: Push;

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<i64>;

var<workgroup> best: array<f32, 256>;
var<workgroup> idx: array<u32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let r = wid.y * pc.stride_y + wid.x;      // one row per workgroup
    if (r >= pc.rows) { return; }
    let j = r % pc.inner;
    let outer = r / pc.inner;
    let base = outer * pc.c * pc.inner + j;

    var bv = -3.4028235e38;
    var bi = 0u;
    // strided scan of the row: 256 lanes cover c/256 elements each.
    for (var k = tid; k < pc.c; k = k + 256u) {
        let v = x[base + k * pc.inner];
        if (v > bv) { bv = v; bi = k; }
    }
    best[tid] = bv;
    idx[tid] = bi;
    workgroupBarrier();
    // tree-reduce the 256 partials (workgroupBarrier per level: proven
    // deterministic on RADV, same pattern as matmul_nbits / GEMV). Ties keep
    // the lowest element index.
    for (var s = 128u; s > 0u; s = s / 2u) {
        if (tid < s) {
            let o = tid + s;
            if (best[o] > best[tid] || (best[o] == best[tid] && idx[o] < idx[tid])) {
                best[tid] = best[o];
                idx[tid] = idx[o];
            }
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        out[r] = i64(idx[0]);
    }
}
"#;

#[cfg(test)]
mod tests {
    #[test]
    fn source_compiles() {
        vk_compute::compile_wgsl(super::ARGMAX).expect("argmax shader valido");
    }
}
