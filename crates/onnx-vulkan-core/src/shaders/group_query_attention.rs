//! Correctness-first `com.microsoft::GroupQueryAttention` for autoregressive
//! LFM2.5 decoding. One workgroup owns one `(batch, query, query-head)`.

pub const BINDINGS: u32 = 10;
pub const PUSH_BYTES: u32 = 36;
pub const MAX_CONTEXT: u32 = 4096;

pub const GQA: &str = r#"
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> past_k: array<f32>;
@group(0) @binding(4) var<storage, read> past_v: array<f32>;
@group(0) @binding(5) var<storage, read> cos_cache: array<f32>;
@group(0) @binding(6) var<storage, read> sin_cache: array<f32>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<storage, read_write> present_k: array<f32>;
@group(0) @binding(9) var<storage, read_write> present_v: array<f32>;

struct Push {
    batch: u32, seq: u32, q_heads: u32, kv_heads: u32, head_dim: u32,
    past_len: u32, total_len: u32, scale: f32, do_rotary: u32,
}
var<immediate> pc: Push;
var<workgroup> scores: array<f32, 4096>;
var<workgroup> reduce: array<f32, 256>;

fn rotate_value(value: f32, other: f32, d: u32, position: u32) -> f32 {
    if (pc.do_rotary == 0u) { return value; }
    let half = pc.head_dim / 2u;
    let cache_index = position * half + (d % half);
    let signed_other = select(-other, other, d >= half);
    return value * cos_cache[cache_index] + signed_other * sin_cache[cache_index];
}

fn q_value(b: u32, s: u32, h: u32, d: u32) -> f32 {
    let base = ((b * pc.seq + s) * pc.q_heads + h) * pc.head_dim;
    let half = pc.head_dim / 2u;
    let other_d = select(d + half, d - half, d >= half);
    return rotate_value(q[base + d], q[base + other_d], d, pc.past_len + s);
}

fn current_k_value(b: u32, s: u32, h: u32, d: u32) -> f32 {
    let base = ((b * pc.seq + s) * pc.kv_heads + h) * pc.head_dim;
    let half = pc.head_dim / 2u;
    let other_d = select(d + half, d - half, d >= half);
    return rotate_value(k[base + d], k[base + other_d], d, pc.past_len + s);
}

fn key_value(b: u32, kh: u32, position: u32, d: u32) -> f32 {
    if (position < pc.past_len) {
        return past_k[((b * pc.kv_heads + kh) * pc.past_len + position) * pc.head_dim + d];
    }
    return current_k_value(b, position - pc.past_len, kh, d);
}

fn value_value(b: u32, kh: u32, position: u32, d: u32) -> f32 {
    if (position < pc.past_len) {
        return past_v[((b * pc.kv_heads + kh) * pc.past_len + position) * pc.head_dim + d];
    }
    let s = position - pc.past_len;
    return v[((b * pc.seq + s) * pc.kv_heads + kh) * pc.head_dim + d];
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wid.x;
    let s = wid.y;
    let b = wid.z;
    let kh = h / (pc.q_heads / pc.kv_heads);
    let limit = pc.past_len + s + 1u;

    var local_max = -3.402823e38;
    var position = lid.x;
    while (position < limit) {
        var dot = 0.0;
        for (var d = 0u; d < pc.head_dim; d = d + 1u) {
            dot = dot + q_value(b, s, h, d) * key_value(b, kh, position, d);
        }
        let score = dot * pc.scale;
        scores[position] = score;
        local_max = max(local_max, score);
        position = position + 256u;
    }
    reduce[lid.x] = local_max;
    workgroupBarrier();
    var stride = 128u;
    while (stride > 0u) {
        if (lid.x < stride) { reduce[lid.x] = max(reduce[lid.x], reduce[lid.x + stride]); }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let maximum = reduce[0];
    var local_sum = 0.0;
    position = lid.x;
    while (position < limit) {
        let weight = exp(scores[position] - maximum);
        scores[position] = weight;
        local_sum = local_sum + weight;
        position = position + 256u;
    }
    reduce[lid.x] = local_sum;
    workgroupBarrier();
    stride = 128u;
    while (stride > 0u) {
        if (lid.x < stride) { reduce[lid.x] = reduce[lid.x] + reduce[lid.x + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let denom = reduce[0];
    var d = lid.x;
    while (d < pc.head_dim) {
        var acc = 0.0;
        for (position = 0u; position < limit; position = position + 1u) {
            acc = acc + scores[position] * value_value(b, kh, position, d);
        }
        out[((b * pc.seq + s) * pc.q_heads + h) * pc.head_dim + d] = acc / denom;
        d = d + 256u;
    }

    if (h < pc.kv_heads) {
        d = lid.x;
        while (d < pc.head_dim) {
            if (s == 0u) {
                for (position = 0u; position < pc.past_len; position = position + 1u) {
                    let old = ((b * pc.kv_heads + h) * pc.past_len + position) * pc.head_dim + d;
                    let dst = ((b * pc.kv_heads + h) * pc.total_len + position) * pc.head_dim + d;
                    present_k[dst] = past_k[old];
                    present_v[dst] = past_v[old];
                }
            }
            let dst = ((b * pc.kv_heads + h) * pc.total_len + pc.past_len + s) * pc.head_dim + d;
            present_k[dst] = current_k_value(b, s, h, d);
            present_v[dst] = v[((b * pc.seq + s) * pc.kv_heads + h) * pc.head_dim + d];
            d = d + 256u;
        }
    }
}
"#;

#[cfg(test)]
mod tests {
    #[test]
    fn source_compiles() {
        vk_compute::compile_wgsl(super::GQA).expect("GQA shader must compile to SPIR-V");
    }
}
