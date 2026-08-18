//! Load-time graph rewrites: patterns of standard operators replaced by the
//! single fused operator the interpreter already implements.
//!
//! This is not the convex partitioning of [`crate::fusion`], which decides
//! *which* nodes a backend claims. Here the node list itself changes: nine
//! dispatches become one, on a graph that was already entirely claimed.
//!
//! The lever is exporter age. `LayerNormalization` only exists from opset 17,
//! so every transformer exported below it carries the normalization written out
//! as `ReduceMean → Sub → Pow → ReduceMean → Add → Sqrt → Div → Mul → Add`.
//! roberta-base (opset 11) repeats that pattern 25 times: 225 nodes doing the
//! work of 25 dispatches, against a kernel that already exists.
//!
//! Two conditions bound what is matched, and both are about not producing a
//! node the kernel would then reject at run time:
//!
//! - the reduction axis must be the **last** one (`axes = [-1]`), because that
//!   is the only axis `core::shaders::layernorm` normalizes over. Exporters
//!   also emit this shape on `axes = [1]` for a channel-wise normalization —
//!   rfdetr does, nine times — and that is a different operator, not fused here;
//! - `scale` and `bias` must be **rank-1 constants of the same length > 1**.
//!   The IR carries no shapes, so this is what stands in for
//!   `scale.elem_count == last_dim`: a rank-1 operand broadcasts over the last
//!   axis by definition, and length 1 is excluded because it would broadcast as
//!   a scalar instead, which the kernel does not accept.

use crate::graph::{AttrValue, ElementType, GraphIr, InitializerIr, NodeIr, constant_outputs};
use crate::host_ops::HostTensor;
use std::collections::{HashMap, HashSet};

/// Producer, consumers and graph outputs of every value, by node index.
struct Index {
    consumers: HashMap<String, Vec<usize>>,
    graph_outputs: HashSet<String>,
}

impl Index {
    fn build(ir: &GraphIr) -> Self {
        let mut consumers: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, node) in ir.nodes.iter().enumerate() {
            for name in node.inputs.iter().filter(|n| !n.is_empty()) {
                let list = consumers.entry(name.clone()).or_default();
                if list.last() != Some(&i) {
                    list.push(i);
                }
            }
        }
        Index {
            consumers,
            graph_outputs: ir.outputs.iter().cloned().collect(),
        }
    }

    /// The single node reading `name`, when the value escapes nowhere else.
    ///
    /// A value observable outside the pattern — read by a node that is not part
    /// of it, or declared as a graph output — cannot be deleted with it.
    fn only_consumer(&self, name: &str) -> Option<usize> {
        if self.graph_outputs.contains(name) {
            return None;
        }
        match self.consumers.get(name)?.as_slice() {
            [only] => Some(*only),
            _ => None,
        }
    }
}

/// A matched pattern: the nodes it consumes and the node that replaces them.
struct Match {
    members: Vec<usize>,
    fused: NodeIr,
}

fn scalar_f32(name: &str, constants: &HashMap<String, InitializerIr>) -> Option<f32> {
    let tensor = constants.get(name)?;
    if tensor.dtype != ElementType::Float32 as i32 {
        return None;
    }
    match HostTensor::new(tensor.dtype, tensor.shape.clone(), tensor.data.clone())
        .to_f32()
        .ok()?
        .as_slice()
    {
        [only] => Some(*only),
        _ => None,
    }
}

/// Length of a rank-1 f32 constant usable as `scale`/`bias`, if it is one.
fn per_channel_len(name: &str, constants: &HashMap<String, InitializerIr>) -> Option<i64> {
    let tensor = constants.get(name)?;
    match (
        tensor.dtype == ElementType::Float32 as i32,
        &tensor.shape[..],
    ) {
        (true, [len]) if *len > 1 => Some(*len),
        _ => None,
    }
}

/// The operand of a two-input node that is not `known`.
fn other_input<'a>(node: &'a NodeIr, known: &str) -> Option<&'a str> {
    match &node.inputs[..] {
        [a, b] if a == known => Some(b),
        [a, b] if b == known => Some(a),
        _ => None,
    }
}

/// `axes` reducing the last dimension, with `keepdims` left at its default.
fn reduces_last_axis(node: &NodeIr) -> bool {
    let axes_last = node.attrs.get("axes").and_then(AttrValue::as_ints) == Some(&[-1][..]);
    let keeps = node.attrs.get("keepdims").and_then(AttrValue::as_i64) != Some(0);
    axes_last && keeps
}

fn is_op(node: &NodeIr, op: &str) -> bool {
    node.op == op && node.domain.is_empty()
}

/// Matches the decomposed pattern rooted at the first `ReduceMean`.
fn match_layernorm(
    nodes: &[NodeIr],
    root: usize,
    index: &Index,
    constants: &HashMap<String, InitializerIr>,
) -> Option<Match> {
    let mean = &nodes[root];
    if !is_op(mean, "ReduceMean") || !reduces_last_axis(mean) {
        return None;
    }
    let x = mean.inputs.first()?;

    let i_sub = index.only_consumer(&mean.outputs[0])?;
    let sub = &nodes[i_sub];
    if !is_op(sub, "Sub") || sub.inputs != [x.clone(), mean.outputs[0].clone()] {
        return None;
    }
    // `x - mean` feeds both the variance branch and the division: two readers,
    // and they must be exactly the pattern's own two.
    let deviation = &sub.outputs[0];
    if index.graph_outputs.contains(deviation) {
        return None;
    }
    let readers = index.consumers.get(deviation)?;
    if readers.len() != 2 {
        return None;
    }
    let i_pow = *readers.iter().find(|&&i| is_op(&nodes[i], "Pow"))?;
    let pow = &nodes[i_pow];
    if pow.inputs.first() != Some(deviation) || scalar_f32(&pow.inputs[1], constants)? != 2.0 {
        return None;
    }

    let i_var = index.only_consumer(&pow.outputs[0])?;
    let var = &nodes[i_var];
    if !is_op(var, "ReduceMean")
        || var.inputs.first() != Some(&pow.outputs[0])
        || !reduces_last_axis(var)
    {
        return None;
    }

    let i_add_eps = index.only_consumer(&var.outputs[0])?;
    let add_eps = &nodes[i_add_eps];
    if !is_op(add_eps, "Add") {
        return None;
    }
    let epsilon = scalar_f32(other_input(add_eps, &var.outputs[0])?, constants)?;

    let i_sqrt = index.only_consumer(&add_eps.outputs[0])?;
    let sqrt = &nodes[i_sqrt];
    if !is_op(sqrt, "Sqrt") || sqrt.inputs.first() != Some(&add_eps.outputs[0]) {
        return None;
    }

    let i_div = index.only_consumer(&sqrt.outputs[0])?;
    let div = &nodes[i_div];
    if !is_op(div, "Div") || div.inputs != [deviation.clone(), sqrt.outputs[0].clone()] {
        return None;
    }
    // the division must be the pattern's other reader of `x - mean`
    if !readers.contains(&i_div) {
        return None;
    }

    let i_mul = index.only_consumer(&div.outputs[0])?;
    let mul = &nodes[i_mul];
    if !is_op(mul, "Mul") {
        return None;
    }
    let scale = other_input(mul, &div.outputs[0])?.to_string();
    let channels = per_channel_len(&scale, constants)?;

    // The bias is optional: without it the kernel runs with `has_bias = 0`.
    let mut members = vec![root, i_sub, i_pow, i_var, i_add_eps, i_sqrt, i_div, i_mul];
    let mut inputs = vec![x.clone(), scale];
    let mut output = mul.outputs[0].clone();
    if let Some(i_add_bias) = index.only_consumer(&mul.outputs[0])
        && is_op(&nodes[i_add_bias], "Add")
        && let Some(bias) = other_input(&nodes[i_add_bias], &mul.outputs[0])
        && per_channel_len(bias, constants) == Some(channels)
    {
        inputs.push(bias.to_string());
        output = nodes[i_add_bias].outputs[0].clone();
        members.push(i_add_bias);
    }

    let fused = NodeIr {
        domain: String::new(),
        op: "LayerNormalization".into(),
        // the operator's own opset, independent of the graph's: the pattern
        // exists precisely because the graph predates it
        since_version: 17,
        name: format!("{}_fused_LayerNormalization", output),
        inputs,
        outputs: vec![output],
        attrs: HashMap::from([
            ("axis".to_string(), AttrValue::Int(-1)),
            ("epsilon".to_string(), AttrValue::Float(epsilon)),
        ]),
    };
    Some(Match { members, fused })
}

/// Replaces every decomposed layer normalization with a single
/// `LayerNormalization`. Returns how many patterns were fused.
pub fn fuse_layernorm(ir: &mut GraphIr) -> usize {
    let mut constants = constant_outputs(&ir.nodes);
    constants.extend(ir.initializers.clone());
    let index = Index::build(ir);

    let mut claimed = vec![false; ir.nodes.len()];
    let mut matches = Vec::new();
    for root in 0..ir.nodes.len() {
        let Some(m) = match_layernorm(&ir.nodes, root, &index, &constants) else {
            continue;
        };
        if m.members.iter().any(|&i| claimed[i]) {
            continue;
        }
        for &i in &m.members {
            claimed[i] = true;
        }
        matches.push(m);
    }
    if matches.is_empty() {
        return 0;
    }

    // The fused node takes the slot of the pattern's last member: every other
    // member is one of its ancestors, so topological order is preserved.
    let mut replacement: HashMap<usize, NodeIr> = HashMap::new();
    let count = matches.len();
    for m in matches {
        let slot = *m.members.iter().max().expect("a match has members");
        replacement.insert(slot, m.fused);
    }
    let nodes = std::mem::take(&mut ir.nodes);
    ir.nodes = nodes
        .into_iter()
        .enumerate()
        .filter_map(|(i, node)| match (claimed[i], replacement.remove(&i)) {
            (true, fused) => fused,
            (false, _) => Some(node),
        })
        .collect();
    count
}

/// Evaluates a node whose inputs are all constant, on host, at load time.
///
/// Only the operators the measurement points at are here. A generic evaluator
/// would mean a second implementation of the interpreter on host tensors, and
/// the payoff does not need one: across the nine models in the matrix five gain
/// nothing at all, and what the other four fold is weight dequantization and
/// shape plumbing. `resnet50-qdq` re-dequantizes its weights on every run —
/// 107 `DequantizeLinear` reading 25.5 MB of int8 and writing **102.1 MB of
/// fp32** for a result that never changes.
fn evaluate(node: &NodeIr, inputs: &[HostTensor]) -> Option<HostTensor> {
    if !node.domain.is_empty() {
        return None;
    }
    match node.op.as_str() {
        "DequantizeLinear" => {
            // `axis` only means anything for a per-axis scale, which the host
            // implementation rejects; a scalar scale ignores it.
            crate::host_ops::dequantize_linear(&inputs[0], inputs.get(1)?, inputs.get(2)).ok()
        }
        "Cast" => {
            let to = node.attrs.get("to")?.as_i64()? as i32;
            crate::host_ops::cast(&inputs[0], to).ok()
        }
        "Concat" => {
            let axis = node.attrs.get("axis")?.as_i64()?;
            crate::host_ops::concat(inputs, axis).ok()
        }
        // Pure reinterpretations of the same bytes: only the shape changes, and
        // it is already resolved because the shape operand is constant too.
        "Reshape" => {
            let requested = inputs.get(1)?.to_i64().ok()?;
            let shape = resolve_reshape(&inputs[0].shape, &requested)?;
            Some(HostTensor::new(
                inputs[0].dtype,
                shape,
                inputs[0].data.clone(),
            ))
        }
        _ => None,
    }
}

/// `Reshape`'s target shape with `0` (copy the input dim) and `-1` (infer)
/// resolved, or `None` if it does not describe the same element count.
fn resolve_reshape(input: &[i64], requested: &[i64]) -> Option<Vec<i64>> {
    let total: i64 = input.iter().product();
    let mut shape: Vec<i64> = requested
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            if d == 0 {
                *input.get(i).unwrap_or(&0)
            } else {
                d
            }
        })
        .collect();
    let inferred = shape.iter().position(|&d| d == -1);
    if let Some(i) = inferred {
        let known: i64 = shape.iter().filter(|&&d| d != -1).product();
        if known == 0 || total % known != 0 {
            return None;
        }
        shape[i] = total / known;
    }
    (shape.iter().product::<i64>() == total && shape.iter().all(|&d| d >= 0)).then_some(shape)
}

/// Replaces every node whose inputs are all constant with its own result,
/// promoted to an initializer. Returns how many nodes were folded.
///
/// The nodes are visited in topological order, so a folded output is itself
/// constant for whatever reads it downstream — the closure is transitive in a
/// single pass. What the fold removes is *dispatches*, not blocks: every model
/// in the matrix is already at one convex block.
pub fn fold_constants(ir: &mut GraphIr) -> usize {
    let mut constants: HashMap<String, InitializerIr> = constant_outputs(&ir.nodes);
    constants.extend(ir.initializers.clone());

    let graph_outputs: HashSet<&str> = ir.outputs.iter().map(String::as_str).collect();
    let mut folded = Vec::new();
    for (i, node) in ir.nodes.iter().enumerate() {
        // A graph output has to stay a produced value: the run's environment
        // hands back what the nodes computed, not what the initializers hold.
        if node.op == "Constant"
            || node.inputs.is_empty()
            || node.outputs.len() != 1
            || graph_outputs.contains(node.outputs[0].as_str())
        {
            continue;
        }
        let Some(inputs) = node
            .inputs
            .iter()
            .map(|name| {
                let init = constants.get(name)?;
                Some(HostTensor::new(
                    init.dtype,
                    init.shape.clone(),
                    init.data.clone(),
                ))
            })
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        let Some(result) = evaluate(node, &inputs) else {
            continue;
        };
        let value = InitializerIr {
            dtype: result.dtype,
            shape: result.shape,
            data: result.data,
        };
        constants.insert(node.outputs[0].clone(), value.clone());
        folded.push((i, node.outputs[0].clone(), value));
    }
    if folded.is_empty() {
        return 0;
    }

    let count = folded.len();
    let removed: HashSet<usize> = folded.iter().map(|(i, ..)| *i).collect();
    for (_, name, value) in folded {
        ir.initializers.insert(name, value);
    }
    let nodes = std::mem::take(&mut ir.nodes);
    ir.nodes = nodes
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !removed.contains(i))
        .map(|(_, node)| node)
        .collect();
    count
}

/// Names every subgraph, at every depth, reads from its enclosing scope — the
/// free variables whose value lives in the *parent* scope.
///
/// An ONNX subgraph reads a parent-scoped value either as a declared `graph.input`
/// or, as exporters commonly emit, as a **node input** bound by name to the
/// enclosing value (the unrolled depthformer's `If` branches read
/// `depth_linear.weight_Q4` straight from the top-level initializers). Either way
/// the name must stay live in the parent: its initializer must survive and the
/// node that produces it must not be pruned. We approximate "reads but does not
/// define here" — a superset of the true free set — so this can only ever
/// over-keep, never wrongly drop.
fn captured_names(ir: &GraphIr) -> Vec<String> {
    let mut out = Vec::new();
    for node in &ir.nodes {
        for attr in node.attrs.values() {
            if let Some(sub) = attr.as_graph() {
                collect_free_reads(sub, &mut out);
            }
        }
    }
    out
}

fn collect_free_reads(sub: &GraphIr, out: &mut Vec<String>) {
    // Declared `graph.input` names are parent-scoped values by definition.
    out.extend(sub.inputs.iter().cloned());
    let mut defined: HashSet<String> = sub.inputs.iter().cloned().collect();
    for node in &sub.nodes {
        for name in &node.outputs {
            defined.insert(name.clone());
        }
    }
    for node in &sub.nodes {
        for name in &node.inputs {
            if !name.is_empty() && !defined.contains(name) {
                out.push(name.clone());
            }
        }
        for attr in node.attrs.values() {
            if let Some(nested) = attr.as_graph() {
                collect_free_reads(nested, out);
            }
        }
    }
}

/// Drops initializers no node reads. Returns the bytes released.
///
/// Constant folding orphans its own inputs — the int8 weights behind 107
/// `DequantizeLinear` are 25.5 MB nothing reads once their fp32 result is an
/// initializer — and they are held for the session's whole life.
pub fn prune_dead_initializers(ir: &mut GraphIr) -> usize {
    let captured = captured_names(ir);
    let live: HashSet<&str> = ir
        .nodes
        .iter()
        .flat_map(|n| n.inputs.iter())
        .map(String::as_str)
        .chain(ir.outputs.iter().map(String::as_str))
        // Subgraph-captured names resolve against this scope's initializers.
        .chain(captured.iter().map(String::as_str))
        .collect();
    let dead: Vec<String> = ir
        .initializers
        .keys()
        .filter(|name| !live.contains(name.as_str()))
        .cloned()
        .collect();
    dead.iter()
        .filter_map(|name| ir.initializers.remove(name))
        .map(|init| init.data.len())
        .sum()
}

/// Drops nodes whose outputs nobody reads and that no graph output declares.
///
/// Every operator the interpreter runs is pure, so an unread result is work
/// with no observer. Rewriting orphans the constants a pattern carried —
/// roberta's `Pow` exponents and epsilons — and they would otherwise stay in
/// the block as uploads nothing reads. Iterated to a fixpoint because removing
/// a node can orphan its own producers.
pub fn prune_dead_nodes(ir: &mut GraphIr) -> usize {
    let mut removed = 0;
    loop {
        let index = Index::build(ir);
        let captured_owned = captured_names(ir);
        let captured: HashSet<&str> = captured_owned.iter().map(String::as_str).collect();
        let before = ir.nodes.len();
        let nodes = std::mem::take(&mut ir.nodes);
        ir.nodes = nodes
            .into_iter()
            .filter(|node| {
                node.outputs.iter().any(|name| {
                    !name.is_empty()
                        && (index.graph_outputs.contains(name)
                            || index.consumers.contains_key(name)
                            // A subgraph captures this value as a free variable.
                            || captured.contains(name.as_str()))
                })
            })
            .collect();
        removed += before - ir.nodes.len();
        if ir.nodes.len() == before {
            return removed;
        }
    }
}

/// Whether the depthformer constant-index folds are active. Read once, at
/// executor build, from `ONNX_VULKAN_DF_FOLDS`. **On by default** — unlike the
/// parked GQA q/k de-interleave these folds remove measured per-run device→host
/// downloads (the 8 KB `GatherElements` of the depth-slice table, ×8 per call),
/// and they are bit-identical rewrites of provably-constant index math — but a
/// stats-off wall gate still decides whether they stay, so `=0` restores the
/// pre-fold graph for A/B without a rebuild.
pub fn df_folds_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("ONNX_VULKAN_DF_FOLDS").as_deref(), Ok("0")))
}

/// Producer index of every value, for the rewrites that walk *backwards*
/// (the `Index` above only records consumers).
fn producers(nodes: &[NodeIr]) -> HashMap<String, usize> {
    let mut map = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        for output in &node.outputs {
            if !output.is_empty() {
                map.insert(output.clone(), i);
            }
        }
    }
    map
}

/// The int64 content of a constant value (initializer or `Constant` output).
fn int64_constant(name: &str, constants: &HashMap<String, InitializerIr>) -> Option<Vec<i64>> {
    let tensor = constants.get(name)?;
    if tensor.dtype != ElementType::Int64 as i32 {
        return None;
    }
    HostTensor::new(tensor.dtype, tensor.shape.clone(), tensor.data.clone())
        .to_i64()
        .ok()
}

/// A rank-1 int64 initializer built from `vals`.
fn int64_init(vals: &[i64]) -> InitializerIr {
    InitializerIr {
        dtype: ElementType::Int64 as i32,
        shape: vec![vals.len() as i64],
        data: vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

/// The scalar a run of **leftmost** `Unsqueeze`s (each inserting exactly axis
/// 0) applies to, walking back from `name` to a constant source. A value
/// produced this way broadcasts one scalar over every position of its final
/// shape — which is what makes a `GatherElements` on it a plain slice.
fn unsqueezed_scalar_source(
    name: &str,
    nodes: &[NodeIr],
    constants: &HashMap<String, InitializerIr>,
) -> Option<i64> {
    let producers = producers(nodes);
    let mut current = name;
    for _ in 0..16 {
        let Some(&i) = producers.get(current) else {
            // Not produced by any node: must be the constant scalar itself
            // (an initializer or a `Constant` output).
            let value = int64_constant(current, constants)?;
            return (value.len() == 1).then_some(value[0]);
        };
        let node = &nodes[i];
        if !is_op(node, "Unsqueeze") || node.outputs.len() != 1 {
            // Produced by something other than an Unsqueeze (e.g. a
            // `Constant` node): the value must still be a constant scalar.
            let value = int64_constant(current, constants)?;
            return (value.len() == 1).then_some(value[0]);
        }
        let axes = if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
            int64_constant(&node.inputs[1], constants)?
        } else {
            node.attrs.get("axes")?.as_ints()?.to_vec()
        };
        // Only axis-0 insertions keep the value a left-anchored broadcast of
        // one scalar; any other axis would interleave dimensions.
        if axes != [0] {
            return None;
        }
        current = &node.inputs[0];
    }
    None
}

/// Whether `name` is the output of `Equal(constA, constB)` on two int64
/// constants, and if so whether they are equal — i.e. the value of the
/// compile-time `(step_idx == 0)` predicate the exporter left in the graph.
/// At step 0 *both* operands are the constant `0`, so the answer cannot come
/// from "which operand is the zero literal"; it is simply `a == b`.
fn const_is_first(
    name: &str,
    nodes: &[NodeIr],
    constants: &HashMap<String, InitializerIr>,
) -> Option<bool> {
    let i = *producers(nodes).get(name)?;
    let node = &nodes[i];
    if !is_op(node, "Equal") || node.outputs.len() != 1 || node.inputs.len() != 2 {
        return None;
    }
    let a = int64_constant(&node.inputs[0], constants)?;
    let b = int64_constant(&node.inputs[1], constants)?;
    if a.len() != 1 || b.len() != 1 {
        return None;
    }
    Some(a[0] == b[0])
}

/// The last dimension of the (f32) table a lookup ultimately reads from.
///
/// Handles the exporter's two shapes: a direct `Gather(table, idx)` on an
/// initializer table, and the first-step pattern
/// `Gather(Gather(stacked_table, const_step), const_row)` — a
/// step-`[rows, D]` sub-table gathered from a `stacked` initializer, where
/// the sub-table's last dimension is `D` by construction (uniform slices).
/// Non-f32 or rank-deficient tables refuse; any doubt leaves the fold out.
fn lookup_table_last_dim(
    name: &str,
    nodes: &[NodeIr],
    constants: &HashMap<String, InitializerIr>,
) -> Option<i64> {
    let prod = producers(nodes);
    let g = prod.get(name)?;
    let g = &nodes[*g];
    if !is_op(g, "Gather") || g.inputs.len() != 2 {
        return None;
    }
    if let Some(t) = constants.get(&g.inputs[0])
        && t.dtype == ElementType::Float32 as i32
    {
        return t.shape.last().copied();
    }
    // The `Gather(Gather(stacked, const), const)` hop.
    let inner = prod.get(&g.inputs[0])?;
    let inner = &nodes[*inner];
    if !is_op(inner, "Gather") || inner.inputs.len() != 2 {
        return None;
    }
    if let Some(t) = constants.get(&inner.inputs[0])
        && t.dtype == ElementType::Float32 as i32
        && t.shape.len() >= 2
    {
        // The row gather's data is one slice of `stacked`: its last
        // dimension is the slice's last dimension.
        return t.shape.last().copied();
    }
    None
}

/// Collapses the depthformer's per-step `get_slice` selection into a device
/// `Slice`.
///
/// The unrolled depthformer picks row `step` of its `[batch, 8, 1024]`
/// depth-slice table with a fully constant index chain:
/// `step_idx` (scalar initializer 0..7) → three leftmost `Unsqueeze`s →
/// `Expand` to `[batch, 1, 1024]` → `GatherElements(axis = 1)` →
/// `Squeeze(axis = 0)`. `GatherElements` is a host op, so it downloads the
/// whole 8 KB *device* table on every step of every run — eight times per
/// depthformer call — to select a row whose index never changes.
///
/// The identical value is one device `Slice`
/// (`starts = [step]`, `ends = [step + 1]`, `axes = [1]`): zero downloads,
/// one dispatch. Both produce `[batch, 1, 1024]` with element
/// `[b, 0, k] = data[b, step, k]`, so the `Squeeze` downstream is untouched.
///
/// Matched only when the index is *provably* a broadcast of a constant
/// scalar (leftmost-Unsqueeze run on a scalar constant, expanded over a shape
/// whose middle dimension is the constant 1 — either literal or the
/// `Concat(batch, [1], width)` the exporter emits). Any doubt leaves the
/// `GatherElements` in place. Returns the number of chains collapsed.
pub fn fold_const_get_slice(ir: &mut GraphIr) -> usize {
    let mut constants = constant_outputs(&ir.nodes);
    constants.extend(ir.initializers.clone());
    let index = Index::build(ir);
    let prod = producers(&ir.nodes);

    let mut remove: HashSet<usize> = HashSet::new();
    let mut replacements: HashMap<usize, NodeIr> = HashMap::new();
    let mut new_inits: HashMap<String, InitializerIr> = HashMap::new();
    let mut folded = 0;

    for (i, node) in ir.nodes.iter().enumerate() {
        if !is_op(node, "GatherElements") || node.outputs.len() != 1 {
            continue;
        }
        if node.attrs.get("axis").and_then(AttrValue::as_i64) != Some(1) {
            continue;
        }
        let (data, idx) = (&node.inputs[0], &node.inputs[1]);
        // idx = Expand(x, shape)
        let Some(&expand_i) = prod.get(idx) else {
            continue;
        };
        let expand = &ir.nodes[expand_i];
        if !is_op(expand, "Expand") || expand.inputs.len() != 2 {
            continue;
        }
        // x = constant scalar broadcast by leftmost Unsqueezes.
        let step = match unsqueezed_scalar_source(&expand.inputs[0], &ir.nodes, &constants) {
            Some(s) => s,
            None => continue,
        };
        // The expanded shape's middle dimension must be exactly 1: only then
        // is the result one row (at `step`) along axis 1, i.e. a Slice.
        let shape_is_single_row = if let Some(shape) = int64_constant(&expand.inputs[1], &constants)
        {
            shape.len() == 3 && shape[1] == 1
        } else if let Some(c) = prod.get(&expand.inputs[1]) {
            let concat = &ir.nodes[*c];
            is_op(concat, "Concat")
                && concat.inputs.len() == 3
                && int64_constant(&concat.inputs[1], &constants) == Some(vec![1])
        } else {
            false
        };
        if !shape_is_single_row {
            continue;
        }
        // The Expand's output must feed only this GatherElements, so the
        // index chain can be orphaned; the rest dies via prune_dead_nodes.
        if index.only_consumer(&expand.outputs[0]) != Some(i) {
            continue;
        }
        let out = node.outputs[0].clone();
        let starts = int64_init(&[step]);
        let ends = int64_init(&[step + 1]);
        let axes = int64_init(&[1]);
        new_inits.insert(format!("{out}__sl_starts"), starts);
        new_inits.insert(format!("{out}__sl_ends"), ends);
        new_inits.insert(format!("{out}__sl_axes"), axes);
        replacements.insert(
            i,
            NodeIr {
                domain: String::new(),
                op: "Slice".into(),
                since_version: 10,
                name: node.name.clone(),
                inputs: vec![
                    data.clone(),
                    format!("{out}__sl_starts"),
                    format!("{out}__sl_ends"),
                    format!("{out}__sl_axes"),
                ],
                outputs: vec![out],
                attrs: HashMap::new(),
            },
        );
        remove.insert(expand_i);
        folded += 1;
    }
    if folded == 0 {
        return 0;
    }
    for (name, value) in new_inits {
        ir.initializers.insert(name, value);
    }
    let nodes = std::mem::take(&mut ir.nodes);
    ir.nodes = nodes
        .into_iter()
        .enumerate()
        .filter_map(|(i, node)| {
            if remove.contains(&i) {
                None
            } else if let Some(replacement) = replacements.get(&i) {
                Some(replacement.clone())
            } else {
                Some(node)
            }
        })
        .collect();
    folded
}

/// Folds the depthformer's `prev_embed` first-step zeroing mask.
///
/// Every unrolled step zeroes its previous-code embedding on the first step:
/// `masked = lookup * (1 - Cast(step_idx == 0))`. `step_idx` is a
/// compile-time initializer (0..7), so the mask is a *constant* scalar: 0 on
/// step 0, 1 elsewhere. The rewrite replaces `masked` at its consumers —
/// step > 0 points them straight at `lookup` (the `Mul` is an identity), and
/// step 0 substitutes a small zero initializer, which also orphans the
/// step-0 `lookup`/`table` gathers nothing reads afterwards. The whole
/// `Equal → Cast → Neg → Add → Unsqueeze → Mul` chain goes with it.
///
/// Matched only when every chain link has a single consumer (no escape) and
/// the embedding table is an initializer (its last dimension is the zero
/// vector's length). Returns the number of steps folded.
pub fn fold_prev_embed_const_mask(ir: &mut GraphIr) -> usize {
    use crate::graph::ElementType;
    let mut constants = constant_outputs(&ir.nodes);
    constants.extend(ir.initializers.clone());
    let index = Index::build(ir);
    let prod = producers(&ir.nodes);

    let mut remove: HashSet<usize> = HashSet::new();
    let mut repoint: Vec<(usize, usize, String)> = Vec::new(); // (consumer, input_slot, new_value)
    let mut new_inits: HashMap<String, InitializerIr> = HashMap::new();
    let mut folded = 0;

    for (i, node) in ir.nodes.iter().enumerate() {
        if !is_op(node, "Mul") || node.outputs.len() != 1 || node.inputs.len() != 2 {
            continue;
        }
        // One input is Unsqueeze(scalar_mask, [0]); the other is the lookup.
        let (mask_unsq_i, lookup) = if let Some(&c) = prod.get(&node.inputs[0])
            && is_op(&ir.nodes[c], "Unsqueeze")
        {
            (c, node.inputs[1].clone())
        } else if let Some(&c) = prod.get(&node.inputs[1])
            && is_op(&ir.nodes[c], "Unsqueeze")
        {
            (c, node.inputs[0].clone())
        } else {
            continue;
        };
        let mask_unsq = &ir.nodes[mask_unsq_i];
        if !is_op(mask_unsq, "Unsqueeze") || mask_unsq.outputs.len() != 1 {
            continue;
        }
        let mask_axes = if mask_unsq.inputs.len() > 1 && !mask_unsq.inputs[1].is_empty() {
            int64_constant(&mask_unsq.inputs[1], &constants)
        } else {
            mask_unsq
                .attrs
                .get("axes")
                .and_then(AttrValue::as_ints)
                .map(|a| a.to_vec())
        };
        if mask_axes != Some(vec![0]) {
            continue;
        }
        // mask = Add(1.0, Neg(Cast(Equal(step, 0)))), either Add order.
        let scalar_mask = &mask_unsq.inputs[0];
        let Some(&add_i) = prod.get(scalar_mask) else {
            continue;
        };
        let add = &ir.nodes[add_i];
        if !is_op(add, "Add") || add.outputs.len() != 1 || add.inputs.len() != 2 {
            continue;
        }
        let neg_src = if scalar_f32(&add.inputs[0], &constants) == Some(1.0) {
            &add.inputs[1]
        } else if scalar_f32(&add.inputs[1], &constants) == Some(1.0) {
            &add.inputs[0]
        } else {
            continue;
        };
        let Some(&neg_i) = prod.get(neg_src) else {
            continue;
        };
        let neg = &ir.nodes[neg_i];
        if !is_op(neg, "Neg") || neg.outputs.len() != 1 || neg.inputs.len() != 1 {
            continue;
        }
        let Some(&cast_i) = prod.get(&neg.inputs[0]) else {
            continue;
        };
        let cast = &ir.nodes[cast_i];
        if !is_op(cast, "Cast")
            || cast.outputs.len() != 1
            || cast.inputs.len() != 1
            || cast.attrs.get("to").and_then(AttrValue::as_i64) != Some(ElementType::Float32 as i64)
        {
            continue;
        }
        let is_first = match const_is_first(&cast.inputs[0], &ir.nodes, &constants) {
            Some(f) => f,
            None => continue,
        };
        // The Equal that feeds the Cast dies with the chain when nothing
        // else reads it; otherwise it is left for `prune_dead_nodes`.
        let equal_i = (index.only_consumer(&cast.inputs[0]) == Some(cast_i))
            .then(|| *producers(&ir.nodes).get(&cast.inputs[0]).unwrap());
        // Every link must be exclusively ours: the consumer of each chain
        // output is the *next* link (and the Mul for the last one).
        if index.only_consumer(&cast.outputs[0]) != Some(neg_i)
            || index.only_consumer(&neg.outputs[0]) != Some(add_i)
            || index.only_consumer(&add.outputs[0]) != Some(mask_unsq_i)
            || index.only_consumer(&mask_unsq.outputs[0]) != Some(i)
        {
            continue;
        }
        let masked = &node.outputs[0];
        if index.graph_outputs.contains(masked) {
            continue;
        }
        // The replacement value, and its shape when it is a new tensor.
        let (replacement, zeros_needed) = if is_first {
            // masked must become a zero vector of the lookup's shape
            // [1, D]; D is the last dimension of the table the lookup reads
            // (a direct initializer, or one slice of a stacked initializer).
            let Some(d) = lookup_table_last_dim(&lookup, &ir.nodes, &constants) else {
                continue;
            };
            (format!("{masked}__zeros"), Some(vec![1, d]))
        } else {
            (lookup.clone(), None)
        };
        // Repoint every consumer of `masked` at the replacement.
        let Some(consumers) = index.consumers.get(masked) else {
            continue;
        };
        for consumer in consumers {
            let slot = ir.nodes[*consumer]
                .inputs
                .iter()
                .position(|name| name == masked)
                .unwrap_or(0);
            repoint.push((*consumer, slot, replacement.clone()));
        }
        if let Some(shape) = zeros_needed {
            let bytes = shape.iter().product::<i64>() as usize * 4;
            new_inits.insert(
                replacement.clone(),
                InitializerIr {
                    dtype: ElementType::Float32 as i32,
                    shape,
                    data: vec![0u8; bytes],
                },
            );
        }
        remove.extend([i, mask_unsq_i, add_i, neg_i, cast_i]);
        if let Some(e) = equal_i {
            remove.insert(e);
        }
        folded += 1;
    }
    if folded == 0 {
        return 0;
    }
    for (consumer, slot, value) in repoint {
        ir.nodes[consumer].inputs[slot] = value;
    }
    let zeros = std::mem::take(&mut new_inits);
    for (name, value) in zeros {
        ir.initializers.insert(name, value);
    }
    let nodes = std::mem::take(&mut ir.nodes);
    ir.nodes = nodes
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !remove.contains(i))
        .map(|(_, node)| node)
        .collect();
    folded
}

/// Whether the GQA q/k de-interleave fold is active. Read once, at executor
/// build, from `ONNX_VULKAN_GQA_QK_REORDER` (generic core knob) or
/// `LFM25_TRANSPOSE_FUSE` (the LFM2.5 harness alias). Off by default: the fold
/// is a per-model optimization and must stay an explicit A/B lever until it
/// clears a stats-off wall-time gate.
pub fn gqa_qk_reorder_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = |name: &str| {
            std::env::var(name)
                .map(|v| !v.is_empty() && v != "0")
                .unwrap_or(false)
        };
        on("ONNX_VULKAN_GQA_QK_REORDER") || on("LFM25_TRANSPOSE_FUSE")
    })
}

/// Fuses the LFM2.5 depthformer `q_rearr`/`k_rearr` de-interleave into the
/// `GroupQueryAttention` that consumes it.
///
/// The exporter emits `Reshape→Transpose[0,1,2,4,3]→Reshape` before each
/// attention so that a head's 32-dim vector, stored interleaved as `[16,2]`
/// (pair `i` adjacent), is presented in logical order. At decode every such
/// `Transpose` is a ~320 µs dispatch over a tensor that is *immediately*
/// read back by one GQA — 96 of them per depthformer call. `Reshape` is a
/// metadata-only view on host and a flat copy on device, so the `Transpose`
/// is the only physical op in the trio, and GQA is its only consumer.
///
/// The fusion deletes the `Transpose`, re-points the trailing `Reshape` at the
/// pre-transpose (interleaved) buffer, and sets `qk_deint=1` on the consuming
/// GQA, whose kernel then indexes Q/K head dims through the inverse
/// permutation `phys(d) = (d % half) * 2 + d / half` (`half = head_dim / 2`).
/// V is un-rearranged and the KV cache is written by GQA in logical order, so
/// neither is touched — the result is bit-identical, with 96 fewer dispatches.
///
/// A pattern is fused only when **all** of: the perm is exactly `[0,1,2,4,3]`;
/// its producer is a `Reshape` and its sole consumer is a `Reshape`; that
/// consumer's sole reader is a `com.microsoft::GroupQueryAttention` reading it
/// as Q (input 0) or K (input 1); and head_dim (`width / heads`) is 32. Any
/// doubt leaves the `Transpose` in place. Returns the number of nodes removed.
pub fn fuse_gqa_qk_deint(ir: &mut GraphIr) -> usize {
    use crate::graph::ElementType;
    // The Reshape shape operands are `Constant` nodes, which `fold_constants`
    // leaves in `ir.nodes` (they have no inputs), so their values live in
    // `constant_outputs`, not `ir.initializers`. Merge both, as `fuse_layernorm`
    // does, so the head_dim test reads the real constant.
    let mut constants = constant_outputs(&ir.nodes);
    constants.extend(ir.initializers.clone());
    let index = Index::build(ir);
    // Phase 1: decide, over an immutable view, which Transposes to delete and
    // which trailing Reshapes to re-point. A pattern matches only when every
    // structural test passes; any doubt leaves the Transpose in place.
    let mut remove: HashSet<usize> = HashSet::new();
    let mut repoint: Vec<(usize, String)> = Vec::new();
    let mut gqa_flag: HashSet<usize> = HashSet::new();
    for (i, node) in ir.nodes.iter().enumerate() {
        if !is_op(node, "Transpose") || node.outputs.len() != 1 {
            continue;
        }
        if node.attrs.get("perm").and_then(AttrValue::as_ints) != Some(&[0, 1, 2, 4, 3][..]) {
            continue;
        }
        let out = node.outputs[0].clone();
        let first = match index.only_consumer(&out) {
            Some(c) => c,
            None => continue,
        };
        if !is_op(&ir.nodes[first], "Reshape") || ir.nodes[first].inputs.first() != Some(&out) {
            continue;
        }
        // Walk the run of sole-consumer `Reshape`s after the Transpose to the
        // terminal consumer. `Reshape` is a flat row-major reinterpret (it never
        // reorders bytes), so the Transpose is the only op that physically moves
        // data; whatever Reshape count sits between it and the GQA is layout
        // plumbing and can all keep running on the pre-transpose buffer.
        let mut cur = ir.nodes[first].outputs[0].clone();
        let mut gqa_i: Option<usize> = None;
        for _ in 0..64 {
            match index.only_consumer(&cur) {
                Some(c) => {
                    let n = &ir.nodes[c];
                    if is_op(n, "Reshape") && n.inputs.first() == Some(&cur) {
                        cur = n.outputs[0].clone();
                        continue;
                    }
                    gqa_i = Some(c);
                    break;
                }
                // multi-reader or a graph output: the chain is not cleanly ours.
                None => break,
            }
        }
        let gqa_i = match gqa_i {
            Some(g) => g,
            None => continue,
        };
        let gqa = &ir.nodes[gqa_i];
        // com.microsoft::GroupQueryAttention — the de-interleave only feeds the
        // Q/K legs of a real GQA, never some other domain's attention.
        if gqa.op != "GroupQueryAttention" || gqa.domain != "com.microsoft" {
            continue;
        }
        // Only the Q (input 0) and K (input 1) legs are de-interleaved; V is
        // not rearranged by the exporter.
        if !(gqa.inputs.get(0) == Some(&cur) || gqa.inputs.get(1) == Some(&cur)) {
            continue;
        }
        // head_dim must be 32 for the kernel's `[16,2]` de-interleave to hold.
        // The Reshape right after the Transpose is `[batch, 1, heads, head_dim]`;
        // its leading dims may be `0`/`-1` (dynamic), so only the LAST dim — the
        // head dim — must be the constant 32. The shape operand is a `Constant`
        // node kept in `ir.nodes` by `fold_constants`, so `constants` (merged
        // above) resolves it.
        let head_dim = ir.nodes[first]
            .inputs
            .get(1)
            .and_then(|c| constants.get(c))
            .filter(|t| t.dtype == ElementType::Int64 as i32)
            .and_then(|t| {
                let h = HostTensor::new(t.dtype, t.shape.clone(), t.data.clone());
                let v = h.to_i64().ok()?;
                v.last().copied().filter(|&d| d > 0)
            });
        if head_dim != Some(32) {
            continue;
        }
        remove.insert(i);
        // Re-point the FIRST trailing Reshape at the pre-transpose (interleaved)
        // buffer; the rest of the run follow it automatically.
        repoint.push((first, node.inputs[0].clone()));
        gqa_flag.insert(gqa_i);
    }
    if remove.is_empty() {
        return 0;
    }
    // Phase 2: apply the mutations on a mutable view.
    for (i, src) in &repoint {
        ir.nodes[*i].inputs[0] = src.clone();
    }
    for &g in &gqa_flag {
        ir.nodes[g]
            .attrs
            .entry("qk_reorder".to_string())
            .or_insert_with(|| AttrValue::Int(1))
            .clone_from(&AttrValue::Int(1));
    }
    let before = ir.nodes.len();
    let nodes = std::mem::take(&mut ir.nodes);
    ir.nodes = nodes
        .into_iter()
        .enumerate()
        .filter_map(|(i, n)| (!remove.contains(&i)).then_some(n))
        .collect();
    let removed = before - ir.nodes.len();
    let left = ir.nodes.iter().filter(|n| is_op(n, "Transpose")).count();
    log::info!(
        "tfuse: {g} GQA nodes flagged qk-deint, removed {removed} rearr Transposes ({left} left)",
        g = gqa_flag.len()
    );
    removed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(op: &str, inputs: &[&str], output: &str) -> NodeIr {
        NodeIr {
            domain: String::new(),
            op: op.into(),
            since_version: 11,
            name: output.into(),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            outputs: vec![output.into()],
            attrs: HashMap::new(),
        }
    }

    fn reduce_mean(input: &str, output: &str, axes: i64) -> NodeIr {
        let mut n = node("ReduceMean", &[input], output);
        n.attrs.insert("axes".into(), AttrValue::Ints(vec![axes]));
        n
    }

    fn scalar(value: f32) -> InitializerIr {
        InitializerIr {
            dtype: ElementType::Float32 as i32,
            shape: vec![],
            data: value.to_le_bytes().to_vec(),
        }
    }

    fn vector(len: usize) -> InitializerIr {
        InitializerIr {
            dtype: ElementType::Float32 as i32,
            shape: vec![len as i64],
            data: vec![0; len * 4],
        }
    }

    /// The pattern roberta-base emits 25 times, `axes = [-1]`.
    fn decomposed(axes: i64) -> GraphIr {
        GraphIr {
            nodes: vec![
                reduce_mean("x", "m", axes),
                node("Sub", &["x", "m"], "d"),
                node("Pow", &["d", "two"], "p"),
                reduce_mean("p", "v", axes),
                node("Add", &["v", "eps"], "ve"),
                node("Sqrt", &["ve"], "s"),
                node("Div", &["d", "s"], "n"),
                node("Mul", &["n", "scale"], "ns"),
                node("Add", &["ns", "bias"], "y"),
            ],
            initializers: HashMap::from([
                ("two".into(), scalar(2.0)),
                ("eps".into(), scalar(1e-5)),
                ("scale".into(), vector(768)),
                ("bias".into(), vector(768)),
            ]),
            inputs: vec!["x".into()],
            outputs: vec!["y".into()],
        }
    }

    #[test]
    fn fuses_the_decomposed_pattern() {
        let mut ir = decomposed(-1);
        assert_eq!(fuse_layernorm(&mut ir), 1);
        assert_eq!(ir.nodes.len(), 1);
        let fused = &ir.nodes[0];
        assert_eq!(fused.op, "LayerNormalization");
        assert_eq!(fused.inputs, ["x", "scale", "bias"]);
        assert_eq!(fused.outputs, ["y"]);
        assert_eq!(fused.attrs["axis"], AttrValue::Int(-1));
        assert_eq!(fused.attrs["epsilon"], AttrValue::Float(1e-5));
    }

    /// rfdetr writes the same nine nodes over the channel axis. That is a
    /// different operator and the kernel normalizes the last axis only.
    fn int64_vec(vals: &[i64]) -> InitializerIr {
        InitializerIr {
            dtype: ElementType::Int64 as i32,
            shape: vec![vals.len() as i64],
            data: vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
        }
    }

    /// A `Constant` node whose `value` is an int64 tensor — how the real graph
    /// carries the Reshape shape operands (they never land in `initializers`).
    fn constant(name: &str, vals: &[i64]) -> NodeIr {
        let mut n = node("Constant", &[], name);
        n.attrs
            .insert("value".into(), AttrValue::Tensor(int64_vec(vals)));
        n
    }

    /// A `com.microsoft::GroupQueryAttention` reading `q`/`k`/`v` plus the KV
    /// cache and rotary tables — the minimum surface the matcher checks.
    fn gqa(q: &str, k: &str, v: &str) -> NodeIr {
        let mut n = NodeIr {
            domain: "com.microsoft".into(),
            op: "GroupQueryAttention".into(),
            since_version: 21,
            name: "gqa".into(),
            inputs: [q, k, v, "pk", "pv", "seqlens", "total", "cos", "sin"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            outputs: vec!["attn".into(), "pres_k".into(), "pres_v".into()],
            attrs: HashMap::new(),
        };
        n.attrs.insert("num_heads".into(), AttrValue::Int(32));
        n.attrs.insert("kv_num_heads".into(), AttrValue::Int(8));
        n.attrs
            .insert("scale".into(), AttrValue::Float(1.0 / 32.0_f32.sqrt()));
        n
    }

    /// `q_rearr` as the real graph emits it: `Reshape5d → Transpose[0,1,2,4,3]`
    /// → **two** flat `Reshape`s → GQA.q, with the shape Constants in `ir.nodes`
    /// (never in `initializers`).
    fn rearr_graph() -> GraphIr {
        let mut perm_node = node("Transpose", &["in5"], "tout");
        perm_node
            .attrs
            .insert("perm".into(), AttrValue::Ints(vec![0, 1, 2, 4, 3]));
        GraphIr {
            nodes: vec![
                constant("shape5d", &[0, 1, 32, 16, 2]),
                constant("shape4d", &[0, 1, 32, 32]),
                constant("shape1024", &[0, 1, 1024]),
                node("Reshape", &["ln_out", "shape5d"], "in5"),
                perm_node,
                node("Reshape", &["tout", "shape4d"], "q3d"),
                node("Reshape", &["q3d", "shape1024"], "qflat"),
                gqa("qflat", "k3d", "v3d"),
            ],
            initializers: HashMap::new(),
            inputs: vec![
                "ln_out".into(),
                "k3d".into(),
                "v3d".into(),
                "pk".into(),
                "pv".into(),
                "seqlens".into(),
                "total".into(),
                "cos".into(),
                "sin".into(),
            ],
            outputs: vec!["attn".into(), "pres_k".into(), "pres_v".into()],
        }
    }

    #[test]
    fn fuses_the_q_rearr_into_gqa() {
        let mut ir = rearr_graph();
        assert_eq!(
            fuse_gqa_qk_deint(&mut ir),
            1,
            "the rearr Transpose must be removed"
        );
        // The trailing Reshape now views the PRE-transpose buffer.
        let reshaped = ir.nodes.iter().find(|n| n.name == "q3d").unwrap();
        assert_eq!(reshaped.inputs, ["in5", "shape4d"]);
        // The GQA is flagged to de-interleave q/k at the load.
        let g = ir
            .nodes
            .iter()
            .find(|n| n.op == "GroupQueryAttention")
            .unwrap();
        assert_eq!(g.attrs["qk_reorder"], AttrValue::Int(1));
        // No Transpose left.
        assert!(!ir.nodes.iter().any(|n| n.op == "Transpose"));
        // 8 nodes (3 Constant + Reshape + Transpose + Reshape + Reshape + GQA)
        // -> 7 after the Transpose is removed.
        assert_eq!(ir.nodes.len(), 7);
    }

    #[test]
    fn leaves_a_non_rearr_perm_alone() {
        let mut ir = rearr_graph();
        // Change the perm to something the kernel does not understand.
        let t = ir.nodes.iter_mut().find(|n| n.op == "Transpose").unwrap();
        t.attrs
            .insert("perm".into(), AttrValue::Ints(vec![0, 1, 2, 3, 4]));
        assert_eq!(fuse_gqa_qk_deint(&mut ir), 0);
        assert!(ir.nodes.iter().any(|n| n.op == "Transpose"));
    }

    #[test]
    fn leaves_a_double_consume_rearr_alone() {
        let mut ir = rearr_graph();
        // A second top-level reader of the transpose output means the value is
        // not the pattern's sole consumer, so it cannot be deleted.
        ir.nodes.push(node("Identity", &["tout"], "tout_copy"));
        ir.outputs.push("tout_copy".into());
        assert_eq!(fuse_gqa_qk_deint(&mut ir), 0);
        assert!(ir.nodes.iter().any(|n| n.op == "Transpose"));
    }

    /// Regression for the unrolled depthformer: an initializer that no
    /// *top-level* node reads but that a subgraph captures as a free variable
    /// (`depth_linear.weight_Q4`, read by all eight `If` branches) must survive
    /// `prune_dead_initializers`, and the node producing a captured value must
    /// survive `prune_dead_nodes`.
    #[test]
    fn keeps_subgraph_captured_values_alive() {
        // A subgraph that reads the parent initializer `w` in TWO ways:
        //  (a) as a declared free variable (`graph.input`), and
        //  (b) as a bare node input bound by name to the parent value — how
        //      the unrolled depthformer's `If` branches read `depth_linear.weight_Q4`.
        // Both must keep `w` alive in the parent scope.
        let sub = GraphIr {
            nodes: vec![
                node("Identity", &["w"], "w_use_declared"),
                node("Identity", &["w"], "w_use_bare"),
            ],
            initializers: HashMap::new(),
            inputs: vec!["w".into()],
            outputs: vec!["w_use_declared".into(), "w_use_bare".into()],
        };
        let mut branch = node("If", &["cond"], "branch_out");
        branch
            .attrs
            .insert("then_branch".into(), AttrValue::Graph(Box::new(sub)));

        let mut ir = GraphIr {
            nodes: vec![branch.clone()],
            initializers: {
                let mut m = HashMap::new();
                m.insert(
                    "w".into(),
                    InitializerIr {
                        dtype: ElementType::Float32 as i32,
                        shape: vec![4],
                        data: vec![0u8; 16],
                    },
                );
                m
            },
            inputs: vec!["cond".into()],
            outputs: vec!["branch_out".into()],
        };
        let released = prune_dead_initializers(&mut ir);
        assert_eq!(released, 0, "subgraph-read initializer must not be pruned");
        assert!(ir.initializers.contains_key("w"));
        // A producer of a value the subgraph reads must survive too.
        let mut ir2 = GraphIr {
            nodes: vec![node("Identity", &["x"], "w"), branch.clone()],
            initializers: HashMap::new(),
            inputs: vec!["x".into(), "cond".into()],
            outputs: vec!["branch_out".into()],
        };
        let dropped = prune_dead_nodes(&mut ir2);
        assert!(
            ir2.nodes.iter().any(|n| n.name == "w"),
            "subgraph-read producer must not be pruned (dropped {dropped})"
        );
    }

    #[test]
    fn leaves_a_channel_axis_normalization_alone() {
        let mut ir = decomposed(1);
        assert_eq!(fuse_layernorm(&mut ir), 0);
        assert_eq!(ir.nodes.len(), 9);
    }

    #[test]
    fn fuses_without_the_bias_add() {
        let mut ir = decomposed(-1);
        ir.nodes.pop();
        ir.outputs = vec!["ns".into()];
        assert_eq!(fuse_layernorm(&mut ir), 1);
        assert_eq!(ir.nodes[0].inputs, ["x", "scale"]);
        assert_eq!(ir.nodes[0].outputs, ["ns"]);
    }

    /// An intermediate someone else reads cannot be deleted with the pattern.
    #[test]
    fn refuses_when_an_intermediate_escapes() {
        let mut ir = decomposed(-1);
        ir.nodes.push(node("Neg", &["s"], "leak"));
        ir.outputs.push("leak".into());
        assert_eq!(fuse_layernorm(&mut ir), 0);

        let mut ir = decomposed(-1);
        ir.outputs.push("v".into());
        assert_eq!(fuse_layernorm(&mut ir), 0);
    }

    /// A scalar scale broadcasts over every axis, not per channel: the kernel
    /// would reject it, so the rewrite must not build it.
    #[test]
    fn refuses_a_scalar_scale() {
        let mut ir = decomposed(-1);
        ir.initializers.insert("scale".into(), vector(1));
        assert_eq!(fuse_layernorm(&mut ir), 0);
    }

    #[test]
    fn refuses_an_exponent_that_is_not_two() {
        let mut ir = decomposed(-1);
        ir.initializers.insert("two".into(), scalar(3.0));
        assert_eq!(fuse_layernorm(&mut ir), 0);
    }

    /// Two patterns in a row, the second reading the first's output.
    #[test]
    fn fuses_consecutive_patterns_and_keeps_topological_order() {
        let mut ir = decomposed(-1);
        let second: Vec<NodeIr> = decomposed(-1)
            .nodes
            .into_iter()
            .map(|mut n| {
                let rename = |s: &String| match s.as_str() {
                    "x" => "y".to_string(),
                    "scale" | "bias" | "two" | "eps" => s.clone(),
                    other => format!("{other}2"),
                };
                n.inputs = n.inputs.iter().map(rename).collect();
                n.outputs = n.outputs.iter().map(rename).collect();
                n
            })
            .collect();
        ir.nodes.extend(second);
        ir.outputs = vec!["y2".into()];
        assert_eq!(fuse_layernorm(&mut ir), 2);
        assert_eq!(ir.nodes.len(), 2);
        assert_eq!(ir.nodes[0].outputs, ["y"]);
        assert_eq!(ir.nodes[1].inputs[0], "y");
    }

    fn int8(shape: Vec<i64>, values: &[i8]) -> InitializerIr {
        InitializerIr {
            dtype: ElementType::Int8 as i32,
            shape,
            data: values.iter().map(|&v| v as u8).collect(),
        }
    }

    /// The 107 nodes `resnet50-qdq` re-runs every inference on a constant.
    #[test]
    fn folds_a_constant_weight_dequantization() {
        let mut ir = GraphIr {
            nodes: vec![
                node("DequantizeLinear", &["w_q", "w_scale", "w_zp"], "w"),
                node("Conv", &["x", "w"], "y"),
            ],
            initializers: HashMap::from([
                ("w_q".into(), int8(vec![2, 2], &[-2, -1, 1, 2])),
                ("w_scale".into(), scalar(0.5)),
                ("w_zp".into(), int8(vec![], &[-1])),
            ]),
            inputs: vec!["x".into()],
            outputs: vec!["y".into()],
        };
        assert_eq!(fold_constants(&mut ir), 1);
        assert_eq!(ir.nodes.len(), 1);
        assert_eq!(ir.nodes[0].op, "Conv");

        let w = &ir.initializers["w"];
        assert_eq!(w.dtype, ElementType::Float32 as i32);
        assert_eq!(w.shape, [2, 2]);
        let values = HostTensor::new(w.dtype, w.shape.clone(), w.data.clone())
            .to_f32()
            .unwrap();
        assert_eq!(values, [-0.5, 0.0, 1.0, 1.5]);

        // the int8 source and its quantization parameters are now unread
        assert_eq!(prune_dead_initializers(&mut ir), 4 + 4 + 1);
        assert!(!ir.initializers.contains_key("w_q"));
        assert!(ir.initializers.contains_key("w"));
    }

    /// A fold whose result feeds another foldable node closes transitively in
    /// one pass, because the nodes are visited in topological order.
    #[test]
    fn folds_transitively_in_one_pass() {
        let mut ir = GraphIr {
            nodes: vec![
                node("DequantizeLinear", &["w_q", "w_scale"], "w"),
                node("Reshape", &["w", "shape"], "w2"),
                node("Conv", &["x", "w2"], "y"),
            ],
            initializers: HashMap::from([
                ("w_q".into(), int8(vec![2, 2], &[1, 2, 3, 4])),
                ("w_scale".into(), scalar(1.0)),
                (
                    "shape".into(),
                    InitializerIr {
                        dtype: ElementType::Int64 as i32,
                        shape: vec![3],
                        data: [1i64, 4, -1].iter().flat_map(|v| v.to_le_bytes()).collect(),
                    },
                ),
            ]),
            inputs: vec!["x".into()],
            outputs: vec!["y".into()],
        };
        assert_eq!(fold_constants(&mut ir), 2);
        assert_eq!(ir.nodes.len(), 1);
        assert_eq!(ir.initializers["w2"].shape, [1, 4, 1]);
    }

    /// Per-axis quantization is refused rather than folded wrongly.
    #[test]
    fn refuses_a_per_axis_dequantization() {
        let mut ir = GraphIr {
            nodes: vec![node("DequantizeLinear", &["w_q", "w_scale"], "w")],
            initializers: HashMap::from([
                ("w_q".into(), int8(vec![2, 2], &[1, 2, 3, 4])),
                ("w_scale".into(), vector(2)),
            ]),
            inputs: vec![],
            outputs: vec!["w".into()],
        };
        assert_eq!(fold_constants(&mut ir), 0);
    }

    /// A node producing a graph output stays: the run hands back computed
    /// values, not initializers.
    #[test]
    fn keeps_a_fold_that_produces_a_graph_output() {
        let mut ir = GraphIr {
            nodes: vec![node("DequantizeLinear", &["w_q", "w_scale"], "w")],
            initializers: HashMap::from([
                ("w_q".into(), int8(vec![2], &[1, 2])),
                ("w_scale".into(), scalar(1.0)),
            ]),
            inputs: vec![],
            outputs: vec!["w".into()],
        };
        assert_eq!(fold_constants(&mut ir), 0);
    }

    #[test]
    fn prunes_the_constants_the_rewrite_orphans() {
        let mut ir = decomposed(-1);
        ir.initializers.remove("two");
        ir.initializers.remove("eps");
        let mut two = node("Constant", &[], "two");
        two.attrs
            .insert("value".into(), AttrValue::Tensor(scalar(2.0)));
        let mut eps = node("Constant", &[], "eps");
        eps.attrs
            .insert("value".into(), AttrValue::Tensor(scalar(1e-5)));
        ir.nodes.insert(0, two);
        ir.nodes.insert(1, eps);

        assert_eq!(fuse_layernorm(&mut ir), 1);
        assert_eq!(ir.nodes.len(), 3, "the two Constant nodes are now orphans");
        assert_eq!(prune_dead_nodes(&mut ir), 2);
        assert_eq!(ir.nodes.len(), 1);
        assert_eq!(ir.nodes[0].op, "LayerNormalization");
    }

    // ---- depthformer constant-index folds --------------------------------

    fn f32_init(shape: &[i64]) -> InitializerIr {
        InitializerIr {
            dtype: ElementType::Float32 as i32,
            shape: shape.to_vec(),
            data: vec![0u8; (shape.iter().product::<i64>() as usize) * 4],
        }
    }

    fn int64_scalar(value: i64) -> InitializerIr {
        InitializerIr {
            dtype: ElementType::Int64 as i32,
            shape: vec![],
            data: value.to_le_bytes().to_vec(),
        }
    }

    /// One unrolled depthformer `get_slice` chain, `step` baked in. The
    /// `Expand` shape is the exporter's `Concat(batch, [1], [1024])`.
    fn get_slice_chain(_step: i64) -> Vec<NodeIr> {
        vec![
            node("Unsqueeze", &["step", "[0]"], "u1"),
            node("Unsqueeze", &["u1", "[0]"], "u2"),
            node("Unsqueeze", &["u2", "[0]"], "u3"),
            node("Concat", &["b", "[1]", "[1024]"], "es"),
            node("Expand", &["u3", "es"], "ex"),
            {
                let mut g = node("GatherElements", &["tbl", "ex"], "sel");
                g.attrs.insert("axis".into(), AttrValue::Int(1));
                g
            },
            node("Squeeze", &["sel", "[1]"], "sq"),
        ]
    }

    #[test]
    fn folds_a_const_get_slice_into_a_device_slice() {
        let chain = get_slice_chain(3);
        let mut ir = GraphIr {
            nodes: chain,
            initializers: HashMap::from([
                ("step".into(), int64_scalar(3)),
                ("[0]".into(), int64_init(&[0])),
                ("b".into(), int64_init(&[1])),
                ("[1]".into(), int64_init(&[1])),
                ("[1024]".into(), int64_init(&[1024])),
            ]),
            inputs: vec!["tbl".into()],
            outputs: vec!["sq".into()],
        };
        assert_eq!(fold_const_get_slice(&mut ir), 1, "the chain must match");
        // The GatherElements slot now holds a device Slice on `tbl`.
        let slice = ir.nodes.iter().find(|n| n.op == "Slice").unwrap();
        assert_eq!(slice.inputs[0], "tbl");
        assert_eq!(slice.outputs[0], "sel", "same output name downstream");
        let starts = &ir.initializers[&format!("{}__sl_starts", slice.outputs[0])];
        let ends = &ir.initializers[&format!("{}__sl_ends", slice.outputs[0])];
        let axes = &ir.initializers[&format!("{}__sl_axes", slice.outputs[0])];
        assert_eq!(starts.data, 3i64.to_le_bytes());
        assert_eq!(ends.data, 4i64.to_le_bytes());
        assert_eq!(axes.data, 1i64.to_le_bytes());
        // The Expand (index chain) is gone; the Squeeze survives.
        assert!(!ir.nodes.iter().any(|n| n.op == "Expand"));
        assert!(ir.nodes.iter().any(|n| n.op == "Squeeze"));
        assert!(!ir.nodes.iter().any(|n| n.op == "GatherElements"));
    }

    #[test]
    fn refuses_a_nonconst_get_slice_index() {
        let chain = get_slice_chain(3);
        let mut ir = GraphIr {
            nodes: chain,
            initializers: HashMap::from([
                // `step` is NOT a constant: it is a runtime graph input.
                ("[0]".into(), int64_init(&[0])),
                ("b".into(), int64_init(&[1])),
                ("[1]".into(), int64_init(&[1])),
                ("[1024]".into(), int64_init(&[1024])),
            ]),
            inputs: vec!["tbl".into(), "step".into()],
            outputs: vec!["sq".into()],
        };
        assert_eq!(
            fold_const_get_slice(&mut ir),
            0,
            "a runtime index must never become a constant Slice"
        );
        assert!(ir.nodes.iter().any(|n| n.op == "GatherElements"));
    }

    #[test]
    fn refuses_a_multirow_get_slice_expand() {
        // Middle dimension of the Expand shape is 2, not 1: a two-row
        // selection is not a one-row Slice and must stay a GatherElements.
        let mut chain = get_slice_chain(3);
        for n in chain.iter_mut() {
            if n.name == "es" {
                n.inputs[1] = "[2]".into();
            }
        }
        let mut ir = GraphIr {
            nodes: chain,
            initializers: HashMap::from([
                ("step".into(), int64_scalar(3)),
                ("[0]".into(), int64_init(&[0])),
                ("b".into(), int64_init(&[1])),
                ("[1]".into(), int64_init(&[1])),
                ("[2]".into(), int64_init(&[2])),
            ]),
            inputs: vec!["tbl".into()],
            outputs: vec!["sq".into()],
        };
        assert_eq!(fold_const_get_slice(&mut ir), 0);
        assert!(ir.nodes.iter().any(|n| n.op == "GatherElements"));
    }

    /// One unrolled `prev_embed` masked-embedding chain. The `Equal` operands
    /// drive the compile-time `(step_idx == 0)` predicate: both `0` on the
    /// first step, `step` vs `0` elsewhere.
    fn prev_embed_chain(_step: i64) -> Vec<NodeIr> {
        vec![
            node("Equal", &["step", "z"], "eq"),
            node("Cast", &["eq"], "f"),
            node("Neg", &["f"], "neg"),
            node("Add", &["one", "neg"], "mask"),
            node("Unsqueeze", &["mask", "[0]"], "masku"),
            node("Mul", &["lookup", "masku"], "masked"),
        ]
    }

    fn prev_embed_graph(step: i64) -> GraphIr {
        let mut nodes = prev_embed_chain(step);
        // The consumer of `masked`.
        nodes.push(node("Add", &["sel", "masked"], "out"));
        for n in nodes.iter_mut() {
            if n.name == "f" {
                n.attrs
                    .insert("to".into(), AttrValue::Int(ElementType::Float32 as i64));
            }
        }
        // lookup = Gather(table, prev)
        let mut table_gather = node("Gather", &["table", "prev"], "lookup");
        table_gather.attrs.insert("axis".into(), AttrValue::Int(0));
        nodes.insert(0, table_gather);
        GraphIr {
            nodes,
            initializers: HashMap::from([
                ("step".into(), int64_scalar(step)),
                ("z".into(), int64_scalar(0)),
                ("one".into(), scalar(1.0)),
                ("[0]".into(), int64_init(&[0])),
                ("table".into(), f32_init(&[8, 1024])),
            ]),
            inputs: vec!["prev".into(), "sel".into()],
            outputs: vec!["out".into()],
        }
    }

    #[test]
    fn folds_a_step_gt0_prev_embed_mask_to_the_lookup() {
        let mut ir = prev_embed_graph(5);
        assert_eq!(fold_prev_embed_const_mask(&mut ir), 1);
        // `masked` is gone; its consumer reads `lookup` directly.
        assert!(!ir.nodes.iter().any(|n| n.name == "masked"));
        let consumer = ir.nodes.iter().find(|n| n.name == "out").unwrap();
        assert_eq!(consumer.inputs, ["sel", "lookup"]);
        // The whole const chain is removed.
        assert!(!ir.nodes.iter().any(|n| n.name == "eq"));
        assert!(!ir.nodes.iter().any(|n| n.name == "f"));
        assert!(!ir.nodes.iter().any(|n| n.name == "neg"));
        assert!(!ir.nodes.iter().any(|n| n.name == "mask"));
    }

    #[test]
    fn folds_a_step0_prev_embed_mask_to_zeros() {
        let mut ir = prev_embed_graph(0);
        assert_eq!(fold_prev_embed_const_mask(&mut ir), 1);
        let consumer = ir.nodes.iter().find(|n| n.name == "out").unwrap();
        assert_eq!(consumer.inputs, ["sel", "masked__zeros"]);
        let zeros = &ir.initializers["masked__zeros"];
        assert_eq!(zeros.shape, [1, 1024]);
        assert!(zeros.data.iter().all(|&b| b == 0));
    }

    #[test]
    fn refuses_a_prev_embed_mask_with_a_runtime_step() {
        let mut ir = prev_embed_graph(0);
        // `step` becomes a runtime input: the fold must refuse.
        ir.initializers.remove("step");
        ir.inputs.push("step".into());
        assert_eq!(
            fold_prev_embed_const_mask(&mut ir),
            0,
            "a runtime step is a runtime mask"
        );
        assert!(ir.nodes.iter().any(|n| n.name == "masked"));
    }
}
