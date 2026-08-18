//! The real LFM2.5 depthformer (8-codebook unrolled) through the load-time
//! rewrite sequence: the constant-index folds must fire exactly 8 times each,
//! remove their host-download sites, and leave a graph the interpreter still
//! supports end to end. The fixture is the production artifact, so a
//! regression in the exporter's shape of these chains is caught at parse
//! time, not at a five-minute Deck wall.
//!
//! The model's weights live in the sibling `vocoder_depthformer_q4.onnx_data`
//! (187 MB, shared with the looped model; not committed — pull it from
//! `lfm25-audio-onnx/dist/models/lfm25-audio-q4-flat/` or the Deck's CAS
//! release). Without it the tests skip rather than fail.

use onnx_vulkan_core::graph::AttrValue;
use std::collections::HashMap;
use std::path::Path;

fn fixture() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/models/vocoder_depthformer_q4_unrolled.onnx")
        .to_path_buf()
}

fn model() -> Option<onnx_vulkan_core::GraphIr> {
    let path = fixture();
    let data = path.with_file_name("vocoder_depthformer_q4.onnx_data");
    if !path.exists() || !data.exists() {
        eprintln!(
            "skip depthformer_folds: fixture or its 187 MB external-data file \
             ({}) is not present; restore it from the lfm25-audio-onnx dist or \
             the Deck CAS release to enable this test",
            path.display()
        );
        return None;
    }
    Some(
        onnx_vulkan_frontend::load(&path)
            .expect("parses the production depthformer")
            .graph,
    )
}

fn apply_folds_and_prune(ir: &mut onnx_vulkan_core::GraphIr) -> (usize, usize) {
    let s = onnx_vulkan_core::rewrite::fold_const_get_slice(ir);
    let m = onnx_vulkan_core::rewrite::fold_prev_embed_const_mask(ir);
    let pruned = onnx_vulkan_core::rewrite::prune_dead_nodes(ir);
    let released = onnx_vulkan_core::rewrite::prune_dead_initializers(ir);
    eprintln!(
        "folds: slices={s} masks={m} pruned={pruned} released={released} nodes={}",
        ir.nodes.len()
    );
    (s, m)
}

fn init_i64(ir: &onnx_vulkan_core::GraphIr, name: &str) -> Option<Vec<i64>> {
    let t = ir.initializers.get(name)?;
    onnx_vulkan_core::HostTensor::new(t.dtype, t.shape.clone(), t.data.clone())
        .to_i64()
        .ok()
}

#[test]
fn folds_fire_exactly_once_per_codebook_step() {
    let Some(mut ir) = model() else {
        return; // fixture not present
    };
    let before = ir.nodes.len();
    let (s, m) = apply_folds_and_prune(&mut ir);
    assert_eq!(s, 8, "one get_slice fold per codebook step");
    assert_eq!(m, 8, "one prev_embed mask fold per codebook step");
    assert!(
        before > ir.nodes.len(),
        "the folds must shrink the graph ({before} -> {})",
        ir.nodes.len()
    );
}

#[test]
fn no_gatherelements_remain_and_the_slices_are_const() {
    let Some(mut ir) = model() else {
        return; // fixture not present
    };
    apply_folds_and_prune(&mut ir);
    assert!(
        !ir.nodes.iter().any(|n| n.op == "GatherElements"),
        "every host-download gather is gone"
    );
    let slices: Vec<&onnx_vulkan_core::NodeIr> = ir
        .nodes
        .iter()
        .filter(|n| n.op == "Slice" && n.name.starts_with("step") && n.name.contains("gather"))
        .collect();
    assert_eq!(slices.len(), 8);
    for (i, sl) in slices.iter().enumerate() {
        let starts = init_i64(&ir, &sl.inputs[1]).expect("const starts");
        let ends = init_i64(&ir, &sl.inputs[2]).expect("const ends");
        let axes = init_i64(&ir, &sl.inputs[3]).expect("const axes");
        assert_eq!(
            starts,
            vec![i as i64],
            "slice {} selects row {}",
            sl.name,
            i
        );
        assert_eq!(ends, vec![i as i64 + 1]);
        assert_eq!(axes, vec![1]);
    }
}

#[test]
fn prev_embed_chains_are_dead_and_consumers_repointed() {
    let Some(mut ir) = model() else {
        return; // fixture not present
    };
    apply_folds_and_prune(&mut ir);
    for step in 0..8usize {
        for suffix in ["is_zero", "neg_is_zero", "mask", "mask_unsq"] {
            assert!(
                !ir.nodes
                    .iter()
                    .any(|n| n.name == format!("step{step}//prev_embed/{suffix}")),
                "step{step} {suffix} survives"
            );
        }
        let masked_out = format!("step{step}//prev_embed/masked/output_0");
        if step == 0 {
            // zeros initializer of the embedding row, named after the masked
            // output the consumer was repointed at
            let zeros = ir
                .initializers
                .get(&format!("{masked_out}__zeros"))
                .unwrap_or_else(|| panic!("step 0 needs its zero-vector initializer"));
            assert_eq!(zeros.shape, vec![1, 1024]);
            assert!(zeros.data.iter().all(|&b| b == 0));
        }
        // the mask Mul is gone (consumers repointed)
        assert!(
            !ir.nodes
                .iter()
                .any(|n| n.name == format!("step{step}//prev_embed/masked")),
            "step{step} masked Mul survives"
        );
        // and the consumer reads the replacement
        let combined = ir
            .nodes
            .iter()
            .find(|n| n.name == format!("step{step}//input/combined"))
            .expect("the step's combined input survives");
        let expected = if step == 0 {
            format!("{masked_out}__zeros")
        } else {
            format!("step{step}//prev_embed/lookup/output_0")
        };
        assert!(
            combined.inputs.contains(&expected),
            "step{step} combined reads {:?}, expected {expected}",
            combined.inputs
        );
    }
    // Only the 8 If-branch conditionals survive; the 8 mask Equal nodes were
    // removed with their chains.
    let eq: Vec<&onnx_vulkan_core::NodeIr> = ir.nodes.iter().filter(|n| n.op == "Equal").collect();
    assert_eq!(eq.len(), 8, "only the If-branch conditionals remain");
}

#[test]
fn folded_graph_is_still_fully_supported() {
    let Some(mut ir) = model() else {
        return; // fixture not present
    };
    apply_folds_and_prune(&mut ir);
    let unsupported: Vec<&onnx_vulkan_core::NodeIr> = ir
        .nodes
        .iter()
        .filter(|n| !onnx_vulkan_core::is_implemented_node(n))
        .collect();
    assert!(
        unsupported.is_empty(),
        "unsupported after folds: {}",
        unsupported
            .iter()
            .map(|n| n.op.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    // Every input references a live name: an initializer, a graph input, or a
    // node output (the fold's new Slice constants and zeros vector included).
    let live: std::collections::HashSet<&str> = ir
        .initializers
        .keys()
        .map(String::as_str)
        .chain(ir.inputs.iter().map(String::as_str))
        .chain(
            ir.nodes
                .iter()
                .flat_map(|n| n.outputs.iter().map(String::as_str)),
        )
        .collect();
    for n in &ir.nodes {
        for inp in &n.inputs {
            assert!(
                live.contains(inp.as_str()),
                "dangling input {inp} in {}",
                n.name
            );
        }
    }
    // The mask-folds' new initializers are the only `__zeros` names, and they
    // are zero-filled float rows.
    let zeros: Vec<&onnx_vulkan_core::InitializerIr> = ir
        .initializers
        .values()
        .filter(|t| t.data.len() > 0 && t.shape == vec![1, 1024])
        .collect();
    assert_eq!(zeros.len(), 1, "exactly one step needs a zero row");
    assert!(zeros[0].data.iter().all(|&b| b == 0));
    let _ = (HashMap::<String, AttrValue>::new(),);
}
