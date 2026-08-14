//! Real-model coverage gate for LiquidAI/LFM2.5-Audio-1.5B-ONNX.
//!
//! The model is intentionally not downloaded by `cargo test`. Set
//! `LFM25_AUDIO_ONNX_DIR` to the model's `onnx/` directory and run this test
//! explicitly. Loading the real external data is important: MatMulNBits
//! packing attributes alone do not describe every initializer variant.

use onnx_vulkan_core::{AttrValue, GraphIr, is_implemented_node};
use std::collections::BTreeMap;
use std::path::Path;

const Q4_GRAPHS: &[&str] = &[
    "decoder_q4.onnx",
    "audio_encoder_q4.onnx",
    "audio_embedding_q4.onnx",
    "audio_detokenizer_q4.onnx",
    "vocoder_depthformer_q4.onnx",
];

#[test]
#[ignore = "requires the 1.7 GB LFM2.5 Audio Q4 external-data closure"]
fn every_lfm25_audio_q4_node_has_a_native_vulkan_implementation() {
    let root = std::env::var_os("LFM25_AUDIO_ONNX_DIR")
        .expect("set LFM25_AUDIO_ONNX_DIR to the model's onnx directory");
    let root = Path::new(&root);
    let mut unsupported: BTreeMap<(String, String), (usize, String)> = BTreeMap::new();
    let mut inventory = String::from(
        "upstream=https://github.com/automataIA/onnx-vulkan-rs\nupstream_commit=747f97807b5d83eca1abcfc1a3ad572e538cce05\nmodel=LiquidAI/LFM2.5-Audio-1.5B-ONNX\nrevision=62318d95ddf42a65e742cdd6fd33df91874a801d\n",
    );

    for graph_name in Q4_GRAPHS {
        let model = onnx_vulkan_frontend::load(root.join(graph_name))
            .unwrap_or_else(|error| panic!("loading {graph_name}: {error:#}"));
        collect_unsupported(graph_name, &model.graph, &mut unsupported);
        inventory.push_str(&format!("\n[{graph_name}]\n"));
        for input in &model.graph.inputs {
            inventory.push_str(&format!(
                "input={input}:{}\n",
                model
                    .types
                    .get(input)
                    .map(ToString::to_string)
                    .unwrap_or_else(|| "?".into())
            ));
        }
        for output in &model.graph.outputs {
            inventory.push_str(&format!(
                "output={output}:{}\n",
                model
                    .types
                    .get(output)
                    .map(ToString::to_string)
                    .unwrap_or_else(|| "?".into())
            ));
        }
        let mut ops = BTreeMap::new();
        collect_ops(&model.graph, &mut ops);
        for ((domain, op), count) in ops {
            let qualified = if domain.is_empty() {
                op
            } else {
                format!("{domain}::{op}")
            };
            inventory.push_str(&format!("op={qualified}:{count}\n"));
        }
    }

    if std::env::var_os("LFM25_PRINT_INVENTORY").is_some() {
        eprintln!("--- LFM25 INVENTORY ---\n{inventory}--- END INVENTORY ---");
    }
    assert_eq!(
        inventory,
        include_str!("fixtures/lfm25_audio_q4_inventory.txt"),
        "the checked-in Q4 graph inventory changed"
    );

    let report = unsupported
        .iter()
        .map(|((domain, op), (count, first))| {
            let qualified = if domain.is_empty() {
                op.clone()
            } else {
                format!("{domain}::{op}")
            };
            format!("{qualified} x{count} (first {first})")
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        unsupported.is_empty(),
        "LFM2.5 Audio Q4 cannot execute fully on Vulkan:\n{report}"
    );
}

fn collect_ops(graph: &GraphIr, ops: &mut BTreeMap<(String, String), usize>) {
    for node in &graph.nodes {
        *ops.entry((node.domain.clone(), node.op.clone()))
            .or_default() += 1;
        for value in node.attrs.values() {
            if let AttrValue::Graph(branch) = value {
                collect_ops(branch, ops);
            }
        }
    }
}

fn collect_unsupported(
    graph_name: &str,
    graph: &GraphIr,
    unsupported: &mut BTreeMap<(String, String), (usize, String)>,
) {
    for node in &graph.nodes {
        if !is_implemented_node(node) {
            let key = (node.domain.clone(), node.op.clone());
            let entry = unsupported
                .entry(key)
                .or_insert_with(|| (0, format!("{graph_name}:{}", node.name)));
            entry.0 += 1;
        }
        for (attribute, value) in &node.attrs {
            if let AttrValue::Graph(branch) = value {
                collect_unsupported(
                    &format!("{graph_name}:{}[{attribute}]", node.name),
                    branch,
                    unsupported,
                );
            }
        }
    }
}
