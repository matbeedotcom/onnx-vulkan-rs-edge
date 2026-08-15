//! Optional, env-gated host-side profiling (ONNX_VULKAN_TRACE=1).
//!
//! `stats` measures GPU time via timestamp queries; this measures the HOST
//! side of the same runs: how many dispatches a graph records, how long the
//! flush waits on a fence (the sync point), and how long downloads block.
//! Together with the cumulative GPU time they say whether a slow graph is
//! host-record-bound or GPU-execution-bound.
//!
//! Keep the fast path free: every hook is a single branch off a process-global
//! flag, so with the env var unset the cost is one static load.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

pub fn enabled() -> bool {
    *ON.get_or_init(|| std::env::var_os("ONNX_VULKAN_TRACE").is_some())
}

/// Dispatches recorded into the command stream since the last summary/reset.
static DISPATCHES: AtomicUsize = AtomicUsize::new(0);
/// Monotonic sequence of every dispatch recorded this process — the submit
/// trace reports each command buffer's [start..end) dispatch range, which is
/// what lets you map a hung submission back to the node-types report.
static DISPATCH_SEQ: AtomicUsize = AtomicUsize::new(0);
/// Accumulated host ns of the fence waits performed by flush (sync).
static FLUSH_WAIT_NS: AtomicU64 = AtomicU64::new(0);
/// Count of flush calls that blocked on a fence (the sync ones).
static FLUSH_WAITS: AtomicUsize = AtomicUsize::new(0);
/// Accumulated host ns spent in download submit+wait (full sync points).
static DOWNLOAD_NS: AtomicU64 = AtomicU64::new(0);
/// Count of downloads.
static DOWNLOADS: AtomicUsize = AtomicUsize::new(0);

pub fn record_dispatch() {
    if enabled() {
        DISPATCHES.fetch_add(1, Ordering::Relaxed);
    }
    DISPATCH_SEQ.fetch_add(1, Ordering::Relaxed);
}

/// The next dispatch's sequence number (so a command buffer closed now spans
/// `[start, dispatch_seq())`).
pub fn dispatch_seq() -> usize {
    DISPATCH_SEQ.load(Ordering::Relaxed)
}

pub fn record_flush_wait(ns: u64) {
    if enabled() {
        FLUSH_WAIT_NS.fetch_add(ns, Ordering::Relaxed);
        FLUSH_WAITS.fetch_add(1, Ordering::Relaxed);
    }
}

pub fn record_download(ns: u64) {
    if enabled() {
        DOWNLOAD_NS.fetch_add(ns, Ordering::Relaxed);
        DOWNLOADS.fetch_add(1, Ordering::Relaxed);
    }
}

/// GPU ns accumulated across all op types (from [`crate::stats`]) since
/// process start; not reset here so a summary line reflects the whole run.
pub fn gpu_ns() -> u64 {
    crate::stats::gpu_time_total()
}

/// Per-node-type host recording time for one graph: prints the top consumers.
/// The stream is async, so these are host-side costs (descriptor set
/// alloc/update, cmd buffer recording, buffer pool allocs) — NOT GPU time.
pub fn dump_node_types(node_ns: &HashMap<&str, (u64, usize)>) {
    if !enabled() {
        return;
    }
    let mut rows: Vec<(&str, u64, usize)> =
        node_ns.iter().map(|(k, (ns, c))| (*k, *ns, *c)).collect();
    rows.sort_by_key(|r| std::cmp::Reverse(r.1));
    if rows.is_empty() {
        return;
    }
    let mut out = String::new();
    for (op, ns, count) in rows.iter().take(8) {
        out.push_str(&format!("{}:{:.2}ms/{} ", op, *ns as f64 / 1e6, count));
    }
    eprintln!("[trace] node-types host  {out}");
}

/// Reads, logs, and zeroes the host-side counters. Returns true when
/// anything was recorded.
pub fn summary(prefix: &str) -> bool {
    if !enabled() {
        return false;
    }
    let d = DISPATCHES.swap(0, Ordering::Relaxed);
    let fw_ns = FLUSH_WAIT_NS.swap(0, Ordering::Relaxed);
    let fw = FLUSH_WAITS.swap(0, Ordering::Relaxed);
    let dl_ns = DOWNLOAD_NS.swap(0, Ordering::Relaxed);
    let dls = DOWNLOADS.swap(0, Ordering::Relaxed);
    if d == 0 && fw == 0 && dls == 0 {
        return false;
    }
    // eprintln, not log: profiling must work in binaries that install no
    // logger (the parity harness).
    eprintln!(
        "[trace] {} dispatches={} flush_wait={:.3}ms/{} dl={:.3}ms/{} gpu_cum={:.3}ms",
        prefix,
        d,
        fw_ns as f64 / 1e6,
        fw,
        dl_ns as f64 / 1e6,
        dls,
        gpu_ns() as f64 / 1e6,
    );
    true
}
