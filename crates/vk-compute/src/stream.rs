//! Deferred command stream with bounded in-flight submits.
//!
//! The original design submitted one command buffer per `flush()` and waited
//! on its fence immediately. For a stream of small compute graphs that
//! serializes the CPU against the GPU at every graph boundary: the next graph
//! cannot even be *recorded* until the previous one has fully completed.
//!
//! This module keeps a bounded ring of in-flight submissions. `flush()`
//! submits the open command buffer *without* waiting; the fence is only waited
//! when a result is actually needed (`stream_download`) or when the ring is
//! full (back-pressure). That lets the host record the next graph while the
//! current one is still executing on the GPU — the overlap that matters for a
//! decoder→depthformer→audio-decode pipeline where each stage is a separate
//! graph.
//!
//! Correctness for the single-threaded engine that uses this stream in order:
//! every dispatch reads only buffers written by earlier dispatches in the same
//! graph (full memory barriers between dispatches), and a readback of a graph
//! records its copy into a *fresh* command buffer submitted *after* the graph's
//! submission. Queue submission order guarantees the copy runs only after the
//! graph has executed, so the copy's own fence is sufficient to guarantee the
//! readback sees the graph's result — no per-buffer tracking is required.

use crate::buffer::GpuBuffer;
use crate::context::VkContext;
use crate::pipeline::BufferSlice;
use crate::pipeline::ComputePipeline;
use anyhow::{bail, Context as _, Result};
use ash::vk;

/// Maximum number of submissions allowed to be in flight before `flush()`
/// applies back-pressure (waits for the oldest). Bounded so a runaway producer
/// cannot allocate unbounded command buffers / fences, while still giving the
/// GPU several graphs of outstanding work to overlap.
///
/// Overridable at runtime via `LFM25_MAX_IN_FLIGHT` (1..=64). On drivers that
/// are unstable with many concurrent submissions (e.g. RADV on the Steam Deck,
/// which raises a GPU "soft recovery" / context-loss when too much work is
/// queued at once), set this to `1`: the host still records the next graph
/// while the current one executes (the overlap that matters), but only one
/// command buffer is ever queued on the GPU at a time.
const MAX_IN_FLIGHT: usize = 8;

/// Resolves the in-flight submission cap, allowing a runtime override.
fn max_in_flight() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("LFM25_MAX_IN_FLIGHT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| (1..=64).contains(&n))
            .unwrap_or(MAX_IN_FLIGHT)
    })
}

/// Maximum dispatches recorded into a single command buffer before `flush()`
/// closes it (a hard sync boundary). Kept from the previous design: small graphs
/// stay resident, large ones do not monopolize a command buffer.
///
/// Overridable at runtime via `ONNX_VULKAN_MAX_DISPATCHES_PER_SUBMIT` for
/// bisecting GPU hangs: some drivers (RADV/ACO on the Steam Deck) lose the
/// device when one command buffer carries too much work — a submission that
/// exceeds the amdgpu gfx-ring watchdog (~5 s) is killed and the context
/// soft-recovers. Shrinking the chunk moves the sync point earlier and can
/// keep every submission under the watchdog.
pub const MAX_DISPATCHES_PER_SUBMIT: usize = 64;

/// Resolves the per-submission dispatch chunk, allowing a runtime override.
pub fn max_dispatches_per_submit() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("ONNX_VULKAN_MAX_DISPATCHES_PER_SUBMIT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| (1..=1024).contains(&n))
            .unwrap_or(MAX_DISPATCHES_PER_SUBMIT)
    })
}

/// Work to run only after a submission's fence has signaled (GPU done with the
/// resources it references).
enum Deferred {
    /// Storage buffer to destroy (was in the graveyard, freed once safe).
    Buffer(GpuBuffer),
    /// Staging buffer to return to the upload (true) or download (false) pool.
    Staging(GpuBuffer, bool),
}

struct Submission {
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    on_complete: Vec<Deferred>,
    /// Dispatch sequence the command buffer ended at: the submission covers
    /// global dispatches `[start, dispatch_end)`. Trace-only.
    dispatch_end: usize,
    /// Profiling (VULKAN_EP_STATS=1): the command buffer's timestamp query
    /// pool and the op type recorded for each of its dispatches. `None` when
    /// profiling is off. The pool lives exactly as long as the submission, so
    /// it can be destroyed in `wait_and_reap` with no in-flight hazard.
    query_pool: Option<vk::QueryPool>,
    ts_ops: Vec<&'static str>,
}

#[derive(Default)]
pub(crate) struct StreamState {
    buffer: Vec<vk::CommandBuffer>,
    pending_dispatches: usize,
    open: bool,
    finished: bool,
    /// Deferred work (staging-release, buffer-destroy) awaiting the NEXT
    /// submission: there is no open command buffer to attach it to yet, so it is
    /// carried here and moved into the submission `flush()` creates.
    pending_deferred: Vec<Deferred>,
    pending: Vec<Submission>,
    /// Profiling (VULKAN_EP_STATS=1): the timestamp query pool of the OPEN
    /// command buffer, created in `ensure_buffer`. `None` when profiling is
    /// off or no buffer is open.
    open_query_pool: Option<vk::QueryPool>,
    /// Next free slot in `open_query_pool` (also the count of written slots:
    /// slot 0 is the buffer-start marker, then one slot after each dispatch).
    open_query_slots: u32,
    /// Op type recorded for each dispatch of the open buffer, index-aligned
    /// with timestamp slots 1.. (slot `i+1` minus slot `i` = dispatch `i`).
    open_ts_ops: Vec<&'static str>,
}

impl StreamState {
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

impl VkContext {
    /// Single grouped barrier: previous writes (transfer or compute) made
    /// visible to subsequent reads/writes.
    fn stream_barrier(
        &self,
        cmd: vk::CommandBuffer,
        dst_stage: vk::PipelineStageFlags,
        dst_access: vk::AccessFlags,
    ) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(dst_access);
        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
    }

    /// Opens a new command buffer for recording (if one is not already open).
    /// `stream` must already be locked by the caller. Returns the handle of the
    /// open command buffer.
    fn ensure_buffer(&self, stream: &mut StreamState) -> Result<vk::CommandBuffer> {
        if stream.finished {
            bail!("stream finished; cannot record more commands");
        }
        if !stream.open {
            let cmd = unsafe {
                self.device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
            }?[0];
            unsafe {
                self.device.begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )?;
            }
            stream.buffer.push(cmd);
            stream.open = true;
            // Profiling: one timestamp query pool per command buffer. Slot 0
            // marks the buffer start; `stream_dispatch_slices` appends one slot
            // after every dispatch. The pool lives exactly as long as the
            // submission, so `wait_and_reap` can read it after the fence.
            if crate::stats::enabled() && self.timestamp_period > 0.0 {
                let capacity = max_dispatches_per_submit() as u32 + 2;
                let pool = unsafe {
                    self.device
                        .create_query_pool(
                            &vk::QueryPoolCreateInfo::default()
                                .query_type(vk::QueryType::TIMESTAMP)
                                .query_count(capacity),
                            None,
                        )
                }?;
                unsafe {
                    self.device.cmd_write_timestamp(
                        cmd,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        pool,
                        0,
                    );
                }
                stream.open_query_pool = Some(pool);
                stream.open_query_slots = 1;
                stream.open_ts_ops.clear();
            }
        }
        Ok(*stream.buffer.last().unwrap())
    }

    /// Closes the open command buffer (called with the stream lock held).
    fn end_buffer_locked(&self, stream: &mut StreamState) -> Result<()> {
        if !stream.open {
            return Ok(());
        }
        let cmd = *stream.buffer.last().unwrap();
        unsafe { self.device.end_command_buffer(cmd)?; }
        stream.open = false;
        stream.pending_dispatches = 0;
        Ok(())
    }

    /// Upload enqueued into the stream (no submit).
    pub fn stream_upload(&self, dst: &GpuBuffer, data: &[u8]) -> Result<()> {
        self.stream_upload_at(dst, 0, data)
    }

    /// Upload into a target buffer **region**. The copy is recorded into the
    /// open command buffer and the staging buffer is released when the
    /// submission that carries it completes.
    pub fn stream_upload_at(
        &self,
        dst: &GpuBuffer,
        dst_offset: u64,
        data: &[u8],
    ) -> Result<()> {
        anyhow::ensure!(
            dst_offset + data.len() as u64 <= dst.size,
            "upload out of bounds: {dst_offset}+{} on {}",
            data.len(),
            dst.size
        );
        if data.is_empty() {
            return Ok(());
        }
        crate::stats::record_up(data.len() as u64);
        let mut staging = self.acquire_staging_upload(data.len() as u64)?;
        staging.write_mapped(data)?;
        let mut stream = self.stream.lock().unwrap();
        let cmd = self.ensure_buffer(&mut stream)?;
        self.stream_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_WRITE,
        );
        unsafe {
            let region = vk::BufferCopy::default()
                .dst_offset(dst_offset)
                .size(data.len() as u64);
            self.device
                .cmd_copy_buffer(cmd, staging.buffer, dst.buffer, &[region]);
        }
        // Staging is freed once the next submission (which carries this upload)
        // completes. There may be no open command buffer yet, so carry it on the
        // stream until `flush()` attaches it.
        stream.pending_deferred.push(Deferred::Staging(staging, true));
        Ok(())
    }

    /// Dispatch enqueued into the stream (no submit), whole buffers bound.
    pub fn stream_dispatch(
        &self,
        pipeline: &ComputePipeline,
        buffers: &[&GpuBuffer],
        push_constants: &[u8],
        groups: [u32; 3],
    ) -> Result<()> {
        let slices: Vec<BufferSlice<'_>> = buffers.iter().map(|b| (*b).into()).collect();
        self.stream_dispatch_slices(pipeline, &slices, push_constants, groups)
    }

    /// Like [`Self::stream_dispatch`], but each binding starts at an offset
    /// inside its buffer (tensors that are slices of an ORT allocation).
    pub fn stream_dispatch_slices(
        &self,
        pipeline: &ComputePipeline,
        buffers: &[BufferSlice<'_>],
        push_constants: &[u8],
        groups: [u32; 3],
    ) -> Result<()> {
        let should_flush = {
            let mut stream = self.stream.lock().unwrap();
            let cmd = self.ensure_buffer(&mut stream)?;
            self.stream_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            );
            self.record_dispatch(cmd, pipeline, buffers, push_constants, groups)?;
            // Profiling: the barrier before the dispatch guarantees all earlier
            // commands are done, and the timestamp AFTER it brackets exactly
            // this dispatch. Slot i = time before dispatch i; slot i+1 - slot
            // i = dispatch i's GPU time (read in `wait_and_reap`).
            if let Some(pool) = stream.open_query_pool {
                unsafe {
                    self.device
                        .cmd_write_timestamp(cmd, vk::PipelineStageFlags::ALL_COMMANDS, pool, stream.open_query_slots);
                }
                stream.open_query_slots += 1;
                stream.open_ts_ops.push(crate::stats::current_op());
            }
            stream.pending_dispatches += 1;
            crate::trace::record_dispatch();
            stream.pending_dispatches >= max_dispatches_per_submit()
        };
        if should_flush {
            self.flush()?;
        }
        Ok(())
    }

    /// GPU→GPU copy enqueued into the stream.
    pub fn stream_copy(&self, src: &GpuBuffer, dst: &GpuBuffer, bytes: u64) -> Result<()> {
        self.stream_copy_range(src, 0, dst, 0, bytes)
    }

    /// GPU→GPU copy of a **region**: used when a tensor occupies only a portion
    /// of a larger buffer (ORT, with the memory pattern enabled, allocates a
    /// single block and assigns slices to individual tensors).
    pub fn stream_copy_range(
        &self,
        src: &GpuBuffer,
        src_offset: u64,
        dst: &GpuBuffer,
        dst_offset: u64,
        bytes: u64,
    ) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        anyhow::ensure!(
            src_offset + bytes <= src.size && dst_offset + bytes <= dst.size,
            "copy out of bounds: src {src_offset}+{bytes}/{}, dst {dst_offset}+{bytes}/{}",
            src.size,
            dst.size
        );
        let mut stream = self.stream.lock().unwrap();
        let cmd = self.ensure_buffer(&mut stream)?;
        self.stream_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE,
        );
        unsafe {
            let region = vk::BufferCopy::default()
                .src_offset(src_offset)
                .dst_offset(dst_offset)
                .size(bytes);
            self.device
                .cmd_copy_buffer(cmd, src.buffer, dst.buffer, &[region]);
        }
        Ok(())
    }

    /// Destroys the buffer when safe: immediately if the stream is empty,
    /// otherwise after the current (in-flight) submission completes.
    pub fn defer_destroy(&self, buffer: GpuBuffer) {
        let mut stream = self.stream.lock().unwrap();
        // Attach to a real in-flight submission (the most recent) if one exists;
        // its fence signals only after every earlier in-flight submission has
        // executed (queue order), so the buffer is no longer referenced.
        if let Some(last) = stream.pending.last_mut() {
            if last.command_buffer != vk::CommandBuffer::null() {
                last.on_complete.push(Deferred::Buffer(buffer));
                return;
            }
        }
        // No real submission yet: carry it until the next flush attaches it.
        stream.pending_deferred.push(Deferred::Buffer(buffer));
    }

    /// Submits the open command buffer WITHOUT waiting, attaching a fresh
    /// fence. Applies back-pressure: if `MAX_IN_FLIGHT` submissions are already
    /// outstanding, waits for the oldest to complete first. That returns
    /// control to the caller between graphs so the next graph can be
    /// recorded/submitted while the waited one continues on the GPU.
    pub fn flush(&self) -> Result<()> {
        let mut stream = self.stream.lock().unwrap();
        if !stream.open {
            return Ok(()); // empty stream
        }
        if stream.pending.len() >= max_in_flight() {
            let oldest = stream.pending.remove(0);
            drop(stream);
            // Reap the oldest submission (waits its fence, runs its deferred
            // work, AND frees its command buffer + fence). Routing through
            // `wait_and_reap` here — rather than a bare wait + run_completion —
            // avoids leaking one command buffer and fence per back-pressure
            // reap, which accumulates into exhaustion when the in-flight cap
            // is 1 (every submission is reaped this way).
            self.wait_and_reap(oldest)?;
            stream = self.stream.lock().unwrap();
        }

        let cmd = *stream.buffer.last().unwrap();
        unsafe {
            self.device.end_command_buffer(cmd)?;
        }
        stream.open = false;
        stream.pending_dispatches = 0;

        let fence = unsafe {
            self.device
                .create_fence(&vk::FenceCreateInfo::default(), None)?
        };
        let cmds = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&cmds);
        if crate::trace::enabled() {
            eprintln!(
                "[trace] submit#{} dispatches={} (chunk cap {})",
                stream.pending.len(),
                stream.pending.len().saturating_sub(1),
                max_dispatches_per_submit()
            );
        }
        let t_submit = std::time::Instant::now();
        let result = unsafe {
            self.device
                .queue_submit(self.queue, &[submit], fence)
                .context("invio in coda (flush)")
        };
        if crate::trace::enabled() {
            eprintln!(
                "[trace] submit#{} queue_submit={}ms",
                stream.pending.len(),
                t_submit.elapsed().as_millis()
            );
        }
        let mut on_complete: Vec<Deferred> = std::mem::take(&mut stream.pending_deferred);
        let query_pool = stream.open_query_pool.take();
        let ts_ops = std::mem::take(&mut stream.open_ts_ops);
        stream.pending.push(Submission {
            command_buffer: cmd,
            fence,
            on_complete,
            dispatch_end: crate::trace::dispatch_seq(),
            query_pool,
            ts_ops,
        });
        result?;
        Ok(())
    }

    /// Submits (if open) and waits for the most recent submission. Used by the
    /// synchronous `dispatch` path and anywhere an immediate result is needed.
    pub fn flush_wait(&self) -> Result<()> {
        self.flush()?;
        let mut stream = self.stream.lock().unwrap();
        if let Some(sub) = stream.pending.pop() {
            drop(stream);
            unsafe {
                self.wait_and_reap(sub)?;
            }
        }
        Ok(())
    }

    /// Runs the deferred work attached to a completed submission.
    fn run_completion(&self, sub: Submission) -> Result<()> {
        for d in sub.on_complete {
            match d {
                Deferred::Buffer(b) => self.destroy_buffer(b),
                Deferred::Staging(b, is_upload) => self.release_staging(b, is_upload),
            }
        }
        // The submission's command buffer and fence are reclaimed by the caller
        // (wait_and_reap / finish_stream). Reset the descriptor arena only when
        // the stream is fully idle: while submissions are in flight their bound
        // sets must survive, AND so must the sets bound by the currently-OPEN
        // command buffer (it records dispatches against the arena before its
        // next flush). With the in-flight cap at 1, back-pressure reaps the
        // oldest submission precisely when `pending` is momentarily empty while
        // the next buffer is still recording — resetting here would invalidate
        // those live sets and the GPU would read garbage descriptors (the
        // Deck's non-finite depthformer logits).
        let stream = self.stream.lock().unwrap();
        if stream.pending.is_empty() && !stream.open {
            drop(stream);
            self.reset_descriptors();
        }
        Ok(())
    }

    /// Waits for a submission's fence, runs its deferred callbacks, then frees
    /// the fence and command buffer.
    fn wait_and_reap(&self, sub: Submission) -> Result<()> {
        let fence = sub.fence;
        let cmd = sub.command_buffer;
        let dispatch_end = sub.dispatch_end;
        let query_pool = sub.query_pool;
        let ts_ops = sub.ts_ops;
        let t_wait = std::time::Instant::now();
        unsafe {
            self.device
                .wait_for_fences(&[fence], true, u64::MAX)?;
        }
        crate::trace::record_flush_wait(t_wait.elapsed().as_nanos() as u64);
        if crate::trace::enabled() {
            eprintln!(
                "[trace] wait: dispatches<={} took {}ms",
                dispatch_end,
                t_wait.elapsed().as_millis()
            );
        }
        // Profiling: the fence guarantees every timestamp write in this
        // submission has landed, so the pool can be read and destroyed now.
        // Slot i is the GPU time after dispatch i (slot 0 = buffer start), so
        // dispatch i's duration is slot[i+1] - slot[i].
        if let Some(pool) = query_pool {
            let slots = ts_ops.len();
            if slots > 0 {
                // 64-bit query results: the spec returns 32-bit timestamps
                // without VK_QUERY_RESULT_64_BIT, and overflowing 32-bit
                // values wrap/saturate — intervals crossing the wrap read as
                // negative and were silently dropped by `ns.max(0.0)`.
                let mut data = vec![0u64; slots + 1];
                let status = unsafe {
                    self.device.get_query_pool_results(
                        pool,
                        0,
                        &mut data,
                        vk::QueryResultFlags::TYPE_64,
                    )
                };
                // The fence was waited above, so every timestamp in this
                // submission has landed; NOT_READY here would be a bug, not a
                // transient state.
                if status.is_ok() {
                    for (i, op) in ts_ops.iter().enumerate() {
                        // ticks -> ns: delta * period. f64 keeps sub-ns periods
                        // intact (casting the f32 period to u32 would truncate
                        // it to 0 on devices whose period is < 1).
                        let ns = (data[i + 1] as f64 - data[i] as f64)
                            * self.timestamp_period as f64;
                        crate::stats::record_gpu(op, ns.max(0.0) as u64);
                    }
                }
            }
            unsafe {
                self.device.destroy_query_pool(pool, None);
            }
        }
        // Rebuild a Submission holding only the on_complete work (the fields
        // moved out above were consumed), so `run_completion` still runs the
        // deferred buffer/staging releases.
        self.run_completion(Submission {
            command_buffer: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            on_complete: sub.on_complete,
            dispatch_end,
            query_pool: None,
            ts_ops: Vec::new(),
        })?;
        unsafe {
            self.device.destroy_fence(fence, None);
            self.device
                .free_command_buffers(self.command_pool, &[cmd]);
        }
        Ok(())
    }

    /// Marks the stream finished: drains all in-flight submissions (wait +
    /// callbacks + cleanup). Must be called before the `&'static` VkContext is
    /// reused for another session, since its Drop never fires (the context is a
    /// process-global `OnceLock`).
    pub fn finish_stream(&self) -> Result<()> {
        let mut stream = self.stream.lock().unwrap();
        while let Some(sub) = stream.pending.pop() {
            drop(stream);
            unsafe {
                self.wait_and_reap(sub)?;
            }
            stream = self.stream.lock().unwrap();
        }
        stream.open = false;
        Ok(())
    }

    /// Readback of `src` into a fresh host-visible staging buffer. The producer
    /// (open command buffer) is submitted first; the copy is recorded into a
    /// *new* command buffer submitted after it, so queue ordering guarantees
    /// the copy sees the producer's result. We wait only on the copy's own
    /// fence.
    pub fn stream_download(&self, src: &GpuBuffer, bytes: usize) -> Result<Vec<u8>> {
        self.stream_download_at(src, 0, bytes)
    }

    /// Download from a **region** of the source buffer.
    pub fn stream_download_at(
        &self,
        src: &GpuBuffer,
        src_offset: u64,
        bytes: usize,
    ) -> Result<Vec<u8>> {
        anyhow::ensure!(
            src_offset + bytes as u64 <= src.size,
            "download out of bounds: {src_offset}+{bytes} on {}",
            src.size
        );
        crate::stats::record_down(bytes as u64);
        let t_dl = std::time::Instant::now();
        // Submit any open (producer) command buffer so its result exists on the
        // GPU before we copy it. In-order use guarantees the open buffer is the
        // graph that produced `src` (the consumer graph is not yet recorded).
        self.flush()?;
        let _guard = self.submit_lock.lock().unwrap();
        let staging = self.acquire_staging_download(bytes as u64)?;
        unsafe {
            let cmd = self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];
            self.device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
            let region = vk::BufferCopy::default()
                .src_offset(src_offset)
                .size(bytes as u64);
            self.device
                .cmd_copy_buffer(cmd, src.buffer, staging.buffer, &[region]);
            self.device.end_command_buffer(cmd)?;
            let fence = self
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)?;
            let cmds = [cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            self.device.queue_submit(self.queue, &[submit], fence)?;
            self.device.wait_for_fences(&[fence], true, u64::MAX)?;
            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(self.command_pool, &cmds);
        }
        let data = staging.read_mapped(bytes)?;
        self.release_staging(staging, false);
        crate::trace::record_download(t_dl.elapsed().as_nanos() as u64);
        Ok(data)
    }
}
