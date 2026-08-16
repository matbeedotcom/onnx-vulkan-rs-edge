//! Pipeline compute + dispatch sincrono.
//!
//! Pipeline-creation diagnostics (`ONNX_VULKAN_PIPELINE_TRACE=1`, default):
//! every `vkCreateComputePipelines` is wall-timed and, when the device
//! supports `VK_EXT_pipeline_creation_feedback`, reports the driver's own
//! creation duration and whether the APPLICATION pipeline cache supplied a
//! usable pipeline. This is the measurement that settles whether a slow first
//! inference is ACO compilation hiding in pipeline creation or something in
//! the submission/queue path. `ONNX_VULKAN_CACHE_VERIFY=1` additionally
//! creates with `VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED` so a
//! genuine cache hit succeeds while any required compilation is reported as a
//! miss without doing the work — a unit test for the persistent cache.

use crate::buffer::GpuBuffer;
use crate::context::VkContext;
use anyhow::Result;
use ash::vk;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

/// Counter of pipeline creations in this process, for the `[pipeline]` lines.
static PIPELINE_CREATIONS: AtomicUsize = AtomicUsize::new(0);

/// Range of buffer bound to binding: descriptor starts at `offset`,
/// so shader indexes starting from tensor origin. Needed for ORT's memory
/// pattern, which assigns tensor slices within a single allocation.
#[derive(Clone, Copy)]
pub struct BufferSlice<'a> {
    pub buf: &'a GpuBuffer,
    pub offset: u64,
}

impl<'a> From<&'a GpuBuffer> for BufferSlice<'a> {
    fn from(buf: &'a GpuBuffer) -> Self {
        Self { buf, offset: 0 }
    }
}

pub struct ComputePipeline {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    shader_module: vk::ShaderModule,
    num_buffers: u32,
    push_const_size: u32,
}

impl VkContext {
    /// Compute pipeline from SPIR-V: `num_buffers` storage buffers (binding 0..n)
    /// + push constants of `push_const_size` bytes (0 = none).
    pub fn create_pipeline(
        &self,
        spirv: &[u32],
        num_buffers: u32,
        push_const_size: u32,
    ) -> Result<ComputePipeline> {
        self.create_pipeline_forced(spirv, num_buffers, push_const_size, None)
    }

    /// Like [`create_pipeline`], but with a forced subgroup (wave) width via
    /// `VK_EXT_subgroup_size_control`. `required_subgroup_size` must divide the
    /// shader's `@workgroup_size` (all matmul kernels use 256). Used to
    /// benchmark the same kernel against wave32 vs the device-default wave64;
    /// only the matmul pipeline opts in — the integer cooperative-matrix kernel
    /// is compiled for one specific device subgroup size and must not be forced.
    pub fn create_pipeline_forced(
        &self,
        spirv: &[u32],
        num_buffers: u32,
        push_const_size: u32,
        required_subgroup_size: Option<u32>,
    ) -> Result<ComputePipeline> {
        let device = &self.device;
        unsafe {
            let shader_module = device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(spirv), None)?;

            let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..num_buffers)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let set_layout = device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?;

            let set_layouts = [set_layout];
            let mut layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
            let push_range = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .size(push_const_size)];
            if push_const_size > 0 {
                layout_info = layout_info.push_constant_ranges(&push_range);
            }
            let layout = device.create_pipeline_layout(&layout_info, None)?;

            // The size-control struct's address is stored in `stage.p_next` and
            // is dereferenced by the driver during `create_compute_pipelines`
            // below, so it must outlive that call. Declare it in the outer
            // scope; `Option::as_mut` gives a temporary `&mut` for `push_next`
            // only.
            let mut size_info_opt: Option<
                vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo<'_>,
            > = None;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader_module)
                .name(c"main");
            let stage = if let Some(size) = required_subgroup_size {
                anyhow::ensure!(
                    self.subgroup_size_control,
                    "forced subgroup size {size} requested but the device lacks VK_EXT_subgroup_size_control"
                );
                size_info_opt = Some(
                    vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
                        .required_subgroup_size(size),
                );
                stage.push_next(size_info_opt.as_mut().expect("set just above"))
            } else {
                stage
            };
            let info = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout);
            // Reuse the persistent pipeline cache so identical SPIR-V is not
            // recompiled every cold start (RADV/ACO compile is minutes on Deck).
            let cache_handle = self
                .pipeline_cache
                .lock()
                .unwrap()
                .as_ref()
                .copied()
                .unwrap_or(vk::PipelineCache::null());

            // Diagnostics: wall-time the creation (the call where ACO compile
            // would show up) and read the driver's own feedback. The feedback
            // struct's address is stored in `info.p_next` and dereferenced by
            // the driver during the call below, so it must outlive it.
            let pipeline_trace = std::env::var_os("ONNX_VULKAN_PIPELINE_TRACE")
                .map(|v| v != "0")
                .unwrap_or(true);
            let cache_verify = std::env::var_os("ONNX_VULKAN_CACHE_VERIFY").is_some();
            let mut feedback = vk::PipelineCreationFeedback::default();
            let mut feedback_info: Option<
                vk::PipelineCreationFeedbackCreateInfo<'_>,
            > = None;
            if pipeline_trace && self.pipeline_creation_feedback {
                let mut fi = vk::PipelineCreationFeedbackCreateInfo::default();
                fi.p_pipeline_creation_feedback = &mut feedback;
                feedback_info = Some(fi);
                info.push_next(
                    feedback_info
                        .as_mut()
                        .expect("set just above"),
                );
            }
            // ONNX_VULKAN_CACHE_VERIFY (with VK_EXT_pipeline_creation_cache_
            // control): create with FAIL_ON_PIPELINE_COMPILE_REQUIRED so a
            // genuine application-cache hit succeeds while any required
            // compilation fails FAST without doing the work — a unit test for
            // the persistent cache. A reported miss is then created normally.
            let verify_flags = if cache_verify && self.pipeline_creation_cache_control {
                vk::PipelineCreateFlags::FAIL_ON_PIPELINE_COMPILE_REQUIRED
            } else {
                vk::PipelineCreateFlags::empty()
            };
            let pipeline_creation_feedback_attached = feedback_info.is_some();
            let t_create = Instant::now();
            let mut make = |flags: vk::PipelineCreateFlags| -> Result<vk::Pipeline> {
                let mut info = info;
                info.flags = flags;
                device
                    .create_compute_pipelines(cache_handle, &[info], None)
                    .map(|p| p[0])
                    .map_err(|(_, e)| anyhow::anyhow!("vkCreateComputePipelines: {e:#}"))
            };
            let pipeline = match make(verify_flags) {
                Ok(p) => p,
                Err(e) if cache_verify && e.to_string().contains("PIPELINE_COMPILE_REQUIRED") => {
                    let label = crate::stats::current_op();
                    eprintln!(
                        "[pipeline] #{:<4} {} MISS compile-required (wall {}ms)",
                        PIPELINE_CREATIONS.fetch_add(1, Ordering::Relaxed) + 1,
                        label,
                        t_create.elapsed().as_millis(),
                    );
                    make(vk::PipelineCreateFlags::empty())?
                }
                Err(e) => {
                    return Err(anyhow::anyhow!(
                        "vkCreateComputePipelines failed for op '{}': {e:#}",
                        crate::stats::current_op()
                    ))
                }
            };
            drop(size_info_opt); // keep alive until after creation
            if pipeline_trace {
                let label = crate::stats::current_op();
                let idx = PIPELINE_CREATIONS.fetch_add(1, Ordering::Relaxed) + 1;
                let wall_ms = t_create.elapsed().as_millis();
                if pipeline_creation_feedback_attached {
                    let cache_hit = feedback.flags.contains(
                        vk::PipelineCreationFeedbackFlags::APPLICATION_PIPELINE_CACHE_HIT,
                    );
                    eprintln!(
                        "[pipeline] #{:<4} {} create_wall={}ms driver={}ms cache_hit={}",
                        idx,
                        label,
                        wall_ms,
                        feedback.duration as f64 / 1e6,
                        cache_hit,
                    );
                } else {
                    eprintln!(
                        "[pipeline] #{:<4} {} create_wall={}ms (feedback n/a)",
                        idx, label, wall_ms,
                    );
                }
            }
            drop(feedback_info);

            Ok(ComputePipeline {
                pipeline,
                layout,
                set_layout,
                shader_module,
                num_buffers,
                push_const_size,
            })
        }
    }

    /// Records a dispatch in the given command buffer. Descriptor set is
    /// allocated from the persistent arena (reset on flush).
    pub(crate) fn record_dispatch(
        &self,
        cmd: vk::CommandBuffer,
        pipeline: &ComputePipeline,
        buffers: &[BufferSlice<'_>],
        push_constants: &[u8],
        groups: [u32; 3],
    ) -> Result<()> {
        assert_eq!(buffers.len() as u32, pipeline.num_buffers);
        assert_eq!(push_constants.len() as u32, pipeline.push_const_size);
        for b in buffers {
            anyhow::ensure!(
                b.offset % self.storage_offset_alignment == 0,
                "offset {} not aligned to {} (minStorageBufferOffsetAlignment)",
                b.offset,
                self.storage_offset_alignment
            );
            anyhow::ensure!(
                b.offset < b.buf.size,
                "offset {} outside the buffer of {} bytes",
                b.offset,
                b.buf.size
            );
        }
        let set = self.acquire_descriptor_set(pipeline.set_layout)?;
        let device = &self.device;
        unsafe {
            let buffer_infos: Vec<[vk::DescriptorBufferInfo; 1]> = buffers
                .iter()
                .map(|b| {
                    [vk::DescriptorBufferInfo::default()
                        .buffer(b.buf.buffer)
                        .offset(b.offset)
                        .range(vk::WHOLE_SIZE)]
                })
                .collect();
            let writes: Vec<vk::WriteDescriptorSet> = buffer_infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(info)
                })
                .collect();
            device.update_descriptor_sets(&writes, &[]);

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout,
                0,
                &[set],
                &[],
            );
            if !push_constants.is_empty() {
                device.cmd_push_constants(
                    cmd,
                    pipeline.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push_constants,
                );
            }
            device.cmd_dispatch(cmd, groups[0], groups[1], groups[2]);
            Ok(())
        }
    }

    /// Synchronous dispatch (test/standalone use): enqueues into the stream and flushes.
    pub fn dispatch(
        &self,
        pipeline: &ComputePipeline,
        buffers: &[&GpuBuffer],
        push_constants: &[u8],
        groups: [u32; 3],
    ) -> Result<()> {
        self.stream_dispatch(pipeline, buffers, push_constants, groups)?;
        self.flush()
    }

    pub fn destroy_pipeline(&self, pipeline: ComputePipeline) {
        unsafe {
            self.device.destroy_pipeline(pipeline.pipeline, None);
            self.device.destroy_pipeline_layout(pipeline.layout, None);
            self.device
                .destroy_descriptor_set_layout(pipeline.set_layout, None);
            self.device
                .destroy_shader_module(pipeline.shader_module, None);
        }
    }
}
