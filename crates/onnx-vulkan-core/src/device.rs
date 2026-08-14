//! Tensori residenti sul backend Vulkan.

use crate::HostTensor;
use std::sync::Arc;
use vk_compute::GpuBuffer;
use vk_compute::VkContext;

/// A device allocation whose last reference returns it to the Vulkan context.
/// This is used for generation state that must survive one graph run without a
/// host readback (for example transformer KV caches).
pub struct SharedDeviceBuffer<'a> {
    context: &'a VkContext,
    buffer: Option<GpuBuffer>,
}

impl SharedDeviceBuffer<'_> {
    pub fn get(&self) -> &GpuBuffer {
        self.buffer.as_ref().expect("shared Vulkan buffer is alive")
    }
}

impl Drop for SharedDeviceBuffer<'_> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.context.defer_destroy(buffer);
        }
    }
}

/// Vulkan storage owned by runtime or borrowed from host.
pub enum DeviceBuffer<'a> {
    Owned(GpuBuffer),
    Borrowed(&'a GpuBuffer),
    Shared(Arc<SharedDeviceBuffer<'a>>),
}

impl DeviceBuffer<'_> {
    pub fn get(&self) -> &GpuBuffer {
        match self {
            Self::Owned(buffer) => buffer,
            Self::Borrowed(buffer) => buffer,
            Self::Shared(buffer) => buffer.get(),
        }
    }
}

/// Tensore residente in VRAM.
pub struct DeviceTensor<'a> {
    pub dtype: i32,
    pub shape: Vec<i64>,
    pub elem_count: usize,
    pub buf: DeviceBuffer<'a>,
}

impl DeviceTensor<'_> {
    pub fn buffer(&self) -> &GpuBuffer {
        self.buf.get()
    }
}

/// Cloneable device-resident tensor suitable for feeding a later graph run.
#[derive(Clone)]
pub struct PersistentTensor<'a> {
    pub dtype: i32,
    pub shape: Vec<i64>,
    pub elem_count: usize,
    buffer: Arc<SharedDeviceBuffer<'a>>,
}

impl<'a> PersistentTensor<'a> {
    pub fn from_owned(context: &'a VkContext, tensor: DeviceTensor<'_>) -> crate::Result<Self> {
        let DeviceBuffer::Owned(buffer) = tensor.buf else {
            return Err(crate::Error::InvalidTensor(
                "only an owned graph output can become persistent".into(),
            ));
        };
        Ok(Self {
            dtype: tensor.dtype,
            shape: tensor.shape,
            elem_count: tensor.elem_count,
            buffer: Arc::new(SharedDeviceBuffer {
                context,
                buffer: Some(buffer),
            }),
        })
    }

    pub fn as_tensor(&self) -> Tensor<'a> {
        Tensor::Device(DeviceTensor {
            dtype: self.dtype,
            shape: self.shape.clone(),
            elem_count: self.elem_count,
            buf: DeviceBuffer::Shared(Arc::clone(&self.buffer)),
        })
    }
}

/// Execution value, resident on host or device.
pub enum Tensor<'a> {
    Device(DeviceTensor<'a>),
    Host(HostTensor),
}
