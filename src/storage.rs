use std::sync::{Arc, Mutex};

use crate::cuda::CudaFloat32Storage;

use crate::device::Device;
use crate::dtype::DType;
use crate::tensor_error::TensorError;

/// Crate-private float32 storage shared by tensors and autograd snapshots.
pub(crate) struct Storage {
    payload: StoragePayload,
}

enum StoragePayload {
    CpuFloat32(StorageData<f32>),
    CudaFloat32(CudaFloat32Storage),
}

enum StorageData<T> {
    // Scalar reductions can share the Storage allocation with their payload.
    Inline(T),
    Owned(Vec<T>),
    SharedGradient(Mutex<Vec<T>>),
}

impl StorageData<f32> {
    fn len(&self) -> usize {
        self.with_values(<[f32]>::len)
    }

    fn data_ptr(&self) -> *const u8 {
        // Gradient accumulation only updates existing elements, so a pointer
        // obtained while holding the shared-gradient lock remains stable after
        // the guard is released.
        self.with_values(|values| values.as_ptr().cast())
    }

    fn owned_values(&self) -> Option<&[f32]> {
        match self {
            Self::Inline(value) => Some(std::slice::from_ref(value)),
            Self::Owned(values) => Some(values),
            Self::SharedGradient(_) => None,
        }
    }

    fn value(&self, index: usize) -> Option<f32> {
        self.with_values(|values| values.get(index).copied())
    }

    fn try_copy_values<E>(
        &self,
        copy: impl FnOnce(&[f32]) -> Result<Vec<f32>, E>,
    ) -> Result<Vec<f32>, E> {
        self.with_values(copy)
    }

    fn copy_range(&self, start: usize, end: usize) -> Vec<f32> {
        self.with_values(|values| values[start..end].to_vec())
    }

    fn with_shared_gradient_range<R>(
        &self,
        start: usize,
        end: usize,
        read: impl FnOnce(&[f32]) -> R,
    ) -> Option<R> {
        let Self::SharedGradient(values) = self else {
            return None;
        };
        let values = values
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        values.get(start..end).map(read)
    }

    fn into_range(self, start: usize, end: usize) -> Vec<f32> {
        let values = match self {
            Self::Inline(value) => return std::slice::from_ref(&value)[start..end].to_vec(),
            Self::Owned(values) => values,
            Self::SharedGradient(values) => values
                .into_inner()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        };
        if start == 0 && end == values.len() {
            values
        } else {
            values[start..end].to_vec()
        }
    }

    fn try_clone_for_saved<S, E>(
        &self,
        copy: impl FnOnce(&[f32]) -> Result<Vec<f32>, E>,
        reuse: impl FnOnce() -> S,
        from_snapshot: impl FnOnce(Vec<f32>) -> S,
    ) -> Result<S, E> {
        match self {
            Self::Inline(_) | Self::Owned(_) => Ok(reuse()),
            Self::SharedGradient(values) => {
                let values = values
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                copy(&values).map(from_snapshot)
            }
        }
    }

    fn accumulate_shared_gradient(&self, contribution: Vec<f32>) {
        match self {
            Self::Inline(_) | Self::Owned(_) => {
                unreachable!("leaf gradients always use shared gradient storage");
            }
            Self::SharedGradient(existing) => {
                let mut existing = existing
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                debug_assert_eq!(existing.len(), contribution.len());
                for (value, contribution) in existing.iter_mut().zip(contribution) {
                    *value += contribution;
                }
            }
        }
    }

    fn with_values<R>(&self, read: impl FnOnce(&[f32]) -> R) -> R {
        match self {
            Self::Inline(value) => read(std::slice::from_ref(value)),
            Self::Owned(values) => read(values),
            Self::SharedGradient(values) => {
                let values = values
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                read(&values)
            }
        }
    }
}

impl Storage {
    pub(crate) fn from_scalar(value: f32, dtype: DType, device: Device) -> Self {
        Self {
            payload: match (device, dtype) {
                (Device::Cpu, DType::Float32) => {
                    StoragePayload::CpuFloat32(StorageData::Inline(value))
                }
                _ => unreachable!("non-CPU storage must use a device-specific constructor"),
            },
        }
    }

    pub(crate) fn from_owned(data: Vec<f32>, dtype: DType, device: Device) -> Self {
        Self {
            payload: match (device, dtype) {
                (Device::Cpu, DType::Float32) => {
                    StoragePayload::CpuFloat32(StorageData::Owned(data))
                }
                _ => unreachable!("non-CPU storage must use a device-specific constructor"),
            },
        }
    }

    pub(crate) fn from_shared_gradient(data: Vec<f32>, dtype: DType, device: Device) -> Self {
        Self {
            payload: match (device, dtype) {
                (Device::Cpu, DType::Float32) => {
                    StoragePayload::CpuFloat32(StorageData::SharedGradient(Mutex::new(data)))
                }
                _ => unreachable!("non-CPU storage must use a device-specific constructor"),
            },
        }
    }

    pub(crate) fn cuda_zeros_float32(
        elements: usize,
        device_index: usize,
    ) -> Result<Self, TensorError> {
        Ok(Self {
            payload: StoragePayload::CudaFloat32(CudaFloat32Storage::zeros(
                elements,
                device_index,
            )?),
        })
    }

    pub(crate) fn cuda_from_host_float32(
        values: &[f32],
        device_index: usize,
    ) -> Result<Self, TensorError> {
        Ok(Self {
            payload: StoragePayload::CudaFloat32(CudaFloat32Storage::from_host(
                values,
                device_index,
            )?),
        })
    }

    pub(crate) fn cuda_add_float32(
        &self,
        left_offset: usize,
        other: &Self,
        right_offset: usize,
        elements: usize,
    ) -> Result<Self, TensorError> {
        match (&self.payload, &other.payload) {
            (StoragePayload::CudaFloat32(left), StoragePayload::CudaFloat32(right)) => Ok(Self {
                payload: StoragePayload::CudaFloat32(left.add(
                    left_offset,
                    right,
                    right_offset,
                    elements,
                )?),
            }),
            _ => Err(TensorError::UnsupportedCudaAddition {
                reason: "mixed devices",
            }),
        }
    }

    #[cfg(any(feature = "python-bindings", test))]
    pub(crate) fn cuda_negate_float32(
        &self,
        offset: usize,
        elements: usize,
    ) -> Result<Self, TensorError> {
        match &self.payload {
            StoragePayload::CudaFloat32(input) => Ok(Self {
                payload: StoragePayload::CudaFloat32(input.negate(offset, elements)?),
            }),
            StoragePayload::CpuFloat32(_) => Err(TensorError::UnsupportedCudaNegation {
                reason: "input must be CUDA",
            }),
        }
    }

    pub(crate) fn cuda_mul_scalar_float32(
        &self,
        offset: usize,
        elements: usize,
        scalar: f32,
    ) -> Result<Self, TensorError> {
        match &self.payload {
            StoragePayload::CudaFloat32(input) => Ok(Self {
                payload: StoragePayload::CudaFloat32(input.mul_scalar(offset, elements, scalar)?),
            }),
            StoragePayload::CpuFloat32(_) => {
                Err(TensorError::UnsupportedCudaScalarMultiplication {
                    reason: "input must be CUDA",
                })
            }
        }
    }

    pub(crate) fn len(&self) -> usize {
        match &self.payload {
            StoragePayload::CpuFloat32(data) => data.len(),
            StoragePayload::CudaFloat32(data) => data.elements,
        }
    }

    pub(crate) const fn dtype(&self) -> DType {
        match &self.payload {
            StoragePayload::CpuFloat32(_) | StoragePayload::CudaFloat32(_) => DType::Float32,
        }
    }

    pub(crate) const fn device(&self) -> Device {
        match &self.payload {
            StoragePayload::CpuFloat32(_) => Device::Cpu,
            StoragePayload::CudaFloat32(data) => Device::Cuda(data.device_index),
        }
    }

    pub(crate) fn data_ptr(&self) -> *const u8 {
        match &self.payload {
            StoragePayload::CpuFloat32(data) => data.data_ptr(),
            StoragePayload::CudaFloat32(data) => data.data_ptr as *const u8,
        }
    }

    pub(crate) fn with_cpu_values<R>(&self, read: impl FnOnce(&[f32]) -> R) -> R {
        match &self.payload {
            StoragePayload::CpuFloat32(data) => data.with_values(read),
            StoragePayload::CudaFloat32(_) => unreachable!("CPU kernel requires CPU storage"),
        }
    }

    pub(crate) fn owned_values(&self) -> Option<&[f32]> {
        match &self.payload {
            StoragePayload::CpuFloat32(data) => data.owned_values(),
            StoragePayload::CudaFloat32(_) => None,
        }
    }

    pub(crate) fn value(&self, index: usize) -> Option<f32> {
        match &self.payload {
            StoragePayload::CpuFloat32(data) => data.value(index),
            StoragePayload::CudaFloat32(_) => {
                let _ = index;
                None
            }
        }
    }

    pub(crate) fn try_copy_values<E>(
        &self,
        copy: impl FnOnce(&[f32]) -> Result<Vec<f32>, E>,
    ) -> Result<Vec<f32>, E> {
        match &self.payload {
            StoragePayload::CpuFloat32(data) => data.try_copy_values(copy),
            StoragePayload::CudaFloat32(_) => unreachable!(
                "CUDA storage must be copied through the synchronized device-to-host path"
            ),
        }
    }

    pub(crate) fn copy_range(&self, start: usize, end: usize) -> Vec<f32> {
        match &self.payload {
            StoragePayload::CpuFloat32(data) => data.copy_range(start, end),
            StoragePayload::CudaFloat32(_) => unreachable!(
                "CUDA storage ranges must be copied through the synchronized device-to-host path"
            ),
        }
    }

    pub(crate) fn with_shared_gradient_range<R>(
        &self,
        start: usize,
        end: usize,
        read: impl FnOnce(&[f32]) -> R,
    ) -> Option<R> {
        match &self.payload {
            StoragePayload::CpuFloat32(data) => data.with_shared_gradient_range(start, end, read),
            StoragePayload::CudaFloat32(_) => None,
        }
    }

    pub(crate) fn into_range(self, start: usize, end: usize) -> Vec<f32> {
        match self.payload {
            StoragePayload::CpuFloat32(data) => data.into_range(start, end),
            StoragePayload::CudaFloat32(_) => {
                let _ = (start, end);
                unreachable!("CUDA storage cannot be moved into a host range")
            }
        }
    }

    pub(crate) fn try_clone_for_saved<E>(
        storage: &Arc<Self>,
        copy: impl FnOnce(&[f32]) -> Result<Vec<f32>, E>,
    ) -> Result<Arc<Self>, E> {
        match &storage.payload {
            StoragePayload::CpuFloat32(data) => data.try_clone_for_saved(
                copy,
                || Arc::clone(storage),
                |values| {
                    Arc::new(Self {
                        payload: StoragePayload::CpuFloat32(StorageData::Owned(values)),
                    })
                },
            ),
            StoragePayload::CudaFloat32(_) => Ok(Arc::clone(storage)),
        }
    }

    pub(crate) fn accumulate_shared_gradient(&self, contribution: Vec<f32>) {
        match &self.payload {
            StoragePayload::CpuFloat32(data) => data.accumulate_shared_gradient(contribution),
            StoragePayload::CudaFloat32(_) => {
                let _ = contribution;
                unreachable!("CUDA gradients are not supported")
            }
        }
    }

    pub(crate) fn copy_cuda_to_cpu_float32(
        &self,
        start: usize,
        elements: usize,
    ) -> Result<Vec<f32>, TensorError> {
        match &self.payload {
            StoragePayload::CpuFloat32(data) => Ok(data.copy_range(start, start + elements)),
            StoragePayload::CudaFloat32(data) => data.copy_range(start, elements),
        }
    }

    pub(crate) fn copy_cuda_regions_to_cpu_float32(
        &self,
        elements: usize,
        regions: impl IntoIterator<Item = Result<crate::cuda::CudaCopyRegion, TensorError>>,
    ) -> Result<Vec<f32>, TensorError> {
        match &self.payload {
            StoragePayload::CudaFloat32(data) => data.copy_regions(elements, regions),
            StoragePayload::CpuFloat32(_) => Err(TensorError::UnsupportedDevice {
                operation: "CUDA transfer",
                device: Device::Cpu,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::Arc;

    use crate::device::Device;
    use crate::dtype::DType;

    use super::{Storage, StorageData, StoragePayload};

    fn poison_shared_gradient(storage: &Storage) {
        let StoragePayload::CpuFloat32(StorageData::SharedGradient(values)) = &storage.payload
        else {
            panic!("expected shared gradient storage");
        };
        let result = catch_unwind(AssertUnwindSafe(|| {
            let _guard = values.lock().unwrap();
            panic!("poison shared gradient storage");
        }));
        assert!(result.is_err());
    }

    #[test]
    fn owned_float32_payload_preserves_pointer_and_copy_paths() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let pointer = values.as_ptr().cast();
        let storage = Storage::from_owned(values, DType::Float32, Device::Cpu);

        assert_eq!(storage.dtype(), DType::Float32);
        assert_eq!(storage.device(), Device::Cpu);
        assert_eq!(storage.len(), 4);
        assert_eq!(storage.data_ptr(), pointer);
        assert_eq!(
            storage.owned_values(),
            Some([1.0, 2.0, 3.0, 4.0].as_slice())
        );
        assert_eq!(storage.value(2), Some(3.0));
        assert_eq!(storage.value(4), None);
        assert_eq!(storage.with_shared_gradient_range(1, 3, <[f32]>::len), None);
        assert_eq!(
            storage
                .try_copy_values(|values| Ok::<_, Infallible>(values.to_vec()))
                .unwrap(),
            [1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(storage.copy_range(1, 3), [2.0, 3.0]);

        let values = vec![5.0, 6.0, 7.0];
        let pointer = values.as_ptr();
        let values = Storage::from_owned(values, DType::Float32, Device::Cpu).into_range(0, 3);
        assert_eq!(values.as_ptr(), pointer);
        assert_eq!(values, [5.0, 6.0, 7.0]);
    }

    #[test]
    fn inline_float32_payload_preserves_pointer_and_copy_paths() {
        let storage = Storage::from_scalar(-0.0, DType::Float32, Device::Cpu);
        let pointer = storage.data_ptr();

        assert_eq!(storage.dtype(), DType::Float32);
        assert_eq!(storage.device(), Device::Cpu);
        assert_eq!(storage.len(), 1);
        assert_eq!(
            storage.owned_values().unwrap()[0].to_bits(),
            (-0.0_f32).to_bits()
        );
        assert_eq!(storage.value(0).unwrap().to_bits(), (-0.0_f32).to_bits());
        assert_eq!(storage.value(1), None);
        assert_eq!(storage.with_shared_gradient_range(0, 1, <[f32]>::len), None);
        assert_eq!(
            storage
                .try_copy_values(|values| Ok::<_, Infallible>(values.to_vec()))
                .unwrap()[0]
                .to_bits(),
            (-0.0_f32).to_bits()
        );
        assert_eq!(storage.data_ptr(), pointer);
        assert_eq!(storage.copy_range(0, 1)[0].to_bits(), (-0.0_f32).to_bits());

        let values = Storage::from_scalar(-0.0, DType::Float32, Device::Cpu).into_range(0, 1);
        assert_eq!(values[0].to_bits(), (-0.0_f32).to_bits());
    }

    #[test]
    fn shared_gradient_float32_payload_preserves_pointer_and_accumulates() {
        let values = vec![1.0, -2.0, 3.5];
        let pointer = values.as_ptr().cast();
        let storage = Storage::from_shared_gradient(values, DType::Float32, Device::Cpu);

        assert_eq!(storage.dtype(), DType::Float32);
        assert_eq!(storage.device(), Device::Cpu);
        assert_eq!(storage.len(), 3);
        assert_eq!(storage.data_ptr(), pointer);
        assert_eq!(storage.owned_values(), None);
        assert_eq!(storage.copy_range(0, 2), [1.0, -2.0]);
        assert_eq!(
            storage.with_shared_gradient_range(1, 3, <[f32]>::to_vec),
            Some(vec![-2.0, 3.5])
        );
        assert_eq!(storage.with_shared_gradient_range(2, 4, <[f32]>::len), None);

        storage.accumulate_shared_gradient(vec![2.0, 4.0, -1.5]);

        assert_eq!(storage.data_ptr(), pointer);
        assert_eq!(
            storage
                .try_copy_values(|values| Ok::<_, Infallible>(values.to_vec()))
                .unwrap(),
            [3.0, 2.0, 2.0]
        );

        let values = vec![8.0, 9.0];
        let pointer = values.as_ptr();
        let values =
            Storage::from_shared_gradient(values, DType::Float32, Device::Cpu).into_range(0, 2);
        assert_eq!(values.as_ptr(), pointer);
        assert_eq!(values, [8.0, 9.0]);
    }

    #[test]
    fn saved_storage_reuses_owned_payload_and_snapshots_shared_gradient() {
        let owned = Arc::new(Storage::from_owned(
            vec![1.0, 2.0],
            DType::Float32,
            Device::Cpu,
        ));
        let saved = Storage::try_clone_for_saved(&owned, |_| -> Result<Vec<f32>, Infallible> {
            panic!("owned saved storage must not be copied");
        })
        .unwrap();
        assert!(Arc::ptr_eq(&saved, &owned));

        let shared = Arc::new(Storage::from_shared_gradient(
            vec![3.0, 4.0],
            DType::Float32,
            Device::Cpu,
        ));
        let shared_pointer = shared.data_ptr();
        let saved =
            Storage::try_clone_for_saved(&shared, |values| Ok::<_, Infallible>(values.to_vec()))
                .unwrap();

        assert!(!Arc::ptr_eq(&saved, &shared));
        assert_ne!(saved.data_ptr(), shared_pointer);
        assert_eq!(saved.dtype(), DType::Float32);
        assert_eq!(saved.device(), Device::Cpu);
        assert_eq!(saved.owned_values(), Some([3.0, 4.0].as_slice()));

        shared.accumulate_shared_gradient(vec![5.0, 6.0]);
        assert_eq!(saved.owned_values(), Some([3.0, 4.0].as_slice()));
        assert_eq!(
            shared
                .try_copy_values(|values| Ok::<_, Infallible>(values.to_vec()))
                .unwrap(),
            [8.0, 10.0]
        );
    }

    #[test]
    fn shared_gradient_operations_recover_from_poison() {
        let values = vec![1.0, 2.0, 3.0];
        let pointer = values.as_ptr();
        let storage = Arc::new(Storage::from_shared_gradient(
            values,
            DType::Float32,
            Device::Cpu,
        ));
        poison_shared_gradient(&storage);

        assert_eq!(storage.len(), 3);
        assert_eq!(storage.data_ptr(), pointer.cast());
        assert_eq!(storage.value(1), Some(2.0));
        assert_eq!(storage.copy_range(1, 3), [2.0, 3.0]);
        assert_eq!(
            storage.with_shared_gradient_range(0, 3, |values| {
                values
                    .iter()
                    .copied()
                    .fold(0.0_f32, |total, value| total + value)
            }),
            Some(6.0)
        );
        assert_eq!(
            storage
                .try_copy_values(|values| Ok::<_, Infallible>(values.to_vec()))
                .unwrap(),
            [1.0, 2.0, 3.0]
        );

        let saved =
            Storage::try_clone_for_saved(&storage, |values| Ok::<_, Infallible>(values.to_vec()))
                .unwrap();
        assert_eq!(saved.owned_values(), Some([1.0, 2.0, 3.0].as_slice()));

        storage.accumulate_shared_gradient(vec![3.0, 2.0, 1.0]);
        assert_eq!(storage.data_ptr(), pointer.cast());
        assert_eq!(storage.copy_range(0, 3), [4.0, 4.0, 4.0]);

        let Ok(storage) = Arc::try_unwrap(storage) else {
            panic!("test must hold the only shared storage reference");
        };
        let values = storage.into_range(0, 3);
        assert_eq!(values.as_ptr(), pointer);
        assert_eq!(values, [4.0, 4.0, 4.0]);
    }
}
