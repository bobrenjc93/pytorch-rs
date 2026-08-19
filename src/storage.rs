use std::sync::{Arc, Mutex};

use crate::device::Device;
use crate::dtype::DType;

/// Crate-private CPU float32 storage shared by tensors and autograd snapshots.
pub(crate) struct Storage {
    payload: StoragePayload,
}

enum StoragePayload {
    CpuFloat32(StorageData<f32>),
}

enum StorageData<T> {
    Owned(Vec<T>),
    SharedGradient(Mutex<Vec<T>>),
}

impl StorageData<f32> {
    fn from_owned(values: Vec<f32>) -> Self {
        Self::Owned(values)
    }

    fn from_shared_gradient(values: Vec<f32>) -> Self {
        Self::SharedGradient(Mutex::new(values))
    }

    fn with_values<R>(&self, read: impl FnOnce(&[f32]) -> R) -> R {
        match self {
            Self::Owned(values) => read(values),
            Self::SharedGradient(values) => {
                let values = values
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                read(&values)
            }
        }
    }

    fn len(&self) -> usize {
        self.with_values(<[f32]>::len)
    }

    fn data_ptr(&self) -> *const u8 {
        // Gradient accumulation only updates existing elements, so a shared
        // gradient's allocation remains stable after its guard is released.
        self.with_values(|values| values.as_ptr().cast())
    }

    fn owned_values(&self) -> Option<&[f32]> {
        match self {
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

    fn into_range(self, start: usize, end: usize) -> Vec<f32> {
        let values = match self {
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
        storage: &Arc<S>,
        copy: impl FnOnce(&[f32]) -> Result<Vec<f32>, E>,
        from_owned: impl FnOnce(Vec<f32>) -> S,
    ) -> Result<Arc<S>, E> {
        match self {
            Self::Owned(_) => Ok(Arc::clone(storage)),
            Self::SharedGradient(values) => {
                let values = values
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                Ok(Arc::new(from_owned(copy(&values)?)))
            }
        }
    }

    fn accumulate_shared_gradient(&self, contribution: Vec<f32>) {
        match self {
            Self::Owned(_) => {
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
}

impl Storage {
    pub(crate) fn from_owned(data: Vec<f32>, dtype: DType, device: Device) -> Self {
        Self {
            payload: match (device, dtype) {
                (Device::Cpu, DType::Float32) => {
                    StoragePayload::CpuFloat32(StorageData::from_owned(data))
                }
            },
        }
    }

    pub(crate) fn from_shared_gradient(data: Vec<f32>, dtype: DType, device: Device) -> Self {
        Self {
            payload: match (device, dtype) {
                (Device::Cpu, DType::Float32) => {
                    StoragePayload::CpuFloat32(StorageData::from_shared_gradient(data))
                }
            },
        }
    }

    pub(crate) fn len(&self) -> usize {
        match &self.payload {
            StoragePayload::CpuFloat32(values) => values.len(),
        }
    }

    pub(crate) const fn dtype(&self) -> DType {
        match &self.payload {
            StoragePayload::CpuFloat32(_) => DType::Float32,
        }
    }

    pub(crate) const fn device(&self) -> Device {
        match &self.payload {
            StoragePayload::CpuFloat32(_) => Device::Cpu,
        }
    }

    pub(crate) fn data_ptr(&self) -> *const u8 {
        match &self.payload {
            StoragePayload::CpuFloat32(values) => values.data_ptr(),
        }
    }

    pub(crate) fn owned_values(&self) -> Option<&[f32]> {
        match &self.payload {
            StoragePayload::CpuFloat32(values) => values.owned_values(),
        }
    }

    pub(crate) fn value(&self, index: usize) -> Option<f32> {
        match &self.payload {
            StoragePayload::CpuFloat32(values) => values.value(index),
        }
    }

    pub(crate) fn try_copy_values<E>(
        &self,
        copy: impl FnOnce(&[f32]) -> Result<Vec<f32>, E>,
    ) -> Result<Vec<f32>, E> {
        match &self.payload {
            StoragePayload::CpuFloat32(values) => values.try_copy_values(copy),
        }
    }

    pub(crate) fn copy_range(&self, start: usize, end: usize) -> Vec<f32> {
        match &self.payload {
            StoragePayload::CpuFloat32(values) => values.copy_range(start, end),
        }
    }

    pub(crate) fn into_range(self, start: usize, end: usize) -> Vec<f32> {
        match self.payload {
            StoragePayload::CpuFloat32(values) => values.into_range(start, end),
        }
    }

    pub(crate) fn try_clone_for_saved<E>(
        storage: &Arc<Self>,
        copy: impl FnOnce(&[f32]) -> Result<Vec<f32>, E>,
    ) -> Result<Arc<Self>, E> {
        match &storage.payload {
            StoragePayload::CpuFloat32(values) => {
                values.try_clone_for_saved(storage, copy, |data| {
                    Self::from_owned(data, DType::Float32, Device::Cpu)
                })
            }
        }
    }

    pub(crate) fn accumulate_shared_gradient(&self, contribution: Vec<f32>) {
        match &self.payload {
            StoragePayload::CpuFloat32(values) => values.accumulate_shared_gradient(contribution),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::sync::Arc;

    use crate::device::Device;
    use crate::dtype::DType;

    use super::Storage;

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
}
