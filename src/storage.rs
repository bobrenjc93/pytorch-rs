use std::sync::{Arc, Mutex};

use crate::device::Device;
use crate::dtype::DType;

/// Crate-private CPU float32 storage shared by tensors and autograd snapshots.
pub(crate) struct Storage {
    data: StorageData,
    dtype: DType,
    device: Device,
}

enum StorageData {
    Owned(Vec<f32>),
    SharedGradient(Mutex<Vec<f32>>),
}

impl Storage {
    pub(crate) fn from_owned(data: Vec<f32>, dtype: DType, device: Device) -> Self {
        Self {
            data: StorageData::Owned(data),
            dtype,
            device,
        }
    }

    pub(crate) fn from_shared_gradient(data: Vec<f32>, dtype: DType, device: Device) -> Self {
        Self {
            data: StorageData::SharedGradient(Mutex::new(data)),
            dtype,
            device,
        }
    }

    pub(crate) fn len(&self) -> usize {
        match &self.data {
            StorageData::Owned(values) => values.len(),
            StorageData::SharedGradient(values) => values
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .len(),
        }
    }

    pub(crate) const fn dtype(&self) -> DType {
        self.dtype
    }

    pub(crate) const fn device(&self) -> Device {
        self.device
    }

    pub(crate) fn data_ptr(&self) -> *const u8 {
        match &self.data {
            StorageData::Owned(values) => values.as_ptr().cast(),
            // Gradient accumulation only updates existing elements, so the
            // allocation remains stable after this guard is released.
            StorageData::SharedGradient(values) => values
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .as_ptr()
                .cast(),
        }
    }

    pub(crate) fn owned_values(&self) -> Option<&[f32]> {
        match &self.data {
            StorageData::Owned(values) => Some(values),
            StorageData::SharedGradient(_) => None,
        }
    }

    pub(crate) fn value(&self, index: usize) -> Option<f32> {
        match &self.data {
            StorageData::Owned(values) => values.get(index).copied(),
            StorageData::SharedGradient(values) => values
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get(index)
                .copied(),
        }
    }

    pub(crate) fn try_copy_values<E>(
        &self,
        copy: impl FnOnce(&[f32]) -> Result<Vec<f32>, E>,
    ) -> Result<Vec<f32>, E> {
        match &self.data {
            StorageData::Owned(values) => copy(values),
            StorageData::SharedGradient(values) => {
                let values = values
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                copy(&values)
            }
        }
    }

    pub(crate) fn copy_range(&self, start: usize, end: usize) -> Vec<f32> {
        match &self.data {
            StorageData::Owned(values) => values[start..end].to_vec(),
            StorageData::SharedGradient(values) => values
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)[start..end]
                .to_vec(),
        }
    }

    pub(crate) fn into_range(self, start: usize, end: usize) -> Vec<f32> {
        match self.data {
            StorageData::Owned(values) if start == 0 && end == values.len() => values,
            StorageData::Owned(values) => values[start..end].to_vec(),
            StorageData::SharedGradient(values) => {
                let values = values
                    .into_inner()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if start == 0 && end == values.len() {
                    values
                } else {
                    values[start..end].to_vec()
                }
            }
        }
    }

    pub(crate) fn try_clone_for_saved<E>(
        storage: &Arc<Self>,
        copy: impl FnOnce(&[f32]) -> Result<Vec<f32>, E>,
    ) -> Result<Arc<Self>, E> {
        let StorageData::SharedGradient(values) = &storage.data else {
            return Ok(Arc::clone(storage));
        };
        let values = values
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let data = copy(&values)?;
        Ok(Arc::new(Self::from_owned(
            data,
            storage.dtype,
            storage.device,
        )))
    }

    pub(crate) fn accumulate_shared_gradient(&self, contribution: Vec<f32>) {
        let StorageData::SharedGradient(existing) = &self.data else {
            unreachable!("leaf gradients always use shared gradient storage");
        };
        let mut existing = existing
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        debug_assert_eq!(existing.len(), contribution.len());
        for (value, contribution) in existing.iter_mut().zip(contribution) {
            *value += contribution;
        }
    }
}
