use std::collections::{HashMap, HashSet};
use std::fmt::Formatter;
use std::iter::FusedIterator;
use std::sync::{Arc, Mutex};

use crate::autograd_node::AutogradNode;
use crate::device::Device;
use crate::dtype::DType;
use crate::grad_mode::is_grad_enabled;
use crate::memory_format::MemoryFormat;
use crate::storage::Storage;
use crate::tensor_error::TensorError;

const F32_SIGN_MASK: u32 = 0x8000_0000;
#[cfg(feature = "python-bindings")]
const MIN_CONCRETE_SYMINT: i64 = -(1_i64 << 62);
const CONTIGUOUS_MATMUL_ROW_BLOCK: usize = 4;
// Keep latency-sized products on the smaller single-row loop.
const CONTIGUOUS_MATMUL_MIN_RHS_ELEMENTS: usize = 4 * 1024;

static BACKWARD_TRAVERSAL: Mutex<()> = Mutex::new(());

struct AutogradMeta {
    kind: AutogradKind,
}

enum AutogradKind {
    Leaf {
        shape: Vec<usize>,
        dtype: DType,
        device: Device,
        grad: Mutex<Option<Arc<Storage>>>,
    },
    NonLeaf {
        grad_fn: Mutex<Option<GradFn>>,
    },
}

#[derive(Clone)]
struct SavedTensor {
    storage: Option<Arc<Storage>>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
    elements: usize,
    output_nr: usize,
    autograd: Option<Arc<AutogradMeta>>,
}

// Keep dynamic dispatch at node granularity so each operation's element loop
// can inline and vectorize its scalar VJP.
type UnaryVjpKernel = fn(&SavedTensor, &[f32], &mut Vec<f32>);

#[derive(Clone)]
struct SavedInputUnaryNode {
    input: SavedTensor,
    #[cfg_attr(not(feature = "python-bindings"), allow(dead_code))]
    identity: AutogradNode,
    vjp: UnaryVjpKernel,
}

#[derive(Clone)]
struct SavedOutputUnaryNode {
    input: SavedTensor,
    output: SavedTensor,
    #[cfg_attr(not(feature = "python-bindings"), allow(dead_code))]
    identity: AutogradNode,
    vjp: UnaryVjpKernel,
}

#[derive(Clone)]
enum GradFn {
    Multiply {
        left: SavedTensor,
        right: SavedTensor,
        output_shape: Vec<usize>,
        output_elements: usize,
    },
    MultiplyScalar {
        input: SavedTensor,
        scalar: Option<f32>,
    },
    Negate {
        input: SavedTensor,
        #[cfg_attr(not(feature = "python-bindings"), allow(dead_code))]
        node: AutogradNode,
    },
    SavedInputUnary(SavedInputUnaryNode),
    SavedOutputUnary(SavedOutputUnaryNode),
    Sum {
        input: SavedTensor,
    },
    Transform {
        input: SavedTensor,
        mapping: TransformMapping,
        #[cfg_attr(not(feature = "python-bindings"), allow(dead_code))]
        node: AutogradNode,
    },
    Unbind {
        input: SavedTensor,
        output_count: usize,
        output_elements: usize,
    },
}

#[derive(Clone)]
enum TransformMapping {
    Identity,
    Permute {
        dimensions: Vec<usize>,
        output_shape: Vec<usize>,
    },
    Index {
        input_start: usize,
    },
}

impl SavedTensor {
    fn take_parent(&mut self, pending: &mut Vec<Arc<AutogradMeta>>) {
        if let Some(parent) = self.autograd.take() {
            pending.push(parent);
        }
    }
}

impl GradFn {
    fn take_parents(&mut self, pending: &mut Vec<Arc<AutogradMeta>>) {
        match self {
            Self::Multiply { left, right, .. } => {
                left.take_parent(pending);
                right.take_parent(pending);
            }
            Self::MultiplyScalar { input, .. }
            | Self::Negate { input, .. }
            | Self::Sum { input }
            | Self::Transform { input, .. }
            | Self::Unbind { input, .. } => input.take_parent(pending),
            Self::SavedInputUnary(node) => node.input.take_parent(pending),
            Self::SavedOutputUnary(node) => node.input.take_parent(pending),
        }
    }

    fn validate_saved_values(&self) -> Result<(), TensorError> {
        match self {
            Self::Multiply { left, right, .. } => {
                if (left.autograd.is_some() && right.storage.is_none())
                    || (right.autograd.is_some() && left.storage.is_none())
                {
                    return Err(TensorError::BackwardGraphFreed);
                }
            }
            Self::MultiplyScalar { input, scalar } => {
                if input.autograd.is_some() && scalar.is_none() {
                    return Err(TensorError::BackwardGraphFreed);
                }
            }
            Self::SavedInputUnary(node) => {
                if node.input.storage.is_none() {
                    return Err(TensorError::BackwardGraphFreed);
                }
            }
            Self::SavedOutputUnary(node) => {
                if node.output.storage.is_none() {
                    return Err(TensorError::BackwardGraphFreed);
                }
            }
            Self::Negate { .. }
            | Self::Sum { .. }
            | Self::Transform { .. }
            | Self::Unbind { .. } => {}
        }
        Ok(())
    }

    fn consume_saved_values(&mut self) -> Result<(), TensorError> {
        self.validate_saved_values()?;
        match self {
            Self::Multiply { left, right, .. } => {
                left.storage = None;
                right.storage = None;
            }
            Self::MultiplyScalar { scalar, .. } => *scalar = None,
            Self::SavedInputUnary(node) => node.input.storage = None,
            Self::SavedOutputUnary(node) => node.output.storage = None,
            Self::Negate { .. }
            | Self::Sum { .. }
            | Self::Transform { .. }
            | Self::Unbind { .. } => {}
        }
        Ok(())
    }
}

impl AutogradMeta {
    fn take_grad_fn(&mut self) -> Option<GradFn> {
        let AutogradKind::NonLeaf { grad_fn } = &mut self.kind else {
            return None;
        };
        grad_fn
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
    }
}

impl Drop for AutogradMeta {
    fn drop(&mut self) {
        let Some(mut grad_fn) = self.take_grad_fn() else {
            return;
        };
        let mut pending = Vec::new();
        grad_fn.take_parents(&mut pending);
        drop(grad_fn);

        while let Some(parent) = pending.pop() {
            if let Ok(mut parent) = Arc::try_unwrap(parent)
                && let Some(mut grad_fn) = parent.take_grad_fn()
            {
                grad_fn.take_parents(&mut pending);
            }
        }
    }
}

/// A tensor with immutable shared storage and native shape/stride metadata.
///
/// Metadata-only views may have a nonzero storage offset and non-contiguous
/// strides. Materializing operations produce independent storage.
pub struct Tensor {
    storage: Arc<Storage>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
    elements: usize,
    output_nr: usize,
    view_requires_grad: bool,
    autograd: Option<Arc<AutogradMeta>>,
}

/// Iterates over a tensor's values in logical row-major index order.
///
/// The iterator follows tensor strides and storage offsets, so it is suitable
/// for both contiguous tensors and metadata-only views. Contiguous tensors use
/// a direct slice iterator.
pub struct LogicalValues<'a> {
    inner: LogicalValuesInner<'a>,
}

enum LogicalValuesInner<'a> {
    Contiguous(std::iter::Copied<std::slice::Iter<'a, f32>>),
    Strided { tensor: &'a Tensor, next: usize },
}

impl Iterator for LogicalValues<'_> {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            LogicalValuesInner::Contiguous(values) => values.next(),
            LogicalValuesInner::Strided { tensor, next } => {
                if *next == tensor.elements {
                    return None;
                }
                let index = *next;
                *next += 1;
                Some(tensor.value_at_strided_linear_index(index))
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = match &self.inner {
            LogicalValuesInner::Contiguous(values) => values.len(),
            LogicalValuesInner::Strided { tensor, next } => tensor.elements - next,
        };
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for LogicalValues<'_> {}
impl FusedIterator for LogicalValues<'_> {}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        self.try_clone()
            .expect("cloning validated tensor storage should succeed")
    }
}

// Preserve the original value-oriented debug representation; storage identity
// and layout bookkeeping are implementation details.
#[allow(clippy::missing_fields_in_debug)]
impl std::fmt::Debug for Tensor {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Tensor")
            .field("data", &self.logical_values().collect::<Vec<_>>())
            .field("shape", &self.shape)
            .finish()
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape
            && self.dtype() == other.dtype()
            && self.device() == other.device()
            && self.logical_values().eq(other.logical_values())
    }
}

impl Tensor {
    /// Creates a tensor after validating that `shape` describes `data`.
    ///
    /// # Errors
    ///
    /// Returns an error when the element count or contiguous stride overflows,
    /// or when the element count differs from the supplied data length.
    pub fn from_vec(data: Vec<f32>, shape: impl Into<Vec<usize>>) -> Result<Self, TensorError> {
        Self::from_vec_with_metadata(data, shape, DType::Float32, Device::Cpu)
    }

    pub(crate) fn from_vec_with_metadata(
        data: Vec<f32>,
        shape: impl Into<Vec<usize>>,
        dtype: DType,
        device: Device,
    ) -> Result<Self, TensorError> {
        let shape = shape.into();
        let (expected, strides) = validated_layout(&shape)?;
        if data.len() != expected {
            return Err(TensorError::ShapeDataMismatch {
                shape,
                elements: data.len(),
            });
        }
        Ok(Self::from_owned_parts(data, shape, strides, dtype, device))
    }

    /// Creates a zero-filled tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the shape's element count, contiguous stride, or
    /// storage size overflows.
    pub fn zeros(shape: impl Into<Vec<usize>>) -> Result<Self, TensorError> {
        Self::zeros_with_metadata(shape, DType::Float32, Device::Cpu)
    }

    pub(crate) fn zeros_with_metadata(
        shape: impl Into<Vec<usize>>,
        dtype: DType,
        device: Device,
    ) -> Result<Self, TensorError> {
        let shape = shape.into();
        let (elements, strides) = validated_layout(&shape)?;
        let data = filled_storage(elements, 0.0)?;
        Ok(Self::from_owned_parts(data, shape, strides, dtype, device))
    }

    /// Creates a one-filled tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the shape's element count, contiguous stride, or
    /// storage size overflows.
    pub fn ones(shape: impl Into<Vec<usize>>) -> Result<Self, TensorError> {
        Self::ones_with_metadata(shape, DType::Float32, Device::Cpu)
    }

    pub(crate) fn ones_with_metadata(
        shape: impl Into<Vec<usize>>,
        dtype: DType,
        device: Device,
    ) -> Result<Self, TensorError> {
        let shape = shape.into();
        let (elements, strides) = validated_layout(&shape)?;
        let data = filled_storage(elements, 1.0)?;
        Ok(Self::from_owned_parts(data, shape, strides, dtype, device))
    }

    /// Creates the default float32 CPU range `0, 1, ..., elements - 1`.
    ///
    /// # Errors
    ///
    /// Returns an error when the one-dimensional shape or storage allocation
    /// exceeds the platform capacity.
    #[cfg(feature = "python-bindings")]
    pub(crate) fn arange_float32(elements: usize) -> Result<Self, TensorError> {
        validate_storage_capacity(elements)?;

        let mut data = try_result_vector(elements, elements)?;
        for index in 0..elements {
            #[allow(clippy::cast_precision_loss)]
            data.push(index as f32);
        }

        let mut shape = try_result_vector(1, elements)?;
        shape.push(elements);
        let (_, strides) = validated_layout(&shape)?;
        Ok(Self::from_owned_parts(
            data,
            shape,
            strides,
            DType::Float32,
            Device::Cpu,
        ))
    }

    /// Creates a two-dimensional tensor with ones on the main diagonal and
    /// zeros elsewhere.
    ///
    /// Passing [`None`] for `m` creates a square `n` by `n` tensor. A column
    /// count may be passed directly or as [`Some`] to create a rectangular
    /// tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the shape's element count, contiguous stride, or
    /// storage size overflows, or when storage allocation fails.
    pub fn eye(n: usize, m: impl Into<Option<usize>>) -> Result<Self, TensorError> {
        Self::eye_with_metadata(n, m.into().unwrap_or(n), DType::Float32, Device::Cpu)
    }

    pub(crate) fn eye_with_metadata(
        n: usize,
        m: usize,
        dtype: DType,
        device: Device,
    ) -> Result<Self, TensorError> {
        let mut shape = try_result_vector(2, 0)?;
        shape.push(n);
        shape.push(m);
        let (elements, strides) = validated_layout(&shape)?;
        let mut data = filled_storage(elements, 0.0)?;

        let diagonal = n.min(m);
        if diagonal > 0 {
            for (index, row) in data.chunks_exact_mut(m).take(diagonal).enumerate() {
                row[index] = 1.0;
            }
        }

        Ok(Self::from_owned_parts(data, shape, strides, dtype, device))
    }

    /// Creates a tensor filled with `fill_value`.
    ///
    /// # Errors
    ///
    /// Returns an error when the shape's element count, contiguous stride, or
    /// storage size overflows, or when storage allocation fails.
    pub fn full(shape: impl Into<Vec<usize>>, fill_value: f32) -> Result<Self, TensorError> {
        Self::full_with_metadata(shape, fill_value, DType::Float32, Device::Cpu)
    }

    pub(crate) fn full_with_metadata(
        shape: impl Into<Vec<usize>>,
        fill_value: f32,
        dtype: DType,
        device: Device,
    ) -> Result<Self, TensorError> {
        let shape = shape.into();
        let (elements, strides) = validated_layout(&shape)?;
        validate_storage_capacity(elements)?;
        let data = filled_storage(elements, fill_value)?;
        Ok(Self::from_owned_parts(data, shape, strides, dtype, device))
    }

    #[cfg(feature = "python-bindings")]
    pub(crate) fn validate_full_shape(shape: &[usize]) -> Result<usize, TensorError> {
        let (elements, _) = validated_layout(shape)?;
        validate_storage_capacity(elements)?;
        Ok(elements)
    }

    fn from_owned_parts(
        data: Vec<f32>,
        shape: Vec<usize>,
        strides: Vec<usize>,
        dtype: DType,
        device: Device,
    ) -> Self {
        let elements = data.len();
        Self {
            storage: Arc::new(Storage::from_owned(data, dtype, device)),
            shape,
            strides,
            offset: 0,
            elements,
            output_nr: 0,
            view_requires_grad: false,
            autograd: None,
        }
    }

    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the tensor's per-dimension element strides.
    #[must_use]
    pub fn stride(&self) -> &[usize] {
        &self.strides
    }

    /// Returns the first element's offset into the shared storage.
    #[must_use]
    pub fn storage_offset(&self) -> usize {
        self.offset
    }

    /// Returns the address of the tensor's first logical element.
    ///
    /// Empty tensors return zero. Views otherwise add their element offset,
    /// scaled by the storage dtype's byte width, to the stable backing
    /// allocation address.
    ///
    /// # Panics
    ///
    /// Panics if validated internal storage metadata does not fit in the
    /// process address space.
    #[must_use]
    pub fn data_ptr(&self) -> usize {
        if self.elements == 0 {
            return 0;
        }

        let byte_offset = self
            .offset
            .checked_mul(self.element_size())
            .expect("validated tensor storage offset must fit in bytes");
        // Retain the allocation provenance while applying the view offset,
        // then expose it from the final pointer so integer-to-pointer round
        // trips through FFI remain valid under Rust's strict provenance model.
        self.storage
            .data_ptr()
            .wrapping_add(byte_offset)
            .expose_provenance()
    }

    /// Returns the read-only address of the tensor's first logical element.
    ///
    /// This tensor engine has only ordinary, non-copy-on-write storage, so the
    /// result is identical to [`Self::data_ptr`] without materializing or
    /// mutating the tensor.
    #[must_use]
    pub fn const_data_ptr(&self) -> usize {
        self.data_ptr()
    }

    /// Reports whether two tensors refer to the same underlying allocation.
    #[must_use]
    pub fn shares_storage_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.storage, &other.storage)
    }

    /// Reports whether two tensors have identical shapes.
    #[must_use]
    pub fn is_same_size(&self, other: &Self) -> bool {
        self.shape() == other.shape()
    }

    /// Reports whether two tensors point to the exact same logical view.
    ///
    /// Matching views share storage and have identical storage offsets, shapes,
    /// and strides. Autograd metadata and tensor wrapper identity do not affect
    /// the result.
    #[must_use]
    pub fn is_set_to(&self, other: &Self) -> bool {
        self.storage_offset() == other.storage_offset()
            && self.shape() == other.shape()
            && self.stride() == other.stride()
            && self.shares_storage_with(other)
    }

    /// Returns the scalar type physically represented by this tensor's storage.
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.storage.dtype()
    }

    /// Returns the number of bytes used to store one tensor element.
    #[must_use]
    pub fn element_size(&self) -> usize {
        self.dtype().element_size()
    }

    /// Returns the number of dense dimensions in this tensor.
    ///
    /// Every currently supported tensor uses the canonical strided layout, so
    /// all dimensions are dense and this is exactly the tensor rank.
    #[must_use]
    pub fn dense_dim(&self) -> usize {
        self.shape.len()
    }

    /// Returns the number of sparse dimensions in this tensor.
    ///
    /// Every currently supported tensor uses the canonical strided layout, so
    /// no dimensions are sparse.
    #[must_use]
    pub const fn sparse_dim(&self) -> usize {
        0
    }

    /// Reports whether the tensor's native scalar type is floating point.
    #[must_use]
    pub fn is_floating_point(&self) -> bool {
        self.dtype().is_floating_point()
    }

    /// Reports whether the tensor's native scalar type is complex.
    #[must_use]
    pub fn is_complex(&self) -> bool {
        self.dtype().is_complex()
    }

    /// Reports whether the tensor's native scalar type uses a quantized representation.
    #[must_use]
    pub fn is_quantized(&self) -> bool {
        self.dtype().is_quantized()
    }

    /// Reports whether the tensor uses oneDNN (MKLDNN) storage.
    ///
    /// Every currently supported tensor uses the canonical strided layout, so
    /// this metadata query does not inspect the backing storage.
    #[must_use]
    pub const fn is_mkldnn(&self) -> bool {
        false
    }

    /// Reports whether the tensor uses nested tensor storage.
    ///
    /// Every currently supported tensor uses the canonical strided layout, so
    /// this metadata query does not inspect the backing storage.
    #[must_use]
    pub const fn is_nested(&self) -> bool {
        false
    }

    /// Reports whether the tensor uses sparse COO storage.
    ///
    /// Every currently supported tensor uses the canonical strided layout, so
    /// this metadata query does not inspect the backing storage.
    #[must_use]
    pub const fn is_sparse(&self) -> bool {
        false
    }

    /// Reports whether the tensor uses sparse CSR storage.
    ///
    /// Every currently supported tensor uses the canonical strided layout, so
    /// this metadata query does not inspect the backing storage.
    #[must_use]
    pub const fn is_sparse_csr(&self) -> bool {
        false
    }

    /// Reports whether the tensor's native scalar type is signed.
    #[must_use]
    pub fn is_signed(&self) -> bool {
        self.dtype().is_signed()
    }

    /// Reports whether this tensor uses page-locked host memory.
    ///
    /// Every currently supported tensor uses ordinary pageable CPU storage,
    /// so this metadata query does not inspect the backing storage.
    #[must_use]
    pub const fn is_pinned(&self) -> bool {
        false
    }

    /// Reports whether this tensor's device is the CPU.
    #[must_use]
    pub fn is_cpu(&self) -> bool {
        self.device().is_cpu()
    }

    /// Reports whether this tensor's device is a CUDA accelerator.
    #[must_use]
    pub fn is_cuda(&self) -> bool {
        self.device().is_cuda()
    }

    /// Reports whether this tensor's device is a Graphcore IPU accelerator.
    #[must_use]
    pub fn is_ipu(&self) -> bool {
        self.device().is_ipu()
    }

    /// Reports whether this tensor's device is a Meta MTIA accelerator.
    #[must_use]
    pub fn is_mtia(&self) -> bool {
        self.device().is_mtia()
    }

    /// Reports whether this tensor's device is a Maia accelerator.
    #[must_use]
    pub fn is_maia(&self) -> bool {
        self.device().is_maia()
    }

    /// Reports whether this tensor's device is an Intel XPU accelerator.
    #[must_use]
    pub fn is_xpu(&self) -> bool {
        self.device().is_xpu()
    }

    /// Reports whether this tensor's device is an XLA accelerator.
    #[must_use]
    pub fn is_xla(&self) -> bool {
        self.device().is_xla()
    }

    /// Reports whether this tensor's device is an Apple MPS accelerator.
    #[must_use]
    pub fn is_mps(&self) -> bool {
        self.device().is_mps()
    }

    /// Reports whether this tensor's device is a Vulkan accelerator.
    #[must_use]
    pub fn is_vulkan(&self) -> bool {
        self.device().is_vulkan()
    }

    /// Reports whether this tensor is stored on the metadata-only device.
    #[must_use]
    pub fn is_meta(&self) -> bool {
        self.device().is_meta()
    }

    /// Returns the device owning this tensor's storage.
    #[must_use]
    pub fn device(&self) -> Device {
        self.storage.device()
    }

    #[must_use]
    pub fn numel(&self) -> usize {
        self.elements
    }

    /// Returns whether operations on this tensor may require reverse-mode gradients.
    ///
    /// Views made while recording is disabled preserve this property without
    /// retaining a backward edge to their source tensor.
    #[must_use]
    pub fn requires_grad(&self) -> bool {
        self.autograd.is_some() || self.view_requires_grad
    }

    /// Returns whether this tensor has no recorded autograd operation producing it.
    ///
    /// Tensors which do not require gradients are leaves by convention. Views
    /// created while gradient recording is disabled can still report
    /// [`Self::requires_grad`] without carrying a recorded backward edge, and
    /// are leaves for the same reason.
    #[must_use]
    pub fn is_leaf(&self) -> bool {
        !matches!(
            self.autograd.as_deref().map(|metadata| &metadata.kind),
            Some(AutogradKind::NonLeaf { .. })
        )
    }

    #[cfg(feature = "python-bindings")]
    pub(crate) fn grad_fn_name(&self) -> Option<&'static str> {
        let metadata = self.autograd.as_deref()?;
        let AutogradKind::NonLeaf { grad_fn } = &metadata.kind else {
            return None;
        };
        let grad_fn = grad_fn
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let node = match grad_fn.as_ref()? {
            GradFn::Multiply { .. } | GradFn::MultiplyScalar { .. } => AutogradNode::Multiply,
            GradFn::Negate { node, .. } | GradFn::Transform { node, .. } => *node,
            GradFn::SavedInputUnary(node) => node.identity,
            GradFn::SavedOutputUnary(node) => node.identity,
            GradFn::Sum { .. } => AutogradNode::Sum,
            GradFn::Unbind { .. } => AutogradNode::Unbind,
        };
        Some(node.python_name())
    }

    /// Returns whether this tensor was created under inference mode.
    ///
    /// The native engine does not expose inference mode, so every reachable
    /// tensor has ordinary autograd metadata and reports `false`.
    #[must_use]
    pub const fn is_inference(&self) -> bool {
        false
    }

    /// Returns whether gradients are retained for this non-leaf tensor.
    ///
    /// The native engine does not retain non-leaf gradients, so every reachable
    /// tensor reports `false`. Leaf tensors can still have a live gradient
    /// through [`Self::grad`]; the Python leaf-only `retain_grad()` binding is a
    /// no-op and does not change this metadata.
    #[must_use]
    pub const fn retains_grad(&self) -> bool {
        false
    }

    /// Returns this tensor's output index within its producing autograd node.
    ///
    /// Tensors without a recorded producing operation report output zero.
    #[must_use]
    pub const fn output_nr(&self) -> usize {
        self.output_nr
    }

    /// Marks an owned tensor as a gradient-accumulating leaf, or detaches it
    /// when `requires_grad` is false.
    ///
    /// Calling this with `true` on a tensor which already participates in a
    /// graph preserves that graph. Freshly marked tensors accumulate gradients
    /// according to their current logical shape.
    #[must_use]
    pub fn with_requires_grad(mut self, requires_grad: bool) -> Self {
        if !requires_grad {
            self.autograd = None;
            self.output_nr = 0;
            self.view_requires_grad = false;
        } else if self.autograd.is_none() {
            self.autograd = Some(Arc::new(AutogradMeta {
                kind: AutogradKind::Leaf {
                    shape: self.shape.clone(),
                    dtype: self.dtype(),
                    device: self.device(),
                    grad: Mutex::new(None),
                },
            }));
            self.output_nr = 0;
            self.view_requires_grad = false;
        }
        self
    }

    /// Returns a detached, contiguous snapshot of an accumulated leaf gradient.
    ///
    /// Non-leaf tensors and leaves which have not been used by a successful
    /// backward pass return [`None`].
    ///
    /// # Errors
    ///
    /// Returns an error if gradient metadata or storage allocation fails.
    pub fn grad(&self) -> Result<Option<Self>, TensorError> {
        let Some(meta) = &self.autograd else {
            return Ok(None);
        };
        let AutogradKind::Leaf { shape, grad, .. } = &meta.kind else {
            return Ok(None);
        };
        let grad = grad
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(storage) = grad.as_ref() else {
            return Ok(None);
        };
        let elements = storage.len();
        let shape = try_clone_result_shape(shape, elements)?;
        let strides = contiguous_strides(&shape, elements)?;
        let values = storage.try_copy_values(|values| copied_storage(values, elements))?;
        Ok(Some(Self::from_owned_parts(
            values,
            shape,
            strides,
            storage.dtype(),
            storage.device(),
        )))
    }

    #[cfg(any(feature = "python-bindings", test))]
    pub(crate) fn live_grad(&self) -> Result<Option<Self>, TensorError> {
        let Some(meta) = &self.autograd else {
            return Ok(None);
        };
        let AutogradKind::Leaf { shape, grad, .. } = &meta.kind else {
            return Ok(None);
        };
        let grad = grad
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(storage) = grad.as_ref() else {
            return Ok(None);
        };
        let elements = storage.len();
        let shape = try_clone_result_shape(shape, elements)?;
        let strides = contiguous_strides(&shape, elements)?;
        Ok(Some(Self {
            storage: Arc::clone(storage),
            shape,
            strides,
            offset: 0,
            elements,
            output_nr: 0,
            view_requires_grad: false,
            autograd: None,
        }))
    }

    /// Creates a metadata-only alias which shares storage but has no graph
    /// history and never requires gradients.
    ///
    /// # Errors
    ///
    /// Returns an error if result metadata allocation fails.
    pub fn detach(&self) -> Result<Self, TensorError> {
        self.metadata_alias_detached()
    }

    /// Runs eager reverse-mode differentiation from a one-element output.
    ///
    /// Successful calls accumulate into every reachable leaf. Value-dependent
    /// saved tensors are then released, while metadata-only graph edges remain
    /// reusable. Calling backward repeatedly on a one-element leaf also
    /// remains valid and accumulates another unit gradient.
    ///
    /// # Errors
    ///
    /// Returns an error for multi-element outputs, tensors which do not require
    /// gradients, or graphs already consumed by a prior backward pass.
    pub fn backward(&self) -> Result<(), TensorError> {
        let meta = self.implicit_backward_root()?;
        run_backward(meta, self.output_nr)
    }

    fn implicit_backward_root(&self) -> Result<&Arc<AutogradMeta>, TensorError> {
        if !self.requires_grad() {
            return Err(TensorError::DoesNotRequireGrad);
        }
        if self.elements != 1 {
            return Err(TensorError::BackwardRequiresScalar {
                elements: self.elements,
            });
        }
        self.autograd
            .as_ref()
            .ok_or(TensorError::DoesNotRequireGrad)
    }

    #[cfg(any(feature = "python-bindings", test))]
    pub(crate) fn backward_leaf_roots(leaf_roots: &[&Self]) -> Result<(), TensorError> {
        // Keep root validation and gradient commits in one transaction. In
        // particular, views created under no_grad can report requires_grad and
        // leaf status without owning a gradient accumulator.
        let _backward_traversal = BACKWARD_TRAVERSAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut roots = Vec::with_capacity(leaf_roots.len());
        for (index, root) in leaf_roots.iter().copied().enumerate() {
            let meta = root.implicit_backward_root().map_err(|error| {
                if matches!(error, TensorError::DoesNotRequireGrad) {
                    TensorError::DoesNotRequireGradAt { index }
                } else {
                    error
                }
            })?;
            if !matches!(&meta.kind, AutogradKind::Leaf { .. }) {
                return Err(TensorError::DoesNotRequireGradAt { index });
            }
            roots.push((Arc::clone(meta), root.output_nr));
        }

        let mut gradients = HashMap::new();
        let mut unique_roots = Vec::with_capacity(leaf_roots.len());
        for (meta, output_nr) in roots {
            let key = gradient_key(&meta, output_nr);
            if !gradients.contains_key(&key) {
                unique_roots.push((Arc::clone(&meta), output_nr));
            }
            add_gradient(&mut gradients, &meta, output_nr, vec![1.0]);
        }
        for (meta, output_nr) in unique_roots {
            let gradient = gradients
                .remove(&gradient_key(&meta, output_nr))
                .expect("every unique leaf root must have an aggregated gradient");
            accumulate_leaf_gradient(&meta, gradient);
        }
        Ok(())
    }

    fn records_grad(&self) -> bool {
        self.requires_grad() && is_grad_enabled()
    }

    fn is_finite_owned_scalar_leaf(&self) -> bool {
        // Factory-created scalar leaves span their complete one-element
        // allocation. Recorded views are non-leaves, while views created
        // under no_grad carry view_requires_grad without leaf metadata.
        if !self.shape.is_empty()
            || self.offset != 0
            || self.elements != 1
            || self.storage.len() != 1
            || self.dtype() != DType::Float32
            || self.device() != Device::Cpu
            || self.view_requires_grad
        {
            return false;
        }

        let Some(metadata) = self.autograd.as_deref() else {
            return false;
        };
        matches!(&metadata.kind, AutogradKind::Leaf { .. })
            && self.value_at_linear_index(0).is_finite()
    }

    fn record_transform(
        &self,
        output: &mut Self,
        mapping: TransformMapping,
        node: AutogradNode,
    ) -> Result<(), TensorError> {
        if self.records_grad() {
            output.autograd = Some(Arc::new(AutogradMeta {
                kind: AutogradKind::NonLeaf {
                    grad_fn: Mutex::new(Some(GradFn::Transform {
                        input: SavedTensor::try_from_tensor(self, false)?,
                        mapping,
                        node,
                    })),
                },
            }));
        }
        Ok(())
    }

    fn record_view_transform(
        &self,
        output: &mut Self,
        mapping: TransformMapping,
        node: AutogradNode,
    ) -> Result<(), TensorError> {
        output.view_requires_grad = self.requires_grad();
        self.record_transform(output, mapping, node)
    }

    fn finish_view_transform(
        &self,
        mut output: Self,
        mapping: TransformMapping,
        node: AutogradNode,
    ) -> Result<Self, TensorError> {
        self.record_view_transform(&mut output, mapping, node)?;
        Ok(output)
    }

    fn finish_copy_transform(
        &self,
        mut output: Self,
        mapping: TransformMapping,
        node: AutogradNode,
    ) -> Result<Self, TensorError> {
        self.record_transform(&mut output, mapping, node)?;
        Ok(output)
    }

    fn finish_negate_vjp(&self, mut output: Self, node: AutogradNode) -> Result<Self, TensorError> {
        if self.records_grad() {
            output.autograd = Some(Arc::new(AutogradMeta {
                kind: AutogradKind::NonLeaf {
                    grad_fn: Mutex::new(Some(GradFn::Negate {
                        input: SavedTensor::try_from_tensor(self, false)?,
                        node,
                    })),
                },
            }));
        }
        Ok(output)
    }

    fn finish_saved_input_unary_vjp(
        &self,
        mut output: Self,
        identity: AutogradNode,
        vjp: UnaryVjpKernel,
    ) -> Result<Self, TensorError> {
        if self.records_grad() {
            output.autograd = Some(Arc::new(AutogradMeta {
                kind: AutogradKind::NonLeaf {
                    grad_fn: Mutex::new(Some(GradFn::SavedInputUnary(SavedInputUnaryNode {
                        input: SavedTensor::try_from_tensor(self, true)?,
                        identity,
                        vjp,
                    }))),
                },
            }));
        }
        Ok(output)
    }

    fn finish_saved_output_unary_vjp(
        &self,
        mut output: Self,
        identity: AutogradNode,
        vjp: UnaryVjpKernel,
    ) -> Result<Self, TensorError> {
        if self.records_grad() {
            let input = SavedTensor::try_from_tensor(self, false)?;
            let saved_output = SavedTensor::try_from_tensor(&output, true)?;
            output.autograd = Some(Arc::new(AutogradMeta {
                kind: AutogradKind::NonLeaf {
                    grad_fn: Mutex::new(Some(GradFn::SavedOutputUnary(SavedOutputUnaryNode {
                        input,
                        output: saved_output,
                        identity,
                        vjp,
                    }))),
                },
            }));
        }
        Ok(output)
    }

    /// Returns whether logical row-major iteration visits adjacent storage.
    ///
    /// As in `PyTorch`, scalars and tensors with no elements are contiguous,
    /// and strides on singleton dimensions do not affect contiguity.
    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        layout_is_contiguous(&self.shape, &self.strides, self.elements)
    }

    /// Returns whether this tensor is contiguous in the requested memory
    /// format.
    ///
    /// [`MemoryFormat::Preserve`] has the same query semantics as
    /// [`MemoryFormat::Contiguous`], matching `PyTorch`. Channel-last formats
    /// require rank four and five, respectively.
    #[must_use]
    pub fn is_contiguous_with_memory_format(&self, memory_format: MemoryFormat) -> bool {
        match memory_format {
            MemoryFormat::Preserve | MemoryFormat::Contiguous => self.is_contiguous(),
            MemoryFormat::ChannelsLast => {
                layout_is_channels_last_contiguous(&self.shape, &self.strides)
            }
            MemoryFormat::ChannelsLast3d => {
                layout_is_channels_last_3d_contiguous(&self.shape, &self.strides)
            }
        }
    }

    /// Returns whether every logical value occupies a distinct element in one
    /// dense storage interval, independent of dimension order.
    #[must_use]
    pub fn is_non_overlapping_and_dense(&self) -> bool {
        if self.elements == 0 {
            return true;
        }

        let dimensions = self
            .shape
            .iter()
            .filter(|dimension| **dimension > 1)
            .count();
        let mut matched = 0_usize;
        let mut expected_stride = 1_usize;
        while matched < dimensions {
            let mut matching_dimension = None;
            for (axis, (&dimension, &stride)) in
                self.shape.iter().zip(self.strides.iter()).enumerate()
            {
                if dimension > 1 && stride == expected_stride {
                    if matching_dimension.is_some() {
                        return false;
                    }
                    matching_dimension = Some((axis, dimension));
                }
            }
            let Some((_, dimension)) = matching_dimension else {
                return false;
            };
            let Some(next_stride) = expected_stride.checked_mul(dimension) else {
                return false;
            };
            expected_stride = next_stride;
            matched += 1;
        }
        true
    }

    /// Returns logical values in row-major index order.
    #[must_use]
    pub fn logical_values(&self) -> LogicalValues<'_> {
        let inner = self.contiguous_slice().map_or_else(
            || LogicalValuesInner::Strided {
                tensor: self,
                next: 0,
            },
            |values| LogicalValuesInner::Contiguous(values.iter().copied()),
        );
        LogicalValues { inner }
    }

    /// Reorders every dimension without copying storage.
    ///
    /// Each source axis must occur exactly once. The output shape and stride at
    /// a position are taken from the corresponding source axis, while storage,
    /// offset, dtype, device, and element count are retained. In particular,
    /// this operation does not recalculate strides or materialize values. The
    /// requested axis order is still checked for intermediate element-count
    /// overflow to match `PyTorch` view construction.
    ///
    /// # Errors
    ///
    /// Returns an error when the permutation has the wrong length, contains an
    /// invalid or duplicate axis, its reordered element-count multiplication
    /// overflows, or when view metadata allocation fails.
    pub fn permute_axes(&self, dimensions: impl AsRef<[usize]>) -> Result<Self, TensorError> {
        self.permute_axes_with_grad_fn(dimensions, AutogradNode::Permute)
    }

    fn permute_axes_with_grad_fn(
        &self,
        dimensions: impl AsRef<[usize]>,
        node: AutogradNode,
    ) -> Result<Self, TensorError> {
        let dimensions = dimensions.as_ref();
        let rank = self.shape.len();
        if dimensions.len() != rank {
            return Err(TensorError::PermutationRankMismatch {
                dimensions: dimensions.len(),
                rank,
            });
        }

        let mut seen = try_result_vector(rank, self.elements)?;
        seen.resize(rank, false);
        for &dimension in dimensions {
            if dimension >= rank {
                return Err(TensorError::PermutationDimensionOutOfRange { dimension, rank });
            }
            if seen[dimension] {
                return Err(TensorError::DuplicatePermutationDimension { dimension });
            }
            seen[dimension] = true;
        }

        element_count_in_axis_order(&self.shape, dimensions)?;

        let mut shape = try_result_vector(rank, self.elements)?;
        for &dimension in dimensions {
            shape.push(self.shape[dimension]);
        }

        let mut strides = try_result_vector(rank, self.elements)?;
        for &dimension in dimensions {
            strides.push(self.strides[dimension]);
        }

        let mut output = Self {
            storage: Arc::clone(&self.storage),
            shape,
            strides,
            offset: self.offset,
            elements: self.elements,
            output_nr: 0,
            view_requires_grad: self.requires_grad(),
            autograd: None,
        };
        if self.records_grad() {
            let mut saved_dimensions = try_result_vector(dimensions.len(), self.elements)?;
            saved_dimensions.extend_from_slice(dimensions);
            let output_shape = try_clone_result_shape(&output.shape, output.elements)?;
            self.record_transform(
                &mut output,
                TransformMapping::Permute {
                    dimensions: saved_dimensions,
                    output_shape,
                },
                node,
            )?;
        }
        Ok(output)
    }

    /// Reverses all dimensions without copying storage.
    ///
    /// This is the binding-independent primitive implementing NumPy-style
    /// `Tensor.T`. Scalars and vectors therefore retain their metadata, while
    /// higher-rank tensors reverse their complete shape and stride tables.
    ///
    /// # Errors
    ///
    /// Returns an error when view metadata allocation fails.
    pub fn reverse_dimensions(&self) -> Result<Self, TensorError> {
        self.reverse_dimensions_with_grad_fn(AutogradNode::Permute)
    }

    #[cfg(feature = "python-bindings")]
    pub(crate) fn t(&self) -> Result<Self, TensorError> {
        self.reverse_dimensions_with_grad_fn(AutogradNode::MatrixTranspose)
    }

    fn reverse_dimensions_with_grad_fn(&self, node: AutogradNode) -> Result<Self, TensorError> {
        let mut dimensions = try_result_vector(self.shape.len(), self.elements)?;
        dimensions.extend((0..self.shape.len()).rev());
        self.permute_axes_with_grad_fn(dimensions, node)
    }

    /// Swaps two dimensions without copying storage.
    ///
    /// Negative dimensions wrap from the end. Scalars accept dimensions `0`
    /// and `-1`, matching `PyTorch`, and produce another scalar alias.
    ///
    /// # Errors
    ///
    /// Returns an error for an out-of-range dimension, when the reordered
    /// shape's element count overflows, or when view metadata allocation
    /// fails.
    pub fn transpose(&self, dim0: i64, dim1: i64) -> Result<Self, TensorError> {
        let axis0 = normalize_transpose_dimension(dim0, self.shape.len())?;
        let axis1 = normalize_transpose_dimension(dim1, self.shape.len())?;
        let mut dimensions = try_result_vector(self.shape.len(), self.elements)?;
        dimensions.extend(0..self.shape.len());
        if !dimensions.is_empty() {
            dimensions.swap(axis0, axis1);
        }

        self.permute_axes_with_grad_fn(dimensions, AutogradNode::Transpose)
    }

    /// Swaps the final two dimensions without copying storage.
    ///
    /// This is the shared primitive for `PyTorch`'s `mT` property and, for the
    /// currently supported real-valued `float32` dtype, its `H` and `mH`
    /// properties. The Python `H` descriptor separately limits inputs to
    /// matrices. Scalars are identity aliases, vectors are rejected, and
    /// tensors of rank two or greater use the same checked path as
    /// `transpose(-2, -1)`.
    ///
    /// # Errors
    ///
    /// Returns an error for vectors, reordered element-count overflow, or view
    /// metadata allocation failure.
    pub fn matrix_transpose(&self) -> Result<Self, TensorError> {
        match self.shape.len() {
            0 => self.metadata_alias(),
            1 => Err(TensorError::MatrixTransposeRequiresMatrix { rank: 1 }),
            _ => self.transpose(-2, -1),
        }
    }

    #[cfg(feature = "python-bindings")]
    pub(crate) fn unsqueeze_front(&self) -> Result<Self, TensorError> {
        let mut shape = try_result_vector(self.shape.len() + 1, self.elements)?;
        shape.push(1);
        shape.extend_from_slice(&self.shape);

        let leading_stride = match (self.shape.first(), self.strides.first()) {
            (Some(dimension), Some(stride)) => {
                // PyTorch carries sizes and strides through signed 64-bit
                // arithmetic here, including wrapping for zero-element views.
                let leading_stride = signed_wrapping_stride_product_value(*stride, *dimension)?;
                // Packed SymInt values below -2^62 identify symbolic nodes
                // instead of concrete integers, even in an eager stride list.
                if leading_stride < MIN_CONCRETE_SYMINT {
                    return Err(TensorError::NonConcreteInteger);
                }
                if leading_stride < 0 {
                    let mut strides = try_result_vector(self.strides.len() + 1, self.elements)?;
                    strides.push(leading_stride);
                    for &stride in &self.strides {
                        strides.push(
                            i64::try_from(stride)
                                .map_err(|_| TensorError::StrideCalculationOverflow)?,
                        );
                    }
                    return Err(TensorError::NegativeStrides { strides });
                }
                usize::try_from(leading_stride)
                    .map_err(|_| TensorError::StrideCalculationOverflow)?
            }
            (None, None) => 1,
            _ => unreachable!("validated tensor shape and stride ranks must match"),
        };
        let mut strides = try_result_vector(self.strides.len() + 1, self.elements)?;
        strides.push(leading_stride);
        strides.extend_from_slice(&self.strides);

        self.finish_view_transform(
            Self {
                storage: Arc::clone(&self.storage),
                shape,
                strides,
                offset: self.offset,
                elements: self.elements,
                output_nr: 0,
                view_requires_grad: false,
                autograd: None,
            },
            TransformMapping::Identity,
            AutogradNode::Unsqueeze,
        )
    }

    #[cfg(feature = "python-bindings")]
    pub(crate) fn unsqueeze_back(&self) -> Result<Self, TensorError> {
        let mut shape = try_result_vector(self.shape.len() + 1, self.elements)?;
        shape.extend_from_slice(&self.shape);
        shape.push(1);

        let mut strides = try_result_vector(self.strides.len() + 1, self.elements)?;
        strides.extend_from_slice(&self.strides);
        strides.push(1);

        self.finish_view_transform(
            Self {
                storage: Arc::clone(&self.storage),
                shape,
                strides,
                offset: self.offset,
                elements: self.elements,
                output_nr: 0,
                view_requires_grad: false,
                autograd: None,
            },
            TransformMapping::Identity,
            AutogradNode::Unsqueeze,
        )
    }

    /// Removes every singleton dimension without copying storage.
    ///
    /// Shape and stride entries for dimensions of size one are dropped while
    /// the storage offset, dtype, device, and underlying allocation are
    /// retained. Scalars therefore produce another scalar alias.
    ///
    /// # Errors
    ///
    /// Returns an error when view metadata allocation fails.
    pub fn squeeze(&self) -> Result<Self, TensorError> {
        self.squeeze_selected(|_, dimension| dimension == 1, AutogradNode::Squeeze)
    }

    /// Removes one singleton dimension without copying storage.
    ///
    /// A valid dimension whose size is not one returns an alias with unchanged
    /// metadata. Negative dimensions wrap from the end, and scalars accept
    /// dimensions `0` and `-1`, matching `PyTorch`.
    ///
    /// # Errors
    ///
    /// Returns an error for an out-of-range dimension or when view metadata
    /// allocation fails.
    pub fn squeeze_dim(&self, dimension: i64) -> Result<Self, TensorError> {
        let axis = normalize_transpose_dimension(dimension, self.shape.len())?;
        self.squeeze_selected(
            |candidate, size| candidate == axis && size == 1,
            AutogradNode::SqueezeDimension,
        )
    }

    /// Removes the selected singleton dimensions without copying storage.
    ///
    /// Dimensions are normalized against the original rank before any are
    /// removed. An empty list is a metadata-preserving alias. This models
    /// `PyTorch`'s tuple/list overload, including its public 64-dimension
    /// implementation limit.
    ///
    /// # Errors
    ///
    /// Returns an error for ranks above 64, duplicate or out-of-range
    /// dimensions, or when view metadata allocation fails.
    pub fn squeeze_dims(&self, dimensions: impl AsRef<[i64]>) -> Result<Self, TensorError> {
        if self.shape.len() > 64 {
            return Err(TensorError::SqueezeDimensionsRankLimit);
        }

        let mut selected = 0_u64;
        for dimension in dimensions.as_ref() {
            let axis = normalize_transpose_dimension(*dimension, self.shape.len())?;
            let mask = 1_u64 << axis;
            if selected & mask != 0 {
                return Err(TensorError::DuplicateDimension { dimension: axis });
            }
            selected |= mask;
        }

        self.squeeze_selected(
            |axis, size| selected & (1_u64 << axis) != 0 && size == 1,
            AutogradNode::SqueezeDimensions,
        )
    }

    fn squeeze_selected(
        &self,
        remove: impl Fn(usize, usize) -> bool,
        node: AutogradNode,
    ) -> Result<Self, TensorError> {
        let mut shape = try_result_vector(self.shape.len(), self.elements)?;
        let mut strides = try_result_vector(self.strides.len(), self.elements)?;
        for (axis, (&dimension, &stride)) in self.shape.iter().zip(self.strides.iter()).enumerate()
        {
            if !remove(axis, dimension) {
                shape.push(dimension);
                strides.push(stride);
            }
        }
        let mut output = Self {
            storage: Arc::clone(&self.storage),
            shape,
            strides,
            offset: self.offset,
            elements: self.elements,
            output_nr: 0,
            view_requires_grad: false,
            autograd: None,
        };
        self.record_view_transform(&mut output, TransformMapping::Identity, node)?;
        Ok(output)
    }

    /// Collapses an inclusive range of dimensions using view-or-copy semantics.
    ///
    /// `start_dim` and `end_dim` are normalized, non-negative dimension
    /// indexes. When the range can be represented by the existing strides,
    /// the result shares storage and preserves its offset. Otherwise logical
    /// values are eagerly packed into independent contiguous storage. Scalars
    /// use the single logical dimension `0` and become one-element vectors.
    ///
    /// This is the reusable, binding-independent primitive behind
    /// [`Self::flatten`].
    ///
    /// # Errors
    ///
    /// Returns an error for an out-of-range or reversed range, arithmetic
    /// overflow, or metadata/storage allocation failure.
    pub fn collapse_dimensions(
        &self,
        start_dim: usize,
        end_dim: usize,
    ) -> Result<Self, TensorError> {
        let effective_rank = self.shape.len().max(1);
        if start_dim >= effective_rank {
            return Err(TensorError::DimensionOutOfRange {
                dimension: dimension_for_error(start_dim),
                rank: self.shape.len(),
            });
        }
        if end_dim >= effective_rank {
            return Err(TensorError::DimensionOutOfRange {
                dimension: dimension_for_error(end_dim),
                rank: self.shape.len(),
            });
        }
        if start_dim > end_dim {
            return Err(TensorError::FlattenStartAfterEnd);
        }

        if !self.shape.is_empty() && start_dim == end_dim {
            return self.metadata_alias();
        }
        if self.shape.is_empty() {
            return self.reshape([1]);
        }

        let output_rank = self.shape.len() - (end_dim - start_dim);
        let mut shape = try_result_vector(output_rank, self.elements)?;
        for &dimension in &self.shape[..start_dim] {
            shape.push(
                i64::try_from(dimension).map_err(|_| TensorError::StrideCalculationOverflow)?,
            );
        }

        let collapsed =
            self.shape[start_dim..=end_dim]
                .iter()
                .try_fold(1_i64, |elements, &dimension| {
                    let dimension = i64::try_from(dimension)
                        .map_err(|_| TensorError::StrideCalculationOverflow)?;
                    Ok::<_, TensorError>(elements.wrapping_mul(dimension))
                })?;
        shape.push(collapsed);
        for &dimension in &self.shape[end_dim.saturating_add(1)..] {
            shape.push(
                i64::try_from(dimension).map_err(|_| TensorError::StrideCalculationOverflow)?,
            );
        }

        // PyTorch's flatten kernel carries this wrapped product through an
        // unchecked SymInt slot. The minimum signed value is interpreted as
        // non-concrete metadata rather than as an ordinary negative shape.
        if collapsed == i64::MIN {
            return Err(TensorError::FlattenNonConcreteInteger);
        }
        if collapsed < -1 {
            return Err(TensorError::ReshapeInvalidDimension {
                dimension: collapsed,
                index: start_dim,
                shape,
            });
        }
        self.reshape(shape)
    }

    /// Flattens an inclusive range of dimensions.
    ///
    /// Negative dimensions wrap from the end. Compatible strides produce a
    /// metadata-only shared-storage view; incompatible layouts are eagerly
    /// copied into independent contiguous storage. Scalars flatten to shape
    /// `[1]`.
    ///
    /// # Errors
    ///
    /// Returns an error for an out-of-range or reversed range, arithmetic
    /// overflow, or metadata/storage allocation failure.
    pub fn flatten(&self, start_dim: i64, end_dim: i64) -> Result<Self, TensorError> {
        let start_dim = normalize_transpose_dimension(start_dim, self.shape.len())?;
        let end_dim = normalize_transpose_dimension(end_dim, self.shape.len())?;
        self.collapse_dimensions(start_dim, end_dim)
    }

    /// Returns a contiguous one-dimensional tensor using view-or-copy semantics.
    ///
    /// Row-contiguous inputs retain shared storage and their offset. Inputs
    /// which are not contiguous are packed first, so even an already
    /// one-dimensional strided input gets independent storage. The final
    /// collapse is delegated to [`Self::flatten`], including its scalar,
    /// empty, stride, and autograd behavior.
    ///
    /// # Errors
    ///
    /// Returns an error when contiguous storage or result metadata cannot be
    /// created.
    pub fn ravel(&self) -> Result<Self, TensorError> {
        let contiguous = self.try_contiguous(MemoryFormat::Contiguous)?;
        if contiguous.shape.len() == 1 {
            return contiguous.metadata_alias_with_grad_fn(AutogradNode::View);
        }
        contiguous.flatten(0, -1)
    }

    /// Creates an independent copy of this tensor's logical values.
    ///
    /// The returned tensor preserves dense layouts and has a storage offset of
    /// zero. Non-dense views retain their dimension order in a packed layout.
    /// Only the logical range of a view is
    /// copied; unused values in the view's backing allocation are not retained.
    ///
    /// # Errors
    ///
    /// Returns an error when result metadata or storage allocation fails.
    pub fn try_clone(&self) -> Result<Self, TensorError> {
        self.try_clone_with_memory_format(MemoryFormat::Preserve)
    }

    /// Creates an independent copy using the requested storage layout.
    ///
    /// [`MemoryFormat::Preserve`] retains dense strides and packs non-dense
    /// views in the same dimension order. [`MemoryFormat::Contiguous`]
    /// recalculates canonical row-major strides. [`MemoryFormat::ChannelsLast`]
    /// and [`MemoryFormat::ChannelsLast3d`] recalculate canonical channel-last
    /// strides for rank-four and rank-five tensors, respectively. Preserve
    /// clones retain an existing dense channel-last layout.
    ///
    /// # Errors
    ///
    /// Returns an error when result metadata or storage allocation fails,
    /// stride calculation overflows, or a channel-last request has the wrong
    /// rank.
    pub fn try_clone_with_memory_format(
        &self,
        memory_format: MemoryFormat,
    ) -> Result<Self, TensorError> {
        let expected_rank = match memory_format {
            MemoryFormat::ChannelsLast => Some(4),
            MemoryFormat::ChannelsLast3d => Some(5),
            MemoryFormat::Preserve | MemoryFormat::Contiguous => None,
        };
        if let Some(expected_rank) = expected_rank
            && self.shape.len() != expected_rank
        {
            // PyTorch validates canonical destination metadata before
            // reporting a channel-format rank mismatch. Preserve-format
            // clones intentionally skip this preflight so representable
            // extreme empty layouts can retain their source strides.
            let _ = contiguous_strides(&self.shape, self.elements)?;
            return Err(TensorError::ContiguousMemoryFormatRankMismatch {
                memory_format,
                expected_rank,
                actual_rank: self.shape.len(),
            });
        }

        let elements = self.elements;
        let shape = try_clone_result_shape(&self.shape, elements)?;
        let strides = match memory_format {
            MemoryFormat::Preserve if elements == 0 || self.is_non_overlapping_and_dense() => {
                try_clone_result_shape(&self.strides, elements)?
            }
            MemoryFormat::Preserve => elementwise_output_strides(
                &shape,
                &[ElementwiseLayout::from_tensor(self)],
                elements,
            )?,
            MemoryFormat::Contiguous => contiguous_strides(&shape, elements)?,
            MemoryFormat::ChannelsLast => channels_last_strides(&shape, elements)?,
            MemoryFormat::ChannelsLast3d => channels_last_3d_strides(&shape, elements)?,
        };
        let data = self.materialize_with_strides(&strides, |value| value)?;
        let mut output = Self::from_owned_parts(data, shape, strides, self.dtype(), self.device());
        self.record_transform(&mut output, TransformMapping::Identity, AutogradNode::Clone)?;
        Ok(output)
    }

    /// Returns a tensor contiguous in the requested storage layout.
    ///
    /// An already-matching tensor returns a shared-storage metadata alias, so
    /// callers which own an object wrapper can preserve object identity.
    /// Otherwise this copies logical values into independent storage, resets
    /// the storage offset to zero, and assigns canonical strides for the
    /// requested format. This is the checked packing primitive reused by
    /// reshape and flatten when existing strides cannot represent a result.
    ///
    /// [`MemoryFormat::Preserve`] is accepted only by the row-contiguous
    /// identity path, matching `PyTorch`'s contiguous operator. Channel-last
    /// layouts require rank four or five, respectively.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid format/rank combination, a nontrivial
    /// preserve-format request, checked stride overflow, or allocation
    /// failure.
    pub fn try_contiguous(&self, memory_format: MemoryFormat) -> Result<Self, TensorError> {
        self.try_contiguous_impl(memory_format, true, AutogradNode::Clone)
    }

    #[cfg(feature = "python-bindings")]
    pub(crate) fn suggested_memory_format(&self) -> MemoryFormat {
        if layout_is_strides_like_channels_last(&self.shape, &self.strides) {
            MemoryFormat::ChannelsLast
        } else if layout_is_strides_like_channels_last_3d(&self.shape, &self.strides) {
            MemoryFormat::ChannelsLast3d
        } else {
            MemoryFormat::Contiguous
        }
    }

    #[cfg(feature = "python-bindings")]
    pub(crate) fn try_copy_with_memory_format(
        &self,
        memory_format: MemoryFormat,
    ) -> Result<Self, TensorError> {
        // PyTorch's device-copy path validates canonical destination metadata
        // before checking a requested channel-last format's rank. Keep this
        // copy-specific preflight separate from contiguous's existing
        // identity and rank-validation path.
        let _ = contiguous_strides(&self.shape, self.elements)?;
        self.try_contiguous_impl(memory_format, false, AutogradNode::Copy)
    }

    fn try_contiguous_impl(
        &self,
        memory_format: MemoryFormat,
        reuse_matching_storage: bool,
        node: AutogradNode,
    ) -> Result<Self, TensorError> {
        let expected_rank = match memory_format {
            MemoryFormat::ChannelsLast => Some(4),
            MemoryFormat::ChannelsLast3d => Some(5),
            MemoryFormat::Preserve | MemoryFormat::Contiguous => None,
        };
        if let Some(expected_rank) = expected_rank
            && self.shape.len() != expected_rank
        {
            return Err(TensorError::ContiguousMemoryFormatRankMismatch {
                memory_format,
                expected_rank,
                actual_rank: self.shape.len(),
            });
        }

        if reuse_matching_storage && self.is_contiguous_with_memory_format(memory_format) {
            let mut output = Self {
                storage: Arc::clone(&self.storage),
                shape: try_clone_result_shape(&self.shape, self.elements)?,
                strides: try_clone_result_shape(&self.strides, self.elements)?,
                offset: self.offset,
                elements: self.elements,
                output_nr: 0,
                view_requires_grad: false,
                autograd: None,
            };
            self.record_view_transform(&mut output, TransformMapping::Identity, node)?;
            return Ok(output);
        }

        let strides = match memory_format {
            MemoryFormat::Preserve => {
                return Err(TensorError::ContiguousPreserveFormatUnsupported);
            }
            MemoryFormat::Contiguous => contiguous_strides(&self.shape, self.elements)?,
            MemoryFormat::ChannelsLast => channels_last_strides(&self.shape, self.elements)?,
            MemoryFormat::ChannelsLast3d => channels_last_3d_strides(&self.shape, self.elements)?,
        };
        let shape = try_clone_result_shape(&self.shape, self.elements)?;
        let data = self.materialize_with_strides(&strides, |value| value)?;
        let mut output = Self::from_owned_parts(data, shape, strides, self.dtype(), self.device());
        self.record_transform(&mut output, TransformMapping::Identity, node)?;
        Ok(output)
    }

    pub(crate) fn metadata_alias(&self) -> Result<Self, TensorError> {
        self.metadata_alias_with_grad_fn(AutogradNode::Alias)
    }

    /// Applies an exact full slice to the leading dimension.
    ///
    /// For rank-one-or-higher tensors this is a metadata-only identity view
    /// with the slice-specific autograd node. Scalars preserve `PyTorch`'s
    /// dedicated indexing diagnostic.
    ///
    /// # Errors
    ///
    /// Returns an error for scalar tensors or if result metadata allocation
    /// fails.
    #[cfg_attr(not(any(feature = "python-bindings", test)), allow(dead_code))]
    pub(crate) fn index_full_slice(&self) -> Result<Self, TensorError> {
        if self.shape.is_empty() {
            return Err(TensorError::SliceCannotApplyToScalar);
        }
        self.metadata_alias_with_grad_fn(AutogradNode::Slice)
    }

    fn metadata_alias_with_grad_fn(&self, node: AutogradNode) -> Result<Self, TensorError> {
        let mut output = self.metadata_alias_detached()?;
        self.record_view_transform(&mut output, TransformMapping::Identity, node)?;
        Ok(output)
    }

    fn metadata_alias_detached(&self) -> Result<Self, TensorError> {
        Ok(Self {
            storage: Arc::clone(&self.storage),
            shape: try_clone_result_shape(&self.shape, self.elements)?,
            strides: try_clone_result_shape(&self.strides, self.elements)?,
            offset: self.offset,
            elements: self.elements,
            output_nr: 0,
            view_requires_grad: false,
            autograd: None,
        })
    }

    #[must_use]
    /// Returns the tensor's logical values as one borrowed slice when its
    /// layout is contiguous.
    ///
    /// # Panics
    ///
    /// Panics if this tensor is a non-contiguous view. Use
    /// [`Self::logical_values`] or [`Self::try_to_vec`] for arbitrary layouts.
    pub fn as_slice(&self) -> &[f32] {
        self.contiguous_slice()
            .expect("as_slice requires a contiguous tensor")
    }

    #[must_use]
    pub fn into_vec(self) -> Vec<f32> {
        if !self.is_contiguous() {
            return self.logical_values().collect();
        }
        let Self {
            storage,
            offset,
            elements,
            ..
        } = self;
        if elements == 0 {
            return Vec::new();
        }
        let end = offset + elements;
        match Arc::try_unwrap(storage) {
            Ok(storage) => storage.into_range(offset, end),
            Err(storage) => storage.copy_range(offset, end),
        }
    }

    /// Copies logical values into a contiguous row-major vector.
    ///
    /// # Errors
    ///
    /// Returns an error if result allocation fails.
    pub fn try_to_vec(&self) -> Result<Vec<f32>, TensorError> {
        if let Some(values) = self.contiguous_slice() {
            return copied_storage(values, self.elements);
        }
        let mut values = try_result_vector(self.elements, self.elements)?;
        values.extend(self.logical_values());
        Ok(values)
    }

    fn contiguous_slice(&self) -> Option<&[f32]> {
        if self.elements == 0 {
            return Some(&[]);
        }
        if !self.is_contiguous() {
            return None;
        }
        let values = self.storage.owned_values()?;
        let end = self.offset.checked_add(self.elements)?;
        values.get(self.offset..end)
    }

    fn dense_physical_slice(&self) -> Option<&[f32]> {
        // A non-overlapping dense view covers one contiguous physical interval
        // even when its logical dimension order is permuted.
        if !self.is_non_overlapping_and_dense() {
            return None;
        }
        let values = self.storage.owned_values()?;
        let end = self.offset.checked_add(self.elements)?;
        values.get(self.offset..end)
    }

    fn value_at_linear_index(&self, index: usize) -> f32 {
        if let Some(values) = self.contiguous_slice() {
            return values[index];
        }
        self.value_at_strided_linear_index(index)
    }

    fn value_at_strided_linear_index(&self, index: usize) -> f32 {
        let offset =
            logical_offset_for_linear_index(&self.shape, &self.strides, self.offset, index)
                .expect("validated tensor logical offset must fit in usize");
        self.storage
            .value(offset)
            .expect("validated tensor logical offset must address storage")
    }

    fn materialize_with_strides(
        &self,
        output_strides: &[usize],
        operation: impl Fn(f32) -> f32,
    ) -> Result<Vec<f32>, TensorError> {
        let mut output = try_result_vector(self.elements, self.elements)?;
        if layout_is_contiguous(&self.shape, output_strides, self.elements) {
            if let Some(values) = self.contiguous_slice() {
                output.extend(values.iter().copied().map(&operation));
            } else {
                output.extend(self.logical_values().map(&operation));
            }
            return Ok(output);
        }
        // Matching dense layouts have the same physical iteration order. An
        // immutable owned slice can therefore bypass per-element logical
        // offset decoding without changing the output layout.
        if self.strides == output_strides
            && let Some(values) = self.dense_physical_slice()
        {
            output.extend(values.iter().copied().map(&operation));
            return Ok(output);
        }
        output.resize(self.elements, 0.0);
        for (linear_index, value) in self.logical_values().enumerate() {
            let output_offset =
                logical_offset_for_linear_index(&self.shape, output_strides, 0, linear_index)?;
            let slot = output
                .get_mut(output_offset)
                .ok_or(TensorError::IndexCalculationOverflow)?;
            *slot = operation(value);
        }
        Ok(output)
    }

    /// Selects the leading dimension with one integer, returning a shared-storage view.
    ///
    /// Negative indices wrap from the end of the dimension. This entry point
    /// preserves the distinct `PyTorch` diagnostic for indexing a scalar with
    /// the integer zero; [`Self::index`] models tuple indexing instead.
    ///
    /// # Errors
    ///
    /// Returns an error when this tensor is scalar, the index is out of
    /// bounds, offset arithmetic overflows, or view metadata allocation fails.
    pub fn index_integer(&self, index: i64) -> Result<Self, TensorError> {
        if self.shape.is_empty() {
            return if index == 0 {
                Err(TensorError::InvalidScalarIndex)
            } else {
                Err(TensorError::IndexOutOfBounds {
                    index,
                    dimension: 0,
                    size: 0,
                })
            };
        }
        self.index_dimensions(&[index])
    }

    /// Selects consecutive leading dimensions with integer indices.
    ///
    /// The returned tensor is a metadata-only view that retains the remaining
    /// shape and strides and shares storage, dtype, and device with `self`.
    /// Passing no indices returns an alias with identical metadata.
    ///
    /// # Errors
    ///
    /// Returns an error for too many indices, an out-of-bounds index, checked
    /// arithmetic overflow, or view metadata allocation failure.
    pub fn index(&self, indices: impl AsRef<[i64]>) -> Result<Self, TensorError> {
        self.index_dimensions(indices.as_ref())
    }

    fn index_dimensions(&self, indices: &[i64]) -> Result<Self, TensorError> {
        self.index_dimensions_impl(indices, true)
    }

    #[cfg_attr(not(any(feature = "python-bindings", test)), allow(dead_code))]
    pub(crate) fn unbind_first_dimension(&self) -> Result<Vec<Self>, TensorError> {
        let Some(&output_count) = self.shape.first() else {
            return Err(TensorError::InvalidScalarIndex);
        };
        let mut outputs = try_result_vector(output_count, self.elements)?;
        for output_nr in 0..output_count {
            let index =
                i64::try_from(output_nr).map_err(|_| TensorError::IndexCalculationOverflow)?;
            outputs.push(self.index_dimensions_impl(&[index], false)?);
        }
        if self.records_grad() && !outputs.is_empty() {
            let output_elements = outputs[0].elements;
            let autograd = Arc::new(AutogradMeta {
                kind: AutogradKind::NonLeaf {
                    grad_fn: Mutex::new(Some(GradFn::Unbind {
                        input: SavedTensor::try_from_tensor(self, false)?,
                        output_count,
                        output_elements,
                    })),
                },
            });
            for (output_nr, output) in outputs.iter_mut().enumerate() {
                output.autograd = Some(Arc::clone(&autograd));
                output.output_nr = output_nr;
            }
        }
        Ok(outputs)
    }

    fn index_dimensions_impl(
        &self,
        indices: &[i64],
        record_history: bool,
    ) -> Result<Self, TensorError> {
        if indices.len() > self.shape.len() {
            return Err(TensorError::TooManyIndices {
                dimensions: self.shape.len(),
            });
        }

        let mut offset = self.offset;
        for (dimension, index) in indices.iter().copied().enumerate() {
            offset = self.checked_index_offset(offset, dimension, index)?;
        }

        let shape = try_clone_result_shape(&self.shape[indices.len()..], self.elements)?;
        let strides = try_clone_result_shape(&self.strides[indices.len()..], self.elements)?;
        let elements = element_count(&shape)?;
        validate_view_bounds(&shape, &strides, offset, elements, self.storage.len())?;
        let mut output = Self {
            storage: Arc::clone(&self.storage),
            shape,
            strides,
            offset,
            elements,
            output_nr: 0,
            view_requires_grad: self.requires_grad(),
            autograd: None,
        };
        if record_history && self.records_grad() {
            let mut logical_prefix = 0_usize;
            for (dimension, index) in indices.iter().copied().enumerate() {
                let size = self.shape[dimension];
                let signed_size =
                    i64::try_from(size).map_err(|_| TensorError::IndexCalculationOverflow)?;
                let normalized = if index < 0 {
                    signed_size
                        .checked_add(index)
                        .ok_or(TensorError::IndexCalculationOverflow)?
                } else {
                    index
                };
                let normalized = usize::try_from(normalized)
                    .map_err(|_| TensorError::IndexCalculationOverflow)?;
                logical_prefix = logical_prefix
                    .checked_mul(size)
                    .and_then(|prefix| prefix.checked_add(normalized))
                    .ok_or(TensorError::IndexCalculationOverflow)?;
            }
            let input_start = logical_prefix
                .checked_mul(elements)
                .ok_or(TensorError::IndexCalculationOverflow)?;
            self.record_transform(
                &mut output,
                TransformMapping::Index { input_start },
                AutogradNode::Select,
            )?;
        }
        Ok(output)
    }

    pub(crate) fn checked_index_offset(
        &self,
        offset: usize,
        dimension: usize,
        index: i64,
    ) -> Result<usize, TensorError> {
        let size = self
            .shape
            .get(dimension)
            .copied()
            .ok_or(TensorError::TooManyIndices {
                dimensions: self.shape.len(),
            })?;
        let signed_size = i64::try_from(size).map_err(|_| TensorError::IndexCalculationOverflow)?;
        if index < -signed_size || index >= signed_size {
            return Err(TensorError::IndexOutOfBounds {
                index,
                dimension,
                size,
            });
        }
        let normalized = if index < 0 {
            signed_size
                .checked_add(index)
                .ok_or(TensorError::IndexCalculationOverflow)?
        } else {
            index
        };
        let normalized =
            usize::try_from(normalized).map_err(|_| TensorError::IndexCalculationOverflow)?;
        let contribution = normalized
            .checked_mul(self.strides[dimension])
            .ok_or(TensorError::IndexCalculationOverflow)?;
        let offset = offset
            .checked_add(contribution)
            .ok_or(TensorError::IndexCalculationOverflow)?;
        if i64::try_from(offset).is_err() {
            let offset = i64::try_from(offset.cast_signed())
                .expect("an isize storage offset must fit in i64");
            return Err(TensorError::InvalidStorageOffset { offset });
        }
        Ok(offset)
    }

    /// Returns a tensor with the same logical values and a new shape.
    ///
    /// One dimension may be `-1`, in which case it is inferred from the
    /// tensor's element count. Storage is shared whenever the existing strides
    /// can represent the requested shape; otherwise logical values are copied
    /// into a new contiguous allocation, as with `PyTorch::reshape`.
    ///
    /// # Errors
    ///
    /// Returns an error for negative dimensions other than `-1`, multiple
    /// inferred dimensions, incompatible element counts, ambiguous inference
    /// for an empty tensor, arithmetic overflow, or metadata allocation
    /// failure.
    pub fn reshape(&self, shape: impl AsRef<[i64]>) -> Result<Self, TensorError> {
        let resolved = self.resolve_reshape_shape(shape.as_ref())?;
        self.reshape_resolved(resolved)
    }

    /// Returns a metadata-only tensor view with a new shape.
    ///
    /// One dimension may be `-1`, in which case it is inferred from the
    /// tensor's element count. The result always shares storage and preserves
    /// the source offset. Unlike [`Self::reshape`], layouts whose existing
    /// strides cannot represent the requested shape return an error instead of
    /// being copied into contiguous storage.
    ///
    /// # Errors
    ///
    /// Returns an error for negative dimensions other than `-1`, multiple
    /// inferred dimensions, incompatible element counts, ambiguous inference
    /// for an empty tensor, a stride-incompatible layout, arithmetic overflow,
    /// or metadata allocation failure.
    pub fn view(&self, shape: impl AsRef<[i64]>) -> Result<Self, TensorError> {
        let resolved = self.resolve_reshape_shape(shape.as_ref())?;
        self.view_resolved(resolved)
    }

    fn resolve_reshape_shape(&self, requested: &[i64]) -> Result<Vec<usize>, TensorError> {
        let mut inferred_index = None;

        for (index, dimension) in requested.iter().copied().enumerate() {
            if dimension == -1 {
                if inferred_index.replace(index).is_some() {
                    return Err(TensorError::ReshapeMultipleInferredDimensions);
                }
                continue;
            }
            if dimension < 0 {
                return Err(TensorError::ReshapeInvalidDimension {
                    dimension,
                    index,
                    shape: try_clone_reshape_shape(requested, self.elements)?,
                });
            }
        }

        let mut resolved = try_result_vector(requested.len(), self.elements)?;
        for dimension in requested.iter().copied() {
            let dimension = if dimension == -1 {
                1
            } else {
                usize::try_from(dimension).map_err(|_| TensorError::ElementCountOverflow)?
            };
            resolved.push(dimension);
        }

        if let Some(index) = inferred_index {
            let specified_elements = requested
                .iter()
                .copied()
                .filter(|dimension| *dimension != -1)
                .fold(1_i64, i64::wrapping_mul);
            let elements =
                i64::try_from(self.elements).map_err(|_| TensorError::ElementCountOverflow)?;
            if !((specified_elements > 0 && elements % specified_elements == 0)
                || elements == specified_elements)
            {
                return Err(TensorError::ReshapeElementCountMismatch {
                    shape: try_clone_reshape_shape(requested, self.elements)?,
                    elements: self.elements,
                });
            }
            if specified_elements == 0 {
                if self.elements == 0 {
                    return Err(TensorError::ReshapeAmbiguousZeroElements {
                        shape: try_clone_reshape_shape(requested, self.elements)?,
                    });
                }
                return Err(TensorError::ReshapeElementCountMismatch {
                    shape: try_clone_reshape_shape(requested, self.elements)?,
                    elements: self.elements,
                });
            }
            resolved[index] = usize::try_from(elements / specified_elements)
                .map_err(|_| TensorError::ElementCountOverflow)?;
        }

        let resolved_elements = element_count(&resolved)?;
        if resolved_elements != self.elements {
            return Err(TensorError::ReshapeElementCountMismatch {
                shape: try_clone_reshape_shape(requested, self.elements)?,
                elements: self.elements,
            });
        }

        Ok(resolved)
    }

    fn reshape_resolved(&self, resolved: Vec<usize>) -> Result<Self, TensorError> {
        if let Some(strides) = self.reshape_view_strides(&resolved)? {
            return self.finish_reshape_view(resolved, strides);
        }

        let strides = contiguous_strides(&resolved, self.elements)?;
        let packed = self.try_contiguous(MemoryFormat::Contiguous)?;
        self.finish_copy_transform(
            Self {
                storage: packed.storage,
                shape: resolved,
                strides,
                offset: 0,
                elements: self.elements,
                output_nr: 0,
                view_requires_grad: false,
                autograd: None,
            },
            TransformMapping::Identity,
            AutogradNode::View,
        )
    }

    fn view_resolved(&self, resolved: Vec<usize>) -> Result<Self, TensorError> {
        let strides = self
            .reshape_view_strides(&resolved)?
            .ok_or(TensorError::ViewIncompatibleLayout)?;
        self.finish_reshape_view(resolved, strides)
    }

    fn reshape_view_strides(&self, resolved: &[usize]) -> Result<Option<Vec<usize>>, TensorError> {
        if self.elements == 0 {
            let strides = if resolved == self.shape {
                try_clone_result_shape(&self.strides, self.elements)?
            } else {
                reshape_strides(resolved, self.elements)?
            };
            return Ok(Some(strides));
        }

        compute_reshape_view_strides(&self.shape, &self.strides, resolved, self.elements)
    }

    fn finish_reshape_view(
        &self,
        shape: Vec<usize>,
        strides: Vec<usize>,
    ) -> Result<Self, TensorError> {
        self.finish_view_transform(
            Self {
                storage: Arc::clone(&self.storage),
                shape,
                strides,
                offset: self.offset,
                elements: self.elements,
                output_nr: 0,
                view_requires_grad: false,
                autograd: None,
            },
            TransformMapping::Identity,
            AutogradNode::View,
        )
    }

    /// Adds tensors element by element with trailing-dimension broadcasting.
    ///
    /// # Errors
    ///
    /// Returns an error when the shapes are not broadcastable or when result
    /// shape calculation or allocation fails.
    pub fn add(&self, other: &Self) -> Result<Self, TensorError> {
        self.zip_map(other, |left, right| left + right)
    }

    /// Subtracts tensors element by element with trailing-dimension broadcasting.
    ///
    /// # Errors
    ///
    /// Returns an error when the shapes are not broadcastable or when result
    /// shape calculation or allocation fails.
    pub fn sub(&self, other: &Self) -> Result<Self, TensorError> {
        self.zip_map(other, |left, right| left - right)
    }

    /// Multiplies tensors element by element with trailing-dimension broadcasting.
    ///
    /// # Errors
    ///
    /// Returns an error when the shapes are not broadcastable or when result
    /// shape calculation or allocation fails.
    pub fn mul(&self, other: &Self) -> Result<Self, TensorError> {
        let mut output = self.multiply_values(other)?;
        if (self.requires_grad() || other.requires_grad()) && is_grad_enabled() {
            let left_has_edge = self.autograd.is_some();
            let right_has_edge = other.autograd.is_some();
            let output_shape = try_clone_result_shape(&output.shape, output.elements)?;
            let grad_fn = GradFn::Multiply {
                left: SavedTensor::try_from_tensor(self, right_has_edge)?,
                right: SavedTensor::try_from_tensor(other, left_has_edge)?,
                output_shape,
                output_elements: output.elements,
            };
            output.autograd = Some(Arc::new(AutogradMeta {
                kind: AutogradKind::NonLeaf {
                    grad_fn: Mutex::new(Some(grad_fn)),
                },
            }));
        }
        Ok(output)
    }

    /// Computes the absolute value of every element using unary output layout planning.
    ///
    /// # Errors
    ///
    /// Returns an error when result metadata or storage allocation fails.
    #[cfg(any(feature = "python-bindings", test))]
    pub(crate) fn absolute(&self) -> Result<Self, TensorError> {
        self.unary_map(f32::abs)
    }

    /// Squares every element through the shared-operand multiplication kernel.
    ///
    /// # Errors
    ///
    /// Returns an error when result metadata or storage allocation fails.
    #[cfg(any(feature = "python-bindings", test))]
    pub(crate) fn square(&self) -> Result<Self, TensorError> {
        let output = self.multiply_values(self)?;
        self.finish_saved_input_unary_vjp(output, AutogradNode::Power, apply_square_vjp)
    }

    /// Squares a materialized dense tensor while retaining its exact strides.
    ///
    /// This supports composed operators whose preceding binary `TensorIterator`
    /// layout remains observable in the final output. Standalone square keeps
    /// using its normal unary layout canonicalization.
    ///
    /// # Errors
    ///
    /// Returns an error when result metadata or storage allocation fails.
    #[cfg(any(feature = "python-bindings", test))]
    pub(crate) fn square_preserving_strides(&self) -> Result<Self, TensorError> {
        debug_assert_eq!(self.offset, 0);
        debug_assert!(self.elements == 0 || self.is_non_overlapping_and_dense());
        let strides = try_clone_result_shape(&self.strides, self.elements)?;
        let mut output = self.square()?;
        // For a dense materialized input, unary planning can only alter the
        // strides of singleton or zero-sized axes. Restoring those strides
        // therefore leaves every stored element at the same physical offset.
        debug_assert_eq!(output.offset, 0);
        output.strides = strides;
        Ok(output)
    }

    /// Divides tensors element by element using IEEE 754 true division and
    /// trailing-dimension broadcasting.
    ///
    /// # Errors
    ///
    /// Returns an error when the shapes are not broadcastable or when result
    /// shape calculation or allocation fails.
    pub fn div(&self, other: &Self) -> Result<Self, TensorError> {
        self.zip_map(other, |left, right| left / right)
    }

    /// Adds a scalar to every element.
    ///
    /// # Errors
    ///
    /// Returns an error when result allocation fails.
    pub fn add_scalar(&self, scalar: f32) -> Result<Self, TensorError> {
        let output = self.map_scalar(scalar, |value, scalar| value + scalar)?;
        self.finish_copy_transform(output, TransformMapping::Identity, AutogradNode::Add)
    }

    /// Subtracts a scalar from every element.
    ///
    /// # Errors
    ///
    /// Returns an error when result allocation fails.
    pub fn sub_scalar(&self, scalar: f32) -> Result<Self, TensorError> {
        let output = self.map_scalar(scalar, |value, scalar| value - scalar)?;
        self.finish_copy_transform(output, TransformMapping::Identity, AutogradNode::Subtract)
    }

    /// Multiplies every element by a scalar.
    ///
    /// # Errors
    ///
    /// Returns an error when result allocation fails.
    pub fn mul_scalar(&self, scalar: f32) -> Result<Self, TensorError> {
        let mut output = self.map_scalar(scalar, |value, scalar| value * scalar)?;
        if self.requires_grad() && is_grad_enabled() {
            let input = SavedTensor::try_from_tensor(self, false)?;
            let scalar = input.autograd.is_some().then_some(scalar);
            output.autograd = Some(Arc::new(AutogradMeta {
                kind: AutogradKind::NonLeaf {
                    grad_fn: Mutex::new(Some(GradFn::MultiplyScalar { input, scalar })),
                },
            }));
        }
        Ok(output)
    }

    /// Negates every element by toggling its IEEE 754 sign bit.
    ///
    /// # Errors
    ///
    /// Returns an error when result metadata or storage allocation fails.
    #[cfg(any(feature = "python-bindings", test))]
    pub(crate) fn negate(&self) -> Result<Self, TensorError> {
        let output = self.unary_map(negate_value)?;
        self.finish_negate_vjp(output, AutogradNode::Negate)
    }

    /// Divides every element by a scalar using IEEE 754 true division.
    ///
    /// # Errors
    ///
    /// Returns an error when result allocation fails.
    pub fn div_scalar(&self, scalar: f32) -> Result<Self, TensorError> {
        self.map_scalar(scalar, |value, scalar| value / scalar)
    }

    /// Subtracts every element from a scalar.
    ///
    /// # Errors
    ///
    /// Returns an error when result allocation fails.
    pub fn scalar_sub(&self, scalar: f32) -> Result<Self, TensorError> {
        let output = self.map_scalar(scalar, |value, scalar| scalar - value)?;
        self.finish_negate_vjp(output, AutogradNode::ReflectedSubtract)
    }

    /// Divides a scalar by every element using `PyTorch`'s float32 reciprocal
    /// multiplication semantics.
    ///
    /// # Errors
    ///
    /// Returns an error when result allocation fails.
    pub fn scalar_div(&self, scalar: f32) -> Result<Self, TensorError> {
        let elements = self.elements;
        let shape = try_clone_result_shape(&self.shape, elements)?;
        // PyTorch plans reflected division like a reciprocal followed by a
        // scalar pointwise operation. The second pass is observable for empty
        // singleton dimensions even though the numerical work remains fused.
        let reciprocal_strides = self.unary_output_strides(&shape, elements)?;
        let strides = elementwise_output_strides(
            &shape,
            &[ElementwiseLayout {
                shape: &shape,
                strides: &reciprocal_strides,
            }],
            elements,
        )?;
        self.scalar_div_with_output_layout(scalar, shape, strides)
    }

    /// Computes the reciprocal of every element using unary output layout planning.
    ///
    /// # Errors
    ///
    /// Returns an error when gradient recording is enabled for this tensor, or
    /// when result metadata or storage allocation fails.
    pub fn reciprocal(&self) -> Result<Self, TensorError> {
        if self.records_grad() {
            return Err(TensorError::AutogradRecordingUnsupported {
                operation: "reciprocal",
            });
        }
        self.unary_map(|value| 1.0 * value.recip())
    }

    fn scalar_div_with_output_layout(
        &self,
        scalar: f32,
        shape: Vec<usize>,
        strides: Vec<usize>,
    ) -> Result<Self, TensorError> {
        let data = self.materialize_with_strides(&strides, |value| scalar * value.recip())?;
        Ok(Self::from_owned_parts(
            data,
            shape,
            strides,
            self.dtype(),
            self.device(),
        ))
    }

    /// Applies rectified linear activation element by element.
    ///
    /// # Errors
    ///
    /// Returns an error when result metadata or storage allocation fails.
    pub fn relu(&self) -> Result<Self, TensorError> {
        let output = self.unary_map(relu_value)?;
        self.finish_saved_input_unary_vjp(output, AutogradNode::Relu, apply_relu_vjp)
    }

    /// Computes the sine of every element in radians.
    ///
    /// # Errors
    ///
    /// Returns an error when result metadata or storage allocation fails.
    pub fn sin(&self) -> Result<Self, TensorError> {
        let output = self.unary_map(f32::sin)?;
        self.finish_saved_input_unary_vjp(output, AutogradNode::Sin, apply_sin_vjp)
    }

    /// Computes the base-e exponential of every element.
    ///
    /// # Errors
    ///
    /// Returns an error when result metadata or storage allocation fails.
    pub fn exp(&self) -> Result<Self, TensorError> {
        let output = self.unary_map(f32::exp)?;
        self.finish_saved_output_unary_vjp(output, AutogradNode::Exp, apply_exp_vjp)
    }

    /// Rounds every element down to the nearest integer.
    ///
    /// # Errors
    ///
    /// Returns an error when gradient recording is enabled for this tensor, or
    /// when result metadata or storage allocation fails.
    pub fn floor(&self) -> Result<Self, TensorError> {
        if self.records_grad() {
            return Err(TensorError::AutogradRecordingUnsupported { operation: "floor" });
        }
        self.unary_map(floor_value)
    }

    /// Computes the hyperbolic tangent of every element.
    ///
    /// # Errors
    ///
    /// Returns an error when gradient recording is enabled for an input other
    /// than a finite, owned, rank-zero CPU float32 leaf, or when result
    /// metadata or storage allocation fails.
    pub fn tanh(&self) -> Result<Self, TensorError> {
        if self.records_grad() && !self.is_finite_owned_scalar_leaf() {
            return Err(TensorError::AutogradRecordingUnsupported { operation: "tanh" });
        }
        let output = self.unary_map(tanh_value)?;
        self.finish_saved_output_unary_vjp(output, AutogradNode::Tanh, apply_tanh_vjp)
    }

    /// Computes the nonnegative square root of every element.
    ///
    /// # Errors
    ///
    /// Returns an error when result metadata or storage allocation fails.
    pub fn sqrt(&self) -> Result<Self, TensorError> {
        let output = self.unary_map(sqrt_value)?;
        self.finish_saved_input_unary_vjp(output, AutogradNode::Sqrt, apply_sqrt_vjp)
    }

    #[must_use]
    pub fn sum(&self) -> Self {
        let mut output = Self::from_owned_parts(
            vec![self.logical_values().sum()],
            Vec::new(),
            Vec::new(),
            self.dtype(),
            self.device(),
        );
        if self.requires_grad() && is_grad_enabled() {
            output.autograd = Some(Arc::new(AutogradMeta {
                kind: AutogradKind::NonLeaf {
                    grad_fn: Mutex::new(Some(GradFn::Sum {
                        input: SavedTensor::from_tensor_metadata(self),
                    })),
                },
            }));
        }
        output
    }

    /// Extracts the value of a one-element tensor.
    ///
    /// # Errors
    ///
    /// Returns an error unless the tensor contains exactly one element.
    pub fn item(&self) -> Result<f32, TensorError> {
        if self.elements != 1 {
            return Err(TensorError::ItemRequiresOneElement {
                elements: self.elements,
            });
        }
        Ok(self.value_at_linear_index(0))
    }

    /// Multiplies two rank-2 matrices.
    ///
    /// # Errors
    ///
    /// Returns an error unless both tensors are matrices with compatible inner
    /// dimensions.
    pub fn matmul(&self, other: &Self) -> Result<Self, TensorError> {
        self.matmul_with_initializer(other, |_, _, output_elements| {
            filled_storage(output_elements, 0.0)
        })
    }

    /// Multiplies two rank-2 matrices after broadcasting a rank-1 bias across
    /// the output rows.
    ///
    /// Bias values seed the accumulators before products are added, matching
    /// `addmm` ordering while reusing the ordinary matrix multiplication loop.
    ///
    /// # Errors
    ///
    /// Returns an error unless both matrix operands and the row bias have
    /// compatible shapes, or when result allocation fails.
    #[cfg(any(feature = "python-bindings", test))]
    pub(crate) fn matmul_with_row_bias(
        &self,
        other: &Self,
        bias: &Self,
    ) -> Result<Self, TensorError> {
        self.matmul_with_initializer(other, |rows, columns, output_elements| {
            if bias.shape.len() != 1 || bias.shape[0] != columns {
                let mut expected_bias_shape = try_result_vector(1, output_elements)?;
                expected_bias_shape.push(columns);
                return Err(TensorError::ShapeMismatch {
                    left: expected_bias_shape,
                    right: try_clone_result_shape(&bias.shape, bias.elements)?,
                });
            }

            let bias_values = bias.try_to_vec()?;
            let mut output = try_result_vector(output_elements, output_elements)?;
            for _ in 0..rows {
                output.extend_from_slice(&bias_values);
            }
            Ok(output)
        })
    }

    fn matmul_with_initializer(
        &self,
        other: &Self,
        initialize: impl FnOnce(usize, usize, usize) -> Result<Vec<f32>, TensorError>,
    ) -> Result<Self, TensorError> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(TensorError::MatmulRequiresMatrices {
                left: self.shape.clone(),
                right: other.shape.clone(),
            });
        }
        let (rows, inner) = (self.shape[0], self.shape[1]);
        let (other_inner, columns) = (other.shape[0], other.shape[1]);
        if inner != other_inner {
            return Err(TensorError::MatmulInnerDimensionMismatch {
                left: self.shape.clone(),
                right: other.shape.clone(),
            });
        }

        let mut output_shape = try_result_vector(2, 0)?;
        output_shape.push(rows);
        output_shape.push(columns);
        let (output_elements, output_strides) = validated_layout(&output_shape)?;
        let mut output = initialize(rows, columns, output_elements)?;
        debug_assert_eq!(output.len(), output_elements);
        if let (Some(left_data), Some(right_data)) =
            (self.contiguous_slice(), other.contiguous_slice())
        {
            accumulate_contiguous_matmul(left_data, right_data, &mut output, rows, inner, columns);
        } else if output_elements != 0
            && inner != 0
            && let (Some(left_data), Some(right_data)) =
                (self.storage.owned_values(), other.storage.owned_values())
        {
            // Immutable owned storage can be borrowed for the whole kernel.
            // Check each monotonic row span once before incrementing within it.
            let left_depth_stride = self.strides[1];
            let right_column_stride = other.strides[1];
            for (row, output_row) in output.chunks_exact_mut(columns).enumerate() {
                let mut left_offset = checked_matrix_row_base(self, row, inner, left_data.len())?;
                for depth in 0..inner {
                    let left = left_data[left_offset];
                    let mut right_offset =
                        checked_matrix_row_base(other, depth, columns, right_data.len())?;
                    let mut column = 0;
                    loop {
                        output_row[column] += left * right_data[right_offset];
                        column += 1;
                        if column == columns {
                            break;
                        }
                        right_offset += right_column_stride;
                    }
                    if depth + 1 != inner {
                        left_offset += left_depth_stride;
                    }
                }
            }
        } else {
            for row in 0..rows {
                for depth in 0..inner {
                    let left_offset = checked_matrix_offset(self, row, depth)?;
                    let left = self
                        .storage
                        .value(left_offset)
                        .ok_or(TensorError::IndexCalculationOverflow)?;
                    for column in 0..columns {
                        let right_offset = checked_matrix_offset(other, depth, column)?;
                        let right = other
                            .storage
                            .value(right_offset)
                            .ok_or(TensorError::IndexCalculationOverflow)?;
                        output[row * columns + column] += left * right;
                    }
                }
            }
        }
        Ok(Self::from_owned_parts(
            output,
            output_shape,
            output_strides,
            self.dtype(),
            self.device(),
        ))
    }

    fn zip_map(
        &self,
        other: &Self,
        operation: impl Fn(f32, f32) -> f32,
    ) -> Result<Self, TensorError> {
        if self.shape == other.shape {
            return self.zip_map_same_shape(other, operation);
        }

        if let Some((data, plan)) =
            materialize_contiguous_trailing_broadcast(self, other, &operation)?
        {
            return Ok(Self::from_owned_parts(
                data,
                plan.shape,
                plan.strides,
                self.dtype(),
                self.device(),
            ));
        }

        self.zip_map_broadcast(other, operation)
    }

    fn multiply_values(&self, other: &Self) -> Result<Self, TensorError> {
        self.zip_map(other, |left, right| left * right)
    }

    fn zip_map_broadcast(
        &self,
        other: &Self,
        operation: impl Fn(f32, f32) -> f32,
    ) -> Result<Self, TensorError> {
        let plan = BroadcastPlan::new(self, other)?;
        if plan.elements == 0 {
            let data = try_result_vector(plan.elements, plan.elements)?;
            return Ok(Self::from_owned_parts(
                data,
                plan.shape,
                plan.strides,
                self.dtype(),
                self.device(),
            ));
        }
        if self.shape.is_empty() != other.shape.is_empty() {
            let scalar = if self.shape.is_empty() {
                self.value_at_linear_index(0)
            } else {
                other.value_at_linear_index(0)
            };
            // A NaN scalar can meet another NaN in the materialized operand.
            // Keep that case in the original loop because payload precedence
            // depends on the target's scalar instruction selection.
            if !scalar.is_nan() {
                return self.zip_map_rank_zero(other, plan, scalar, operation);
            }
        }

        let data = if let (Some(left_values), Some(right_values)) =
            (self.storage.owned_values(), other.storage.owned_values())
        {
            materialize_broadcast(
                &plan,
                self.offset,
                other.offset,
                |offset| left_values[offset],
                |offset| right_values[offset],
                operation,
            )?
        } else {
            materialize_broadcast(
                &plan,
                self.offset,
                other.offset,
                |offset| {
                    self.storage
                        .value(offset)
                        .expect("validated broadcast offset must address left storage")
                },
                |offset| {
                    other
                        .storage
                        .value(offset)
                        .expect("validated broadcast offset must address right storage")
                },
                operation,
            )?
        };

        Ok(Self::from_owned_parts(
            data,
            plan.shape,
            plan.strides,
            self.dtype(),
            self.device(),
        ))
    }

    fn zip_map_same_shape(
        &self,
        other: &Self,
        operation: impl Fn(f32, f32) -> f32,
    ) -> Result<Self, TensorError> {
        let elements = self.elements;
        let shape = try_clone_result_shape(&self.shape, elements)?;
        // TensorIterator prioritizes canonical contiguous formats before an
        // arbitrary dense layout, which normalizes strides on singleton axes.
        let strides = if self.is_contiguous() && other.is_contiguous() {
            contiguous_strides(&shape, elements)?
        } else if self.is_channels_last_contiguous() && other.is_channels_last_contiguous() {
            channels_last_strides(&shape, elements)?
        } else if self.is_non_overlapping_and_dense()
            && other.is_non_overlapping_and_dense()
            && self.strides == other.strides
        {
            try_clone_result_shape(&self.strides, elements)?
        } else {
            elementwise_output_strides(
                &shape,
                &[
                    ElementwiseLayout::from_tensor(self),
                    ElementwiseLayout::from_tensor(other),
                ],
                elements,
            )?
        };
        if let (Some(left), Some(right)) = (self.contiguous_slice(), other.contiguous_slice()) {
            let mut data = try_result_vector(elements, elements)?;
            data.extend(
                left.iter()
                    .copied()
                    .zip(right.iter().copied())
                    .map(|(left, right)| operation(left, right)),
            );
            return Ok(Self::from_owned_parts(
                data,
                shape,
                strides,
                self.dtype(),
                self.device(),
            ));
        }
        // Identical dense strides give both inputs and the output the same
        // physical iteration order, avoiding repeated logical index decoding.
        if self.strides == strides
            && other.strides == strides
            && let (Some(left), Some(right)) =
                (self.dense_physical_slice(), other.dense_physical_slice())
            // Vector code may choose a different NaN payload than the original
            // scalar loop. Preserve target-specific behavior for NaN pairs.
            && !left
                .iter()
                .zip(right)
                .any(|(left, right)| left.is_nan() && right.is_nan())
        {
            let mut data = try_result_vector(elements, elements)?;
            data.extend(
                left.iter()
                    .copied()
                    .zip(right.iter().copied())
                    .map(|(left, right)| operation(left, right)),
            );
            return Ok(Self::from_owned_parts(
                data,
                shape,
                strides,
                self.dtype(),
                self.device(),
            ));
        }
        let mut data = filled_storage(elements, 0.0)?;
        for linear_index in 0..elements {
            let output_offset = logical_offset_for_linear_index(&shape, &strides, 0, linear_index)?;
            data[output_offset] = operation(
                self.value_at_linear_index(linear_index),
                other.value_at_linear_index(linear_index),
            );
        }
        Ok(Self::from_owned_parts(
            data,
            shape,
            strides,
            self.dtype(),
            self.device(),
        ))
    }

    fn map_scalar(
        &self,
        scalar: f32,
        operation: impl Fn(f32, f32) -> f32,
    ) -> Result<Self, TensorError> {
        let elements = self.elements;
        let shape = try_clone_result_shape(&self.shape, elements)?;
        let strides =
            elementwise_output_strides(&shape, &[ElementwiseLayout::from_tensor(self)], elements)?;
        let data = self.materialize_with_strides(&strides, |value| operation(value, scalar))?;
        Ok(Self::from_owned_parts(
            data,
            shape,
            strides,
            self.dtype(),
            self.device(),
        ))
    }

    fn unary_map(&self, operation: impl Fn(f32) -> f32) -> Result<Self, TensorError> {
        let elements = self.elements;
        let shape = try_clone_result_shape(&self.shape, elements)?;
        let strides = self.unary_output_strides(&shape, elements)?;
        let data = self.materialize_with_strides(&strides, operation)?;
        Ok(Self::from_owned_parts(
            data,
            shape,
            strides,
            self.dtype(),
            self.device(),
        ))
    }

    fn unary_output_strides(
        &self,
        shape: &[usize],
        elements: usize,
    ) -> Result<Vec<usize>, TensorError> {
        // Match TensorIterator's fast-setup precedence for a single operand.
        if self.is_contiguous() {
            contiguous_strides(shape, elements)
        } else if self.is_channels_last_contiguous() {
            channels_last_strides(shape, elements)
        } else if self.is_non_overlapping_and_dense() {
            try_clone_result_shape(&self.strides, elements)
        } else {
            elementwise_output_strides(shape, &[ElementwiseLayout::from_tensor(self)], elements)
        }
    }

    fn zip_map_rank_zero(
        &self,
        other: &Self,
        plan: BroadcastPlan,
        scalar: f32,
        operation: impl Fn(f32, f32) -> f32,
    ) -> Result<Self, TensorError> {
        debug_assert!(!scalar.is_nan());
        let data = if self.shape.is_empty() {
            other.materialize_with_strides(&plan.strides, |value| operation(scalar, value))?
        } else {
            debug_assert!(other.shape.is_empty());
            self.materialize_with_strides(&plan.strides, |value| operation(value, scalar))?
        };
        Ok(Self::from_owned_parts(
            data,
            plan.shape,
            plan.strides,
            self.dtype(),
            self.device(),
        ))
    }

    fn is_channels_last_contiguous(&self) -> bool {
        layout_is_channels_last_contiguous(&self.shape, &self.strides)
    }
}

impl SavedTensor {
    fn from_tensor_metadata(tensor: &Tensor) -> Self {
        Self {
            storage: None,
            shape: tensor.shape.clone(),
            strides: tensor.strides.clone(),
            offset: tensor.offset,
            elements: tensor.elements,
            output_nr: tensor.output_nr,
            autograd: tensor.autograd.as_ref().map(Arc::clone),
        }
    }

    fn try_from_tensor(tensor: &Tensor, save_values: bool) -> Result<Self, TensorError> {
        Ok(Self {
            storage: if save_values {
                Some(Storage::try_clone_for_saved(&tensor.storage, |values| {
                    copied_storage(values, values.len())
                })?)
            } else {
                None
            },
            shape: try_clone_result_shape(&tensor.shape, tensor.elements)?,
            strides: try_clone_result_shape(&tensor.strides, tensor.elements)?,
            offset: tensor.offset,
            elements: tensor.elements,
            output_nr: tensor.output_nr,
            autograd: tensor.autograd.as_ref().map(Arc::clone),
        })
    }

    fn contiguous_slice(&self) -> Option<&[f32]> {
        if self.elements == 0 {
            return Some(&[]);
        }
        if !layout_is_contiguous(&self.shape, &self.strides, self.elements) {
            return None;
        }
        let values = self.storage.as_ref()?.owned_values()?;
        let end = self.offset.checked_add(self.elements)?;
        values.get(self.offset..end)
    }

    fn value_at_linear_index(&self, index: usize) -> f32 {
        let offset =
            logical_offset_for_linear_index(&self.shape, &self.strides, self.offset, index)
                .expect("validated saved tensor logical offset must fit in usize");
        self.storage
            .as_ref()
            .expect("value-dependent derivative must retain saved tensor storage")
            .value(offset)
            .expect("validated saved tensor logical offset must address storage")
    }

    fn broadcast_position(&self, output_shape: &[usize], coordinates: &[usize]) -> (usize, usize) {
        let leading = output_shape.len() - self.shape.len();
        let mut logical_index = 0_usize;
        let mut storage_offset = self.offset;
        for (axis, (&dimension, &stride)) in self.shape.iter().zip(self.strides.iter()).enumerate()
        {
            let coordinate = if dimension == 1 {
                0
            } else {
                coordinates[leading + axis]
            };
            logical_index = logical_index
                .checked_mul(dimension)
                .and_then(|index| index.checked_add(coordinate))
                .expect("validated broadcast logical index must fit in usize");
            storage_offset = storage_offset
                .checked_add(
                    coordinate
                        .checked_mul(stride)
                        .expect("validated broadcast stride must fit in usize"),
                )
                .expect("validated broadcast storage offset must fit in usize");
        }
        (logical_index, storage_offset)
    }
}

type Topology = Vec<(Arc<AutogradMeta>, Option<GradFn>)>;
type GradientKey = (usize, usize);
type Gradients = HashMap<GradientKey, Vec<f32>>;

fn run_backward(root: &Arc<AutogradMeta>, root_output_nr: usize) -> Result<(), TensorError> {
    // Saved values form a transaction: a traversal must either consume all of
    // them and commit its leaf gradients, or consume none. Serializing the
    // complete operation prevents concurrent roots which share intermediates
    // from each consuming a different subset of the same graph.
    let _backward_traversal = BACKWARD_TRAVERSAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let topology = collect_topology(root)?;

    let mut gradients = HashMap::new();
    gradients.insert(gradient_key(root, root_output_nr), vec![1.0]);
    let mut leaf_gradients = Vec::new();

    for (meta, grad_fn) in topology.iter().rev() {
        match grad_fn {
            None => {
                if let Some(upstream) = gradients.remove(&gradient_key(meta, 0)) {
                    leaf_gradients.push((Arc::clone(meta), upstream));
                }
            }
            Some(GradFn::Unbind {
                input,
                output_count,
                output_elements,
            }) => {
                apply_unbind_grad_fn(meta, input, *output_count, *output_elements, &mut gradients)?;
            }
            Some(grad_fn) => {
                let Some(upstream) = gradients.remove(&gradient_key(meta, 0)) else {
                    continue;
                };
                apply_grad_fn(grad_fn, &upstream, &mut gradients)?;
            }
        }
    }

    // Release only value-dependent saved state after all fallible gradient
    // calculations have succeeded. Metadata-only edges remain available for
    // repeated backward traversals.
    for (meta, grad_fn) in &topology {
        if grad_fn.is_some()
            && let AutogradKind::NonLeaf { grad_fn } = &meta.kind
        {
            let mut grad_fn = grad_fn
                .lock()
                .expect("gradient function mutex must not be poisoned");
            grad_fn
                .as_mut()
                .ok_or(TensorError::BackwardGraphFreed)?
                .consume_saved_values()?;
        }
    }
    for (meta, gradient) in leaf_gradients {
        accumulate_leaf_gradient(&meta, gradient);
    }
    Ok(())
}

enum TopologyFrame {
    Enter(Arc<AutogradMeta>),
    Exit(Arc<AutogradMeta>, Box<Option<GradFn>>),
}

fn collect_topology(root: &Arc<AutogradMeta>) -> Result<Topology, TensorError> {
    let mut seen = HashSet::new();
    let mut topology = Vec::new();
    let mut stack = vec![TopologyFrame::Enter(Arc::clone(root))];
    while let Some(frame) = stack.pop() {
        match frame {
            TopologyFrame::Exit(meta, grad_fn) => topology.push((meta, *grad_fn)),
            TopologyFrame::Enter(meta) => {
                if !seen.insert(autograd_id(&meta)) {
                    continue;
                }
                let grad_fn = match &meta.kind {
                    AutogradKind::Leaf { .. } => None,
                    AutogradKind::NonLeaf { grad_fn } => {
                        let grad_fn = grad_fn
                            .lock()
                            .expect("gradient function mutex must not be poisoned")
                            .clone()
                            .ok_or(TensorError::BackwardGraphFreed)?;
                        grad_fn.validate_saved_values()?;
                        Some(grad_fn)
                    }
                };
                stack.push(TopologyFrame::Exit(
                    Arc::clone(&meta),
                    Box::new(grad_fn.clone()),
                ));
                if let Some(grad_fn) = &grad_fn {
                    match grad_fn {
                        GradFn::Multiply { left, right, .. } => {
                            push_saved_parent(&mut stack, right);
                            push_saved_parent(&mut stack, left);
                        }
                        GradFn::MultiplyScalar { input, .. }
                        | GradFn::Negate { input, .. }
                        | GradFn::Sum { input }
                        | GradFn::Transform { input, .. }
                        | GradFn::Unbind { input, .. } => {
                            push_saved_parent(&mut stack, input);
                        }
                        GradFn::SavedInputUnary(node) => {
                            push_saved_parent(&mut stack, &node.input);
                        }
                        GradFn::SavedOutputUnary(node) => {
                            push_saved_parent(&mut stack, &node.input);
                        }
                    }
                }
            }
        }
    }
    Ok(topology)
}

fn push_saved_parent(stack: &mut Vec<TopologyFrame>, tensor: &SavedTensor) {
    if let Some(meta) = &tensor.autograd {
        stack.push(TopologyFrame::Enter(Arc::clone(meta)));
    }
}

fn apply_grad_fn(
    grad_fn: &GradFn,
    upstream: &[f32],
    gradients: &mut Gradients,
) -> Result<(), TensorError> {
    match grad_fn {
        GradFn::Sum { input } => {
            if let Some(meta) = &input.autograd {
                let gradient = filled_storage(input.elements, upstream[0])?;
                add_gradient(gradients, meta, input.output_nr, gradient);
            }
        }
        GradFn::MultiplyScalar { input, scalar } => {
            if let Some(meta) = &input.autograd {
                let scalar = scalar.ok_or(TensorError::BackwardGraphFreed)?;
                let mut gradient = try_result_vector(input.elements, input.elements)?;
                gradient.extend(upstream.iter().map(|value| value * scalar));
                add_gradient(gradients, meta, input.output_nr, gradient);
            }
        }
        GradFn::Negate { input, .. } => {
            if let Some(meta) = &input.autograd {
                debug_assert_eq!(input.elements, upstream.len());
                let mut gradient = try_result_vector(input.elements, input.elements)?;
                gradient.extend(upstream.iter().copied().map(negate_value));
                add_gradient(gradients, meta, input.output_nr, gradient);
            }
        }
        GradFn::SavedInputUnary(node) => apply_saved_input_unary(node, upstream, gradients)?,
        GradFn::SavedOutputUnary(node) => apply_saved_output_unary(node, upstream, gradients)?,
        GradFn::Multiply {
            left,
            right,
            output_shape,
            output_elements,
        } => {
            debug_assert_eq!(*output_elements, upstream.len());
            let mut left_gradient = if left.autograd.is_some() {
                Some(GradientAccumulator::new(
                    left.elements,
                    left.elements == *output_elements,
                )?)
            } else {
                None
            };
            let mut right_gradient = if right.autograd.is_some() {
                Some(GradientAccumulator::new(
                    right.elements,
                    right.elements == *output_elements,
                )?)
            } else {
                None
            };
            let mut coordinates = try_result_vector(output_shape.len(), *output_elements)?;
            coordinates.resize(output_shape.len(), 0_usize);

            for (output_index, &output_gradient) in upstream.iter().enumerate() {
                let mut remaining = output_index;
                for axis in (0..output_shape.len()).rev() {
                    coordinates[axis] = remaining % output_shape[axis];
                    remaining /= output_shape[axis];
                }
                let (left_index, left_offset) = left.broadcast_position(output_shape, &coordinates);
                let (right_index, right_offset) =
                    right.broadcast_position(output_shape, &coordinates);
                if let Some(gradient) = &mut left_gradient {
                    gradient.add(
                        left_index,
                        output_gradient
                            * right
                                .storage
                                .as_ref()
                                .expect("left derivative must save right operand values")
                                .value(right_offset)
                                .expect("saved right operand offset must address storage"),
                    );
                }
                if let Some(gradient) = &mut right_gradient {
                    gradient.add(
                        right_index,
                        output_gradient
                            * left
                                .storage
                                .as_ref()
                                .expect("right derivative must save left operand values")
                                .value(left_offset)
                                .expect("saved left operand offset must address storage"),
                    );
                }
            }
            if let (Some(meta), Some(gradient)) = (&left.autograd, left_gradient) {
                add_gradient(gradients, meta, left.output_nr, gradient.values);
            }
            if let (Some(meta), Some(gradient)) = (&right.autograd, right_gradient) {
                add_gradient(gradients, meta, right.output_nr, gradient.values);
            }
        }
        GradFn::Transform { input, mapping, .. } => {
            if let Some(meta) = &input.autograd {
                let gradient = transform_backward(input, mapping, upstream)?;
                add_gradient(gradients, meta, input.output_nr, gradient);
            }
        }
        GradFn::Unbind { .. } => unreachable!(),
    }
    Ok(())
}

fn apply_unbind_grad_fn(
    node: &Arc<AutogradMeta>,
    input: &SavedTensor,
    output_count: usize,
    output_elements: usize,
    gradients: &mut Gradients,
) -> Result<(), TensorError> {
    let mut assembled = None;
    for output_nr in 0..output_count {
        let Some(output_gradient) = gradients.remove(&gradient_key(node, output_nr)) else {
            continue;
        };
        if output_gradient.len() != output_elements {
            return Err(TensorError::IndexCalculationOverflow);
        }
        let gradient = match &mut assembled {
            Some(gradient) => gradient,
            None => assembled.insert(filled_storage(input.elements, 0.0)?),
        };
        let start = output_nr
            .checked_mul(output_elements)
            .ok_or(TensorError::IndexCalculationOverflow)?;
        let end = start
            .checked_add(output_elements)
            .ok_or(TensorError::IndexCalculationOverflow)?;
        gradient
            .get_mut(start..end)
            .ok_or(TensorError::IndexCalculationOverflow)?
            .copy_from_slice(&output_gradient);
    }
    if let (Some(meta), Some(gradient)) = (&input.autograd, assembled) {
        add_gradient(gradients, meta, input.output_nr, gradient);
    }
    Ok(())
}

fn apply_saved_input_unary(
    node: &SavedInputUnaryNode,
    upstream: &[f32],
    gradients: &mut Gradients,
) -> Result<(), TensorError> {
    let input = &node.input;
    let Some(meta) = &input.autograd else {
        return Ok(());
    };
    debug_assert_eq!(input.elements, upstream.len());
    let mut gradient = try_result_vector(input.elements, input.elements)?;
    (node.vjp)(input, upstream, &mut gradient);
    add_gradient(gradients, meta, input.output_nr, gradient);
    Ok(())
}

fn apply_saved_output_unary(
    node: &SavedOutputUnaryNode,
    upstream: &[f32],
    gradients: &mut Gradients,
) -> Result<(), TensorError> {
    let input = &node.input;
    let Some(meta) = &input.autograd else {
        return Ok(());
    };
    debug_assert_eq!(input.elements, node.output.elements);
    debug_assert_eq!(input.elements, upstream.len());
    let mut gradient = try_result_vector(input.elements, input.elements)?;
    (node.vjp)(&node.output, upstream, &mut gradient);
    add_gradient(gradients, meta, input.output_nr, gradient);
    Ok(())
}

fn apply_relu_vjp(input: &SavedTensor, upstream: &[f32], gradient: &mut Vec<f32>) {
    // Borrow one exact saved range for row-contiguous layouts, including
    // nonzero-offset views, instead of resolving layout and storage per value.
    if let Some(saved_values) = input.contiguous_slice() {
        debug_assert_eq!(saved_values.len(), upstream.len());
        gradient.extend(saved_values.iter().zip(upstream).map(
            |(&saved_value, &upstream_value)| relu_backward_value(saved_value, upstream_value),
        ));
    } else {
        gradient.extend(
            upstream.iter().enumerate().map(|(index, &value)| {
                relu_backward_value(input.value_at_linear_index(index), value)
            }),
        );
    }
}

fn apply_sin_vjp(input: &SavedTensor, upstream: &[f32], gradient: &mut Vec<f32>) {
    gradient.extend(
        upstream
            .iter()
            .enumerate()
            .map(|(index, value)| value * input.value_at_linear_index(index).cos()),
    );
}

fn apply_exp_vjp(output: &SavedTensor, upstream: &[f32], gradient: &mut Vec<f32>) {
    // Borrow one exact saved range for row-contiguous layouts, including
    // nonzero-offset views, instead of resolving layout and storage per value.
    if let Some(saved_values) = output.contiguous_slice() {
        debug_assert_eq!(saved_values.len(), upstream.len());
        gradient.extend(saved_values.iter().zip(upstream).map(
            |(&saved_value, &upstream_value)| exp_backward_value(saved_value, upstream_value),
        ));
    } else {
        gradient.extend(
            upstream.iter().enumerate().map(|(index, &value)| {
                exp_backward_value(output.value_at_linear_index(index), value)
            }),
        );
    }
}

fn apply_tanh_vjp(output: &SavedTensor, upstream: &[f32], gradient: &mut Vec<f32>) {
    // Scalar tanh autograd always saves one contiguous output today. Keep the
    // generic saved-output iteration shape so widening support does not need a
    // different graph representation.
    if let Some(saved_values) = output.contiguous_slice() {
        debug_assert_eq!(saved_values.len(), upstream.len());
        gradient.extend(saved_values.iter().zip(upstream).map(
            |(&saved_value, &upstream_value)| tanh_backward_value(saved_value, upstream_value),
        ));
    } else {
        gradient.extend(upstream.iter().enumerate().map(|(index, &value)| {
            tanh_backward_value(output.value_at_linear_index(index), value)
        }));
    }
}

#[cfg(any(feature = "python-bindings", test))]
fn apply_square_vjp(input: &SavedTensor, upstream: &[f32], gradient: &mut Vec<f32>) {
    // PyTorch's PowBackward0 doubles the saved input before applying the
    // upstream gradient. Preserve that operation order because distributing
    // the gradient over two multiply contributions changes float32 overflow
    // and subnormal rounding.
    if let Some(saved_values) = input.contiguous_slice() {
        debug_assert_eq!(saved_values.len(), upstream.len());
        gradient.extend(saved_values.iter().zip(upstream).map(
            |(&saved_value, &upstream_value)| square_backward_value(saved_value, upstream_value),
        ));
    } else {
        gradient.extend(upstream.iter().enumerate().map(|(index, &value)| {
            square_backward_value(input.value_at_linear_index(index), value)
        }));
    }
}

fn apply_sqrt_vjp(input: &SavedTensor, upstream: &[f32], gradient: &mut Vec<f32>) {
    // Borrow one exact saved range for row-contiguous layouts, including
    // nonzero-offset views, instead of resolving layout and storage per value.
    if let Some(saved_values) = input.contiguous_slice() {
        debug_assert_eq!(saved_values.len(), upstream.len());
        gradient.extend(saved_values.iter().zip(upstream).map(
            |(&saved_value, &upstream_value)| sqrt_backward_value(saved_value, upstream_value),
        ));
    } else {
        gradient.extend(
            upstream.iter().enumerate().map(|(index, &value)| {
                sqrt_backward_value(input.value_at_linear_index(index), value)
            }),
        );
    }
}

struct GradientAccumulator {
    values: Vec<f32>,
    initialized: Vec<bool>,
}

impl GradientAccumulator {
    fn new(elements: usize, preserve_first: bool) -> Result<Self, TensorError> {
        let values = filled_storage(elements, 0.0)?;
        let mut initialized = try_result_vector(elements, elements)?;
        initialized.resize(elements, !preserve_first);
        Ok(Self {
            values,
            initialized,
        })
    }

    fn add(&mut self, index: usize, contribution: f32) {
        if self.initialized[index] {
            self.values[index] += contribution;
        } else {
            self.values[index] = contribution;
            self.initialized[index] = true;
        }
    }
}

fn transform_backward(
    input: &SavedTensor,
    mapping: &TransformMapping,
    upstream: &[f32],
) -> Result<Vec<f32>, TensorError> {
    match mapping {
        TransformMapping::Identity => copied_storage(upstream, input.elements),
        TransformMapping::Index { input_start } => {
            let mut gradient = filled_storage(input.elements, 0.0)?;
            let end = input_start
                .checked_add(upstream.len())
                .ok_or(TensorError::IndexCalculationOverflow)?;
            gradient
                .get_mut(*input_start..end)
                .ok_or(TensorError::IndexCalculationOverflow)?
                .copy_from_slice(upstream);
            Ok(gradient)
        }
        TransformMapping::Permute {
            dimensions,
            output_shape,
        } => {
            if dimensions.as_slice() == [1, 0] && input.shape.len() == 2 {
                let rows = input.shape[0];
                let columns = input.shape[1];
                debug_assert_eq!(output_shape.as_slice(), [columns, rows]);
                debug_assert_eq!(upstream.len(), input.elements);

                let mut gradient = try_result_vector(input.elements, input.elements)?;
                if upstream.is_empty() {
                    return Ok(gradient);
                }
                // Push in input-row order while reading each transposed output column.
                for row in 0..rows {
                    for column in 0..columns {
                        gradient.push(upstream[column * rows + row]);
                    }
                }
                return Ok(gradient);
            }

            let mut gradient = filled_storage(input.elements, 0.0)?;
            if upstream.is_empty() {
                return Ok(gradient);
            }
            let input_strides = contiguous_strides(&input.shape, input.elements)?;
            let mut coordinates = try_result_vector(output_shape.len(), input.elements)?;
            coordinates.resize(output_shape.len(), 0_usize);
            for (output_index, &value) in upstream.iter().enumerate() {
                let mut remaining = output_index;
                for axis in (0..output_shape.len()).rev() {
                    coordinates[axis] = remaining % output_shape[axis];
                    remaining /= output_shape[axis];
                }
                let input_index = dimensions.iter().enumerate().try_fold(
                    0_usize,
                    |input_index, (output_axis, &input_axis)| {
                        let contribution = coordinates[output_axis]
                            .checked_mul(input_strides[input_axis])
                            .ok_or(TensorError::IndexCalculationOverflow)?;
                        input_index
                            .checked_add(contribution)
                            .ok_or(TensorError::IndexCalculationOverflow)
                    },
                )?;
                gradient[input_index] = value;
            }
            Ok(gradient)
        }
    }
}

fn add_gradient(
    gradients: &mut Gradients,
    meta: &Arc<AutogradMeta>,
    output_nr: usize,
    contribution: Vec<f32>,
) {
    gradients
        .entry(gradient_key(meta, output_nr))
        .and_modify(|gradient| {
            debug_assert_eq!(gradient.len(), contribution.len());
            for (value, contribution) in gradient.iter_mut().zip(&contribution) {
                *value += contribution;
            }
        })
        .or_insert(contribution);
}

fn accumulate_leaf_gradient(meta: &AutogradMeta, contribution: Vec<f32>) {
    let AutogradKind::Leaf {
        dtype,
        device,
        grad,
        ..
    } = &meta.kind
    else {
        unreachable!("only leaf nodes are queued for gradient accumulation");
    };
    let mut grad = grad
        .lock()
        .expect("leaf gradient mutex must not be poisoned");
    if let Some(storage) = grad.as_ref() {
        storage.accumulate_shared_gradient(contribution);
    } else {
        *grad = Some(Arc::new(Storage::from_shared_gradient(
            contribution,
            *dtype,
            *device,
        )));
    }
}

fn autograd_id(meta: &Arc<AutogradMeta>) -> usize {
    Arc::as_ptr(meta) as usize
}

fn gradient_key(meta: &Arc<AutogradMeta>, output_nr: usize) -> GradientKey {
    (autograd_id(meta), output_nr)
}

#[derive(Clone, Copy)]
struct BroadcastDimension {
    left_step: usize,
    right_step: usize,
    left_rewind: usize,
    right_rewind: usize,
}

struct BroadcastPlan {
    shape: Vec<usize>,
    strides: Vec<usize>,
    dimensions: Vec<BroadcastDimension>,
    elements: usize,
}

#[inline(never)]
fn apply_binary_operation_scalar(
    operation: &impl Fn(f32, f32) -> f32,
    left: f32,
    right: f32,
) -> f32 {
    operation(left, right)
}

#[inline(never)]
fn materialize_contiguous_trailing_singleton_broadcast(
    left: &Tensor,
    right: &Tensor,
    operation: &impl Fn(f32, f32) -> f32,
) -> Result<Option<(Vec<f32>, BroadcastPlan)>, TensorError> {
    if left.shape.len() != right.shape.len() || left.shape.is_empty() {
        return Ok(None);
    }
    let prefix_end = left.shape.len() - 1;
    if left.shape[..prefix_end] != right.shape[..prefix_end] {
        return Ok(None);
    }

    let (rows, row_scalars, scalars_on_left) =
        if left.shape[prefix_end] != 1 && right.shape[prefix_end] == 1 {
            (left, right, false)
        } else if right.shape[prefix_end] != 1 && left.shape[prefix_end] == 1 {
            (right, left, true)
        } else {
            return Ok(None);
        };

    if rows.elements == 0 {
        return Ok(None);
    }
    let (Some(row_values), Some(scalar_values)) =
        (rows.contiguous_slice(), row_scalars.contiguous_slice())
    else {
        return Ok(None);
    };
    let plan = BroadcastPlan::new(left, right)?;
    if !layout_is_contiguous(&plan.shape, &plan.strides, plan.elements) {
        return Ok(None);
    }

    let row_width = rows.shape[prefix_end];
    debug_assert_ne!(row_width, 0);
    debug_assert_eq!(row_values.len(), plan.elements);
    debug_assert_eq!(row_values.len() / row_width, scalar_values.len());

    let mut data = try_result_vector(plan.elements, plan.elements)?;
    if scalars_on_left {
        for (&scalar, row) in scalar_values.iter().zip(row_values.chunks_exact(row_width)) {
            data.extend(row.iter().copied().map(|value| operation(scalar, value)));
        }
    } else {
        for (row, &scalar) in row_values.chunks_exact(row_width).zip(scalar_values) {
            data.extend(row.iter().copied().map(|value| operation(value, scalar)));
        }
    }

    // Keep paired-NaN payload selection identical to the scalar odometer
    // fallback without inhibiting vectorization for ordinary rows.
    if scalar_values.iter().any(|value| value.is_nan()) {
        for ((output_row, row), &scalar) in data
            .chunks_exact_mut(row_width)
            .zip(row_values.chunks_exact(row_width))
            .zip(scalar_values)
        {
            if scalar.is_nan() {
                for (output, &value) in output_row.iter_mut().zip(row) {
                    if value.is_nan() {
                        let (left, right) = if scalars_on_left {
                            (scalar, value)
                        } else {
                            (value, scalar)
                        };
                        *output = apply_binary_operation_scalar(operation, left, right);
                    }
                }
            }
        }
    }
    debug_assert_eq!(data.len(), plan.elements);
    Ok(Some((data, plan)))
}

#[inline(never)]
fn materialize_contiguous_trailing_broadcast(
    left: &Tensor,
    right: &Tensor,
    operation: &impl Fn(f32, f32) -> f32,
) -> Result<Option<(Vec<f32>, BroadcastPlan)>, TensorError> {
    let (broad, vector, vector_on_left) = if left.shape.len() > 1
        && right.shape.len() == 1
        && left.shape.last() == right.shape.first()
    {
        (left, right, false)
    } else if right.shape.len() > 1
        && left.shape.len() == 1
        && right.shape.last() == left.shape.first()
    {
        (right, left, true)
    } else if left.shape.len() == right.shape.len() && !left.shape.is_empty() {
        let prefix_end = left.shape.len() - 1;
        let trailing_singleton = (left.shape[prefix_end] != 1 && right.shape[prefix_end] == 1)
            || (right.shape[prefix_end] != 1 && left.shape[prefix_end] == 1);
        if trailing_singleton && left.shape[..prefix_end] == right.shape[..prefix_end] {
            return materialize_contiguous_trailing_singleton_broadcast(left, right, operation);
        }
        return Ok(None);
    } else {
        return Ok(None);
    };

    if broad.elements == 0 {
        return Ok(None);
    }
    let (Some(broad_values), Some(vector_values)) =
        (broad.contiguous_slice(), vector.contiguous_slice())
    else {
        return Ok(None);
    };
    let plan = BroadcastPlan::new(left, right)?;
    if !layout_is_contiguous(&plan.shape, &plan.strides, plan.elements) {
        return Ok(None);
    }
    debug_assert_eq!(broad_values.len(), plan.elements);
    debug_assert!(!vector_values.is_empty());

    let mut data = try_result_vector(plan.elements, plan.elements)?;
    if vector_on_left {
        for broad_slice in broad_values.chunks_exact(vector_values.len()) {
            data.extend(
                vector_values
                    .iter()
                    .copied()
                    .zip(broad_slice.iter().copied())
                    .map(|(vector, broad)| operation(vector, broad)),
            );
        }
    } else {
        for broad_slice in broad_values.chunks_exact(vector_values.len()) {
            data.extend(
                broad_slice
                    .iter()
                    .copied()
                    .zip(vector_values.iter().copied())
                    .map(|(broad, vector)| operation(broad, vector)),
            );
        }
    }

    // Vectorized arithmetic can select a different NaN payload when both
    // operands are NaNs. Recompute only those rare lanes through a scalar call
    // so the slice path remains bitwise identical to the odometer fallback.
    if vector_values.iter().any(|value| value.is_nan()) {
        for (output_slice, broad_slice) in data
            .chunks_exact_mut(vector_values.len())
            .zip(broad_values.chunks_exact(vector_values.len()))
        {
            for ((output, &broad), &vector) in
                output_slice.iter_mut().zip(broad_slice).zip(vector_values)
            {
                if broad.is_nan() && vector.is_nan() {
                    let (left, right) = if vector_on_left {
                        (vector, broad)
                    } else {
                        (broad, vector)
                    };
                    *output = apply_binary_operation_scalar(operation, left, right);
                }
            }
        }
    }
    debug_assert_eq!(data.len(), plan.elements);
    Ok(Some((data, plan)))
}

#[inline]
fn materialize_broadcast(
    plan: &BroadcastPlan,
    mut left_offset: usize,
    mut right_offset: usize,
    left_value: impl Fn(usize) -> f32,
    right_value: impl Fn(usize) -> f32,
    operation: impl Fn(f32, f32) -> f32,
) -> Result<Vec<f32>, TensorError> {
    let mut data = try_result_vector(plan.elements, plan.elements)?;
    let mut coordinates = try_result_vector(plan.shape.len(), plan.elements)?;
    coordinates.resize(plan.shape.len(), 0_usize);
    let contiguous_output = layout_is_contiguous(&plan.shape, &plan.strides, plan.elements);

    if !contiguous_output {
        data.resize(plan.elements, 0.0);
    }
    for output_index in 0..plan.elements {
        let value = operation(left_value(left_offset), right_value(right_offset));
        if contiguous_output {
            data.push(value);
        } else {
            let output_offset =
                logical_offset_for_linear_index(&plan.shape, &plan.strides, 0, output_index)?;
            data[output_offset] = value;
        }
        if output_index + 1 == plan.elements {
            break;
        }

        for axis in (0..plan.shape.len()).rev() {
            coordinates[axis] = coordinates[axis]
                .checked_add(1)
                .ok_or(TensorError::StrideCalculationOverflow)?;
            if coordinates[axis] < plan.shape[axis] {
                left_offset = left_offset
                    .checked_add(plan.dimensions[axis].left_step)
                    .ok_or(TensorError::StrideCalculationOverflow)?;
                right_offset = right_offset
                    .checked_add(plan.dimensions[axis].right_step)
                    .ok_or(TensorError::StrideCalculationOverflow)?;
                break;
            }

            coordinates[axis] = 0;
            left_offset = left_offset
                .checked_sub(plan.dimensions[axis].left_rewind)
                .ok_or(TensorError::StrideCalculationOverflow)?;
            right_offset = right_offset
                .checked_sub(plan.dimensions[axis].right_rewind)
                .ok_or(TensorError::StrideCalculationOverflow)?;
        }
    }
    Ok(data)
}

impl BroadcastPlan {
    fn new(left: &Tensor, right: &Tensor) -> Result<Self, TensorError> {
        let rank = left.shape.len().max(right.shape.len());
        for axis in 0..rank {
            let left_dimension = aligned_dimension(&left.shape, rank, axis);
            let right_dimension = aligned_dimension(&right.shape, rank, axis);
            if broadcast_dimension(left_dimension, right_dimension).is_none() {
                return Err(TensorError::ShapeMismatch {
                    left: try_clone_result_shape(&left.shape, left.elements)?,
                    right: try_clone_result_shape(&right.shape, right.elements)?,
                });
            }
        }

        let mut elements = 1_usize;
        for axis in 0..rank {
            let dimension = broadcast_dimension(
                aligned_dimension(&left.shape, rank, axis),
                aligned_dimension(&right.shape, rank, axis),
            )
            .expect("broadcast compatibility was checked above");
            elements = elements
                .checked_mul(dimension)
                .ok_or(TensorError::ElementCountOverflow)?;
        }
        validate_storage_capacity(elements)?;

        let mut shape = try_result_vector(rank, elements)?;
        for axis in 0..rank {
            shape.push(
                broadcast_dimension(
                    aligned_dimension(&left.shape, rank, axis),
                    aligned_dimension(&right.shape, rank, axis),
                )
                .expect("broadcast compatibility was checked above"),
            );
        }
        let strides = elementwise_output_strides(
            &shape,
            &[
                ElementwiseLayout::from_tensor(left),
                ElementwiseLayout::from_tensor(right),
            ],
            elements,
        )?;

        let mut dimensions = try_result_vector(rank, elements)?;
        if elements == 0 {
            dimensions.resize(
                rank,
                BroadcastDimension {
                    left_step: 0,
                    right_step: 0,
                    left_rewind: 0,
                    right_rewind: 0,
                },
            );
            return Ok(Self {
                shape,
                strides,
                dimensions,
                elements,
            });
        }

        for axis in (0..rank).rev() {
            let output_dimension = shape[axis];
            let left_step = aligned_broadcast_stride(left, rank, axis, output_dimension);
            let right_step = aligned_broadcast_stride(right, rank, axis, output_dimension);
            let repeats = output_dimension.saturating_sub(1);
            let left_rewind = left_step
                .checked_mul(repeats)
                .ok_or(TensorError::StrideCalculationOverflow)?;
            let right_rewind = right_step
                .checked_mul(repeats)
                .ok_or(TensorError::StrideCalculationOverflow)?;
            dimensions.push(BroadcastDimension {
                left_step,
                right_step,
                left_rewind,
                right_rewind,
            });
        }
        dimensions.reverse();

        Ok(Self {
            shape,
            strides,
            dimensions,
            elements,
        })
    }
}

fn layout_is_contiguous(shape: &[usize], strides: &[usize], elements: usize) -> bool {
    if elements == 0 {
        return true;
    }

    let mut expected_stride = 1_usize;
    for axis in (0..shape.len()).rev() {
        let dimension = shape[axis];
        if dimension == 1 {
            continue;
        }
        if strides[axis] != expected_stride {
            return false;
        }
        let Some(next_stride) = expected_stride.checked_mul(dimension) else {
            return false;
        };
        expected_stride = next_stride;
    }
    true
}

fn layout_is_channels_last_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    layout_is_contiguous_in_order(shape, strides, &[1, 3, 2, 0])
}

fn layout_is_channels_last_3d_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    layout_is_contiguous_in_order(shape, strides, &[1, 4, 3, 2, 0])
}

#[cfg(feature = "python-bindings")]
fn layout_is_strides_like_channels_last(shape: &[usize], strides: &[usize]) -> bool {
    layout_is_strides_like_channels_order(shape, strides, &[1, 3, 2, 0])
}

#[cfg(feature = "python-bindings")]
fn layout_is_strides_like_channels_last_3d(shape: &[usize], strides: &[usize]) -> bool {
    layout_is_strides_like_channels_order(shape, strides, &[1, 4, 3, 2, 0])
}

#[cfg(feature = "python-bindings")]
fn layout_is_strides_like_channels_order(
    shape: &[usize],
    strides: &[usize],
    order: &[usize],
) -> bool {
    if shape.len() != order.len() || strides.len() != order.len() || strides[1] == 0 {
        return false;
    }

    let mut minimum_stride = 0_usize;
    for &axis in order {
        if shape[axis] == 0 || strides[axis] < minimum_stride {
            return false;
        }
        // PyTorch defaults fully ambiguous N111 layouts to row-major.
        if axis == 0 && minimum_stride == strides[1] {
            return false;
        }
        minimum_stride = strides[axis];
        if shape[axis] > 1 {
            let Some(next_minimum_stride) = minimum_stride.checked_mul(shape[axis]) else {
                return false;
            };
            minimum_stride = next_minimum_stride;
        }
    }
    true
}

fn layout_is_contiguous_in_order(shape: &[usize], strides: &[usize], order: &[usize]) -> bool {
    if shape.len() != order.len() || strides.len() != order.len() {
        return false;
    }

    let is_empty = shape.contains(&0);
    let mut expected_stride = 1_usize;
    for &axis in order {
        let dimension = shape[axis];
        if dimension == 1 {
            continue;
        }
        if strides[axis] != expected_stride {
            return false;
        }
        expected_stride = match expected_stride.checked_mul(dimension) {
            Some(next_stride) => next_stride,
            // PyTorch treats strides on empty tensors as arbitrary signed-64
            // metadata. Its channel-order contiguity check lets an overflowing
            // product wrap, allowing a later zero-sized axis to match zero.
            // Materialization remains separately checked before allocating.
            None if is_empty => expected_stride.wrapping_mul(dimension),
            None => return false,
        };
    }
    true
}

fn normalize_transpose_dimension(dimension: i64, rank: usize) -> Result<usize, TensorError> {
    let effective_rank = rank.max(1);
    let signed_rank = i64::try_from(effective_rank)
        .map_err(|_| TensorError::DimensionOutOfRange { dimension, rank })?;
    if dimension < -signed_rank || dimension >= signed_rank {
        return Err(TensorError::DimensionOutOfRange { dimension, rank });
    }
    if rank == 0 {
        return Ok(0);
    }
    usize::try_from(if dimension < 0 {
        dimension + signed_rank
    } else {
        dimension
    })
    .map_err(|_| TensorError::DimensionOutOfRange { dimension, rank })
}

fn dimension_for_error(dimension: usize) -> i64 {
    i64::try_from(dimension).unwrap_or(i64::MAX)
}

fn logical_offset_for_linear_index(
    shape: &[usize],
    strides: &[usize],
    base_offset: usize,
    linear_index: usize,
) -> Result<usize, TensorError> {
    let mut remaining = linear_index;
    let mut offset = base_offset;
    for axis in (0..shape.len()).rev() {
        let dimension = shape[axis];
        if dimension == 0 {
            return Err(TensorError::IndexCalculationOverflow);
        }
        let coordinate = remaining % dimension;
        remaining /= dimension;
        let contribution = coordinate
            .checked_mul(strides[axis])
            .ok_or(TensorError::IndexCalculationOverflow)?;
        offset = offset
            .checked_add(contribution)
            .ok_or(TensorError::IndexCalculationOverflow)?;
    }
    if remaining != 0 {
        return Err(TensorError::IndexCalculationOverflow);
    }
    Ok(offset)
}

fn validate_view_bounds(
    shape: &[usize],
    strides: &[usize],
    offset: usize,
    elements: usize,
    storage_elements: usize,
) -> Result<(), TensorError> {
    if elements == 0 {
        return Ok(());
    }
    let maximum_offset =
        shape
            .iter()
            .zip(strides)
            .try_fold(offset, |maximum_offset, (&dimension, &stride)| {
                let contribution = dimension
                    .saturating_sub(1)
                    .checked_mul(stride)
                    .ok_or(TensorError::IndexCalculationOverflow)?;
                maximum_offset
                    .checked_add(contribution)
                    .ok_or(TensorError::IndexCalculationOverflow)
            })?;
    if maximum_offset >= storage_elements {
        return Err(TensorError::IndexCalculationOverflow);
    }
    Ok(())
}

fn checked_matrix_offset(tensor: &Tensor, row: usize, column: usize) -> Result<usize, TensorError> {
    let row_offset = row
        .checked_mul(tensor.strides[0])
        .ok_or(TensorError::IndexCalculationOverflow)?;
    let column_offset = column
        .checked_mul(tensor.strides[1])
        .ok_or(TensorError::IndexCalculationOverflow)?;
    tensor
        .offset
        .checked_add(row_offset)
        .and_then(|offset| offset.checked_add(column_offset))
        .filter(|offset| *offset < tensor.storage.len())
        .ok_or(TensorError::IndexCalculationOverflow)
}

// Isolate contiguous code generation from the unchanged strided dispatch.
#[inline(never)]
fn accumulate_contiguous_matmul(
    left: &[f32],
    right: &[f32],
    output: &mut [f32],
    rows: usize,
    inner: usize,
    columns: usize,
) {
    if output.is_empty() || inner == 0 {
        return;
    }
    if rows >= CONTIGUOUS_MATMUL_ROW_BLOCK && right.len() >= CONTIGUOUS_MATMUL_MIN_RHS_ELEMENTS {
        contiguous_matmul_row_blocked(left, right, output, rows, inner, columns);
        return;
    }

    for (left_row, output_row) in left
        .chunks_exact(inner)
        .zip(output.chunks_exact_mut(columns))
    {
        for (&left, right_row) in left_row.iter().zip(right.chunks_exact(columns)) {
            for (output_value, &right) in output_row.iter_mut().zip(right_row) {
                *output_value += left * right;
            }
        }
    }
}

// Keep the unrolled kernel separate from the latency-sized contiguous loop.
#[inline(never)]
fn contiguous_matmul_row_blocked(
    left: &[f32],
    right: &[f32],
    output: &mut [f32],
    rows: usize,
    inner: usize,
    columns: usize,
) {
    let blocked_rows = rows / CONTIGUOUS_MATMUL_ROW_BLOCK * CONTIGUOUS_MATMUL_ROW_BLOCK;
    let blocked_left_elements = blocked_rows * inner;
    let blocked_output_elements = blocked_rows * columns;
    let (blocked_left, tail_left) = left.split_at(blocked_left_elements);
    let (blocked_output, tail_output) = output.split_at_mut(blocked_output_elements);

    for (left_tile, output_tile) in blocked_left
        .chunks_exact(inner * CONTIGUOUS_MATMUL_ROW_BLOCK)
        .zip(blocked_output.chunks_exact_mut(columns * CONTIGUOUS_MATMUL_ROW_BLOCK))
    {
        let (output_row_0, remaining) = output_tile.split_at_mut(columns);
        let (output_row_1, remaining) = remaining.split_at_mut(columns);
        let (output_row_2, output_row_3) = remaining.split_at_mut(columns);
        for (depth, right_row) in right.chunks_exact(columns).enumerate() {
            let left_0 = left_tile[depth];
            let left_1 = left_tile[inner + depth];
            let left_2 = left_tile[2 * inner + depth];
            let left_3 = left_tile[3 * inner + depth];
            // Depth stays outermost, preserving every result's addition order.
            for column in 0..columns {
                let right = right_row[column];
                output_row_0[column] += left_0 * right;
                output_row_1[column] += left_1 * right;
                output_row_2[column] += left_2 * right;
                output_row_3[column] += left_3 * right;
            }
        }
    }

    for (left_row, output_row) in tail_left
        .chunks_exact(inner)
        .zip(tail_output.chunks_exact_mut(columns))
    {
        for (&left, right_row) in left_row.iter().zip(right.chunks_exact(columns)) {
            for (output_value, &right) in output_row.iter_mut().zip(right_row) {
                *output_value += left * right;
            }
        }
    }
}

fn checked_matrix_row_base(
    tensor: &Tensor,
    row: usize,
    columns: usize,
    storage_elements: usize,
) -> Result<usize, TensorError> {
    debug_assert_ne!(columns, 0);
    let row_offset = row
        .checked_mul(tensor.strides[0])
        .ok_or(TensorError::IndexCalculationOverflow)?;
    let base = tensor
        .offset
        .checked_add(row_offset)
        .ok_or(TensorError::IndexCalculationOverflow)?;
    let final_column = columns
        .checked_sub(1)
        .ok_or(TensorError::IndexCalculationOverflow)?;
    let final_column_offset = final_column
        .checked_mul(tensor.strides[1])
        .ok_or(TensorError::IndexCalculationOverflow)?;
    // Strides are non-negative, so a valid final address covers every
    // increment from the base as well.
    base.checked_add(final_column_offset)
        .filter(|offset| *offset < storage_elements)
        .map(|_| base)
        .ok_or(TensorError::IndexCalculationOverflow)
}

fn compute_reshape_view_strides(
    old_shape: &[usize],
    old_strides: &[usize],
    new_shape: &[usize],
    elements: usize,
) -> Result<Option<Vec<usize>>, TensorError> {
    if old_shape.is_empty() {
        return contiguous_strides(new_shape, elements).map(Some);
    }

    let mut new_strides = try_result_vector(new_shape.len(), elements)?;
    new_strides.resize(new_shape.len(), 0);
    let mut view_dimension = new_shape.len();
    let mut chunk_base_stride = *old_strides
        .last()
        .ok_or(TensorError::StrideCalculationOverflow)?;
    let mut tensor_elements = 1_usize;
    let mut view_elements = 1_usize;

    for tensor_dimension in (0..old_shape.len()).rev() {
        tensor_elements = tensor_elements
            .checked_mul(old_shape[tensor_dimension])
            .ok_or(TensorError::ElementCountOverflow)?;
        let chunk_ends = tensor_dimension == 0
            || (old_shape[tensor_dimension - 1] != 1
                && old_strides[tensor_dimension - 1]
                    != tensor_elements
                        .checked_mul(chunk_base_stride)
                        .ok_or(TensorError::StrideCalculationOverflow)?);
        if !chunk_ends {
            continue;
        }

        while view_dimension > 0
            && (view_elements < tensor_elements || new_shape[view_dimension - 1] == 1)
        {
            view_dimension -= 1;
            new_strides[view_dimension] = view_elements
                .checked_mul(chunk_base_stride)
                .ok_or(TensorError::StrideCalculationOverflow)?;
            view_elements = view_elements
                .checked_mul(new_shape[view_dimension])
                .ok_or(TensorError::ElementCountOverflow)?;
        }
        if view_elements != tensor_elements {
            return Ok(None);
        }
        if tensor_dimension > 0 {
            chunk_base_stride = old_strides[tensor_dimension - 1];
            tensor_elements = 1;
            view_elements = 1;
        }
    }

    if view_dimension == 0 {
        Ok(Some(new_strides))
    } else {
        Ok(None)
    }
}

fn aligned_dimension(shape: &[usize], output_rank: usize, output_axis: usize) -> usize {
    let leading_dimensions = output_rank - shape.len();
    if output_axis < leading_dimensions {
        1
    } else {
        shape[output_axis - leading_dimensions]
    }
}

fn broadcast_dimension(left: usize, right: usize) -> Option<usize> {
    if left == right {
        Some(left)
    } else if left == 1 {
        Some(right)
    } else if right == 1 {
        Some(left)
    } else {
        None
    }
}

fn aligned_broadcast_stride(
    tensor: &Tensor,
    output_rank: usize,
    output_axis: usize,
    output_dimension: usize,
) -> usize {
    let leading_dimensions = output_rank - tensor.shape.len();
    if output_axis < leading_dimensions {
        return 0;
    }

    let input_axis = output_axis - leading_dimensions;
    let input_dimension = tensor.shape[input_axis];
    if input_dimension == 1 && output_dimension != 1 {
        0
    } else {
        tensor.strides[input_axis]
    }
}

#[derive(Clone, Copy)]
struct ElementwiseLayout<'a> {
    shape: &'a [usize],
    strides: &'a [usize],
}

impl<'a> ElementwiseLayout<'a> {
    fn from_tensor(tensor: &'a Tensor) -> Self {
        Self {
            shape: &tensor.shape,
            strides: &tensor.strides,
        }
    }

    fn aligned_broadcast_stride(
        self,
        output_rank: usize,
        output_axis: usize,
        output_dimension: usize,
    ) -> usize {
        let leading_dimensions = output_rank - self.shape.len();
        if output_axis < leading_dimensions {
            return 0;
        }

        let input_axis = output_axis - leading_dimensions;
        let input_dimension = self.shape[input_axis];
        if input_dimension == 1 && output_dimension != 1 {
            0
        } else {
            self.strides[input_axis]
        }
    }
}

fn aligned_broadcast_stride_bytes(
    layout: ElementwiseLayout<'_>,
    output_rank: usize,
    output_axis: usize,
    output_dimension: usize,
) -> i64 {
    // TensorIterator compares byte strides stored in signed 64-bit integers.
    // Preserve its wrapping conversion at this boundary: an extreme but valid
    // empty view can therefore change the recovered output permutation without
    // accessing any storage.
    let stride = layout
        .aligned_broadcast_stride(output_rank, output_axis, output_dimension)
        .cast_signed();
    let element_size =
        i64::try_from(DType::Float32.element_size()).expect("an f32 element size must fit in i64");
    i64::try_from(stride)
        .expect("an isize stride must fit in a signed 64-bit TensorIterator stride")
        .wrapping_mul(element_size)
}

fn try_result_vector<T>(capacity: usize, elements: usize) -> Result<Vec<T>, TensorError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| TensorError::AllocationFailed { elements })?;
    Ok(values)
}

fn try_clone_result_shape(shape: &[usize], elements: usize) -> Result<Vec<usize>, TensorError> {
    let mut cloned = try_result_vector(shape.len(), elements)?;
    cloned.extend_from_slice(shape);
    Ok(cloned)
}

fn try_clone_reshape_shape(shape: &[i64], elements: usize) -> Result<Vec<i64>, TensorError> {
    let mut cloned = try_result_vector(shape.len(), elements)?;
    cloned.extend_from_slice(shape);
    Ok(cloned)
}

fn element_count(shape: &[usize]) -> Result<usize, TensorError> {
    shape.iter().try_fold(1_usize, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or(TensorError::ElementCountOverflow)
    })
}

fn element_count_in_axis_order(
    shape: &[usize],
    dimensions: &[usize],
) -> Result<usize, TensorError> {
    dimensions.iter().try_fold(1_usize, |count, &dimension| {
        count
            .checked_mul(shape[dimension])
            .ok_or(TensorError::ElementCountOverflow)
    })
}

fn validated_layout(shape: &[usize]) -> Result<(usize, Vec<usize>), TensorError> {
    let elements = element_count(shape)?;
    let strides = contiguous_strides(shape, elements)?;
    Ok((elements, strides))
}

fn contiguous_strides(shape: &[usize], elements: usize) -> Result<Vec<usize>, TensorError> {
    let mut strides = try_result_vector(shape.len(), elements)?;
    strides.resize(shape.len(), 0);
    let mut stride = 1_usize;
    for axis in (0..shape.len()).rev() {
        strides[axis] = stride;
        if axis > 0 {
            stride = checked_stride_product(stride, shape[axis])?;
        }
    }
    Ok(strides)
}

fn channels_last_strides(shape: &[usize], elements: usize) -> Result<Vec<usize>, TensorError> {
    if shape.len() != 4 {
        return Err(TensorError::StrideCalculationOverflow);
    }
    strides_in_physical_order(shape, elements, &[1, 3, 2, 0])
}

fn channels_last_3d_strides(shape: &[usize], elements: usize) -> Result<Vec<usize>, TensorError> {
    if shape.len() != 5 {
        return Err(TensorError::StrideCalculationOverflow);
    }
    strides_in_physical_order(shape, elements, &[1, 4, 3, 2, 0])
}

fn strides_in_physical_order(
    shape: &[usize],
    elements: usize,
    order: &[usize],
) -> Result<Vec<usize>, TensorError> {
    // Tensor allocation validates the shape's canonical right-to-left stride
    // products even when a zero dimension makes the allocation empty. Keep
    // that checked boundary before calculating the requested physical order.
    let _ = contiguous_strides(shape, elements)?;
    let mut strides = try_result_vector(shape.len(), elements)?;
    strides.resize(shape.len(), 0);
    let mut stride = 1_usize;
    for (position, &axis) in order.iter().enumerate() {
        strides[axis] = stride;
        if position + 1 < order.len() {
            // Unlike canonical row-major strides, PyTorch's channel-last
            // restriding lets a zero-sized physical dimension zero every
            // subsequent stride.
            stride = checked_physical_stride_product(stride, shape[axis])?;
        }
    }
    Ok(strides)
}

fn elementwise_output_strides(
    shape: &[usize],
    operands: &[ElementwiseLayout<'_>],
    elements: usize,
) -> Result<Vec<usize>, TensorError> {
    let rank = shape.len();
    let mut permutation = try_result_vector(rank, elements)?;
    permutation.extend((0..rank).rev());

    for index in 1..rank {
        let mut dimension_1 = index;
        for dimension_0 in (0..index).rev() {
            let comparison = compare_elementwise_dimensions(
                shape,
                operands,
                permutation[dimension_0],
                permutation[dimension_1],
            );
            if comparison > 0 {
                permutation.swap(dimension_0, dimension_1);
                dimension_1 = dimension_0;
            } else if comparison < 0 {
                break;
            }
        }
    }

    if permutation
        .iter()
        .enumerate()
        .all(|(index, axis)| *axis == rank - index - 1)
    {
        return contiguous_strides(shape, elements);
    }

    let element_size =
        i64::try_from(DType::Float32.element_size()).expect("an f32 element size must fit in i64");
    let mut byte_strides = try_result_vector(rank, elements)?;
    byte_strides.resize(rank, 0_i64);
    let mut next_byte_stride = element_size;
    for (position, axis) in permutation.into_iter().enumerate() {
        byte_strides[axis] = next_byte_stride;
        if position + 1 < rank {
            let dimension =
                i64::try_from(shape[axis]).map_err(|_| TensorError::StrideCalculationOverflow)?;
            next_byte_stride = next_byte_stride.wrapping_mul(dimension);
        }
    }

    let mut strides = try_result_vector(rank, elements)?;
    strides.resize(rank, 0);
    for axis in (0..rank).rev() {
        let stride = byte_strides[axis] / element_size;
        if stride >= 0 {
            strides[axis] =
                usize::try_from(stride).map_err(|_| TensorError::StrideCalculationOverflow)?;
        } else if axis + 1 == rank {
            strides[axis] = 1;
        } else {
            strides[axis] = checked_stride_product(strides[axis + 1], shape[axis + 1])?;
        }
    }
    Ok(strides)
}

fn compare_elementwise_dimensions(
    shape: &[usize],
    operands: &[ElementwiseLayout<'_>],
    dimension_0: usize,
    dimension_1: usize,
) -> i8 {
    for layout in operands {
        let stride_0 =
            aligned_broadcast_stride_bytes(*layout, shape.len(), dimension_0, shape[dimension_0]);
        let stride_1 =
            aligned_broadcast_stride_bytes(*layout, shape.len(), dimension_1, shape[dimension_1]);
        if stride_0 == 0 || stride_1 == 0 {
            continue;
        }
        if stride_0 < stride_1 {
            return -1;
        }
        if stride_0 > stride_1 || shape[dimension_0] > shape[dimension_1] {
            return 1;
        }
    }
    0
}

fn reshape_strides(shape: &[usize], elements: usize) -> Result<Vec<usize>, TensorError> {
    if elements != 0 {
        return contiguous_strides(shape, elements);
    }

    let mut strides = try_result_vector(shape.len(), elements)?;
    strides.resize(shape.len(), 0);
    let mut stride = 1_usize;
    for axis in (0..shape.len()).rev() {
        strides[axis] = stride;
        if axis > 0 {
            // PyTorch treats empty-view strides as arbitrary metadata and its
            // resize-style calculation permits non-negative signed wrapping.
            stride = signed_wrapping_stride_product(stride, shape[axis].max(1))?;
        }
    }
    Ok(strides)
}

fn signed_wrapping_stride_product(stride: usize, dimension: usize) -> Result<usize, TensorError> {
    let product = signed_wrapping_stride_product_value(stride, dimension)?;
    usize::try_from(product).map_err(|_| TensorError::StrideCalculationOverflow)
}

fn signed_wrapping_stride_product_value(
    stride: usize,
    dimension: usize,
) -> Result<i64, TensorError> {
    if stride == 0 || dimension == 0 {
        return Ok(0);
    }
    let stride = i64::try_from(stride).map_err(|_| TensorError::StrideCalculationOverflow)?;
    let dimension = i64::try_from(dimension).map_err(|_| TensorError::StrideCalculationOverflow)?;
    Ok(stride.wrapping_mul(dimension))
}

fn checked_stride_product(stride: usize, dimension: usize) -> Result<usize, TensorError> {
    stride
        .checked_mul(dimension.max(1))
        .filter(|product| *product <= isize::MAX.unsigned_abs())
        .ok_or(TensorError::StrideCalculationOverflow)
}

fn checked_physical_stride_product(stride: usize, dimension: usize) -> Result<usize, TensorError> {
    stride
        .checked_mul(dimension)
        .filter(|product| *product <= isize::MAX.unsigned_abs())
        .ok_or(TensorError::StrideCalculationOverflow)
}

fn negate_value(value: f32) -> f32 {
    f32::from_bits(value.to_bits() ^ F32_SIGN_MASK)
}

fn relu_value(value: f32) -> f32 {
    // Only exact zeros bypass the established max path, so FTZ/DAZ cannot
    // classify a subnormal as zero and NaN behavior remains unchanged.
    if (value.to_bits() & !F32_SIGN_MASK) == 0 {
        value
    } else {
        value.max(0.0)
    }
}

fn floor_value(value: f32) -> f32 {
    const QUIET_NAN_MASK: u32 = 0x0040_0000;

    let bits = value.to_bits();
    if bits & !F32_SIGN_MASK > f32::INFINITY.to_bits() {
        // PyTorch quiets signaling NaNs while retaining their sign and payload.
        f32::from_bits(bits | QUIET_NAN_MASK)
    } else {
        value.floor()
    }
}

fn sqrt_value(value: f32) -> f32 {
    // PyTorch canonicalizes domain errors to a positive quiet NaN while
    // preserving signed zero and the payload and sign of NaN inputs.
    let bits = value.to_bits();
    let magnitude = bits & !F32_SIGN_MASK;
    let is_nan = magnitude > f32::INFINITY.to_bits();
    let is_negative_nonzero = bits & F32_SIGN_MASK != 0 && magnitude != 0;
    if is_negative_nonzero && !is_nan {
        f32::NAN
    } else {
        value.sqrt()
    }
}

fn tanh_value(value: f32) -> f32 {
    const QUIET_NAN_MASK: u32 = 0x0040_0000;
    const SATURATION_MAGNITUDE_BITS: u32 = 0x4110_2c67;

    let bits = value.to_bits();
    let magnitude = bits & !F32_SIGN_MASK;
    if magnitude <= f32::MIN_POSITIVE.to_bits() {
        // PyTorch preserves signed zero and every subnormal input exactly.
        value
    } else if magnitude > f32::INFINITY.to_bits() {
        // Quiet signaling NaNs while retaining their sign and payload.
        f32::from_bits(bits | QUIET_NAN_MASK)
    } else if magnitude >= SATURATION_MAGNITUDE_BITS {
        // PyTorch's float32 CPU kernel rounds every finite value from this
        // boundary outward to exactly +/-1.0.
        f32::from_bits((bits & F32_SIGN_MASK) | 1.0_f32.to_bits())
    } else {
        value.tanh()
    }
}

#[inline]
fn sqrt_backward_value(input: f32, upstream: f32) -> f32 {
    upstream / (2.0 * sqrt_value(input))
}

#[inline]
fn exp_backward_value(output: f32, upstream: f32) -> f32 {
    upstream * output
}

#[inline]
fn tanh_backward_value(output: f32, upstream: f32) -> f32 {
    upstream * (-output).mul_add(output, 1.0)
}

#[inline]
#[cfg(any(feature = "python-bindings", test))]
fn square_backward_value(input: f32, upstream: f32) -> f32 {
    (2.0 * input) * upstream
}

#[inline]
fn relu_backward_value(input: f32, upstream: f32) -> f32 {
    let bits = input.to_bits();
    let magnitude = bits & !F32_SIGN_MASK;
    let is_nan = magnitude > f32::INFINITY.to_bits();
    let is_positive_nonzero = bits & F32_SIGN_MASK == 0 && magnitude != 0;
    if is_nan || is_positive_nonzero {
        upstream
    } else {
        0.0
    }
}

fn filled_storage(elements: usize, fill_value: f32) -> Result<Vec<f32>, TensorError> {
    validate_storage_capacity(elements)?;

    let mut data = Vec::new();
    data.try_reserve_exact(elements)
        .map_err(|_| TensorError::AllocationFailed { elements })?;
    data.resize(elements, fill_value);
    Ok(data)
}

fn copied_storage(values: &[f32], elements: usize) -> Result<Vec<f32>, TensorError> {
    validate_storage_capacity(elements)?;

    let mut data = try_result_vector(elements, elements)?;
    data.extend_from_slice(values);
    Ok(data)
}

fn validate_storage_capacity(elements: usize) -> Result<(), TensorError> {
    let maximum_elements = isize::MAX.unsigned_abs() / DType::Float32.element_size();
    if elements > maximum_elements {
        return Err(TensorError::StorageCapacityOverflow { elements });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::sync::Arc;

    use crate::storage::Storage;

    use super::{
        CONTIGUOUS_MATMUL_MIN_RHS_ELEMENTS, CONTIGUOUS_MATMUL_ROW_BLOCK, DType, Device,
        F32_SIGN_MASK, SavedTensor, Tensor, TensorError, materialize_contiguous_trailing_broadcast,
        sqrt_value, try_result_vector,
    };

    fn shared_gradient_copy(tensor: &Tensor) -> Tensor {
        Tensor {
            storage: Arc::new(Storage::from_shared_gradient(
                tensor.storage.owned_values().unwrap().to_vec(),
                tensor.dtype(),
                tensor.device(),
            )),
            shape: tensor.shape.clone(),
            strides: tensor.strides.clone(),
            offset: tensor.offset,
            elements: tensor.elements,
            output_nr: 0,
            view_requires_grad: false,
            autograd: None,
        }
    }

    #[test]
    fn square_root_matches_pytorch_float32_edge_bits() {
        let inputs = [
            0x0000_0000,
            0x8000_0000,
            0x0000_0001,
            0x8000_0001,
            0x0080_0000,
            0x8080_0000,
            0x3f80_0000,
            0x4000_0000,
            0x4080_0000,
            0x7f7f_ffff,
            0xff7f_ffff,
            0x7f80_0000,
            0xff80_0000,
            0x7f81_2345,
            0xff81_2345,
            0x7fc1_2345,
            0xffc5_4321,
        ];
        let expected = [
            0x0000_0000,
            0x8000_0000,
            0x1a35_04f3,
            0x7fc0_0000,
            0x2000_0000,
            0x7fc0_0000,
            0x3f80_0000,
            0x3fb5_04f3,
            0x4000_0000,
            0x5f7f_ffff,
            0x7fc0_0000,
            0x7f80_0000,
            0x7fc0_0000,
            0x7fc1_2345,
            0xffc1_2345,
            0x7fc1_2345,
            0xffc5_4321,
        ];

        assert_eq!(
            inputs.map(|bits| sqrt_value(f32::from_bits(bits)).to_bits()),
            expected
        );
    }

    #[test]
    fn square_backward_matches_pow_order_for_overflow_and_subnormals() {
        let inputs = [
            0x7f7f_ffff,
            0xff7f_ffff,
            0x0000_0001,
            0x8000_0001,
            0x0000_0002,
            0x8000_0002,
            0x0080_0000,
            0x8080_0000,
        ];
        let upstream = [
            0x3e80_0000,
            0x3e80_0000,
            0x3f00_0000,
            0x3f00_0000,
            0x3e80_0000,
            0x3e80_0000,
            0x0000_0001,
            0x0000_0001,
        ];
        let expected = [
            0x7f80_0000,
            0xff80_0000,
            0x0000_0001,
            0x8000_0001,
            0x0000_0001,
            0x8000_0001,
            0x0000_0000,
            0x8000_0000,
        ];

        let leaf = Tensor::from_vec(inputs.map(f32::from_bits).to_vec(), [inputs.len()])
            .unwrap()
            .with_requires_grad(true);
        let weights =
            Tensor::from_vec(upstream.map(f32::from_bits).to_vec(), [upstream.len()]).unwrap();
        leaf.square()
            .unwrap()
            .mul(&weights)
            .unwrap()
            .sum()
            .backward()
            .unwrap();

        assert!(
            leaf.grad()
                .unwrap()
                .unwrap()
                .logical_values()
                .map(f32::to_bits)
                .eq(expected)
        );
    }

    #[test]
    fn square_can_preserve_a_binary_output_singleton_stride() {
        let input = Tensor::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [2, 1, 3]).unwrap();
        let target = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], [3, 1, 2])
            .unwrap()
            .permute_axes([2, 1, 0])
            .unwrap();
        let difference = input.sub(&target).unwrap();

        assert_eq!(input.stride(), &[3, 3, 1]);
        assert_eq!(target.stride(), &[1, 2, 2]);
        assert_eq!(difference.stride(), &[3, 6, 1]);

        let ordinary = difference.square().unwrap();
        let preserved = difference.square_preserving_strides().unwrap();
        assert_eq!(ordinary.stride(), &[3, 3, 1]);
        assert_eq!(preserved.stride(), &[3, 6, 1]);
        assert!(
            ordinary
                .logical_values()
                .map(f32::to_bits)
                .eq(preserved.logical_values().map(f32::to_bits))
        );
        assert_ne!(preserved.data_ptr(), difference.data_ptr());
    }

    #[test]
    fn absolute_preserves_dense_strides_and_clears_float32_sign_bits() {
        let bits = [
            0x0000_0000,
            0x8000_0000,
            0x3f80_0000,
            0xbf80_0000,
            0x7f80_0000,
            0xff80_0000,
            0x7fc1_2345,
            0xffc5_4321,
        ];
        let input = Tensor::from_vec(bits.map(f32::from_bits).to_vec(), [2, 4])
            .unwrap()
            .permute_axes([1, 0])
            .unwrap();
        let expected = input
            .logical_values()
            .map(|value| value.to_bits() & !F32_SIGN_MASK)
            .collect::<Vec<_>>();

        let output = input.absolute().unwrap();

        assert_eq!(output.shape(), input.shape());
        assert_eq!(output.stride(), input.stride());
        assert_eq!(output.storage_offset(), 0);
        assert!(output.logical_values().map(f32::to_bits).eq(expected));
        assert_ne!(output.data_ptr(), input.data_ptr());
    }

    #[test]
    fn full_slice_is_an_identity_view_and_rejects_scalars() {
        let source = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]).unwrap();
        let alias = source.index_full_slice().unwrap();

        assert!(alias.shares_storage_with(&source));
        assert_eq!(alias.shape(), source.shape());
        assert_eq!(alias.stride(), source.stride());
        assert_eq!(alias.storage_offset(), source.storage_offset());
        assert_eq!(alias.try_to_vec().unwrap(), source.try_to_vec().unwrap());

        let scalar = Tensor::from_vec(vec![1.0], []).unwrap();
        assert_eq!(
            scalar.index_full_slice(),
            Err(TensorError::SliceCannotApplyToScalar)
        );
    }

    #[test]
    fn first_dimension_unbind_tracks_output_numbers_only_with_autograd_history() {
        let source = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2])
            .unwrap()
            .with_requires_grad(true);
        let outputs = source.unbind_first_dimension().unwrap();

        assert_eq!(outputs.len(), 3);
        assert_eq!(
            outputs.iter().map(Tensor::output_nr).collect::<Vec<_>>(),
            [0, 1, 2]
        );
        assert!(Arc::ptr_eq(
            outputs[0].autograd.as_ref().unwrap(),
            outputs[1].autograd.as_ref().unwrap()
        ));
        for (index, output) in outputs.iter().enumerate() {
            assert!(output.shares_storage_with(&source));
            assert_eq!(output.shape(), [2]);
            assert_eq!(output.stride(), [1]);
            assert_eq!(output.storage_offset(), index * 2);
            assert_eq!(
                source
                    .index_integer(i64::try_from(index).unwrap())
                    .unwrap()
                    .output_nr(),
                0
            );
        }

        outputs[1].sum().backward().unwrap();
        assert_eq!(
            source.grad().unwrap().unwrap().as_slice(),
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
        );

        let no_grad_outputs = {
            let _guard = crate::no_grad();
            source.unbind_first_dimension().unwrap()
        };
        assert!(no_grad_outputs.iter().all(|output| output.output_nr() == 0));

        let ordinary = Tensor::zeros([3, 2]).unwrap();
        assert!(
            ordinary
                .unbind_first_dimension()
                .unwrap()
                .iter()
                .all(|output| output.output_nr() == 0)
        );
        assert!(
            Tensor::zeros([0, 2])
                .unwrap()
                .unbind_first_dimension()
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            Tensor::zeros([]).unwrap().unbind_first_dimension(),
            Err(TensorError::InvalidScalarIndex)
        );

        let signed_source = Tensor::from_vec(vec![1.0, -0.0], [2, 1])
            .unwrap()
            .with_requires_grad(true);
        let signed_outputs = signed_source.unbind_first_dimension().unwrap();
        signed_outputs[0]
            .mul(&signed_outputs[1])
            .unwrap()
            .sum()
            .backward()
            .unwrap();
        let gradient = signed_source.grad().unwrap().unwrap();
        assert_eq!(gradient.as_slice()[0].to_bits(), (-0.0_f32).to_bits());
        assert_eq!(gradient.as_slice()[1].to_bits(), 1.0_f32.to_bits());
    }

    #[test]
    fn saved_contiguous_slice_borrows_an_offset_range_and_rejects_strides() {
        let source = Tensor::from_vec((0_u8..16).map(f32::from).collect(), [2, 2, 4])
            .unwrap()
            .with_requires_grad(true);
        let offset = source.index_integer(1).unwrap();
        let saved = SavedTensor::try_from_tensor(&offset, true).unwrap();
        let saved_values = saved.contiguous_slice().unwrap();

        assert_eq!(offset.storage_offset(), 8);
        assert!(Arc::ptr_eq(
            saved.storage.as_ref().unwrap(),
            &source.storage
        ));
        assert_eq!(
            saved_values.as_ptr(),
            source.storage.owned_values().unwrap()[8..].as_ptr()
        );
        assert_eq!(saved_values, [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);

        let strided = offset.transpose(0, 1).unwrap();
        let saved_strided = SavedTensor::try_from_tensor(&strided, true).unwrap();
        assert!(saved_strided.contiguous_slice().is_none());
        assert!(
            (0..saved_strided.elements)
                .map(|index| saved_strided.value_at_linear_index(index))
                .eq([8.0, 12.0, 9.0, 13.0, 10.0, 14.0, 11.0, 15.0])
        );
    }

    #[cfg(feature = "python-bindings")]
    #[test]
    fn saved_unary_nodes_preserve_operation_specific_python_names() {
        let source = Tensor::from_vec(vec![-1.0, 0.5], [2])
            .unwrap()
            .with_requires_grad(true);

        assert_eq!(source.relu().unwrap().grad_fn_name(), Some("ReluBackward0"));
        assert_eq!(source.sin().unwrap().grad_fn_name(), Some("SinBackward0"));
        assert_eq!(source.exp().unwrap().grad_fn_name(), Some("ExpBackward0"));
        assert_eq!(
            source.tanh(),
            Err(TensorError::AutogradRecordingUnsupported { operation: "tanh" })
        );
        let scalar = Tensor::from_vec(vec![0.5], [])
            .unwrap()
            .with_requires_grad(true);
        assert_eq!(scalar.tanh().unwrap().grad_fn_name(), Some("TanhBackward0"));
        assert_eq!(
            source.square().unwrap().grad_fn_name(),
            Some("PowBackward0")
        );
        assert_eq!(source.sqrt().unwrap().grad_fn_name(), Some("SqrtBackward0"));
    }

    fn binary_outputs(left: &Tensor, right: &Tensor) -> [Tensor; 4] {
        [
            left.add(right).unwrap(),
            left.sub(right).unwrap(),
            left.mul(right).unwrap(),
            left.div(right).unwrap(),
        ]
    }

    fn add_values(left: f32, right: f32) -> f32 {
        left + right
    }

    fn offset_contiguous_tensor(bits: &[u32], shape: &[usize]) -> Tensor {
        let elements = shape.iter().product::<usize>();
        assert_eq!(bits.len(), elements);
        let mut values = vec![0.0; elements];
        values.extend(bits.iter().copied().map(f32::from_bits));
        let mut source_shape = vec![2];
        source_shape.extend_from_slice(shape);
        Tensor::from_vec(values, source_shape)
            .unwrap()
            .index_integer(1)
            .unwrap()
    }

    fn offset_strided_matrix(bits: [u32; 9]) -> Tensor {
        let mut values = vec![0.0; 9];
        values.extend(bits.map(f32::from_bits));
        Tensor::from_vec(values, [2, 3, 3])
            .unwrap()
            .index_integer(1)
            .unwrap()
            .transpose(0, 1)
            .unwrap()
    }

    #[test]
    fn contiguous_matmul_is_bitwise_identical_to_shared_gradient_fallback() {
        let left = Tensor::from_vec(
            [
                0x60ad_78ec,
                0xe0ad_78ec,
                0x3f80_0000,
                0xff80_0000,
                0x8000_0000,
                0x7fc1_2345,
            ]
            .map(f32::from_bits)
            .to_vec(),
            [2, 3],
        )
        .unwrap();
        let right = Tensor::from_vec(
            [
                0x3f80_0000,
                0xbf80_0000,
                0x7f80_0000,
                0x3f80_0000,
                0x3f80_0000,
                0xff80_0000,
                0x3f80_0000,
                0x0000_0000,
                0x3f80_0000,
            ]
            .map(f32::from_bits)
            .to_vec(),
            [3, 3],
        )
        .unwrap();
        let shared_left = shared_gradient_copy(&left);
        let shared_right = shared_gradient_copy(&right);
        let expected = shared_left.matmul(&shared_right).unwrap();
        assert_eq!(expected.as_slice()[0].to_bits(), 0x3f80_0000);

        for actual in [
            left.matmul(&right).unwrap(),
            left.matmul(&shared_right).unwrap(),
            shared_left.matmul(&right).unwrap(),
        ] {
            assert!(
                actual
                    .logical_values()
                    .map(f32::to_bits)
                    .eq(expected.logical_values().map(f32::to_bits))
            );
        }
    }

    #[test]
    fn row_biased_matmul_seeds_signed_zero_accumulators() {
        let bias = Tensor::from_vec(vec![-0.0], [1]).unwrap();
        let negative_zero_product = Tensor::from_vec(vec![0.0], [1, 1])
            .unwrap()
            .matmul_with_row_bias(&Tensor::from_vec(vec![-0.0], [1, 1]).unwrap(), &bias)
            .unwrap();
        assert_eq!(
            negative_zero_product.as_slice()[0].to_bits(),
            (-0.0_f32).to_bits()
        );

        let empty_inner = Tensor::zeros([1, 0])
            .unwrap()
            .matmul_with_row_bias(&Tensor::zeros([0, 1]).unwrap(), &bias)
            .unwrap();
        assert_eq!(empty_inner.as_slice()[0].to_bits(), (-0.0_f32).to_bits());
    }

    #[test]
    fn blocked_contiguous_matmul_preserves_each_result_bit_pattern() {
        let rows = CONTIGUOUS_MATMUL_ROW_BLOCK + 1;
        let inner = 64;
        let columns = 64;
        assert!(inner * columns >= CONTIGUOUS_MATMUL_MIN_RHS_ELEMENTS);

        let finite_bits = [
            0x60ad_78ec,
            0xe0ad_78ec,
            0x3f80_0000,
            0xbf80_0000,
            0x0000_0000,
            0x8000_0000,
            0x0000_0001,
            0x8000_0001,
        ];
        let left = Tensor::from_vec(
            (0..rows * inner)
                .map(|index| f32::from_bits(finite_bits[(index * 5 + index / inner) % 8]))
                .collect(),
            [rows, inner],
        )
        .unwrap();
        let right = Tensor::from_vec(
            (0..inner * columns)
                .map(|index| {
                    let depth = index / columns;
                    let column = index % columns;
                    let bits = match column % 8 {
                        3 if depth == 7 => 0x7f80_0000,
                        4 if depth == 11 => 0xff80_0000,
                        5 if depth == 13 => 0x7fc1_2345,
                        _ => finite_bits[(depth * 3 + column) % 8],
                    };
                    f32::from_bits(bits)
                })
                .collect(),
            [inner, columns],
        )
        .unwrap();
        let expected = shared_gradient_copy(&left)
            .matmul(&shared_gradient_copy(&right))
            .unwrap();
        let actual = left.matmul(&right).unwrap();

        assert!(
            actual
                .logical_values()
                .map(f32::to_bits)
                .eq(expected.logical_values().map(f32::to_bits))
        );
    }

    #[test]
    fn contiguous_matmul_preserves_empty_dimension_outputs() {
        let no_rows = Tensor::zeros([0, 3])
            .unwrap()
            .matmul(&Tensor::ones([3, 4]).unwrap())
            .unwrap();
        assert_eq!(no_rows.shape(), [0, 4]);
        assert!(no_rows.as_slice().is_empty());

        let no_columns = Tensor::ones([2, 3])
            .unwrap()
            .matmul(&Tensor::zeros([3, 0]).unwrap())
            .unwrap();
        assert_eq!(no_columns.shape(), [2, 0]);
        assert!(no_columns.as_slice().is_empty());

        let no_inner = Tensor::ones([2, 0])
            .unwrap()
            .matmul(&Tensor::zeros([0, 4]).unwrap())
            .unwrap();
        assert_eq!(no_inner.shape(), [2, 4]);
        assert!(no_inner.as_slice().iter().all(|value| value.to_bits() == 0));
    }

    #[test]
    fn owned_strided_matmul_is_bitwise_identical_to_shared_gradient_fallback() {
        let left = offset_strided_matrix([
            0x7f80_0000,
            0x8000_0000,
            0x3f80_0000,
            0xff80_0000,
            0x0000_0000,
            0xbf80_0000,
            0x7fc1_2345,
            0x8000_0000,
            0x4000_0000,
        ]);
        let right = offset_strided_matrix([
            0x3f80_0000,
            0xbf80_0000,
            0x4000_0000,
            0x7f80_0000,
            0x3f80_0000,
            0xff80_0000,
            0x7fc6_789a,
            0x0000_0000,
            0x3f80_0000,
        ]);
        assert!(!left.is_contiguous());
        assert!(!right.is_contiguous());
        assert_ne!(left.storage_offset(), 0);
        assert_ne!(right.storage_offset(), 0);

        let shared_left = shared_gradient_copy(&left);
        let shared_right = shared_gradient_copy(&right);
        let expected = shared_left.matmul(&shared_right).unwrap();
        assert_eq!(expected.as_slice()[3].to_bits(), 0x0000_0000);
        for actual in [
            left.matmul(&right).unwrap(),
            left.matmul(&shared_right).unwrap(),
            shared_left.matmul(&right).unwrap(),
        ] {
            assert!(
                actual
                    .logical_values()
                    .map(f32::to_bits)
                    .eq(expected.logical_values().map(f32::to_bits))
            );
        }
    }

    #[test]
    fn owned_dense_binary_fast_path_is_bitwise_identical_to_shared_gradient_fallback() {
        let left = offset_strided_matrix([
            0x7f81_2345,
            0xffc5_4321,
            0x7f80_0000,
            0xff80_0000,
            0x0000_0000,
            0x8000_0000,
            0x0000_0001,
            0x8000_0001,
            0x3f80_0000,
        ]);
        let right = offset_strided_matrix([
            0xff85_4321,
            0x7fc1_2345,
            0xff80_0000,
            0x7f80_0000,
            0x8000_0000,
            0x0000_0000,
            0x8000_0001,
            0x0000_0001,
            0xbf80_0000,
        ]);
        let shared_left = shared_gradient_copy(&left);
        let shared_right = shared_gradient_copy(&right);
        let cases = [
            (
                left.add(&right).unwrap(),
                shared_left.add(&shared_right).unwrap(),
            ),
            (
                left.sub(&right).unwrap(),
                shared_left.sub(&shared_right).unwrap(),
            ),
            (
                left.mul(&right).unwrap(),
                shared_left.mul(&shared_right).unwrap(),
            ),
            (
                left.div(&right).unwrap(),
                shared_left.div(&shared_right).unwrap(),
            ),
        ];

        for (actual, expected) in cases {
            assert_eq!(actual.shape(), expected.shape());
            assert_eq!(actual.stride(), expected.stride());
            assert!(
                actual
                    .logical_values()
                    .map(f32::to_bits)
                    .eq(expected.logical_values().map(f32::to_bits))
            );
        }
    }

    #[test]
    fn owned_general_broadcast_is_bitwise_identical_to_shared_gradient_fallback() {
        let verify = |left: &Tensor, right: &Tensor| {
            assert_ne!(left.shape(), right.shape());
            let shared_left = shared_gradient_copy(left);
            let shared_right = shared_gradient_copy(right);
            let expected = binary_outputs(&shared_left, &shared_right);

            for actual in [
                binary_outputs(left, right),
                binary_outputs(left, &shared_right),
                binary_outputs(&shared_left, right),
            ] {
                for (actual, expected) in actual.into_iter().zip(&expected) {
                    assert_eq!(actual.shape(), expected.shape());
                    assert_eq!(actual.stride(), expected.stride());
                    assert_eq!(actual.storage_offset(), expected.storage_offset());
                    assert_eq!(actual.dtype(), expected.dtype());
                    assert_eq!(actual.device(), expected.device());
                    assert!(
                        actual
                            .logical_values()
                            .map(f32::to_bits)
                            .eq(expected.logical_values().map(f32::to_bits))
                    );
                }
            }
        };

        let left = Tensor::from_vec(
            [
                0x7f81_2345,
                0xffc5_4321,
                0x7f80_0000,
                0xff80_0000,
                0x0000_0000,
                0x8000_0000,
            ]
            .map(f32::from_bits)
            .to_vec(),
            [2, 1, 3],
        )
        .unwrap();
        let right = Tensor::from_vec(
            [0xff85_4321, 0x8000_0001].map(f32::from_bits).to_vec(),
            [1, 2, 1],
        )
        .unwrap();
        verify(&left, &right);

        let strided = offset_strided_matrix([
            0x7f81_2345,
            0xffc5_4321,
            0x7f80_0000,
            0xff80_0000,
            0x0000_0000,
            0x8000_0000,
            0x0000_0001,
            0x8000_0001,
            0x3f80_0000,
        ]);
        let row = Tensor::from_vec(
            [0_u32, 0, 0, 0xff85_4321, 0x8000_0001, 0xbf80_0000]
                .map(f32::from_bits)
                .to_vec(),
            [2, 3],
        )
        .unwrap()
        .index_integer(1)
        .unwrap();
        assert_eq!(strided.add(&row).unwrap().stride(), [1, 3]);
        verify(&strided, &row);

        let nan_scalar = Tensor::from_vec([0_u32, 0xff81_2345].map(f32::from_bits).to_vec(), [2])
            .unwrap()
            .index_integer(1)
            .unwrap();
        verify(&nan_scalar, &strided);
        verify(&strided, &nan_scalar);
    }

    #[test]
    fn contiguous_trailing_vector_broadcast_is_bitwise_identical_to_fallback() {
        let assert_matches_fallback = |left: &Tensor, right: &Tensor| {
            let shared_left = shared_gradient_copy(left);
            let shared_right = shared_gradient_copy(right);
            let expected = binary_outputs(&shared_left, &shared_right);
            let actual = binary_outputs(left, right);

            for (actual, expected) in actual.into_iter().zip(expected) {
                assert_eq!(actual.shape(), expected.shape());
                assert_eq!(actual.stride(), expected.stride());
                assert_eq!(actual.storage_offset(), expected.storage_offset());
                assert_eq!(actual.dtype(), expected.dtype());
                assert_eq!(actual.device(), expected.device());
                assert!(
                    actual
                        .logical_values()
                        .map(f32::to_bits)
                        .eq(expected.logical_values().map(f32::to_bits))
                );
            }
        };
        let assert_uses_fast_path = |left: &Tensor, right: &Tensor| {
            let output =
                materialize_contiguous_trailing_broadcast(left, right, &add_values).unwrap();
            assert!(output.is_some());
        };

        let broad = offset_contiguous_tensor(
            &[
                0x7f81_2345,
                0x0000_0000,
                0x8000_0000,
                0x7f80_0000,
                0x0000_0001,
                0xbf80_0000,
                0xff80_0000,
                0x7fc5_4321,
                0x8000_0000,
                0x8000_0001,
            ],
            &[2, 1, 5],
        );
        let vector = offset_contiguous_tensor(
            &[
                0x3f80_0000,
                0xffc6_789a,
                0x8000_0000,
                0x7f80_0000,
                0x7f85_6789,
            ],
            &[5],
        );
        assert_eq!(broad.storage_offset(), 10);
        assert_eq!(vector.storage_offset(), 5);
        for (left, right) in [(&broad, &vector), (&vector, &broad)] {
            assert_uses_fast_path(left, right);
            assert_matches_fallback(left, right);
        }

        let singleton_broad =
            offset_contiguous_tensor(&[0x0000_0000, 0x8000_0000, 0x7fc1_2345], &[3, 1, 1]);
        let singleton_vector = offset_contiguous_tensor(&[0x8000_0000], &[1]);
        for (left, right) in [
            (&singleton_broad, &singleton_vector),
            (&singleton_vector, &singleton_broad),
        ] {
            assert_uses_fast_path(left, right);
            assert_matches_fallback(left, right);
        }

        let paired_nan_broad = offset_contiguous_tensor(
            &[
                0x7f81_2345,
                0xffc1_2345,
                0x7fc5_4321,
                0xff85_4321,
                0xffc6_789a,
                0x7f86_789a,
                0xff81_abcd,
                0x7fc2_abcd,
            ],
            &[4, 2],
        );
        let paired_nan_vector = offset_contiguous_tensor(&[0xffc5_4321, 0x7f85_6789], &[2]);
        for (left, right) in [
            (&paired_nan_broad, &paired_nan_vector),
            (&paired_nan_vector, &paired_nan_broad),
        ] {
            assert_uses_fast_path(left, right);
            assert_matches_fallback(left, right);
        }

        let empty_broad = Tensor::zeros([2, 4, 0]).unwrap().index_integer(1).unwrap();
        let empty_vector = Tensor::zeros([0]).unwrap();
        assert_matches_fallback(&empty_broad, &empty_vector);
        assert_matches_fallback(&empty_vector, &empty_broad);
    }

    #[test]
    fn contiguous_trailing_singleton_broadcast_is_bitwise_identical_to_fallback() {
        let assert_matches_fallback = |left: &Tensor, right: &Tensor| {
            let shared_left = shared_gradient_copy(left);
            let shared_right = shared_gradient_copy(right);
            let expected = binary_outputs(&shared_left, &shared_right);
            let actual = binary_outputs(left, right);

            for (actual, expected) in actual.into_iter().zip(expected) {
                assert_eq!(actual.shape(), expected.shape());
                assert_eq!(actual.stride(), expected.stride());
                assert_eq!(actual.storage_offset(), expected.storage_offset());
                assert!(
                    actual
                        .logical_values()
                        .map(f32::to_bits)
                        .eq(expected.logical_values().map(f32::to_bits))
                );
            }
        };
        let assert_uses_fast_path = |left: &Tensor, right: &Tensor| {
            let output =
                materialize_contiguous_trailing_broadcast(left, right, &add_values).unwrap();
            assert!(output.is_some());
        };

        let rows = offset_contiguous_tensor(
            &[
                0x7f81_2345,
                0x0000_0000,
                0x8000_0000,
                0x7f80_0000,
                0x0000_0001,
                0xbf80_0000,
                0xff80_0000,
                0x7fc5_4321,
                0x8000_0000,
                0x8000_0001,
            ],
            &[2, 1, 5],
        );
        let row_scalars = offset_contiguous_tensor(&[0xffc6_789a, 0x8000_0000], &[2, 1, 1]);
        assert_eq!(rows.storage_offset(), 10);
        assert_eq!(row_scalars.storage_offset(), 2);
        for (left, right) in [(&rows, &row_scalars), (&row_scalars, &rows)] {
            assert_uses_fast_path(left, right);
            assert_matches_fallback(left, right);
        }

        let vector = offset_contiguous_tensor(
            &[
                0x0000_0000,
                0x8000_0000,
                0x7f80_0000,
                0xff80_0000,
                0x7fc1_2345,
            ],
            &[5],
        );
        let scalar = offset_contiguous_tensor(&[0xffc5_4321], &[1]);
        for (left, right) in [(&vector, &scalar), (&scalar, &vector)] {
            assert_uses_fast_path(left, right);
            assert_matches_fallback(left, right);
        }

        let singleton_rows =
            offset_contiguous_tensor(&[0x0000_0000, 0x8000_0000, 0x7fc1_2345], &[3, 1, 1]);
        let singleton_scalars =
            offset_contiguous_tensor(&[0x8000_0000, 0x7f85_6789, 0xff80_0000], &[3, 1, 1]);
        assert_matches_fallback(&singleton_rows, &singleton_scalars);
        assert_matches_fallback(&singleton_scalars, &singleton_rows);

        let empty_rows = Tensor::zeros([2, 0, 5]).unwrap();
        let empty_scalars = Tensor::zeros([2, 0, 1]).unwrap();
        assert_matches_fallback(&empty_rows, &empty_scalars);
        assert_matches_fallback(&empty_scalars, &empty_rows);

        let trailing_empty = Tensor::zeros([2, 0]).unwrap();
        let trailing_scalar = Tensor::zeros([2, 1]).unwrap();
        assert_matches_fallback(&trailing_empty, &trailing_scalar);
        assert_matches_fallback(&trailing_scalar, &trailing_empty);

        let strided_rows = Tensor::ones([2, 3, 5]).unwrap().transpose(0, 1).unwrap();
        let strided_scalars = Tensor::ones([3, 2, 1]).unwrap();
        assert!(
            materialize_contiguous_trailing_broadcast(&strided_rows, &strided_scalars, &add_values)
                .unwrap()
                .is_none()
        );
        assert_matches_fallback(&strided_rows, &strided_scalars);
        assert_matches_fallback(&strided_scalars, &strided_rows);

        let general_left = Tensor::ones([2, 1, 5]).unwrap();
        let general_right = Tensor::ones([1, 3, 1]).unwrap();
        assert!(
            materialize_contiguous_trailing_broadcast(&general_left, &general_right, &add_values)
                .unwrap()
                .is_none()
        );
        assert_matches_fallback(&general_left, &general_right);
        assert_matches_fallback(&general_right, &general_left);
    }

    #[test]
    fn dense_materialization_fast_path_requires_owned_matching_strides() {
        let bits = [
            0x3f80_0000,
            0x4000_0000,
            0x4040_0000,
            0x4080_0000,
            0x40a0_0000,
            0x40c0_0000,
            0x40e0_0000,
            0x4100_0000,
            0x4110_0000,
        ];
        let tensor = offset_strided_matrix(bits);
        let physical_bits = bits.to_vec();
        let logical_bits = tensor
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        assert_ne!(physical_bits, logical_bits);

        let matched_visits = RefCell::new(Vec::new());
        let matched = tensor
            .materialize_with_strides(tensor.stride(), |value| {
                matched_visits.borrow_mut().push(value.to_bits());
                value
            })
            .unwrap();
        assert_eq!(matched_visits.into_inner(), physical_bits);
        assert!(matched.into_iter().map(f32::to_bits).eq(bits));

        let shared = shared_gradient_copy(&tensor);
        let shared_visits = RefCell::new(Vec::new());
        let shared_output = shared
            .materialize_with_strides(shared.stride(), |value| {
                shared_visits.borrow_mut().push(value.to_bits());
                value
            })
            .unwrap();
        assert_eq!(shared_visits.into_inner(), logical_bits);
        assert!(shared_output.into_iter().map(f32::to_bits).eq(bits));

        let contiguous_strides = [3, 1];
        let mismatched_visits = RefCell::new(Vec::new());
        let mismatched = tensor
            .materialize_with_strides(&contiguous_strides, |value| {
                mismatched_visits.borrow_mut().push(value.to_bits());
                value
            })
            .unwrap();
        assert_eq!(mismatched_visits.into_inner(), logical_bits);
        assert_eq!(
            mismatched.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
            logical_bits
        );

        assert_eq!(
            tensor.materialize_with_strides(&[usize::MAX, 1], |value| value),
            Err(TensorError::IndexCalculationOverflow)
        );
    }

    #[test]
    fn owned_dense_materialization_is_bitwise_identical_to_shared_gradient_fallback() {
        let tensor = offset_strided_matrix([
            0x0000_0000,
            0x8000_0000,
            0x0000_0001,
            0x8000_0001,
            0x7f80_0000,
            0xff80_0000,
            0x7fc1_2345,
            0xffc5_4321,
            0xbf80_0000,
        ]);
        let shared = shared_gradient_copy(&tensor);
        let cases = [
            (tensor.negate().unwrap(), shared.negate().unwrap()),
            (tensor.relu().unwrap(), shared.relu().unwrap()),
            (tensor.sin().unwrap(), shared.sin().unwrap()),
            (tensor.exp().unwrap(), shared.exp().unwrap()),
            (tensor.floor().unwrap(), shared.floor().unwrap()),
            (tensor.tanh().unwrap(), shared.tanh().unwrap()),
            (tensor.sqrt().unwrap(), shared.sqrt().unwrap()),
            (
                tensor.add_scalar(1.25).unwrap(),
                shared.add_scalar(1.25).unwrap(),
            ),
            (
                tensor.sub_scalar(1.25).unwrap(),
                shared.sub_scalar(1.25).unwrap(),
            ),
            (
                tensor.mul_scalar(-0.0).unwrap(),
                shared.mul_scalar(-0.0).unwrap(),
            ),
            (
                tensor.div_scalar(-2.0).unwrap(),
                shared.div_scalar(-2.0).unwrap(),
            ),
            (
                tensor.scalar_sub(1.25).unwrap(),
                shared.scalar_sub(1.25).unwrap(),
            ),
            (
                tensor.scalar_div(-2.0).unwrap(),
                shared.scalar_div(-2.0).unwrap(),
            ),
            (tensor.try_clone().unwrap(), shared.try_clone().unwrap()),
        ];

        for (actual, expected) in cases {
            assert_eq!(actual.shape(), expected.shape());
            assert_eq!(actual.stride(), expected.stride());
            assert_eq!(actual.storage_offset(), expected.storage_offset());
            assert_eq!(actual.dtype(), expected.dtype());
            assert_eq!(actual.device(), expected.device());
            assert!(
                actual
                    .logical_values()
                    .map(f32::to_bits)
                    .eq(expected.logical_values().map(f32::to_bits))
            );
        }
    }

    #[test]
    fn owned_strided_matmul_preserves_fallback_index_errors() {
        let invalid = Tensor {
            storage: Arc::new(Storage::from_owned(vec![1.0], DType::Float32, Device::Cpu)),
            shape: vec![2, 2],
            strides: vec![2, 1],
            offset: 0,
            elements: 4,
            output_nr: 0,
            view_requires_grad: false,
            autograd: None,
        };
        let right = Tensor::ones([2, 1]).unwrap();

        assert_eq!(
            invalid.matmul(&right),
            shared_gradient_copy(&invalid).matmul(&right)
        );
        assert_eq!(
            invalid.matmul(&right),
            Err(TensorError::IndexCalculationOverflow)
        );
    }

    #[test]
    fn multiply_snapshots_live_gradient_operands_for_backward() {
        let source = Tensor::from_vec(vec![4.0, 5.0], [2])
            .unwrap()
            .with_requires_grad(true);
        source.sum().backward().unwrap();
        let live_gradient = source.live_grad().unwrap().unwrap();
        let weights = Tensor::from_vec(vec![2.0, 3.0], [2])
            .unwrap()
            .with_requires_grad(true);
        let saved_loss = weights.mul(&live_gradient).unwrap().sum();

        source.sum().backward().unwrap();
        assert_eq!(live_gradient.try_to_vec().unwrap(), [2.0, 2.0]);
        saved_loss.backward().unwrap();
        assert_eq!(weights.grad().unwrap().unwrap().as_slice(), [1.0, 1.0]);
    }

    #[test]
    fn broadcast_multiply_snapshots_live_gradient_operands_for_backward() {
        let source = Tensor::from_vec(vec![4.0, 5.0], [2, 1])
            .unwrap()
            .with_requires_grad(true);
        source.sum().backward().unwrap();
        let live_gradient = source.live_grad().unwrap().unwrap();
        let weights = Tensor::from_vec(vec![2.0, 3.0, 4.0], [1, 3])
            .unwrap()
            .with_requires_grad(true);
        let saved_loss = weights.mul(&live_gradient).unwrap().sum();

        source.sum().backward().unwrap();
        assert_eq!(live_gradient.try_to_vec().unwrap(), [2.0, 2.0]);
        saved_loss.backward().unwrap();
        assert_eq!(weights.grad().unwrap().unwrap().as_slice(), [2.0; 3]);
    }

    #[test]
    fn relu_snapshots_contiguous_live_gradient_storage_for_backward() {
        let source = Tensor::ones([4]).unwrap().with_requires_grad(true);
        let initial_weights =
            Tensor::from_vec(vec![-1.0, 2.0, -0.0, f32::from_bits(0x7fc1_2345)], [4]).unwrap();
        source
            .mul(&initial_weights)
            .unwrap()
            .sum()
            .backward()
            .unwrap();
        let live_gradient = source
            .live_grad()
            .unwrap()
            .unwrap()
            .with_requires_grad(true);
        let saved_loss = live_gradient.relu().unwrap().sum();

        let later_weights = Tensor::from_vec(vec![3.0, -4.0, 1.0, 0.0], [4]).unwrap();
        source
            .mul(&later_weights)
            .unwrap()
            .sum()
            .backward()
            .unwrap();
        assert!(
            live_gradient
                .logical_values()
                .take(3)
                .map(f32::to_bits)
                .eq([2.0_f32.to_bits(), (-2.0_f32).to_bits(), 1.0_f32.to_bits()])
        );

        saved_loss.backward().unwrap();
        assert!(
            live_gradient
                .grad()
                .unwrap()
                .unwrap()
                .logical_values()
                .map(f32::to_bits)
                .eq([
                    0.0_f32.to_bits(),
                    1.0_f32.to_bits(),
                    0.0_f32.to_bits(),
                    1.0_f32.to_bits(),
                ])
        );
        assert_eq!(saved_loss.backward(), Err(TensorError::BackwardGraphFreed));
    }

    #[test]
    fn leaf_root_batch_aggregates_duplicate_seeds_before_existing_gradient() {
        let leaf = Tensor::from_vec(vec![1.0], [1])
            .unwrap()
            .with_requires_grad(true);
        leaf.mul_scalar(16_777_216.0).unwrap().backward().unwrap();

        Tensor::backward_leaf_roots(&[&leaf, &leaf]).unwrap();

        assert_eq!(leaf.grad().unwrap().unwrap().as_slice(), [16_777_218.0]);
    }

    #[test]
    fn three_leaf_root_batch_accumulates_distinct_and_duplicate_roots() {
        let first = Tensor::from_vec(vec![2.0], [])
            .unwrap()
            .with_requires_grad(true);
        let second = Tensor::from_vec(vec![3.0], [1])
            .unwrap()
            .with_requires_grad(true);

        Tensor::backward_leaf_roots(&[&first, &second, &first]).unwrap();
        assert_eq!(first.grad().unwrap().unwrap().as_slice(), [2.0]);
        assert_eq!(second.grad().unwrap().unwrap().as_slice(), [1.0]);

        Tensor::backward_leaf_roots(&[&first, &first, &first]).unwrap();
        assert_eq!(first.grad().unwrap().unwrap().as_slice(), [5.0]);
        assert_eq!(second.grad().unwrap().unwrap().as_slice(), [1.0]);
    }

    #[test]
    fn three_duplicate_leaf_roots_aggregate_before_existing_gradient() {
        let leaf = Tensor::from_vec(vec![1.0], [1])
            .unwrap()
            .with_requires_grad(true);
        leaf.mul_scalar(16_777_216.0).unwrap().backward().unwrap();

        Tensor::backward_leaf_roots(&[&leaf, &leaf, &leaf]).unwrap();

        assert_eq!(leaf.grad().unwrap().unwrap().as_slice(), [16_777_220.0]);
    }

    #[test]
    fn leaf_root_batch_rejects_no_grad_view_before_committing() {
        let first = Tensor::from_vec(vec![3.0], [])
            .unwrap()
            .with_requires_grad(true);
        let source = Tensor::from_vec(vec![1.0, 2.0], [1, 2])
            .unwrap()
            .with_requires_grad(true);
        let view = {
            let _guard = crate::no_grad();
            source.transpose(0, 1).unwrap().index([1]).unwrap()
        };
        assert_eq!(view.shape(), [1]);
        assert_eq!(view.stride(), [2]);
        assert_eq!(view.storage_offset(), 1);
        assert!(view.requires_grad());
        assert!(view.is_leaf());

        assert_eq!(
            Tensor::backward_leaf_roots(&[&first, &view]),
            Err(TensorError::DoesNotRequireGradAt { index: 1 })
        );
        assert!(first.grad().unwrap().is_none());
        assert!(source.grad().unwrap().is_none());
        assert!(view.grad().unwrap().is_none());

        first.backward().unwrap();
        assert_eq!(first.grad().unwrap().unwrap().as_slice(), [1.0]);
    }

    #[test]
    fn three_leaf_root_batch_validates_every_root_before_committing() {
        let first = Tensor::from_vec(vec![3.0], [])
            .unwrap()
            .with_requires_grad(true);
        let second = Tensor::from_vec(vec![4.0], [1])
            .unwrap()
            .with_requires_grad(true);
        let source = Tensor::from_vec(vec![1.0, 2.0], [1, 2])
            .unwrap()
            .with_requires_grad(true);
        let view = {
            let _guard = crate::no_grad();
            source.transpose(0, 1).unwrap().index([1]).unwrap()
        };

        assert_eq!(
            Tensor::backward_leaf_roots(&[&first, &second, &view]),
            Err(TensorError::DoesNotRequireGradAt { index: 2 })
        );
        assert!(first.grad().unwrap().is_none());
        assert!(second.grad().unwrap().is_none());
        assert!(source.grad().unwrap().is_none());
        assert!(view.grad().unwrap().is_none());
    }

    #[test]
    fn negation_toggles_every_float_sign_bit() {
        let bits = [
            0x0000_0000,
            0x8000_0000,
            0x0000_0001,
            0x8000_0001,
            0x7f80_0000,
            0xff80_0000,
            0x7fc1_2345,
            0xffc5_4321,
            0x7f81_2345,
            0xff85_4321,
        ];
        let values = bits.map(f32::from_bits);
        let input = Tensor::from_vec(values.to_vec(), [bits.len()]).unwrap();
        let output = input.negate().unwrap();
        assert!(!output.shares_storage_with(&input));
        assert!(
            output
                .logical_values()
                .map(f32::to_bits)
                .eq(bits.map(|value| value ^ F32_SIGN_MASK))
        );
    }

    #[test]
    fn negation_gradient_edge_is_reusable_shared_and_bitwise() {
        let leaf = Tensor::from_vec(vec![2.0, -3.0], [2])
            .unwrap()
            .with_requires_grad(true);
        let repeated = leaf.negate().unwrap().sum();
        repeated.backward().unwrap();
        repeated.backward().unwrap();
        assert_eq!(leaf.grad().unwrap().unwrap().as_slice(), [-2.0, -2.0]);

        let shared_leaf = Tensor::from_vec(vec![5.0, 7.0], [2])
            .unwrap()
            .with_requires_grad(true);
        let shared_negative = shared_leaf.negate().unwrap();
        let first_root = shared_negative.sum();
        let second_root = shared_negative.sum();
        first_root.backward().unwrap();
        second_root.backward().unwrap();
        assert_eq!(
            shared_leaf.grad().unwrap().unwrap().as_slice(),
            [-2.0, -2.0]
        );

        let nan_leaf = Tensor::from_vec(vec![1.0, 2.0], [2])
            .unwrap()
            .with_requires_grad(true);
        let weights = Tensor::from_vec(
            vec![f32::from_bits(0x7fc1_2345), f32::from_bits(0xffc5_4321)],
            [2],
        )
        .unwrap();
        nan_leaf
            .negate()
            .unwrap()
            .mul(&weights)
            .unwrap()
            .sum()
            .backward()
            .unwrap();
        assert!(
            nan_leaf
                .grad()
                .unwrap()
                .unwrap()
                .logical_values()
                .map(f32::to_bits)
                .eq([0xffc1_2345, 0x7fc5_4321])
        );
    }

    #[test]
    fn binary_result_reservation_failures_return_tensor_errors() {
        let elements = 17;
        let expected = Err(TensorError::AllocationFailed { elements });

        assert_eq!(try_result_vector::<f32>(usize::MAX, elements), expected);
        assert_eq!(
            try_result_vector::<usize>(usize::MAX, elements),
            Err(TensorError::AllocationFailed { elements })
        );
    }

    #[test]
    fn inference_unary_functions_propagate_result_allocation_overflow() {
        let elements = usize::MAX;
        // Failure injection deliberately bypasses the validated constructors:
        // no real tensor can own this many f32 values, so the kernel must fail
        // its output reservation before attempting to read the empty fixture.
        let tensor = Tensor {
            storage: Arc::new(Storage::from_owned(Vec::new(), DType::Float32, Device::Cpu)),
            shape: vec![elements],
            strides: vec![1],
            offset: 0,
            elements,
            output_nr: 0,
            view_requires_grad: false,
            autograd: None,
        };

        assert_eq!(
            tensor.exp(),
            Err(TensorError::AllocationFailed { elements })
        );
        assert_eq!(
            tensor.floor(),
            Err(TensorError::AllocationFailed { elements })
        );
        assert_eq!(
            tensor.reciprocal(),
            Err(TensorError::AllocationFailed { elements })
        );
        assert_eq!(
            tensor.tanh(),
            Err(TensorError::AllocationFailed { elements })
        );
        assert_eq!(
            tensor.sqrt(),
            Err(TensorError::AllocationFailed { elements })
        );
    }

    #[test]
    fn metadata_only_autograd_edges_release_intermediate_storage() {
        let leaf = Tensor::ones([16_384]).unwrap().with_requires_grad(true);
        let mut output = leaf.mul_scalar(1.0).unwrap();
        for _ in 0..128 {
            let previous_storage = Arc::downgrade(&output.storage);
            output = output.mul_scalar(1.0).unwrap();
            assert!(
                previous_storage.upgrade().is_none(),
                "scalar autograd edges must not retain operand values"
            );
        }
        assert!(output.requires_grad());
    }

    #[test]
    fn no_grad_view_multiply_does_not_retain_operand_storage_without_edges() {
        let source = Tensor::ones([16_384]).unwrap().with_requires_grad(true);
        let source_storage = Arc::downgrade(&source.storage);
        let view = {
            let _guard = crate::no_grad();
            source.reshape([128, 128]).unwrap()
        };
        let output = view.mul(&view).unwrap();

        drop(view);
        drop(source);
        assert!(
            source_storage.upgrade().is_none(),
            "no-edge operands must not be retained for an unreachable derivative"
        );

        let loss = output.sum();
        loss.backward().unwrap();
        loss.backward().unwrap();
    }
}
