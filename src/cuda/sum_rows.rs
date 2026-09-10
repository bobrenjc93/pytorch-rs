//! Geometry for contiguous f32 row reductions. The arithmetic order follows
//! `PyTorch` 2.13's Reduce.cuh (four accumulators, x then y, CTA partials y then x).
//! Device occupancy inputs are queried on the guarded device, never assumed.
use super::{Driver, Kernel, TensorError, c_int, c_void, driver};

/// One contiguous sub-iterator in `TensorIterator`'s lower-half-first order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RowSumSlice {
    pub(crate) input_offset: usize,
    pub(crate) output_offset: usize,
    pub(crate) rows: usize,
    pub(crate) columns: usize,
    pub(crate) accumulate: bool,
}

impl RowSumSlice {
    /// The caller has checked the original matrix's element count and bounds.
    pub(crate) fn for_each(
        rows: usize,
        columns: usize,
        mut visit: impl FnMut(Self) -> Result<(), TensorError>,
    ) -> Result<(), TensorError> {
        Self {
            input_offset: 0,
            output_offset: 0,
            rows,
            columns,
            accumulate: false,
        }
        .visit(&mut visit)
    }

    fn visit(
        self,
        visit: &mut impl FnMut(Self) -> Result<(), TensorError>,
    ) -> Result<(), TensorError> {
        // TensorIterator checks numel <= INT32_MAX and, for each operand,
        // 1 + sum((size - 1) * byte_stride) <= INT32_MAX. For a nonempty
        // contiguous f32 region the input byte span is the strongest bound.
        // Empty reductions only write zeros and need no ordering split.
        const MAX_ELEMENTS: usize = (i32::MAX as usize) / 4 + 1;
        if self.rows * self.columns <= MAX_ELEMENTS {
            return visit(self);
        }
        let (mut lower, mut upper) = (self, self);
        // The largest extent decides TensorIterator's split dimension. For
        // contiguous rows, (rows - 1) * columns exceeds columns - 1 whenever
        // rows > 1. Thus column splits only occur in single-row regions and
        // every child remains contiguous (no new strided-input support).
        if self.rows > 1 {
            lower.rows = self.rows / 2;
            upper.rows -= lower.rows;
            upper.input_offset += lower.rows * self.columns;
            upper.output_offset += lower.rows;
        } else {
            lower.columns = self.columns / 2;
            upper.columns -= lower.columns;
            upper.input_offset += lower.columns;
            upper.accumulate = true;
        }
        // Recursion is bounded by the bit width of the checked element count.
        lower.visit(visit)?;
        upper.visit(visit)
    }
}

pub(crate) struct RowSumConfig {
    width: u32,
    height: u32,
    reduce_y: bool,
    vectorize: bool,
    pub(crate) ctas: usize,
}

impl RowSumConfig {
    pub(crate) fn current(rows: usize, columns: usize) -> Result<Self, TensorError> {
        let driver = driver()?;
        // Initializes the context on cold threads before querying its device.
        driver.function(Kernel::SumRows)?;
        let mut device = 0;
        let mut multiprocessors = 0;
        let mut threads_per_mp = 0;
        // SAFETY: CUDA driver ABI, writable attributes, current guarded context.
        unsafe {
            driver.check((driver.device)(&raw mut device), "cuCtxGetDevice")?;
            driver.check(
                (driver.attribute)(&raw mut multiprocessors, 16, device),
                "cuDeviceGetAttribute(MULTIPROCESSOR_COUNT)",
            )?;
            driver.check(
                (driver.attribute)(&raw mut threads_per_mp, 39, device),
                "cuDeviceGetAttribute(MAX_THREADS_PER_MULTIPROCESSOR)",
            )?;
        }
        let positive = |value: c_int| {
            usize::try_from(value)
                .ok()
                .filter(|v| *v > 0)
                .ok_or_else(|| TensorError::CudaRuntimeError {
                    operation: "row reduction configuration",
                    message: "invalid CUDA device properties".into(),
                })
        };
        Ok(Self::new(
            rows,
            columns,
            positive(multiprocessors)?,
            positive(threads_per_mp)?,
        ))
    }

    fn new(rows: usize, columns: usize, multiprocessors: usize, threads_per_mp: usize) -> Self {
        let vectorize = columns >= 128;
        let floor_power = |n: usize| 1_usize << n.clamp(1, 512).ilog2();
        let dim0 = floor_power(if vectorize { columns / 4 } else { columns });
        let dim1 = floor_power(rows);
        let width = dim0.min(32);
        let height = dim1.min(512 / width);
        let width = dim0.min(512 / height);
        let reduce_y = columns.div_ceil(width) >= (height * 16).min(256);
        let step = width * if reduce_y { height } else { 1 };
        let values_per_thread = columns.div_ceil(step);
        let grid = rows.div_ceil(if reduce_y { 1 } else { height });
        let target_grid = multiprocessors * (threads_per_mp / (width * height));
        let ctas = if reduce_y && values_per_thread >= 256 && grid <= target_grid {
            target_grid
                .div_ceil(grid)
                .min(values_per_thread.div_ceil(16))
                .max(values_per_thread.div_ceil(256))
        } else {
            1
        };
        Self {
            width: u32::try_from(width).expect("at most 512 threads"),
            height: u32::try_from(height).expect("at most 512 threads"),
            reduce_y,
            vectorize,
            ctas,
        }
    }
}

/// # Safety
/// Live contiguous input/output and optional `rows * config.ctas` scratch
/// floats on the guarded device. Retain all three until stream completion,
/// including after either launch fails. Input may be null only for zero width.
/// For an accumulating slice, output must contain the preceding slices' sums.
pub(crate) unsafe fn launch_sum_rows(
    mut input: u64,
    mut output: u64,
    mut scratch: u64,
    slice: RowSumSlice,
    config: &RowSumConfig,
) -> Result<(), TensorError> {
    let driver = driver()?;
    let mut rows = slice.rows as u64;
    let mut columns = slice.columns as u64;
    let mut ctas = config.ctas as u64;
    let mut reduce_y = u32::from(config.reduce_y);
    let mut vectorize = u32::from(config.vectorize);
    let mut accumulate = u32::from(slice.accumulate);
    let mut arguments = [
        (&raw mut input).cast(),
        (&raw mut output).cast(),
        (&raw mut scratch).cast(),
        (&raw mut rows).cast(),
        (&raw mut columns).cast(),
        (&raw mut ctas).cast(),
        (&raw mut reduce_y).cast(),
        (&raw mut vectorize).cast(),
        (&raw mut accumulate).cast(),
    ];
    let row_step = if config.reduce_y {
        1
    } else {
        u64::from(config.height)
    };
    let blocks = u32::try_from(rows.div_ceil(row_step).min(65535)).expect("bounded grid");
    // The kernel strides over CTA partials as well as rows if either grid
    // dimension exceeds the portable CUDA launch limit.
    let partial_blocks = u32::try_from(ctas.min(65535)).expect("bounded grid");
    unsafe {
        launch(
            driver,
            Kernel::SumRows,
            [blocks, partial_blocks],
            config,
            &mut arguments,
        )?;
    }
    if config.ctas > 1 {
        unsafe {
            launch(
                driver,
                Kernel::SumRowsFinalize,
                [blocks, 1],
                config,
                &mut arguments,
            )?;
        }
    }
    Ok(())
}

unsafe fn launch(
    driver: &Driver,
    kernel: Kernel,
    grid: [u32; 2],
    config: &RowSumConfig,
    arguments: &mut [*mut c_void],
) -> Result<(), TensorError> {
    let function = driver.function(kernel)?;
    // SAFETY: caller retains allocations and arguments through copy/completion;
    // the function is context-local and uses the legacy stream like transfers.
    driver.check(
        unsafe {
            (driver.launch)(
                function as *mut c_void,
                grid[0],
                grid[1],
                1,
                config.width,
                config.height,
                1,
                0,
                std::ptr::without_provenance_mut(1),
                arguments.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        },
        "cuLaunchKernel",
    )
}

#[cfg(test)]
mod tests {
    use super::{RowSumConfig, RowSumSlice};

    fn slices(rows: usize, columns: usize) -> Vec<RowSumSlice> {
        let mut result = Vec::new();
        RowSumSlice::for_each(rows, columns, |slice| {
            result.push(slice);
            Ok(())
        })
        .unwrap();
        result
    }

    #[test]
    fn indexing_split_preserves_byte_boundary_offsets_and_accumulation_order() {
        const LIMIT: usize = 1 << 29;
        assert_eq!(slices(1, LIMIT).len(), 1);
        let split = slices(1, LIMIT + 4);
        assert_eq!(
            split,
            vec![
                RowSumSlice {
                    input_offset: 0,
                    output_offset: 0,
                    rows: 1,
                    columns: LIMIT / 2 + 2,
                    accumulate: false
                },
                RowSumSlice {
                    input_offset: LIMIT / 2 + 2,
                    output_offset: 0,
                    rows: 1,
                    columns: LIMIT / 2 + 2,
                    accumulate: true
                },
            ]
        );
        let nested = slices(1, 2 * LIMIT + 3);
        assert_eq!(
            nested.iter().map(|s| s.columns).collect::<Vec<_>>(),
            [LIMIT / 2, LIMIT / 2 + 1, LIMIT / 2 + 1, LIMIT / 2 + 1]
        );
        assert_eq!(
            nested.iter().map(|s| s.accumulate).collect::<Vec<_>>(),
            [false, true, true, true]
        );
        let mut end = 0;
        for slice in nested {
            assert_eq!(slice.input_offset, end);
            end += slice.columns;
        }
        assert_eq!(end, 2 * LIMIT + 3);
    }

    #[test]
    fn indexing_split_partitions_rows_before_columns_and_resets_each_output() {
        const LIMIT: usize = 1 << 29;
        let rows = slices(5, LIMIT / 2);
        assert_eq!(rows.iter().map(|s| s.rows).collect::<Vec<_>>(), [2, 1, 2]);
        assert_eq!(
            rows.iter().map(|s| s.output_offset).collect::<Vec<_>>(),
            [0, 2, 3]
        );
        assert!(rows.iter().all(|s| !s.accumulate));
        for (row, pair) in slices(3, LIMIT + 5).chunks_exact(2).enumerate() {
            assert_eq!(pair[0].output_offset, row);
            assert_eq!(pair[1].output_offset, row);
            assert_eq!(pair[0].input_offset, row * (LIMIT + 5));
            assert_eq!(pair[1].input_offset, row * (LIMIT + 5) + LIMIT / 2 + 2);
            assert!(!pair[0].accumulate);
            assert!(pair[1].accumulate);
            assert_eq!(pair[0].columns + pair[1].columns, LIMIT + 5);
        }
        assert_eq!(slices(0, usize::MAX).len(), 1);
        assert_eq!(slices(usize::MAX, 0).len(), 1);
    }

    #[test]
    fn geometry_uses_device_capacity_and_row_count() {
        let small = RowSumConfig::new(1, 1_000_003, 10, 1024);
        let large = RowSumConfig::new(1, 1_000_003, 120, 2048);
        assert!(small.ctas < large.ctas);
        assert!(small.reduce_y && large.vectorize);
        let many = RowSumConfig::new(32777, 1, 120, 2048);
        assert!(!many.reduce_y);
        assert_eq!(many.ctas, 1);
        assert_eq!((many.width, many.height), (1, 512));
        for rows in [1, 2, 17, 513] {
            for columns in [0, 4, 127, 128, 65539, 1_000_003] {
                let config = RowSumConfig::new(rows, columns, 120, 2048);
                assert!(config.width * config.height <= 512);
                assert!(config.width.is_power_of_two() && config.height.is_power_of_two());
            }
        }
    }
}
