//! Layout-aware float32 dimension sums. Four-level blocked accumulation keeps
//! long axes from becoming one serial dependency chain. Contiguous axes use
//! four groups of eight lanes, matching `ATen`'s float32 CPU sum ordering.

// These helpers must inline into the runtime-selected AVX2 entry point so
// fixed-size slice arithmetic uses that target's vector instructions.
#![allow(clippy::inline_always)]

#[inline(always)]
fn add<const N: usize>(left: &mut [f32; N], right: &[f32; N]) {
    for (left, right) in left.iter_mut().zip(right) {
        *left += right;
    }
}

#[inline(always)]
fn cascade<const N: usize, const CONTIGUOUS: bool>(
    values: &[f32],
    count: usize,
    step: usize,
    lane_stride: usize,
) -> [f32; N] {
    cascade_rows(count, |partial: &mut [f32; N], row| {
        let start = row * step;
        if CONTIGUOUS {
            for (sum, value) in partial.iter_mut().zip(&values[start..start + N]) {
                *sum += value;
            }
        } else {
            for lane in 0..N {
                partial[lane] += values[start + lane * lane_stride];
            }
        }
    })
}

#[inline(always)]
fn cascade_rows<const N: usize>(
    count: usize,
    mut add_row: impl FnMut(&mut [f32; N], usize),
) -> [f32; N] {
    let level_power = ((usize::BITS - count.saturating_sub(1).leading_zeros()) / 4).max(4) as usize;
    let block = 1usize << level_power;
    let mut levels = [[0.0; N]; 4];
    let mut position = 0;
    while count - position >= block {
        let mut partial = [0.0; N];
        for row in position..position + block {
            add_row(&mut partial, row);
        }
        add(&mut levels[1], &partial);
        position += block;
        for level in 1..3 {
            if (position >> (level_power * level)) & (block - 1) != 0 {
                break;
            }
            let partial = levels[level];
            add(&mut levels[level + 1], &partial);
            levels[level] = [0.0; N];
        }
    }
    for row in position..count {
        add_row(&mut levels[0], row);
    }
    let mut result = levels[0];
    for level in &levels[1..] {
        add(&mut result, level);
    }
    result
}

#[inline(always)]
fn scalar_row(values: &[f32], count: usize, stride: usize) -> f32 {
    let mut lanes = cascade::<4, false>(values, count / 4, 4 * stride, stride);
    for index in count / 4 * 4..count {
        lanes[0] += values[index * stride];
    }
    ((lanes[0] + lanes[1]) + lanes[2]) + lanes[3]
}

#[inline(always)]
fn outer_eight(values: &[f32], count: usize, stride: usize) -> [f32; 8] {
    // Each vector lane has four accumulators along the reduced axis, just
    // like scalar_row. Full 32-column tiles instead accumulate each column
    // independently; those two orders differ under cancellation and overflow.
    let mut partial = cascade_rows(count / 4, |partial: &mut [f32; 32], group| {
        for row in 0..4 {
            let start = (group * 4 + row) * stride;
            for lane in 0..8 {
                partial[row * 8 + lane] += values[start + lane];
            }
        }
    });
    for row in count / 4 * 4..count {
        for lane in 0..8 {
            partial[lane] += values[row * stride + lane];
        }
    }
    std::array::from_fn(|lane| {
        ((partial[lane] + partial[8 + lane]) + partial[16 + lane]) + partial[24 + lane]
    })
}

#[inline(always)]
fn vectorized_outer_tail(values: &[f32], output: &mut [f32], count: usize, stride: usize) {
    let mut index = 0;
    while output.len() - index >= 8 {
        let sums = outer_eight(&values[index..], count, stride);
        output[index..index + 8].copy_from_slice(&sums);
        index += 8;
    }
    for (index, value) in output.iter_mut().enumerate().skip(index) {
        *value = scalar_row(&values[index..], count, stride);
    }
}

#[inline(always)]
fn inner(values: &[f32]) -> f32 {
    if values.len() < 8 {
        return scalar_row(values, values.len(), 1);
    }
    let mut lanes = cascade::<32, true>(values, values.len() / 32, 32, 1);
    for vector in (values.len() / 32 * 32..values.len() / 8 * 8).step_by(8) {
        for lane in 0..8 {
            lanes[lane] += values[vector + lane];
        }
    }
    for lane in 0..8 {
        lanes[lane] = ((lanes[lane] + lanes[8 + lane]) + lanes[16 + lane]) + lanes[24 + lane];
    }
    let mut total = values[values.len() / 8 * 8..]
        .iter()
        .copied()
        .fold(0.0, |sum, value| sum + value);
    for value in &lanes[..8] {
        total += value;
    }
    total
}

#[inline(always)]
fn dimension_sum_impl<const CONTIGUOUS: bool>(
    values: &[f32],
    output: &mut [f32],
    count: usize,
    reduce_stride: usize,
    output_stride: usize,
) {
    if reduce_stride == 1 {
        for (index, value) in output.iter_mut().enumerate() {
            let start = index * output_stride;
            *value = inner(&values[start..start + count]);
        }
    } else if !CONTIGUOUS && reduce_stride < output_stride {
        // A strided inner reduction uses four scalar accumulators along the
        // reduced axis. A strided outer reduction groups independent outputs
        // instead. Preserve that distinction for selected rank-three views
        // whose two remaining strides are both greater than 1.
        for (index, value) in output.iter_mut().enumerate() {
            *value = scalar_row(&values[index * output_stride..], count, reduce_stride);
        }
    } else {
        // Adjacent outputs share a traversal of the reduced axis, avoiding a
        // full cache-line walk per column. Fixed lanes let LLVM vectorize adds.
        let mut index = 0;
        while output.len() - index >= 32 {
            let sums = cascade::<32, CONTIGUOUS>(
                &values[index * output_stride..],
                count,
                reduce_stride,
                output_stride,
            );
            output[index..index + 32].copy_from_slice(&sums);
            index += 32;
        }
        if CONTIGUOUS && output.len() >= 8 {
            return vectorized_outer_tail(
                &values[index..],
                &mut output[index..],
                count,
                reduce_stride,
            );
        }
        // Scalar outer reductions use groups of four outputs only when
        // the operation did not enter the eight-lane vector path.
        while output.len() - index >= 4 {
            let sums = cascade::<4, CONTIGUOUS>(
                &values[index * output_stride..],
                count,
                reduce_stride,
                output_stride,
            );
            output[index..index + 4].copy_from_slice(&sums);
            index += 4;
        }
        for (index, value) in output.iter_mut().enumerate().skip(index) {
            *value = scalar_row(&values[index * output_stride..], count, reduce_stride);
        }
    }
}

// Runtime dispatch preserves portable binaries while allowing the fixed-lane
// kernels to compile to AVX2 on capable hosts.
pub(crate) fn dimension_sum(
    values: &[f32],
    output: &mut [f32],
    count: usize,
    reduce_stride: usize,
    output_stride: usize,
) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: checked the required CPU capability; all accesses use slices.
        #[allow(unsafe_code)]
        unsafe {
            return avx2_sum(values, output, count, reduce_stride, output_stride);
        }
    }
    dispatch_layout(values, output, count, reduce_stride, output_stride);
}

#[inline(always)]
fn dispatch_layout(
    values: &[f32],
    output: &mut [f32],
    count: usize,
    reduce_stride: usize,
    output_stride: usize,
) {
    if output_stride == 1 {
        dimension_sum_impl::<true>(values, output, count, reduce_stride, output_stride);
    } else {
        dimension_sum_impl::<false>(values, output, count, reduce_stride, output_stride);
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
#[target_feature(enable = "avx2")]
unsafe fn avx2_sum(
    values: &[f32],
    output: &mut [f32],
    count: usize,
    reduce_stride: usize,
    output_stride: usize,
) {
    if reduce_stride == 1 {
        for (index, value) in output.iter_mut().enumerate() {
            let start = index * output_stride;
            // SAFETY: the caller checked AVX2 and the row slice bounds every load.
            *value = unsafe { inner_avx2(&values[start..start + count]) };
        }
    } else if output_stride == 1 {
        let mut index = 0;
        while output.len() - index >= 32 {
            // SAFETY: AVX2 is enabled and the helper checks the source bounds.
            // The destination tile contains 32 writable floats.
            unsafe {
                let sums = cascade_avx2(&values[index..], count, reduce_stride);
                for (lane, sum) in sums.iter().enumerate() {
                    std::arch::x86_64::_mm256_storeu_ps(
                        output.as_mut_ptr().add(index + lane * 8),
                        *sum,
                    );
                }
            }
            index += 32;
        }
        if output.len() < 8 {
            dimension_sum_impl::<true>(values, output, count, reduce_stride, 1);
        } else {
            vectorized_outer_tail(&values[index..], &mut output[index..], count, reduce_stride);
        }
    } else {
        dispatch_layout(values, output, count, reduce_stride, output_stride);
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
#[target_feature(enable = "avx2")]
unsafe fn inner_avx2(values: &[f32]) -> f32 {
    use std::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_storeu_ps};
    if values.len() < 8 {
        return scalar_row(values, values.len(), 1);
    }
    let groups = values.len() / 32;
    // SAFETY: AVX2 is enabled; the helper validates all full-vector loads.
    unsafe {
        let mut partial = cascade_avx2(values, groups, 32);
        for index in (groups * 32..values.len() / 8 * 8).step_by(8) {
            partial[0] = _mm256_add_ps(partial[0], _mm256_loadu_ps(values.as_ptr().add(index)));
        }
        let result = _mm256_add_ps(
            _mm256_add_ps(_mm256_add_ps(partial[0], partial[1]), partial[2]),
            partial[3],
        );
        let mut lanes = [0.0_f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), result);
        let mut total = values[values.len() / 8 * 8..]
            .iter()
            .copied()
            .fold(0.0, |sum, value| sum + value);
        for lane in lanes {
            total += lane;
        }
        total
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
#[target_feature(enable = "avx2")]
unsafe fn cascade_avx2(
    values: &[f32],
    groups: usize,
    step: usize,
) -> [std::arch::x86_64::__m256; 4] {
    use std::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_setzero_ps};
    assert!(
        groups == 0
            || (groups - 1)
                .checked_mul(step)
                .and_then(|start| start.checked_add(32))
                .is_some_and(|end| end <= values.len())
    );
    let power = ((usize::BITS - groups.saturating_sub(1).leading_zeros()) / 4).max(4);
    let block = 1usize << power;
    let zero = _mm256_setzero_ps();
    let mut low = [zero; 4];
    let mut middle = [zero; 4];
    let mut high = [zero; 4];
    let mut position = 0;
    // SAFETY: all pointer offsets are within full 32-element groups (or the
    // full 8-element tail); output is an initialized eight-element array.
    unsafe {
        while groups - position >= block {
            let mut partial = [zero; 4];
            for group in position..position + block {
                for (lane, sum) in partial.iter_mut().enumerate() {
                    *sum = _mm256_add_ps(
                        *sum,
                        _mm256_loadu_ps(values.as_ptr().add(group * step + lane * 8)),
                    );
                }
            }
            for lane in 0..4 {
                low[lane] = _mm256_add_ps(low[lane], partial[lane]);
            }
            position += block;
            if (position >> power) & (block - 1) == 0 {
                for lane in 0..4 {
                    middle[lane] = _mm256_add_ps(middle[lane], low[lane]);
                }
                low = [zero; 4];
                if (position >> (2 * power)) & (block - 1) == 0 {
                    for lane in 0..4 {
                        high[lane] = _mm256_add_ps(high[lane], middle[lane]);
                    }
                    middle = [zero; 4];
                }
            }
        }
        let mut partial = [zero; 4];
        for group in position..groups {
            for (lane, sum) in partial.iter_mut().enumerate() {
                *sum = _mm256_add_ps(
                    *sum,
                    _mm256_loadu_ps(values.as_ptr().add(group * step + lane * 8)),
                );
            }
        }
        for lane in 0..4 {
            partial[lane] = _mm256_add_ps(
                _mm256_add_ps(_mm256_add_ps(partial[lane], low[lane]), middle[lane]),
                high[lane],
            );
        }
        partial
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn outer_remainder_cancellation_matches_reference() {
        for width in 2..96 {
            let values: Vec<f32> = [-1e8, 1.0, 1.0, 1.0, 1.0, 1e8, 1.0]
                .into_iter()
                .flat_map(|value| std::iter::repeat_n(value, width))
                .collect();
            let independent_columns = if width < 8 {
                width / 4 * 4
            } else {
                width / 32 * 32
            };
            let expected: Vec<f32> = (0..width)
                .map(|index| {
                    if index < independent_columns {
                        1.0
                    } else {
                        4.0
                    }
                })
                .collect();
            let mut actual = vec![0.0; width];
            super::dimension_sum(&values, &mut actual, 7, width, 1);
            assert_eq!(actual, expected, "dispatched width {width}");
            super::dispatch_layout(&values, &mut actual, 7, width, 1);
            assert_eq!(actual, expected, "portable width {width}");
        }
    }

    #[test]
    fn dispatched_and_portable_reductions_agree() {
        for count in [1, 5, 16, 33, 257, 4097] {
            for outputs in [1, 5, 32, 37] {
                let values: Vec<f32> = (0..count * outputs)
                    .map(|index| [1e8, 1.0, -1e8, 1.0, -3.0][index % 5])
                    .collect();
                for (reduce_stride, output_stride) in [(1, count), (outputs, 1)] {
                    let mut actual = vec![0.0; outputs];
                    let mut expected = vec![0.0; outputs];
                    super::dimension_sum(&values, &mut actual, count, reduce_stride, output_stride);
                    super::dispatch_layout(
                        &values,
                        &mut expected,
                        count,
                        reduce_stride,
                        output_stride,
                    );
                    assert_eq!(actual, expected);
                }
            }
        }
    }
}
