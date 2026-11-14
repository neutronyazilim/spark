pub mod avx;
pub mod avx2;
pub mod f16;
pub mod fallback;
pub mod pool;
pub mod q4_0;
pub mod q8_0;
pub mod ternary;

use crate::profile::ProfileStats;
use q4_0::BlockQ4_0;
use q8_0::BlockQ8_0;

const QK: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemvError {
    ColumnsNotMultipleOfBlock { columns: usize },
    ActivationLength { expected: usize, actual: usize },
    OutputLength { expected: usize, actual: usize },
    WeightLength { expected: usize, actual: usize },
}

impl std::fmt::Display for GemvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ColumnsNotMultipleOfBlock { columns } => {
                write!(f, "column count {columns} is not a multiple of {QK}")
            }
            Self::ActivationLength { expected, actual } => {
                write!(f, "activation length {actual}, expected {expected}")
            }
            Self::OutputLength { expected, actual } => {
                write!(f, "output length {actual}, expected {expected}")
            }
            Self::WeightLength { expected, actual } => {
                write!(f, "weight block length {actual}, expected {expected}")
            }
        }
    }
}

impl std::error::Error for GemvError {}

pub fn gemv_q8_0(
    weights: &[BlockQ8_0],
    activations: &[f32],
    output: &mut [f32],
    columns: usize,
) -> Result<(), GemvError> {
    let blocks_per_row = validate_gemv(weights.len(), activations.len(), output.len(), columns)?;
    if std::env::var_os("SPARK_SCALAR").is_some() {
        fallback::gemv_q8_0(weights, activations, output, blocks_per_row);
    } else if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        // SAFETY: feature detection above guarantees the target features.
        unsafe {
            avx2::gemv_q8_0(weights, activations, output, blocks_per_row);
        }
    } else if std::is_x86_feature_detected!("avx") {
        // SAFETY: feature detection above guarantees AVX support.
        unsafe {
            avx::gemv_q8_0(weights, activations, output, blocks_per_row);
        }
    } else {
        fallback::gemv_q8_0(weights, activations, output, blocks_per_row);
    }
    Ok(())
}

pub fn gemv_q4_0(
    weights: &[BlockQ4_0],
    activations: &[f32],
    output: &mut [f32],
    columns: usize,
) -> Result<(), GemvError> {
    let blocks_per_row = validate_gemv(weights.len(), activations.len(), output.len(), columns)?;
    if std::env::var_os("SPARK_SCALAR").is_some() {
        fallback::gemv_q4_0(weights, activations, output, blocks_per_row);
    } else if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        // SAFETY: feature detection above guarantees the target features.
        unsafe {
            avx2::gemv_q4_0(weights, activations, output, blocks_per_row);
        }
    } else if std::is_x86_feature_detected!("avx") {
        // SAFETY: feature detection above guarantees AVX support.
        unsafe {
            avx::gemv_q4_0(weights, activations, output, blocks_per_row);
        }
    } else {
        fallback::gemv_q4_0(weights, activations, output, blocks_per_row);
    }
    Ok(())
}

pub fn gemv_q8_0_parallel(
    pool: &pool::ThreadPool,
    weights: &[BlockQ8_0],
    activations: &[f32],
    output: &mut [f32],
    columns: usize,
    profile: Option<&mut ProfileStats>,
) -> Result<(), GemvError> {
    validate_gemv(weights.len(), activations.len(), output.len(), columns)?;
    pool.gemv_q8_0(weights, activations, output, columns, profile);
    Ok(())
}

pub fn gemv_q4_0_parallel(
    pool: &pool::ThreadPool,
    weights: &[BlockQ4_0],
    activations: &[f32],
    output: &mut [f32],
    columns: usize,
    profile: Option<&mut ProfileStats>,
) -> Result<(), GemvError> {
    validate_gemv(weights.len(), activations.len(), output.len(), columns)?;
    pool.gemv_q4_0(weights, activations, output, columns, profile);
    Ok(())
}

unsafe fn run_q8_rows(
    weights: &[BlockQ8_0],
    activations: &[f32],
    output: &mut [f32],
    blocks_per_row: usize,
) {
    if std::env::var_os("SPARK_SCALAR").is_some() {
        fallback::gemv_q8_0(weights, activations, output, blocks_per_row);
    } else if std::env::var_os("SPARK_FORCE_AVX").is_some() {
        avx::gemv_q8_0(weights, activations, output, blocks_per_row);
    } else if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        avx2::gemv_q8_0(weights, activations, output, blocks_per_row);
    } else if std::is_x86_feature_detected!("avx") {
        avx::gemv_q8_0(weights, activations, output, blocks_per_row);
    } else {
        fallback::gemv_q8_0(weights, activations, output, blocks_per_row);
    }
}

unsafe fn run_fused_ffn_q8_rows(
    gate_weights: &[BlockQ8_0],
    up_weights: &[BlockQ8_0],
    activations: &[f32],
    output: &mut [f32],
    blocks_per_row: usize,
) {
    if std::env::var_os("SPARK_SCALAR").is_some() {
        fallback::fused_ffn_q8_0(gate_weights, up_weights, activations, output, blocks_per_row);
    } else if std::env::var_os("SPARK_FORCE_AVX").is_some() {
        avx::fused_ffn_q8_0(gate_weights, up_weights, activations, output, blocks_per_row);
    } else if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        avx2::fused_ffn_q8_0(gate_weights, up_weights, activations, output, blocks_per_row);
    } else if std::is_x86_feature_detected!("avx") {
        avx::fused_ffn_q8_0(gate_weights, up_weights, activations, output, blocks_per_row);
    } else {
        fallback::fused_ffn_q8_0(gate_weights, up_weights, activations, output, blocks_per_row);
    }
}

unsafe fn run_q4_rows(
    weights: &[BlockQ4_0],
    activations: &[f32],
    output: &mut [f32],
    blocks_per_row: usize,
) {
    if std::env::var_os("SPARK_SCALAR").is_some() {
        fallback::gemv_q4_0(weights, activations, output, blocks_per_row);
    } else if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        avx2::gemv_q4_0(weights, activations, output, blocks_per_row);
    } else if std::is_x86_feature_detected!("avx") {
        avx::gemv_q4_0(weights, activations, output, blocks_per_row);
    } else {
        fallback::gemv_q4_0(weights, activations, output, blocks_per_row);
    }
}

#[allow(dead_code)]
unsafe fn run_fused_ffn_q4_rows(
    gate_weights: &[BlockQ4_0],
    up_weights: &[BlockQ4_0],
    activations: &[f32],
    output: &mut [f32],
    blocks_per_row: usize,
) {
    if std::env::var_os("SPARK_SCALAR").is_some() {
        fallback::fused_ffn_q4_0(gate_weights, up_weights, activations, output, blocks_per_row);
    } else if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        avx2::fused_ffn_q4_0(gate_weights, up_weights, activations, output, blocks_per_row);
    } else {
        fallback::fused_ffn_q4_0(gate_weights, up_weights, activations, output, blocks_per_row);
    }
}

fn validate_gemv(
    weight_blocks: usize,
    activation_len: usize,
    output_len: usize,
    columns: usize,
) -> Result<usize, GemvError> {
    if !columns.is_multiple_of(QK) {
        return Err(GemvError::ColumnsNotMultipleOfBlock { columns });
    }
    if activation_len != columns {
        return Err(GemvError::ActivationLength {
            expected: columns,
            actual: activation_len,
        });
    }
    if weight_blocks != output_len * (columns / QK) {
        return Err(GemvError::WeightLength {
            expected: output_len * (columns / QK),
            actual: weight_blocks,
        });
    }
    if output_len == 0 {
        return Err(GemvError::OutputLength {
            expected: 1,
            actual: 0,
        });
    }
    Ok(columns / QK)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q8_blocks(rows: usize, columns: usize) -> Vec<BlockQ8_0> {
        let blocks = rows * columns / QK;
        (0..blocks)
            .map(|block| {
                let mut result = BlockQ8_0 {
                    d: 0x3c00,
                    qs: [0; QK],
                };
                for (i, q) in result.qs.iter_mut().enumerate() {
                    *q = ((block * QK + i) as i8).wrapping_mul(3);
                }
                result
            })
            .collect()
    }

    fn q4_blocks(rows: usize, columns: usize) -> Vec<BlockQ4_0> {
        let blocks = rows * columns / QK;
        (0..blocks)
            .map(|block| {
                let mut result = BlockQ4_0 {
                    d: 0x3c00,
                    qs: [0; 16],
                };
                for (i, q) in result.qs.iter_mut().enumerate() {
                    *q = ((block + i) as u8 & 0x0f) | (((15 - i) as u8) << 4);
                }
                result
            })
            .collect()
    }

    fn assert_close(left: &[f32], right: &[f32]) {
        assert_eq!(left.len(), right.len());
        for (index, (&a, &b)) in left.iter().zip(right).enumerate() {
            let tolerance = 1e-5 * a.abs().max(b.abs()).max(1.0);
            assert!(
                (a - b).abs() <= tolerance,
                "index {index}: {a} != {b}, tolerance {tolerance}"
            );
        }
    }

    #[test]
    fn q8_dispatch_matches_scalar_reference() {
        let weights = q8_blocks(3, 64);
        let activations: Vec<_> = (0..64).map(|i| (i as f32 - 20.0) * 0.125).collect();
        let mut dispatch = [0.0; 3];
        let mut scalar = [0.0; 3];
        gemv_q8_0(&weights, &activations, &mut dispatch, 64).unwrap();
        fallback::gemv_q8_0(&weights, &activations, &mut scalar, 2);
        assert_close(&dispatch, &scalar);
    }

    #[test]
    fn q8_explicit_simd_kernels_match_scalar_reference() {
        let weights = q8_blocks(3, 64);
        let activations: Vec<_> = (0..64).map(|i| (i as f32 - 20.0) * 0.125).collect();
        let mut scalar = [0.0; 3];
        fallback::gemv_q8_0(&weights, &activations, &mut scalar, 2);

        if std::is_x86_feature_detected!("avx") {
            let mut avx = [0.0; 3];
            // SAFETY: the runtime check above guarantees AVX support.
            unsafe {
                avx::gemv_q8_0(&weights, &activations, &mut avx, 2);
            }
            assert_close(&avx, &scalar);
        }
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            let mut avx2 = [0.0; 3];
            // SAFETY: the runtime checks above guarantee AVX2 and FMA support.
            unsafe {
                avx2::gemv_q8_0(&weights, &activations, &mut avx2, 2);
            }
            assert_close(&avx2, &scalar);
        }
    }

    #[test]
    fn q4_dispatch_matches_scalar_reference() {
        let weights = q4_blocks(3, 64);
        let activations: Vec<_> = (0..64).map(|i| (i as f32 - 20.0) * 0.125).collect();
        let mut dispatch = [0.0; 3];
        let mut scalar = [0.0; 3];
        gemv_q4_0(&weights, &activations, &mut dispatch, 64).unwrap();
        fallback::gemv_q4_0(&weights, &activations, &mut scalar, 2);
        assert_close(&dispatch, &scalar);
    }

    #[test]
    fn q4_explicit_simd_kernels_match_scalar_reference() {
        let weights = q4_blocks(4, 64);
        let activations: Vec<_> = (0..64).map(|i| (i as f32 - 20.0) * 0.125).collect();
        let mut scalar = [0.0; 4];
        fallback::gemv_q4_0(&weights, &activations, &mut scalar, 2);

        if std::is_x86_feature_detected!("avx") {
            let mut avx = [0.0; 4];
            // SAFETY: the runtime check above guarantees AVX support.
            unsafe {
                avx::gemv_q4_0(&weights, &activations, &mut avx, 2);
            }
            assert_close(&avx, &scalar);
        }
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            let mut avx2 = [0.0; 4];
            // SAFETY: the runtime checks above guarantee AVX2 and FMA support.
            unsafe {
                avx2::gemv_q4_0(&weights, &activations, &mut avx2, 2);
            }
            assert_close(&avx2, &scalar);
        }
    }

    #[test]
    fn rejects_invalid_shapes_without_writing_output() {
        let weights = [BlockQ8_0 { d: 0, qs: [0; QK] }];
        let mut output = [7.0];
        assert_eq!(
            gemv_q8_0(&weights, &[0.0; 31], &mut output, 31),
            Err(GemvError::ColumnsNotMultipleOfBlock { columns: 31 })
        );
        assert_eq!(output, [7.0]);
    }
}
