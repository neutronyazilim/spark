use super::f16::f16_to_f32;

pub const QK: usize = 32;
pub const QK_PTQ: usize = 128;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockQ2_0 {
    pub d: u16,        // f16 scale (2 bytes)
    pub qs: [u8; 8],   // 32 packed 2-bit weights (8 bytes)
}

impl BlockQ2_0 {
    pub const BYTE_SIZE: usize = 10;

    #[inline(always)]
    pub fn scale(self) -> f32 {
        f16_to_f32(self.d)
    }

    #[inline(always)]
    pub fn weight(self, index: usize) -> f32 {
        let byte = self.qs[index / 4];
        let code = (byte >> ((index % 4) * 2)) & 0x03;
        match code {
            1 => 1.0,
            2 => -1.0,
            _ => 0.0,
        }
    }
}

/// Official Prism ML / BitNet PTQ1_0 format: 128 weights per 28-byte block (1.75 bpw)
#[repr(C, packed)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockPTQ1_0 {
    pub qs: [u8; 24], // 120 trits packed into 24 bytes (5 trits per byte in base-3)
    pub qh: [u8; 2],  // 8 trits packed into 2 bytes (4 trits per byte, 2 bits each)
    pub d: u16,       // FP16 group scale
}

const fn make_trit_lut() -> [[f32; 5]; 256] {
    let mut lut = [[0.0f32; 5]; 256];
    let mut i = 0;
    while i < 256 {
        let mut v = i as u8;
        let mut j = 0;
        while j < 5 {
            let d = (v % 3) as i32 - 1;
            lut[i][j] = d as f32;
            v /= 3;
            j += 1;
        }
        i += 1;
    }
    lut
}

pub static TRIT_LUT: [[f32; 5]; 256] = make_trit_lut();

impl BlockPTQ1_0 {
    pub const BYTE_SIZE: usize = 28;

    #[inline(always)]
    pub fn scale(self) -> f32 {
        f16_to_f32(self.d)
    }

    #[inline(always)]
    pub fn weight(self, index: usize) -> f32 {
        let s = self.scale();
        if index < 120 {
            let byte_idx = index / 5;
            let trit_idx = index % 5;
            let byte_val = self.qs[byte_idx];
            TRIT_LUT[byte_val as usize][trit_idx] * s
        } else {
            let sub = index - 120;
            let qh_byte = self.qh[sub / 4];
            let code = (qh_byte >> ((sub % 4) * 2)) & 3;
            let val = match code {
                1 => 1.0,
                2 => -1.0,
                _ => 0.0,
            };
            val * s
        }
    }
}

pub fn gemv_ptq1_0(
    weights: &[BlockPTQ1_0],
    activations: &[f32],
    output: &mut [f32],
    blocks_per_row: usize,
) {
    for (row, result) in output.iter_mut().enumerate() {
        let start = row * blocks_per_row;
        if start + blocks_per_row <= weights.len() {
            let row_blocks = &weights[start..start + blocks_per_row];
            let mut total = 0.0f32;
            for (b_idx, block) in row_blocks.iter().enumerate() {
                let act = &activations[b_idx * QK_PTQ..(b_idx + 1) * QK_PTQ];
                let s = block.scale();
                let mut sum = 0.0f32;

                // 24 bytes x 5 trits = 120 trits via precomputed LUT (zero division!)
                for byte_idx in 0..24 {
                    let trits = &TRIT_LUT[block.qs[byte_idx] as usize];
                    let base_act = byte_idx * 5;
                    sum += trits[0] * act[base_act]
                        + trits[1] * act[base_act + 1]
                        + trits[2] * act[base_act + 2]
                        + trits[3] * act[base_act + 3]
                        + trits[4] * act[base_act + 4];
                }

                // Last 8 trits (2 bytes x 4 trits)
                for qh_idx in 0..2 {
                    let qh_byte = block.qh[qh_idx];
                    let base_act = 120 + qh_idx * 4;
                    for t in 0..4 {
                        let code = (qh_byte >> (t * 2)) & 3;
                        let sign = match code {
                            1 => 1.0,
                            2 => -1.0,
                            _ => 0.0,
                        };
                        sum += sign * act[base_act + t];
                    }
                }

                total += sum * s;
            }
            *result = total;
        }
    }
}

#[inline]
pub fn dot_ternary(blocks: &[BlockQ2_0], activations: &[f32]) -> f32 {
    let mut total = 0.0f32;
    for (b_idx, block) in blocks.iter().enumerate() {
        let act = &activations[b_idx * QK..(b_idx + 1) * QK];
        let mut sum = 0.0f32;

        for (i, &x) in act.iter().enumerate() {
            let byte = block.qs[i / 4];
            let code = (byte >> ((i % 4) * 2)) & 0x03;
            if code == 1 {
                sum += x;
            } else if code == 2 {
                sum -= x;
            }
        }
        total += sum * block.scale();
    }
    total
}

pub fn gemv_ternary(
    weights: &[BlockQ2_0],
    activations: &[f32],
    output: &mut [f32],
    blocks_per_row: usize,
) {
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        unsafe {
            gemv_ternary_avx2(weights, activations, output, blocks_per_row);
        }
    } else {
        for (row, result) in output.iter_mut().enumerate() {
            let start = row * blocks_per_row;
            if start + blocks_per_row <= weights.len() {
                *result = dot_ternary(&weights[start..start + blocks_per_row], activations);
            }
        }
    }
}

/// # Safety
///
/// The caller must execute this function only when AVX2 and FMA are available.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemv_ternary_avx2(
    weights: &[BlockQ2_0],
    activations: &[f32],
    output: &mut [f32],
    blocks_per_row: usize,
) {
    use std::arch::x86_64::*;

    let rows = output.len();
    let unroll_limit = rows.saturating_sub(rows % 4);

    let mut row = 0;
    while row < unroll_limit {
        let r0 = row;
        let r1 = row + 1;
        let r2 = row + 2;
        let r3 = row + 3;

        let max_block_idx = (r3 + 1) * blocks_per_row;
        if max_block_idx > weights.len() {
            break;
        }

        let w0 = &weights[r0 * blocks_per_row..(r0 + 1) * blocks_per_row];
        let w1 = &weights[r1 * blocks_per_row..(r1 + 1) * blocks_per_row];
        let w2 = &weights[r2 * blocks_per_row..(r2 + 1) * blocks_per_row];
        let w3 = &weights[r3 * blocks_per_row..(r3 + 1) * blocks_per_row];

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        for b in 0..blocks_per_row {
            let b0 = &w0[b];
            let b1 = &w1[b];
            let b2 = &w2[b];
            let b3 = &w3[b];

            let s0 = _mm256_set1_ps(b0.scale());
            let s1 = _mm256_set1_ps(b1.scale());
            let s2 = _mm256_set1_ps(b2.scale());
            let s3 = _mm256_set1_ps(b3.scale());

            let act_chunk = &activations[b * QK..(b + 1) * QK];

            for c in 0..4 {
                let x = _mm256_loadu_ps(act_chunk.as_ptr().add(c * 8));

                let signs0 = unpack_ternary_signs(b0.qs.as_ptr(), c);
                let signs1 = unpack_ternary_signs(b1.qs.as_ptr(), c);
                let signs2 = unpack_ternary_signs(b2.qs.as_ptr(), c);
                let signs3 = unpack_ternary_signs(b3.qs.as_ptr(), c);

                acc0 = _mm256_fmadd_ps(_mm256_mul_ps(signs0, s0), x, acc0);
                acc1 = _mm256_fmadd_ps(_mm256_mul_ps(signs1, s1), x, acc1);
                acc2 = _mm256_fmadd_ps(_mm256_mul_ps(signs2, s2), x, acc2);
                acc3 = _mm256_fmadd_ps(_mm256_mul_ps(signs3, s3), x, acc3);
            }
        }

        output[r0] = horizontal_sum(acc0);
        output[r1] = horizontal_sum(acc1);
        output[r2] = horizontal_sum(acc2);
        output[r3] = horizontal_sum(acc3);

        row += 4;
    }

    while row < rows && (row + 1) * blocks_per_row <= weights.len() {
        let start = row * blocks_per_row;
        output[row] = dot_ternary(&weights[start..start + blocks_per_row], activations);
        row += 1;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn unpack_ternary_signs(qs: *const u8, chunk_idx: usize) -> std::arch::x86_64::__m256 {
    let pair = *qs.add(chunk_idx * 2) as u32 | ((*qs.add(chunk_idx * 2 + 1) as u32) << 8);
    let mut signs = [0.0f32; 8];
    for (i, sign) in signs.iter_mut().enumerate() {
        let code = (pair >> (i * 2)) & 0x03;
        *sign = if code == 1 {
            1.0
        } else if code == 2 {
            -1.0
        } else {
            0.0
        };
    }
    std::arch::x86_64::_mm256_loadu_ps(signs.as_ptr())
}

#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum(value: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let halves = _mm256_add_ps(value, _mm256_permute2f128_ps(value, value, 1));
    let shuffled = _mm256_shuffle_ps(halves, halves, 0x4e);
    let sums = _mm256_add_ps(halves, shuffled);
    let shuffled = _mm256_shuffle_ps(sums, sums, 0xb1);
    _mm_cvtss_f32(_mm256_castps256_ps128(_mm256_add_ps(sums, shuffled)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ternary_encoding_matches_expected_values() {
        let blk = BlockQ2_0 {
            d: 0x3c00, // 1.0f in f16
            qs: [0b0001_1001, 0, 0, 0, 0, 0, 0, 0], // i=0: 1, i=1: 2 (-1), i=2: 1, i=3: 0
        };
        assert_eq!(blk.weight(0), 1.0);
        assert_eq!(blk.weight(1), -1.0);
        assert_eq!(blk.weight(2), 1.0);
        assert_eq!(blk.weight(3), 0.0);
    }

    #[test]
    fn ptq1_0_block_unpack_matches_specification() {
        let blk = BlockPTQ1_0 {
            qs: [23; 24], // 23 in base 3: 2 + 1*3 + 2*9 -> [-1, 0, 1, -1, -1]
            qh: [0b1001_1001; 2],
            d: 0x3c00, // 1.0f
        };
        assert!((blk.scale() - 1.0).abs() < 1e-4);
        assert_eq!(blk.weight(0), 1.0);
        assert_eq!(blk.weight(1), 0.0);
        assert_eq!(blk.weight(2), 1.0);
    }
}
