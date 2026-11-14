use super::{f16::{f16_to_f32, f32_to_f16}, q8_0::BlockQ8_0};

pub const QK: usize = 128;

/// GGML Q1_0: one FP16 scale and 128 bipolar sign bits.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockQ1_0 {
    pub d: u16,
    pub qs: [u8; 16],
}

impl BlockQ1_0 {
    pub const BYTE_SIZE: usize = 18;

    #[inline(always)]
    pub fn scale(self) -> f32 {
        f16_to_f32(self.d)
    }

    #[inline(always)]
    pub fn value(self, index: usize) -> f32 {
        let bit = (self.qs[index / 8] >> (index % 8)) & 1;
        if bit != 0 { self.scale() } else { -self.scale() }
    }
}

const fn make_sign_lut() -> [[f32; 8]; 256] {
    let mut lut = [[0.0; 8]; 256];
    let mut byte = 0;
    while byte < 256 {
        let mut bit = 0;
        while bit < 8 {
            lut[byte][bit] = if (byte >> bit) & 1 != 0 { 1.0 } else { -1.0 };
            bit += 1;
        }
        byte += 1;
    }
    lut
}

static SIGN_LUT: [[f32; 8]; 256] = make_sign_lut();

const fn make_sign_i8_lut() -> [[i8; 8]; 256] {
    let mut lut = [[0; 8]; 256];
    let mut byte = 0;
    while byte < 256 {
        let mut bit = 0;
        while bit < 8 {
            lut[byte][bit] = if (byte >> bit) & 1 != 0 { 1 } else { -1 };
            bit += 1;
        }
        byte += 1;
    }
    lut
}

static SIGN_I8_LUT: [[i8; 8]; 256] = make_sign_i8_lut();

pub fn quantize_activations_q8_0(input: &[f32]) -> Vec<BlockQ8_0> {
    let mut blocks = Vec::with_capacity(input.len() / 32);
    for values in input.as_chunks::<32>().0 {
        let amax = values.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let d_bits = f32_to_f16(d);
        // GGML quantizes with the original FP32 reciprocal, then stores a
        // rounded FP16 scale for the dot product.
        let inv = if d != 0.0 { d.recip() } else { 0.0 };
        let mut qs = [0i8; 32];
        for (q, &value) in qs.iter_mut().zip(values) {
            *q = (value * inv).round().clamp(-127.0, 127.0) as i8;
        }
        blocks.push(BlockQ8_0 { d: d_bits, qs });
    }
    blocks
}

#[inline]
pub fn dot_q1_0_q8_0(blocks: &[BlockQ1_0], input: &[BlockQ8_0]) -> f32 {
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        unsafe { dot_q1_0_q8_0_avx2(blocks, input) }
    } else {
        let mut total = 0.0f32;
        for (block_index, block) in blocks.iter().enumerate() {
            let d1 = f16_to_f32(block.d);
            for chunk in 0..4 {
                let q8 = &input[block_index * 4 + chunk];
                let mut sum = 0i32;
                for (i, &quant) in q8.qs.iter().enumerate() {
                    let bit_index = chunk * 32 + i;
                    let sign = if (block.qs[bit_index / 8] >> (bit_index % 8)) & 1 != 0 { 1 } else { -1 };
                    sum += sign * quant as i32;
                }
                total += d1 * f16_to_f32(q8.d) * sum as f32;
            }
        }
        total
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q1_0_q8_0_avx2(blocks: &[BlockQ1_0], input: &[BlockQ8_0]) -> f32 {
    use std::arch::x86_64::*;

    let ones8 = _mm256_set1_epi8(1);
    let ones16 = _mm256_set1_epi16(1);
    let mut acc = _mm256_setzero_ps();
    for (block_index, block) in blocks.iter().enumerate() {
        let d1 = f16_to_f32(block.d);
        for chunk in 0..4 {
            let byte = chunk * 4;
            let s0 = std::ptr::read_unaligned(SIGN_I8_LUT[block.qs[byte] as usize].as_ptr().cast::<i64>());
            let s1 = std::ptr::read_unaligned(SIGN_I8_LUT[block.qs[byte + 1] as usize].as_ptr().cast::<i64>());
            let s2 = std::ptr::read_unaligned(SIGN_I8_LUT[block.qs[byte + 2] as usize].as_ptr().cast::<i64>());
            let s3 = std::ptr::read_unaligned(SIGN_I8_LUT[block.qs[byte + 3] as usize].as_ptr().cast::<i64>());
            let signs = _mm256_set_epi64x(s3, s2, s1, s0);
            let q8 = &input[block_index * 4 + chunk];
            let values = _mm256_loadu_si256(q8.qs.as_ptr().cast::<__m256i>());
            let signed = _mm256_sign_epi8(values, signs);
            let sums16 = _mm256_maddubs_epi16(ones8, signed);
            let sums32 = _mm256_madd_epi16(sums16, ones16);
            let scale = _mm256_set1_ps(d1 * f16_to_f32(q8.d));
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sums32), scale, acc);
        }
    }
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let sum = _mm_add_ps(lo, hi);
    let sum = _mm_hadd_ps(sum, sum);
    let sum = _mm_hadd_ps(sum, sum);
    _mm_cvtss_f32(sum)
}

#[inline]
pub fn dot_q1_0(blocks: &[BlockQ1_0], input: &[f32]) -> f32 {
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        unsafe { dot_q1_0_avx2(blocks, input) }
    } else {
        dot_q1_0_scalar(blocks, input)
    }
}

#[inline]
fn dot_q1_0_scalar(blocks: &[BlockQ1_0], input: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (block_index, block) in blocks.iter().copied().enumerate() {
        let x = &input[block_index * QK..(block_index + 1) * QK];
        let scale = block.scale();
        for (byte_index, &packed) in block.qs.iter().enumerate() {
            for bit in 0..8 {
                let sign = if (packed >> bit) & 1 != 0 { scale } else { -scale };
                sum = sign.mul_add(x[byte_index * 8 + bit], sum);
            }
        }
    }
    sum
}

/// # Safety
/// Caller must ensure AVX2 and FMA are available.
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q1_0_avx2(blocks: &[BlockQ1_0], input: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    for (block_index, block) in blocks.iter().enumerate() {
        let scale = _mm256_set1_ps(f16_to_f32(block.d));
        let x = input.as_ptr().add(block_index * QK);
        for byte in (0..16).step_by(4) {
            let s0 = _mm256_mul_ps(
                _mm256_loadu_ps(SIGN_LUT[block.qs[byte] as usize].as_ptr()),
                scale,
            );
            let s1 = _mm256_mul_ps(
                _mm256_loadu_ps(SIGN_LUT[block.qs[byte + 1] as usize].as_ptr()),
                scale,
            );
            let s2 = _mm256_mul_ps(
                _mm256_loadu_ps(SIGN_LUT[block.qs[byte + 2] as usize].as_ptr()),
                scale,
            );
            let s3 = _mm256_mul_ps(
                _mm256_loadu_ps(SIGN_LUT[block.qs[byte + 3] as usize].as_ptr()),
                scale,
            );
            acc0 = _mm256_fmadd_ps(s0, _mm256_loadu_ps(x.add(byte * 8)), acc0);
            acc1 = _mm256_fmadd_ps(s1, _mm256_loadu_ps(x.add((byte + 1) * 8)), acc1);
            acc2 = _mm256_fmadd_ps(s2, _mm256_loadu_ps(x.add((byte + 2) * 8)), acc2);
            acc3 = _mm256_fmadd_ps(s3, _mm256_loadu_ps(x.add((byte + 3) * 8)), acc3);
        }
    }

    let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let sum = _mm_add_ps(lo, hi);
    let sum = _mm_hadd_ps(sum, sum);
    let sum = _mm_hadd_ps(sum, sum);
    _mm_cvtss_f32(sum)
}

pub fn gemv_q1_0_rows(
    weights: &[BlockQ1_0],
    input: &[f32],
    output: &mut [f32],
    blocks_per_row: usize,
) {
    for (row, value) in output.iter_mut().enumerate() {
        let start = row * blocks_per_row;
        *value = dot_q1_0(&weights[start..start + blocks_per_row], input);
    }
}

pub fn gemv_q1_0_q8_0_rows(
    weights: &[BlockQ1_0],
    input: &[BlockQ8_0],
    output: &mut [f32],
    blocks_per_row: usize,
) {
    for (row, value) in output.iter_mut().enumerate() {
        let start = row * blocks_per_row;
        *value = dot_q1_0_q8_0(&weights[start..start + blocks_per_row], input);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_bipolar_bits() {
        let mut block = BlockQ1_0 { d: 0x3c00, qs: [0; 16] };
        block.qs[0] = 0b1000_0101;
        assert_eq!(block.value(0), 1.0);
        assert_eq!(block.value(1), -1.0);
        assert_eq!(block.value(2), 1.0);
        assert_eq!(block.value(7), 1.0);
        assert_eq!(block.value(8), -1.0);
    }

    #[test]
    fn avx2_matches_scalar() {
        let blocks = vec![BlockQ1_0 { d: 0x3800, qs: [0x5a; 16] }; 3];
        let input: Vec<f32> = (0..3 * QK).map(|i| (i as f32 - 100.0) * 0.01).collect();
        let scalar = dot_q1_0_scalar(&blocks, &input);
        let dispatched = dot_q1_0(&blocks, &input);
        assert!((scalar - dispatched).abs() < 1e-4 * scalar.abs().max(1.0));
    }
}
