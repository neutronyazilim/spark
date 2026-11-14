use super::{q4_0::BlockQ4_0, q8_0::BlockQ8_0};

#[target_feature(enable = "avx")]
unsafe fn unpack_8_i8_to_m256(ptr: *const i8) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    // Load 8 signed bytes into the low half of a 128-bit register
    let q = _mm_loadl_epi64(ptr.cast());
    // Convert low 4 bytes to 4 x 32-bit integers, then to floats
    let lo_i = _mm_cvtepi8_epi32(q);
    let lo_f = _mm_cvtepi32_ps(lo_i);
    // Shift right by 4 bytes to access the high 4 bytes
    let hi_q = _mm_srli_si128(q, 4);
    let hi_i = _mm_cvtepi8_epi32(hi_q);
    let hi_f = _mm_cvtepi32_ps(hi_i);
    // Combine two 128-bit float vectors into one 256-bit YMM vector
    _mm256_insertf128_ps(_mm256_castps128_ps256(lo_f), hi_f, 1)
}

/// # Safety
///
/// The caller must execute this function only when AVX is available.
#[target_feature(enable = "avx")]
pub unsafe fn gemv_q8_0(
    weights: &[BlockQ8_0],
    activations: &[f32],
    output: &mut [f32],
    blocks_per_row: usize,
) {
    let rows = output.len();
    let unroll_limit = rows.saturating_sub(rows % 4);

    let mut row = 0;
    while row < unroll_limit {
        let r0 = row;
        let r1 = row + 1;
        let r2 = row + 2;
        let r3 = row + 3;

        let w0 = &weights[r0 * blocks_per_row..(r0 + 1) * blocks_per_row];
        let w1 = &weights[r1 * blocks_per_row..(r1 + 1) * blocks_per_row];
        let w2 = &weights[r2 * blocks_per_row..(r2 + 1) * blocks_per_row];
        let w3 = &weights[r3 * blocks_per_row..(r3 + 1) * blocks_per_row];

        let mut acc0 = std::arch::x86_64::_mm256_setzero_ps();
        let mut acc1 = std::arch::x86_64::_mm256_setzero_ps();
        let mut acc2 = std::arch::x86_64::_mm256_setzero_ps();
        let mut acc3 = std::arch::x86_64::_mm256_setzero_ps();

        for b in 0..blocks_per_row {
            let b0 = &w0[b];
            let b1 = &w1[b];
            let b2 = &w2[b];
            let b3 = &w3[b];

            let s0 = std::arch::x86_64::_mm256_set1_ps(b0.scale());
            let s1 = std::arch::x86_64::_mm256_set1_ps(b1.scale());
            let s2 = std::arch::x86_64::_mm256_set1_ps(b2.scale());
            let s3 = std::arch::x86_64::_mm256_set1_ps(b3.scale());

            let act_chunk = &activations[b * 32..(b + 1) * 32];

            for c in 0..4 {
                let x = std::arch::x86_64::_mm256_loadu_ps(act_chunk.as_ptr().add(c * 8));

                let w0_f = unpack_8_i8_to_m256(b0.qs.as_ptr().add(c * 8));
                let prod0 = std::arch::x86_64::_mm256_mul_ps(
                    std::arch::x86_64::_mm256_mul_ps(w0_f, s0),
                    x,
                );
                acc0 = std::arch::x86_64::_mm256_add_ps(acc0, prod0);

                let w1_f = unpack_8_i8_to_m256(b1.qs.as_ptr().add(c * 8));
                let prod1 = std::arch::x86_64::_mm256_mul_ps(
                    std::arch::x86_64::_mm256_mul_ps(w1_f, s1),
                    x,
                );
                acc1 = std::arch::x86_64::_mm256_add_ps(acc1, prod1);

                let w2_f = unpack_8_i8_to_m256(b2.qs.as_ptr().add(c * 8));
                let prod2 = std::arch::x86_64::_mm256_mul_ps(
                    std::arch::x86_64::_mm256_mul_ps(w2_f, s2),
                    x,
                );
                acc2 = std::arch::x86_64::_mm256_add_ps(acc2, prod2);

                let w3_f = unpack_8_i8_to_m256(b3.qs.as_ptr().add(c * 8));
                let prod3 = std::arch::x86_64::_mm256_mul_ps(
                    std::arch::x86_64::_mm256_mul_ps(w3_f, s3),
                    x,
                );
                acc3 = std::arch::x86_64::_mm256_add_ps(acc3, prod3);
            }
        }

        output[r0] = horizontal_sum(acc0);
        output[r1] = horizontal_sum(acc1);
        output[r2] = horizontal_sum(acc2);
        output[r3] = horizontal_sum(acc3);

        row += 4;
    }

    while row < rows {
        let start = row * blocks_per_row;
        output[row] = dot_q8_0(&weights[start..start + blocks_per_row], activations);
        row += 1;
    }
}

#[inline(always)]
fn silu(value: f32) -> f32 {
    if value >= 0.0 {
        value / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        value * (exp / (1.0 + exp))
    }
}

/// # Safety
///
/// The caller must execute this function only when AVX is available.
#[target_feature(enable = "avx")]
pub unsafe fn fused_ffn_q8_0(
    gate_weights: &[BlockQ8_0],
    up_weights: &[BlockQ8_0],
    activations: &[f32],
    output: &mut [f32],
    blocks_per_row: usize,
) {
    let rows = output.len();
    let unroll_limit = rows.saturating_sub(rows % 4);

    let mut row = 0;
    while row < unroll_limit {
        let r0 = row;
        let r1 = row + 1;
        let r2 = row + 2;
        let r3 = row + 3;

        let g0 = &gate_weights[r0 * blocks_per_row..(r0 + 1) * blocks_per_row];
        let g1 = &gate_weights[r1 * blocks_per_row..(r1 + 1) * blocks_per_row];
        let g2 = &gate_weights[r2 * blocks_per_row..(r2 + 1) * blocks_per_row];
        let g3 = &gate_weights[r3 * blocks_per_row..(r3 + 1) * blocks_per_row];

        let u0 = &up_weights[r0 * blocks_per_row..(r0 + 1) * blocks_per_row];
        let u1 = &up_weights[r1 * blocks_per_row..(r1 + 1) * blocks_per_row];
        let u2 = &up_weights[r2 * blocks_per_row..(r2 + 1) * blocks_per_row];
        let u3 = &up_weights[r3 * blocks_per_row..(r3 + 1) * blocks_per_row];

        let mut acc_g0 = std::arch::x86_64::_mm256_setzero_ps();
        let mut acc_g1 = std::arch::x86_64::_mm256_setzero_ps();
        let mut acc_g2 = std::arch::x86_64::_mm256_setzero_ps();
        let mut acc_g3 = std::arch::x86_64::_mm256_setzero_ps();

        let mut acc_u0 = std::arch::x86_64::_mm256_setzero_ps();
        let mut acc_u1 = std::arch::x86_64::_mm256_setzero_ps();
        let mut acc_u2 = std::arch::x86_64::_mm256_setzero_ps();
        let mut acc_u3 = std::arch::x86_64::_mm256_setzero_ps();

        for b in 0..blocks_per_row {
            let bg0 = &g0[b];
            let bg1 = &g1[b];
            let bg2 = &g2[b];
            let bg3 = &g3[b];

            let bu0 = &u0[b];
            let bu1 = &u1[b];
            let bu2 = &u2[b];
            let bu3 = &u3[b];

            let sg0 = std::arch::x86_64::_mm256_set1_ps(bg0.scale());
            let sg1 = std::arch::x86_64::_mm256_set1_ps(bg1.scale());
            let sg2 = std::arch::x86_64::_mm256_set1_ps(bg2.scale());
            let sg3 = std::arch::x86_64::_mm256_set1_ps(bg3.scale());

            let su0 = std::arch::x86_64::_mm256_set1_ps(bu0.scale());
            let su1 = std::arch::x86_64::_mm256_set1_ps(bu1.scale());
            let su2 = std::arch::x86_64::_mm256_set1_ps(bu2.scale());
            let su3 = std::arch::x86_64::_mm256_set1_ps(bu3.scale());

            let act_chunk = &activations[b * 32..(b + 1) * 32];

            for c in 0..4 {
                let x = std::arch::x86_64::_mm256_loadu_ps(act_chunk.as_ptr().add(c * 8));

                // Gate
                let wg0_f = unpack_8_i8_to_m256(bg0.qs.as_ptr().add(c * 8));
                let prod_g0 = std::arch::x86_64::_mm256_mul_ps(
                    std::arch::x86_64::_mm256_mul_ps(wg0_f, sg0),
                    x,
                );
                acc_g0 = std::arch::x86_64::_mm256_add_ps(acc_g0, prod_g0);

                let wg1_f = unpack_8_i8_to_m256(bg1.qs.as_ptr().add(c * 8));
                let prod_g1 = std::arch::x86_64::_mm256_mul_ps(
                    std::arch::x86_64::_mm256_mul_ps(wg1_f, sg1),
                    x,
                );
                acc_g1 = std::arch::x86_64::_mm256_add_ps(acc_g1, prod_g1);

                let wg2_f = unpack_8_i8_to_m256(bg2.qs.as_ptr().add(c * 8));
                let prod_g2 = std::arch::x86_64::_mm256_mul_ps(
                    std::arch::x86_64::_mm256_mul_ps(wg2_f, sg2),
                    x,
                );
                acc_g2 = std::arch::x86_64::_mm256_add_ps(acc_g2, prod_g2);

                let wg3_f = unpack_8_i8_to_m256(bg3.qs.as_ptr().add(c * 8));
                let prod_g3 = std::arch::x86_64::_mm256_mul_ps(
                    std::arch::x86_64::_mm256_mul_ps(wg3_f, sg3),
                    x,
                );
                acc_g3 = std::arch::x86_64::_mm256_add_ps(acc_g3, prod_g3);

                // Up
                let wu0_f = unpack_8_i8_to_m256(bu0.qs.as_ptr().add(c * 8));
                let prod_u0 = std::arch::x86_64::_mm256_mul_ps(
                    std::arch::x86_64::_mm256_mul_ps(wu0_f, su0),
                    x,
                );
                acc_u0 = std::arch::x86_64::_mm256_add_ps(acc_u0, prod_u0);

                let wu1_f = unpack_8_i8_to_m256(bu1.qs.as_ptr().add(c * 8));
                let prod_u1 = std::arch::x86_64::_mm256_mul_ps(
                    std::arch::x86_64::_mm256_mul_ps(wu1_f, su1),
                    x,
                );
                acc_u1 = std::arch::x86_64::_mm256_add_ps(acc_u1, prod_u1);

                let wu2_f = unpack_8_i8_to_m256(bu2.qs.as_ptr().add(c * 8));
                let prod_u2 = std::arch::x86_64::_mm256_mul_ps(
                    std::arch::x86_64::_mm256_mul_ps(wu2_f, su2),
                    x,
                );
                acc_u2 = std::arch::x86_64::_mm256_add_ps(acc_u2, prod_u2);

                let wu3_f = unpack_8_i8_to_m256(bu3.qs.as_ptr().add(c * 8));
                let prod_u3 = std::arch::x86_64::_mm256_mul_ps(
                    std::arch::x86_64::_mm256_mul_ps(wu3_f, su3),
                    x,
                );
                acc_u3 = std::arch::x86_64::_mm256_add_ps(acc_u3, prod_u3);
            }
        }

        let val_g0 = horizontal_sum(acc_g0);
        let val_u0 = horizontal_sum(acc_u0);
        output[r0] = silu(val_g0) * val_u0;

        let val_g1 = horizontal_sum(acc_g1);
        let val_u1 = horizontal_sum(acc_u1);
        output[r1] = silu(val_g1) * val_u1;

        let val_g2 = horizontal_sum(acc_g2);
        let val_u2 = horizontal_sum(acc_u2);
        output[r2] = silu(val_g2) * val_u2;

        let val_g3 = horizontal_sum(acc_g3);
        let val_u3 = horizontal_sum(acc_u3);
        output[r3] = silu(val_g3) * val_u3;

        row += 4;
    }

    while row < rows {
        let start = row * blocks_per_row;
        let g = dot_q8_0(&gate_weights[start..start + blocks_per_row], activations);
        let u = dot_q8_0(&up_weights[start..start + blocks_per_row], activations);
        output[row] = silu(g) * u;
        row += 1;
    }
}

/// # Safety
///
/// The caller must execute this function only when AVX is available.
#[target_feature(enable = "avx")]
pub unsafe fn gemv_q4_0(
    weights: &[BlockQ4_0],
    activations: &[f32],
    output: &mut [f32],
    blocks_per_row: usize,
) {
    for (row, result) in output.iter_mut().enumerate() {
        let start = row * blocks_per_row;
        *result = dot_q4_0(&weights[start..start + blocks_per_row], activations);
    }
}

#[target_feature(enable = "avx")]
unsafe fn dot_q8_0(blocks: &[BlockQ8_0], activations: &[f32]) -> f32 {
    let mut total = std::arch::x86_64::_mm256_setzero_ps();
    for (block, input) in blocks.iter().zip(activations.as_chunks::<32>().0) {
        let scale = std::arch::x86_64::_mm256_set1_ps(block.scale());
        for (offset, values) in input.as_chunks::<8>().0.iter().enumerate() {
            let weights = unpack_8_i8_to_m256(block.qs.as_ptr().add(offset * 8));
            let x = std::arch::x86_64::_mm256_loadu_ps(values.as_ptr());
            total = std::arch::x86_64::_mm256_add_ps(
                total,
                std::arch::x86_64::_mm256_mul_ps(
                    std::arch::x86_64::_mm256_mul_ps(weights, x),
                    scale,
                ),
            );
        }
    }
    horizontal_sum(total)
}

#[target_feature(enable = "avx")]
unsafe fn dot_q4_0(blocks: &[BlockQ4_0], activations: &[f32]) -> f32 {
    let mut total = std::arch::x86_64::_mm256_setzero_ps();
    for (block, input) in blocks.iter().zip(activations.as_chunks::<32>().0) {
        let scale = std::arch::x86_64::_mm256_set1_ps(block.scale());
        for (offset, values) in input.as_chunks::<8>().0.iter().enumerate() {
            let weights = std::arch::x86_64::_mm256_set_ps(
                block.quantized(offset * 8 + 7) as f32,
                block.quantized(offset * 8 + 6) as f32,
                block.quantized(offset * 8 + 5) as f32,
                block.quantized(offset * 8 + 4) as f32,
                block.quantized(offset * 8 + 3) as f32,
                block.quantized(offset * 8 + 2) as f32,
                block.quantized(offset * 8 + 1) as f32,
                block.quantized(offset * 8) as f32,
            );
            let x = std::arch::x86_64::_mm256_loadu_ps(values.as_ptr());
            total = std::arch::x86_64::_mm256_add_ps(
                total,
                std::arch::x86_64::_mm256_mul_ps(
                    std::arch::x86_64::_mm256_mul_ps(weights, x),
                    scale,
                ),
            );
        }
    }
    horizontal_sum(total)
}

#[target_feature(enable = "avx")]
unsafe fn horizontal_sum(value: std::arch::x86_64::__m256) -> f32 {
    let halves = std::arch::x86_64::_mm256_add_ps(
        value,
        std::arch::x86_64::_mm256_permute2f128_ps(value, value, 1),
    );
    let shuffled = std::arch::x86_64::_mm256_shuffle_ps(halves, halves, 0x4e);
    let sums = std::arch::x86_64::_mm256_add_ps(halves, shuffled);
    let shuffled = std::arch::x86_64::_mm256_shuffle_ps(sums, sums, 0xb1);
    std::arch::x86_64::_mm_cvtss_f32(std::arch::x86_64::_mm256_castps256_ps128(
        std::arch::x86_64::_mm256_add_ps(sums, shuffled),
    ))
}
