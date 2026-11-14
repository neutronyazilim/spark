use super::{q4_0::BlockQ4_0, q8_0::BlockQ8_0};

/// # Safety
///
/// The caller must execute this function only when AVX2 and FMA are available.
#[target_feature(enable = "avx2,fma")]
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

            // 4 chunks of 8 floats per block
            for c in 0..4 {
                let x = std::arch::x86_64::_mm256_loadu_ps(act_chunk.as_ptr().add(c * 8));

                let q0 = std::arch::x86_64::_mm_loadl_epi64(b0.qs.as_ptr().add(c * 8).cast());
                let i0 = std::arch::x86_64::_mm256_cvtepi8_epi32(q0);
                let w0_ps = std::arch::x86_64::_mm256_cvtepi32_ps(i0);
                acc0 = std::arch::x86_64::_mm256_fmadd_ps(
                    std::arch::x86_64::_mm256_mul_ps(w0_ps, s0),
                    x,
                    acc0,
                );

                let q1 = std::arch::x86_64::_mm_loadl_epi64(b1.qs.as_ptr().add(c * 8).cast());
                let i1 = std::arch::x86_64::_mm256_cvtepi8_epi32(q1);
                let w1_ps = std::arch::x86_64::_mm256_cvtepi32_ps(i1);
                acc1 = std::arch::x86_64::_mm256_fmadd_ps(
                    std::arch::x86_64::_mm256_mul_ps(w1_ps, s1),
                    x,
                    acc1,
                );

                let q2 = std::arch::x86_64::_mm_loadl_epi64(b2.qs.as_ptr().add(c * 8).cast());
                let i2 = std::arch::x86_64::_mm256_cvtepi8_epi32(q2);
                let w2_ps = std::arch::x86_64::_mm256_cvtepi32_ps(i2);
                acc2 = std::arch::x86_64::_mm256_fmadd_ps(
                    std::arch::x86_64::_mm256_mul_ps(w2_ps, s2),
                    x,
                    acc2,
                );

                let q3 = std::arch::x86_64::_mm_loadl_epi64(b3.qs.as_ptr().add(c * 8).cast());
                let i3 = std::arch::x86_64::_mm256_cvtepi8_epi32(q3);
                let w3_ps = std::arch::x86_64::_mm256_cvtepi32_ps(i3);
                acc3 = std::arch::x86_64::_mm256_fmadd_ps(
                    std::arch::x86_64::_mm256_mul_ps(w3_ps, s3),
                    x,
                    acc3,
                );
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
/// The caller must execute this function only when AVX2 and FMA are available.
#[target_feature(enable = "avx2,fma")]
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

                // Gate rows
                let qg0 = std::arch::x86_64::_mm_loadl_epi64(bg0.qs.as_ptr().add(c * 8).cast());
                let ig0 = std::arch::x86_64::_mm256_cvtepi8_epi32(qg0);
                let wg0_ps = std::arch::x86_64::_mm256_cvtepi32_ps(ig0);
                acc_g0 = std::arch::x86_64::_mm256_fmadd_ps(
                    std::arch::x86_64::_mm256_mul_ps(wg0_ps, sg0),
                    x,
                    acc_g0,
                );

                let qg1 = std::arch::x86_64::_mm_loadl_epi64(bg1.qs.as_ptr().add(c * 8).cast());
                let ig1 = std::arch::x86_64::_mm256_cvtepi8_epi32(qg1);
                let wg1_ps = std::arch::x86_64::_mm256_cvtepi32_ps(ig1);
                acc_g1 = std::arch::x86_64::_mm256_fmadd_ps(
                    std::arch::x86_64::_mm256_mul_ps(wg1_ps, sg1),
                    x,
                    acc_g1,
                );

                let qg2 = std::arch::x86_64::_mm_loadl_epi64(bg2.qs.as_ptr().add(c * 8).cast());
                let ig2 = std::arch::x86_64::_mm256_cvtepi8_epi32(qg2);
                let wg2_ps = std::arch::x86_64::_mm256_cvtepi32_ps(ig2);
                acc_g2 = std::arch::x86_64::_mm256_fmadd_ps(
                    std::arch::x86_64::_mm256_mul_ps(wg2_ps, sg2),
                    x,
                    acc_g2,
                );

                let qg3 = std::arch::x86_64::_mm_loadl_epi64(bg3.qs.as_ptr().add(c * 8).cast());
                let ig3 = std::arch::x86_64::_mm256_cvtepi8_epi32(qg3);
                let wg3_ps = std::arch::x86_64::_mm256_cvtepi32_ps(ig3);
                acc_g3 = std::arch::x86_64::_mm256_fmadd_ps(
                    std::arch::x86_64::_mm256_mul_ps(wg3_ps, sg3),
                    x,
                    acc_g3,
                );

                // Up rows
                let qu0 = std::arch::x86_64::_mm_loadl_epi64(bu0.qs.as_ptr().add(c * 8).cast());
                let iu0 = std::arch::x86_64::_mm256_cvtepi8_epi32(qu0);
                let wu0_ps = std::arch::x86_64::_mm256_cvtepi32_ps(iu0);
                acc_u0 = std::arch::x86_64::_mm256_fmadd_ps(
                    std::arch::x86_64::_mm256_mul_ps(wu0_ps, su0),
                    x,
                    acc_u0,
                );

                let qu1 = std::arch::x86_64::_mm_loadl_epi64(bu1.qs.as_ptr().add(c * 8).cast());
                let iu1 = std::arch::x86_64::_mm256_cvtepi8_epi32(qu1);
                let wu1_ps = std::arch::x86_64::_mm256_cvtepi32_ps(iu1);
                acc_u1 = std::arch::x86_64::_mm256_fmadd_ps(
                    std::arch::x86_64::_mm256_mul_ps(wu1_ps, su1),
                    x,
                    acc_u1,
                );

                let qu2 = std::arch::x86_64::_mm_loadl_epi64(bu2.qs.as_ptr().add(c * 8).cast());
                let iu2 = std::arch::x86_64::_mm256_cvtepi8_epi32(qu2);
                let wu2_ps = std::arch::x86_64::_mm256_cvtepi32_ps(iu2);
                acc_u2 = std::arch::x86_64::_mm256_fmadd_ps(
                    std::arch::x86_64::_mm256_mul_ps(wu2_ps, su2),
                    x,
                    acc_u2,
                );

                let qu3 = std::arch::x86_64::_mm_loadl_epi64(bu3.qs.as_ptr().add(c * 8).cast());
                let iu3 = std::arch::x86_64::_mm256_cvtepi8_epi32(qu3);
                let wu3_ps = std::arch::x86_64::_mm256_cvtepi32_ps(iu3);
                acc_u3 = std::arch::x86_64::_mm256_fmadd_ps(
                    std::arch::x86_64::_mm256_mul_ps(wu3_ps, su3),
                    x,
                    acc_u3,
                );
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

#[target_feature(enable = "avx2,fma")]
unsafe fn unpack_q4_0_block(
    qs: *const u8,
) -> (
    std::arch::x86_64::__m256,
    std::arch::x86_64::__m256,
    std::arch::x86_64::__m256,
    std::arch::x86_64::__m256,
) {
    use std::arch::x86_64::*;
    let raw = _mm_loadu_si128(qs.cast());
    let mask = _mm_set1_epi8(0x0f);
    let bias = _mm_set1_epi8(8);

    let lo = _mm_sub_epi8(_mm_and_si128(raw, mask), bias);
    let hi = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(raw, 4), mask), bias);

    let c0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo));
    let c1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8)));
    let c2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi));
    let c3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8)));

    (c0, c1, c2, c3)
}

/// # Safety
///
/// The caller must execute this function only when AVX2 and FMA are available.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemv_q4_0(
    weights: &[BlockQ4_0],
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

            let (w0_0, w0_1, w0_2, w0_3) = unpack_q4_0_block(b0.qs.as_ptr());
            let (w1_0, w1_1, w1_2, w1_3) = unpack_q4_0_block(b1.qs.as_ptr());
            let (w2_0, w2_1, w2_2, w2_3) = unpack_q4_0_block(b2.qs.as_ptr());
            let (w3_0, w3_1, w3_2, w3_3) = unpack_q4_0_block(b3.qs.as_ptr());

            let act_chunk = &activations[b * 32..(b + 1) * 32];

            let x0 = std::arch::x86_64::_mm256_loadu_ps(act_chunk.as_ptr());
            acc0 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(w0_0, s0), x0, acc0);
            acc1 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(w1_0, s1), x0, acc1);
            acc2 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(w2_0, s2), x0, acc2);
            acc3 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(w3_0, s3), x0, acc3);

            let x1 = std::arch::x86_64::_mm256_loadu_ps(act_chunk.as_ptr().add(8));
            acc0 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(w0_1, s0), x1, acc0);
            acc1 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(w1_1, s1), x1, acc1);
            acc2 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(w2_1, s2), x1, acc2);
            acc3 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(w3_1, s3), x1, acc3);

            let x2 = std::arch::x86_64::_mm256_loadu_ps(act_chunk.as_ptr().add(16));
            acc0 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(w0_2, s0), x2, acc0);
            acc1 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(w1_2, s1), x2, acc1);
            acc2 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(w2_2, s2), x2, acc2);
            acc3 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(w3_2, s3), x2, acc3);

            let x3 = std::arch::x86_64::_mm256_loadu_ps(act_chunk.as_ptr().add(24));
            acc0 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(w0_3, s0), x3, acc0);
            acc1 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(w1_3, s1), x3, acc1);
            acc2 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(w2_3, s2), x3, acc2);
            acc3 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(w3_3, s3), x3, acc3);
        }

        output[r0] = horizontal_sum(acc0);
        output[r1] = horizontal_sum(acc1);
        output[r2] = horizontal_sum(acc2);
        output[r3] = horizontal_sum(acc3);

        row += 4;
    }

    while row < rows {
        let start = row * blocks_per_row;
        output[row] = dot_q4_0(&weights[start..start + blocks_per_row], activations);
        row += 1;
    }
}

/// # Safety
///
/// The caller must execute this function only when AVX2 and FMA are available.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn fused_ffn_q4_0(
    gate_weights: &[BlockQ4_0],
    up_weights: &[BlockQ4_0],
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

            let (wg0_0, wg0_1, wg0_2, wg0_3) = unpack_q4_0_block(bg0.qs.as_ptr());
            let (wg1_0, wg1_1, wg1_2, wg1_3) = unpack_q4_0_block(bg1.qs.as_ptr());
            let (wg2_0, wg2_1, wg2_2, wg2_3) = unpack_q4_0_block(bg2.qs.as_ptr());
            let (wg3_0, wg3_1, wg3_2, wg3_3) = unpack_q4_0_block(bg3.qs.as_ptr());

            let (wu0_0, wu0_1, wu0_2, wu0_3) = unpack_q4_0_block(bu0.qs.as_ptr());
            let (wu1_0, wu1_1, wu1_2, wu1_3) = unpack_q4_0_block(bu1.qs.as_ptr());
            let (wu2_0, wu2_1, wu2_2, wu2_3) = unpack_q4_0_block(bu2.qs.as_ptr());
            let (wu3_0, wu3_1, wu3_2, wu3_3) = unpack_q4_0_block(bu3.qs.as_ptr());

            let x0 = std::arch::x86_64::_mm256_loadu_ps(act_chunk.as_ptr());
            acc_g0 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(wg0_0, sg0), x0, acc_g0);
            acc_g1 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(wg1_0, sg1), x0, acc_g1);
            acc_g2 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(wg2_0, sg2), x0, acc_g2);
            acc_g3 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(wg3_0, sg3), x0, acc_g3);
            acc_u0 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(wu0_0, su0), x0, acc_u0);
            acc_u1 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(wu1_0, su1), x0, acc_u1);
            acc_u2 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(wu2_0, su2), x0, acc_u2);
            acc_u3 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(wu3_0, su3), x0, acc_u3);

            let x1 = std::arch::x86_64::_mm256_loadu_ps(act_chunk.as_ptr().add(8));
            acc_g0 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(wg0_1, sg0), x1, acc_g0);
            acc_g1 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(wg1_1, sg1), x1, acc_g1);
            acc_g2 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(wg2_1, sg2), x1, acc_g2);
            acc_g3 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(wg3_1, sg3), x1, acc_g3);
            acc_u0 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(wu0_1, su0), x1, acc_u0);
            acc_u1 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(wu1_1, su1), x1, acc_u1);
            acc_u2 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(wu2_1, su2), x1, acc_u2);
            acc_u3 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(wu3_1, su3), x1, acc_u3);

            let x2 = std::arch::x86_64::_mm256_loadu_ps(act_chunk.as_ptr().add(16));
            acc_g0 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(wg0_2, sg0), x2, acc_g0);
            acc_g1 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(wg1_2, sg1), x2, acc_g1);
            acc_g2 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(wg2_2, sg2), x2, acc_g2);
            acc_g3 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(wg3_2, sg3), x2, acc_g3);
            acc_u0 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(wu0_2, su0), x2, acc_u0);
            acc_u1 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(wu1_2, su1), x2, acc_u1);
            acc_u2 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(wu2_2, su2), x2, acc_u2);
            acc_u3 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(wu3_2, su3), x2, acc_u3);

            let x3 = std::arch::x86_64::_mm256_loadu_ps(act_chunk.as_ptr().add(24));
            acc_g0 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(wg0_3, sg0), x3, acc_g0);
            acc_g1 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(wg1_3, sg1), x3, acc_g1);
            acc_g2 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(wg2_3, sg2), x3, acc_g2);
            acc_g3 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(wg3_3, sg3), x3, acc_g3);
            acc_u0 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(wu0_3, su0), x3, acc_u0);
            acc_u1 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(wu1_3, su1), x3, acc_u1);
            acc_u2 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(wu2_3, su2), x3, acc_u2);
            acc_u3 = std::arch::x86_64::_mm256_fmadd_ps(std::arch::x86_64::_mm256_mul_ps(wu3_3, su3), x3, acc_u3);
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
        let g = dot_q4_0(&gate_weights[start..start + blocks_per_row], activations);
        let u = dot_q4_0(&up_weights[start..start + blocks_per_row], activations);
        output[row] = silu(g) * u;
        row += 1;
    }
}

#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn dot_q8_0(blocks: &[BlockQ8_0], activations: &[f32]) -> f32 {
    let mut total = std::arch::x86_64::_mm256_setzero_ps();
    for (block, input) in blocks.iter().zip(activations.as_chunks::<32>().0) {
        let scale = std::arch::x86_64::_mm256_set1_ps(block.scale());
        for (q, values) in block
            .qs
            .as_chunks::<8>()
            .0
            .iter()
            .zip(input.as_chunks::<8>().0)
        {
            let packed = std::arch::x86_64::_mm_loadl_epi64(q.as_ptr().cast());
            let ints = std::arch::x86_64::_mm256_cvtepi8_epi32(packed);
            let weights = std::arch::x86_64::_mm256_cvtepi32_ps(ints);
            let x = std::arch::x86_64::_mm256_loadu_ps(values.as_ptr());
            total = std::arch::x86_64::_mm256_fmadd_ps(
                std::arch::x86_64::_mm256_mul_ps(weights, scale),
                x,
                total,
            );
        }
    }
    horizontal_sum(total)
}

#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q4_0(blocks: &[BlockQ4_0], activations: &[f32]) -> f32 {
    let mut total = std::arch::x86_64::_mm256_setzero_ps();
    for (block, input) in blocks.iter().zip(activations.as_chunks::<32>().0) {
        let scale = std::arch::x86_64::_mm256_set1_ps(block.scale());
        for (offset, values) in input.as_chunks::<8>().0.iter().enumerate() {
            let weights =
                std::arch::x86_64::_mm256_cvtepi32_ps(std::arch::x86_64::_mm256_set_epi32(
                    block.quantized(offset * 8 + 7),
                    block.quantized(offset * 8 + 6),
                    block.quantized(offset * 8 + 5),
                    block.quantized(offset * 8 + 4),
                    block.quantized(offset * 8 + 3),
                    block.quantized(offset * 8 + 2),
                    block.quantized(offset * 8 + 1),
                    block.quantized(offset * 8),
                ));
            let x = std::arch::x86_64::_mm256_loadu_ps(values.as_ptr());
            total = std::arch::x86_64::_mm256_fmadd_ps(
                std::arch::x86_64::_mm256_mul_ps(weights, scale),
                x,
                total,
            );
        }
    }
    horizontal_sum(total)
}

#[target_feature(enable = "avx2,fma")]
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
