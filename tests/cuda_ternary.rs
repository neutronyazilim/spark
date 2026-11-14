use spark::{
    backend::CudaDevice,
    kernel::{
        f16::f16_to_f32,
        q1_0::{dot_q1_0_q8_0, quantize_activations_q8_0, BlockQ1_0},
    },
};

fn bytes_of_f32(values: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4) }
}

fn bytes_of_f32_mut(values: &mut [f32]) -> &mut [u8] {
    unsafe {
        std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), values.len() * 4)
    }
}

fn bytes_of<T>(values: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            values.as_ptr().cast::<u8>(),
            std::mem::size_of_val(values),
        )
    }
}

fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 2.0e-4 * e.abs().max(1.0);
        assert!(
            (a - e).abs() <= tolerance,
            "{label} row {i}: GPU={a}, CPU={e}, diff={}",
            (a - e).abs()
        );
    }
}

#[test]
#[allow(clippy::needless_range_loop)]
fn cuda_q1_q2_and_ptq1_match_scalar_multiblock_reference() {
    let Some(device) = CudaDevice::init() else {
        eprintln!("CUDA unavailable; skipping ternary GPU validation");
        return;
    };

    // Q2_0: three blocks per row catches packed-byte index corruption after
    // block zero. Five rows also exercises a partial final CUDA block.
    let q2_rows = 5usize;
    let q2_cols = 96usize;
    let q2_blocks = q2_cols / 32;
    let q2_scales = [0x3c00u16, 0x3800, 0x4000]; // 1.0, 0.5, 2.0
    let mut q2_weights = vec![0u8; q2_rows * q2_blocks * 10];
    for row in 0..q2_rows {
        for block in 0..q2_blocks {
            let base = (row * q2_blocks + block) * 10;
            q2_weights[base..base + 2].copy_from_slice(&q2_scales[block].to_le_bytes());
            for byte in 0..8 {
                let mut packed = 0u8;
                for trit in 0..4 {
                    let code = ((row * 5 + block * 3 + byte + trit) % 4) as u8;
                    packed |= code << (trit * 2);
                }
                q2_weights[base + 2 + byte] = packed;
            }
        }
    }
    let q2_input: Vec<f32> = (0..q2_cols)
        .map(|i| ((i as i32 % 17) - 8) as f32 * 0.125)
        .collect();
    let mut q2_expected = vec![0.0f32; q2_rows];
    for row in 0..q2_rows {
        for block in 0..q2_blocks {
            let base = (row * q2_blocks + block) * 10;
            let scale = f16_to_f32(u16::from_le_bytes([
                q2_weights[base],
                q2_weights[base + 1],
            ]));
            for i in 0..32 {
                let packed = q2_weights[base + 2 + i / 4];
                let code = (packed >> ((i % 4) * 2)) & 3;
                let sign = match code {
                    1 => 1.0,
                    2 => -1.0,
                    _ => 0.0,
                };
                q2_expected[row] += sign * scale * q2_input[block * 32 + i];
            }
        }
    }

    let d_q2_weights = device.alloc(q2_weights.len()).unwrap();
    let d_q2_input = device.alloc(q2_input.len() * 4).unwrap();
    let d_q2_output = device.alloc(q2_rows * 4).unwrap();
    device.copy_to_device(d_q2_weights, &q2_weights).unwrap();
    device
        .copy_to_device(d_q2_input, bytes_of_f32(&q2_input))
        .unwrap();
    device
        .launch_gemv_q2_0(d_q2_weights, d_q2_input, d_q2_output, q2_cols, q2_rows)
        .unwrap();
    let mut q2_actual = vec![0.0f32; q2_rows];
    device
        .copy_from_device(bytes_of_f32_mut(&mut q2_actual), d_q2_output)
        .unwrap();
    assert_close(&q2_actual, &q2_expected, "Q2_0");
    device.free(d_q2_weights);
    device.free(d_q2_input);
    device.free(d_q2_output);

    // PTQ1_0: two 128-weight blocks per row exercise all 24 base-3 bytes,
    // both packed tail bytes, per-block scales, and row strides.
    let ptq_rows = 5usize;
    let ptq_cols = 256usize;
    let ptq_blocks = ptq_cols / 128;
    let ptq_scales = [0x3c00u16, 0x3800];
    let mut ptq_weights = vec![0u8; ptq_rows * ptq_blocks * 28];
    for row in 0..ptq_rows {
        for block in 0..ptq_blocks {
            let base = (row * ptq_blocks + block) * 28;
            for byte in 0..24 {
                // Valid packed base-3 values are 0..242.
                ptq_weights[base + byte] = ((row * 47 + block * 29 + byte * 11) % 243) as u8;
            }
            for byte in 0..2 {
                let mut packed = 0u8;
                for trit in 0..4 {
                    let code = ((row + block * 2 + byte + trit) % 4) as u8;
                    packed |= code << (trit * 2);
                }
                ptq_weights[base + 24 + byte] = packed;
            }
            ptq_weights[base + 26..base + 28]
                .copy_from_slice(&ptq_scales[block].to_le_bytes());
        }
    }
    let ptq_input: Vec<f32> = (0..ptq_cols)
        .map(|i| ((i as i32 % 23) - 11) as f32 * 0.0625)
        .collect();
    let mut ptq_expected = vec![0.0f32; ptq_rows];
    for row in 0..ptq_rows {
        for block in 0..ptq_blocks {
            let base = (row * ptq_blocks + block) * 28;
            let scale = f16_to_f32(u16::from_le_bytes([
                ptq_weights[base + 26],
                ptq_weights[base + 27],
            ]));
            let mut block_sum = 0.0f32;
            for byte in 0..24 {
                let mut packed = ptq_weights[base + byte];
                for trit in 0..5 {
                    let sign = (packed % 3) as i32 - 1;
                    packed /= 3;
                    block_sum += sign as f32 * ptq_input[block * 128 + byte * 5 + trit];
                }
            }
            for byte in 0..2 {
                let packed = ptq_weights[base + 24 + byte];
                for trit in 0..4 {
                    let code = (packed >> (trit * 2)) & 3;
                    let sign = match code {
                        1 => 1.0,
                        2 => -1.0,
                        _ => 0.0,
                    };
                    block_sum += sign * ptq_input[block * 128 + 120 + byte * 4 + trit];
                }
            }
            ptq_expected[row] += block_sum * scale;
        }
    }

    let d_ptq_weights = device.alloc(ptq_weights.len()).unwrap();
    let d_ptq_input = device.alloc(ptq_input.len() * 4).unwrap();
    let d_ptq_output = device.alloc(ptq_rows * 4).unwrap();
    device.copy_to_device(d_ptq_weights, &ptq_weights).unwrap();
    device
        .copy_to_device(d_ptq_input, bytes_of_f32(&ptq_input))
        .unwrap();
    device
        .launch_gemv_ternary(
            d_ptq_weights,
            d_ptq_input,
            d_ptq_output,
            ptq_cols,
            ptq_rows,
        )
        .unwrap();
    let mut ptq_actual = vec![0.0f32; ptq_rows];
    device
        .copy_from_device(bytes_of_f32_mut(&mut ptq_actual), d_ptq_output)
        .unwrap();
    assert_close(&ptq_actual, &ptq_expected, "PTQ1_0");
    device.free(d_ptq_weights);
    device.free(d_ptq_input);
    device.free(d_ptq_output);

    // Q1_0 x Q8_0: validate the exact activation quantization and compact
    // 136-byte input layout consumed by the dp4a PTX kernel.
    let q1_rows = 8usize;
    let q1_cols = 1024usize;
    let blocks_per_row = q1_cols / 128;
    let mut q1_weights = Vec::with_capacity(q1_rows * blocks_per_row);
    for row in 0..q1_rows {
        for block in 0..blocks_per_row {
            let mut qs = [0u8; 16];
            for (i, byte) in qs.iter_mut().enumerate() {
                *byte = (row * 53 + block * 29 + i * 17) as u8;
            }
            q1_weights.push(BlockQ1_0 {
                d: [0x3c00u16, 0x3800][block % 2],
                qs,
            });
        }
    }
    let q1_input: Vec<f32> = (0..q1_cols)
        .map(|i| ((i as i32 % 31) - 15) as f32 * 0.03125)
        .collect();
    let q8_input = quantize_activations_q8_0(&q1_input);
    let mut compact_input = vec![0u8; blocks_per_row * 136];
    for group in 0..blocks_per_row {
        let dst = &mut compact_input[group * 136..(group + 1) * 136];
        for block in 0..4 {
            let src = &q8_input[group * 4 + block];
            dst[block * 2..block * 2 + 2].copy_from_slice(&src.d.to_le_bytes());
            dst[8 + block * 32..8 + (block + 1) * 32]
                .copy_from_slice(bytes_of(&src.qs));
        }
    }
    let q1_expected: Vec<f32> = (0..q1_rows)
        .map(|row| {
            dot_q1_0_q8_0(
                &q1_weights[row * blocks_per_row..(row + 1) * blocks_per_row],
                &q8_input,
            )
        })
        .collect();
    let tiled_q1;
    let q1_bytes = if device.q1_tiled_enabled() {
        let mut out = vec![0u8; std::mem::size_of_val(q1_weights.as_slice())];
        for row_tile in 0..q1_rows / 8 {
            for block in 0..blocks_per_row {
                let d = (row_tile * blocks_per_row + block) * 144;
                for row in 0..8 {
                    let src = &q1_weights[(row_tile * 8 + row) * blocks_per_row + block];
                    out[d + row * 2..d + row * 2 + 2].copy_from_slice(&src.d.to_le_bytes());
                    out[d + 16 + row * 16..d + 16 + (row + 1) * 16]
                        .copy_from_slice(&src.qs);
                }
            }
        }
        tiled_q1 = out;
        tiled_q1.as_slice()
    } else {
        bytes_of(&q1_weights)
    };
    let d_q1_weights = device.alloc(q1_bytes.len()).unwrap();
    let d_q1_input = device.alloc(compact_input.len()).unwrap();
    let d_q1_output = device.alloc(q1_rows * 4).unwrap();
    device.copy_to_device(d_q1_weights, q1_bytes).unwrap();
    device.copy_to_device(d_q1_input, &compact_input).unwrap();
    device
        .launch_gemv_q1_0(d_q1_weights, d_q1_input, d_q1_output, q1_cols, q1_rows)
        .unwrap();
    let mut q1_actual = vec![0.0f32; q1_rows];
    device
        .copy_from_device(bytes_of_f32_mut(&mut q1_actual), d_q1_output)
        .unwrap();
    assert_close(&q1_actual, &q1_expected, "Q1_0");

    // Resident decode quantizes FP32 activations directly on the GPU. Its
    // compact output must exactly match the CPU packing consumed above.
    let d_q1_f32 = device.alloc(q1_input.len() * 4).unwrap();
    let d_q1_compact = device.alloc(compact_input.len()).unwrap();
    device
        .copy_to_device(d_q1_f32, bytes_of_f32(&q1_input))
        .unwrap();
    device
        .launch_quantize_q8_0_128(d_q1_f32, d_q1_compact, q1_input.len())
        .unwrap();
    let mut gpu_compact = vec![0u8; compact_input.len()];
    device
        .copy_from_device(&mut gpu_compact, d_q1_compact)
        .unwrap();
    assert_eq!(gpu_compact, compact_input, "resident Q8 activation packing");

    device.free(d_q1_weights);
    device.free(d_q1_input);
    device.free(d_q1_output);
    device.free(d_q1_f32);
    device.free(d_q1_compact);
}
