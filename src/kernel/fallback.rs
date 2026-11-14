use super::{q4_0::BlockQ4_0, q8_0::BlockQ8_0};

#[inline]
pub fn dot_q8_0(blocks: &[BlockQ8_0], activations: &[f32]) -> f32 {
    blocks
        .iter()
        .zip(activations.as_chunks::<32>().0)
        .map(|(block, input)| {
            block.scale()
                * block
                    .qs
                    .iter()
                    .zip(input)
                    .map(|(&q, &x)| q as f32 * x)
                    .sum::<f32>()
        })
        .sum()
}

#[inline]
pub fn dot_q4_0(blocks: &[BlockQ4_0], activations: &[f32]) -> f32 {
    blocks
        .iter()
        .zip(activations.as_chunks::<32>().0)
        .map(|(block, input)| {
            block.scale()
                * (0..32)
                    .map(|index| block.quantized(index) as f32 * input[index])
                    .sum::<f32>()
        })
        .sum()
}

pub fn gemv_q8_0(
    weights: &[BlockQ8_0],
    activations: &[f32],
    output: &mut [f32],
    blocks_per_row: usize,
) {
    for (row, result) in output.iter_mut().enumerate() {
        let start = row * blocks_per_row;
        *result = dot_q8_0(&weights[start..start + blocks_per_row], activations);
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

pub fn fused_ffn_q8_0(
    gate_weights: &[BlockQ8_0],
    up_weights: &[BlockQ8_0],
    activations: &[f32],
    output: &mut [f32],
    blocks_per_row: usize,
) {
    for (row, result) in output.iter_mut().enumerate() {
        let start = row * blocks_per_row;
        let g = dot_q8_0(&gate_weights[start..start + blocks_per_row], activations);
        let u = dot_q8_0(&up_weights[start..start + blocks_per_row], activations);
        *result = silu(g) * u;
    }
}

pub fn fused_ffn_q4_0(
    gate_weights: &[BlockQ4_0],
    up_weights: &[BlockQ4_0],
    activations: &[f32],
    output: &mut [f32],
    blocks_per_row: usize,
) {
    for (row, result) in output.iter_mut().enumerate() {
        let start = row * blocks_per_row;
        let g = dot_q4_0(&gate_weights[start..start + blocks_per_row], activations);
        let u = dot_q4_0(&up_weights[start..start + blocks_per_row], activations);
        *result = silu(g) * u;
    }
}

pub fn gemv_q4_0(
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
