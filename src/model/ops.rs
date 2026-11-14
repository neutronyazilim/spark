pub fn rmsnorm(output: &mut [f32], input: &[f32], weight: &[f32], epsilon: f32) {
    let mean = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let inverse = (mean + epsilon).sqrt().recip();
    for ((destination, &value), &scale) in output.iter_mut().zip(input).zip(weight) {
        *destination = value * inverse * scale;
    }
}

pub fn rope(values: &mut [f32], position: usize, head_dim: usize, base: f32) {
    for head in values.chunks_exact_mut(head_dim) {
        let half = head_dim / 2;
        for index in 0..half {
            let frequency = base.powf(-(2.0 * index as f32 / head_dim as f32));
            let angle = position as f32 * frequency;
            let (sin, cos) = angle.sin_cos();
            let first = head[index];
            let second = head[index + half];
            // Qwen2 uses the NeoX layout: rotate the first and second
            // halves of each head rather than adjacent scalar pairs.
            head[index] = first * cos - second * sin;
            head[index + half] = second * cos + first * sin;
        }
    }
}

pub fn softmax(values: &mut [f32]) {
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0;
    for value in values.iter_mut() {
        *value = (*value - maximum).exp();
        total += *value;
    }
    let inverse = total.recip();
    for value in values {
        *value *= inverse;
    }
}

pub fn swiglu(gate: &mut [f32], up: &[f32]) {
    for (gate_value, &up_value) in gate.iter_mut().zip(up) {
        let value = *gate_value;
        let sigmoid = if value >= 0.0 {
            1.0 / (1.0 + (-value).exp())
        } else {
            let exp = value.exp();
            exp / (1.0 + exp)
        };
        *gate_value = value * sigmoid * up_value;
    }
}

pub fn hadamard_transform_1024(values: &mut [f32]) {
    for chunk in values.as_chunks_mut::<1024>().0 {
        let mut h = 1;
        while h < 1024 {
            for i in (0..1024).step_by(h * 2) {
                for j in i..i + h {
                    let u = chunk[j];
                    let v = chunk[j + h];
                    chunk[j] = u + v;
                    chunk[j + h] = u - v;
                }
            }
            h *= 2;
        }
        let norm = 1.0 / 32.0f32; // 1 / sqrt(1024)
        for x in chunk.iter_mut() {
            *x *= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_scales_by_root_mean_square() {
        let input = [3.0, 4.0];
        let weight = [1.0, 2.0];
        let mut output = [0.0; 2];
        rmsnorm(&mut output, &input, &weight, 0.0);
        assert!((output[0] - 0.84852815).abs() < 1e-6);
        assert!((output[1] - 2.2627416).abs() < 1e-6);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut values = [1.0, 2.0, 3.0];
        softmax(&mut values);
        assert!((values.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(values[2] > values[1] && values[1] > values[0]);
    }

    #[test]
    fn swiglu_is_elementwise() {
        let mut gate = [0.0, 1.0];
        swiglu(&mut gate, &[2.0, 3.0]);
        assert_eq!(gate[0], 0.0);
        assert!((gate[1] - 2.1931758).abs() < 1e-6);
    }

    #[test]
    fn rope_uses_neox_half_rotation() {
        let mut values = [1.0, 2.0, 3.0, 4.0];
        rope(&mut values, 1, 4, 1.0);
        let (sin, cos) = 1.0f32.sin_cos();
        assert!((values[0] - (cos - 3.0 * sin)).abs() < 1e-6);
        assert!((values[1] - (2.0 * cos - 4.0 * sin)).abs() < 1e-6);
        assert!((values[2] - (3.0 * cos + sin)).abs() < 1e-6);
        assert!((values[3] - (4.0 * cos + 2.0 * sin)).abs() < 1e-6);
    }

    #[test]
    fn swiglu_handles_large_magnitudes() {
        let mut gate = [1000.0, -1000.0];
        swiglu(&mut gate, &[1.0, 1.0]);
        assert!(gate.iter().all(|value| value.is_finite()));
        assert!(gate[0] > 999.0);
        assert!(gate[1].abs() < 1e-3);
    }

    #[test]
    fn hadamard_1024_is_unitary_and_invertible() {
        let mut data = vec![1.0f32; 1024];
        hadamard_transform_1024(&mut data);
        assert!((data[0] - 32.0).abs() < 1e-4);
        assert!(data[1].abs() < 1e-4);
        hadamard_transform_1024(&mut data);
        assert!((data[0] - 1.0).abs() < 1e-4);
        assert!((data[1] - 1.0).abs() < 1e-4);
    }
}
