#[derive(Debug, Clone)]
pub struct Sampler {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    history: Vec<u32>,
    history_limit: usize,
    state: u64,
}

impl Sampler {
    pub fn new(temperature: f32, top_p: f32, top_k: usize, repetition_penalty: f32) -> Self {
        Self {
            temperature,
            top_p,
            top_k,
            repetition_penalty: repetition_penalty.max(1.0),
            history: Vec::with_capacity(64),
            history_limit: 64,
            state: 0x9e37_79b9_7f4a_7c15,
        }
    }

    pub fn sample(&mut self, logits: &[f32]) -> usize {
        let selected = if self.temperature <= 0.0 {
            if self.repetition_penalty <= 1.0 || self.history.is_empty() {
                argmax(logits)
            } else {
                let mut best_idx = 0;
                let mut best_val = f32::NEG_INFINITY;
                for (idx, &val) in logits.iter().enumerate() {
                    let penalized = if self.history.contains(&(idx as u32)) {
                        if val > 0.0 {
                            val / self.repetition_penalty
                        } else {
                            val * self.repetition_penalty
                        }
                    } else {
                        val
                    };
                    if penalized > best_val {
                        best_val = penalized;
                        best_idx = idx;
                    }
                }
                best_idx
            }
        } else {
            let mut adjusted = logits.to_vec();
            apply_repetition_penalty(&mut adjusted, &self.history, self.repetition_penalty);
            let temperature = self.temperature.max(f32::MIN_POSITIVE);
            let mut candidates: Vec<(usize, f32)> = adjusted
                .into_iter()
                .enumerate()
                .map(|(index, logit)| (index, logit / temperature))
                .collect();
            top_k(&mut candidates, self.top_k);

            let maximum = candidates
                .iter()
                .map(|(_, logit)| *logit)
                .fold(f32::NEG_INFINITY, f32::max);
            let normalizer: f32 = candidates
                .iter_mut()
                .map(|(_, logit)| {
                    *logit = (*logit - maximum).exp();
                    *logit
                })
                .sum();
            if normalizer <= 0.0 || !normalizer.is_finite() {
                argmax(
                    &candidates
                        .iter()
                        .map(|(_, probability)| *probability)
                        .collect::<Vec<_>>(),
                )
            } else {
                for (_, probability) in &mut candidates {
                    *probability /= normalizer;
                }
                candidates.sort_by(|left, right| right.1.total_cmp(&left.1));
                let cutoff_probability = self.top_p.clamp(0.0, 1.0).max(f32::EPSILON);
                let mut cumulative = 0.0;
                let mut cutoff = candidates.len().saturating_sub(1);
                for (index, (_, probability)) in candidates.iter().enumerate() {
                    cumulative += *probability;
                    if cumulative >= cutoff_probability {
                        cutoff = index;
                        break;
                    }
                }
                let threshold = self.next_random();
                let mut local = 0.0;
                let mut selected = candidates[0].0;
                for &(index, probability) in candidates.iter().take(cutoff + 1) {
                    local += probability;
                    selected = index;
                    if local >= threshold {
                        break;
                    }
                }
                selected
            }
        };
        self.history.push(selected as u32);
        if self.history.len() > self.history_limit {
            let excess = self.history.len() - self.history_limit;
            self.history.drain(..excess);
        }
        selected
    }

    pub fn next_random(&mut self) -> f32 {
        self.state ^= self.state << 7;
        self.state ^= self.state >> 9;
        self.state ^= self.state << 8;
        (self.state as f64 / u64::MAX as f64) as f32
    }
}

pub fn apply_repetition_penalty(logits: &mut [f32], history: &[u32], penalty: f32) {
    if penalty <= 1.0 {
        return;
    }
    for &token in history {
        if let Some(logit) = logits.get_mut(token as usize) {
            if *logit > 0.0 {
                *logit /= penalty;
            } else {
                *logit *= penalty;
            }
        }
    }
}

pub fn top_k(candidates: &mut Vec<(usize, f32)>, k: usize) {
    if k == 0 || k >= candidates.len() {
        return;
    }
    candidates.sort_by(|left, right| right.1.total_cmp(&left.1));
    candidates.truncate(k);
}

pub fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::{apply_repetition_penalty, argmax, top_k, Sampler};

    #[test]
    fn greedy_selects_largest_logit() {
        assert_eq!(argmax(&[1.0, 4.0, 2.0]), 1);
        assert_eq!(Sampler::new(0.0, 1.0, 40, 1.1).sample(&[1.0, 4.0]), 1);
    }

    #[test]
    fn repetition_penalty_handles_positive_and_negative_logits() {
        let mut logits = [2.0, -2.0, 1.0];
        apply_repetition_penalty(&mut logits, &[0, 1], 2.0);
        assert_eq!(logits, [1.0, -4.0, 1.0]);
    }

    #[test]
    fn top_k_keeps_highest_candidates() {
        let mut candidates = vec![(0, 0.1), (1, 0.9), (2, 0.5)];
        top_k(&mut candidates, 2);
        assert_eq!(candidates, vec![(1, 0.9), (2, 0.5)]);
    }
}
