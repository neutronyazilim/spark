use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, Default)]
pub struct ProfileStats {
    pub enabled: bool,
    pub rmsnorm: Duration,
    pub qkv: Duration,
    pub attention: Duration,
    pub attention_output: Duration,
    pub ffn_gate_up: Duration,
    pub swiglu: Duration,
    pub ffn_down: Duration,
    pub final_output: Duration,
    pub pool_overhead: Duration,
    pub sampling: Duration,
    pub detokenize: Duration,
    pub resident_cuda: Duration,
    pub forward: Duration,
}

impl ProfileStats {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            ..Self::default()
        }
    }

    #[inline(always)]
    pub fn start(&self) -> Option<Instant> {
        if self.enabled {
            Some(Instant::now())
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn add(&mut self, field: &mut Duration, started: Option<Instant>) {
        if self.enabled {
            if let Some(started) = started {
                *field += started.elapsed();
            }
        }
    }

    pub fn total(&self) -> Duration {
        self.rmsnorm
            + self.qkv
            + self.attention
            + self.attention_output
            + self.ffn_gate_up
            + self.swiglu
            + self.ffn_down
            + self.final_output
            + self.pool_overhead
            + self.sampling
            + self.detokenize
            + self.resident_cuda
    }

    pub fn print(&self, tokens: usize) {
        if !self.enabled {
            return;
        }
        let total_dur = self.total();
        let total_secs = total_dur.as_secs_f64().max(1e-12);
        let total_ms = total_secs * 1e3;
        let tok_count = tokens.max(1);
        let ms_per_tok = total_ms / tok_count as f64;

        let rows = [
            ("RMSNorm", self.rmsnorm),
            ("QKV Projections (GEMV)", self.qkv),
            ("RoPE & Attention", self.attention),
            ("Attn Output Proj (GEMV)", self.attention_output),
            ("FFN Gate/Up (GEMV)", self.ffn_gate_up),
            ("SwiGLU Activation", self.swiglu),
            ("FFN Down (GEMV)", self.ffn_down),
            ("Output Proj (GEMV)", self.final_output),
            ("Thread Barrier / Sync", self.pool_overhead),
            ("Sampling / Argmax", self.sampling),
            ("Detokenize & I/O", self.detokenize),
            ("Resident CUDA Forward", self.resident_cuda),
        ];

        println!();
        println!("{}", "=".repeat(78));
        println!(
            " Spark Micro-Profiler ({tokens} tokens, {total_ms:.2} ms total, {ms_per_tok:.2} ms/tok)"
        );
        println!("{}", "-".repeat(78));
        println!(
            " {:<28} {:>12} {:>16} {:>14}",
            "Metric / Operation", "Total Time", "Per Token", "Percentage"
        );
        println!("{}", "-".repeat(78));

        for (name, duration) in rows {
            let secs = duration.as_secs_f64();
            let ms = secs * 1e3;
            let us_per_tok = (secs * 1e6) / tok_count as f64;
            let pct = (secs / total_secs) * 100.0;

            let per_tok_str = if us_per_tok >= 1000.0 {
                format!("{:.2} ms", us_per_tok / 1000.0)
            } else {
                format!("{:.1} µs", us_per_tok)
            };

            println!(
                " {:<28} {:>9.3} ms {:>16} {:>13.1} %",
                name, ms, per_tok_str, pct
            );
        }

        println!("{}", "-".repeat(78));
        println!(
            " {:<28} {:>9.3} ms {:>13.2} ms {:>13.1} %",
            "Total Measured", total_ms, ms_per_tok, 100.0
        );
        println!("{}", "=".repeat(78));
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn disabled_profiler_does_not_start_timers() {
        let profiler = ProfileStats::new(false);
        assert!(profiler.start().is_none());
    }

    #[test]
    fn enabled_profiler_accumulates_elapsed_time() {
        let mut profiler = ProfileStats::new(true);
        let started = profiler.start();
        thread::sleep(Duration::from_millis(1));
        let mut bucket = Duration::ZERO;
        profiler.add(&mut bucket, started);
        assert!(!bucket.is_zero());
    }

    #[test]
    fn total_duration_sums_all_fields() {
        let mut profiler = ProfileStats::new(true);
        profiler.rmsnorm = Duration::from_millis(1);
        profiler.qkv = Duration::from_millis(2);
        profiler.attention = Duration::from_millis(3);
        profiler.attention_output = Duration::from_millis(4);
        profiler.ffn_gate_up = Duration::from_millis(5);
        profiler.swiglu = Duration::from_millis(6);
        profiler.ffn_down = Duration::from_millis(7);
        profiler.final_output = Duration::from_millis(8);
        profiler.pool_overhead = Duration::from_millis(9);
        profiler.sampling = Duration::from_millis(10);
        profiler.detokenize = Duration::from_millis(11);

        assert_eq!(profiler.total(), Duration::from_millis(66));
    }
}
