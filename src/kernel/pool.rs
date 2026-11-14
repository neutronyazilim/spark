use std::{
    cell::UnsafeCell,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use super::{q1_0::BlockQ1_0, q4_0::BlockQ4_0, q8_0::BlockQ8_0};
use crate::profile::ProfileStats;

#[repr(align(64))]
pub struct CachePadded<T>(pub T);

impl<T> std::ops::Deref for CachePadded<T> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> std::ops::DerefMut for CachePadded<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Clone, Copy)]
pub struct SingleGemvJob {
    pub weights: usize,
    pub input: usize,
    pub output: usize,
    pub blocks: usize,
    pub rows: usize,
    pub blocks_per_row: usize,
}

#[derive(Clone, Copy)]
pub struct FusedQkvJob {
    pub q_weights: usize,
    pub k_weights: usize,
    pub v_weights: usize,
    pub input: usize,
    pub q_out: usize,
    pub k_out: usize,
    pub v_out: usize,
    pub q_rows: usize,
    pub k_rows: usize,
    pub v_rows: usize,
    pub blocks_per_row: usize,
}

#[derive(Clone, Copy)]
pub struct FusedFfnJob {
    pub gate_weights: usize,
    pub up_weights: usize,
    pub input: usize,
    pub output: usize,
    pub rows: usize,
    pub blocks_per_row: usize,
}

#[derive(Clone, Copy)]
pub struct GatedDeltaJob {
    pub q: usize,
    pub k: usize,
    pub v: usize,
    pub gate: usize,
    pub beta: usize,
    pub state: usize,
    pub output: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub head_dim: usize,
    pub use_avx2: bool,
}

#[derive(Clone, Copy)]
enum TaskKind {
    None,
    Q8(SingleGemvJob),
    Q4(SingleGemvJob),
    PTQ1(SingleGemvJob),
    Q1(SingleGemvJob),
    GatedDelta(GatedDeltaJob),
    FusedQkv(FusedQkvJob),
    FusedFfn(FusedFfnJob),
}

#[derive(Clone, Copy)]
struct Task {
    kind: TaskKind,
    profile: bool,
}

struct Shared {
    generation: CachePadded<AtomicUsize>,
    shutdown: CachePadded<AtomicBool>,
    worker_done: Box<[CachePadded<AtomicUsize>]>,
    worker_compute_ns: Box<[CachePadded<AtomicU64>]>,
    task: CachePadded<UnsafeCell<Task>>,
}

unsafe impl Sync for Shared {}

pub struct ThreadPool {
    workers: Vec<JoinHandle<()>>,
    shared: Arc<Shared>,
    threads: usize,
}

impl ThreadPool {
    pub fn new(threads: usize) -> Result<Self, &'static str> {
        if threads == 0 {
            return Err("thread count must be greater than zero");
        }

        let worker_done = (0..threads)
            .map(|_| CachePadded(AtomicUsize::new(0)))
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let worker_compute_ns = (0..threads)
            .map(|_| CachePadded(AtomicU64::new(0)))
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let shared = Arc::new(Shared {
            generation: CachePadded(AtomicUsize::new(0)),
            shutdown: CachePadded(AtomicBool::new(false)),
            worker_done,
            worker_compute_ns,
            task: CachePadded(UnsafeCell::new(Task {
                kind: TaskKind::None,
                profile: false,
            })),
        });

        // Pin the main thread to core 0 if pinning is enabled
        if std::env::var_os("SPARK_NO_PIN").is_none() {
            pin_thread_to_core(0);
        }

        // Spawn (threads - 1) background workers; worker 0 is executed by the main thread.
        let mut workers = Vec::with_capacity(threads.saturating_sub(1));
        for worker_index in 1..threads {
            let shared = Arc::clone(&shared);
            workers.push(
                thread::Builder::new()
                    .name(format!("spark-gemv-{worker_index}"))
                    .spawn(move || {
                        if std::env::var_os("SPARK_NO_PIN").is_none() {
                            pin_thread_to_core(worker_index);
                        }
                        worker_loop(worker_index, threads, shared);
                    })
                    .map_err(|_| "failed to spawn GEMV worker")?,
            );
        }

        Ok(Self {
            workers,
            shared,
            threads,
        })
    }

    pub fn thread_count(&self) -> usize {
        self.threads
    }

    pub fn gemv_q8_0(
        &self,
        weights: &[BlockQ8_0],
        activations: &[f32],
        output: &mut [f32],
        columns: usize,
        profile: Option<&mut ProfileStats>,
    ) {
        self.dispatch(
            TaskKind::Q8(SingleGemvJob {
                weights: weights.as_ptr() as usize,
                input: activations.as_ptr() as usize,
                output: output.as_mut_ptr() as usize,
                blocks: weights.len(),
                rows: output.len(),
                blocks_per_row: columns / 32,
            }),
            profile,
        );
    }

    pub fn gemv_q4_0(
        &self,
        weights: &[BlockQ4_0],
        activations: &[f32],
        output: &mut [f32],
        columns: usize,
        profile: Option<&mut ProfileStats>,
    ) {
        self.dispatch(
            TaskKind::Q4(SingleGemvJob {
                weights: weights.as_ptr() as usize,
                input: activations.as_ptr() as usize,
                output: output.as_mut_ptr() as usize,
                blocks: weights.len(),
                rows: output.len(),
                blocks_per_row: columns / 32,
            }),
            profile,
        );
    }

    pub fn gemv_ptq1_0(
        &self,
        weights: &[crate::kernel::ternary::BlockPTQ1_0],
        activations: &[f32],
        output: &mut [f32],
        columns: usize,
        profile: Option<&mut ProfileStats>,
    ) {
        self.dispatch(
            TaskKind::PTQ1(SingleGemvJob {
                weights: weights.as_ptr() as usize,
                input: activations.as_ptr() as usize,
                output: output.as_mut_ptr() as usize,
                blocks: weights.len(),
                rows: output.len(),
                blocks_per_row: columns / 128,
            }),
            profile,
        );
    }

    pub fn gemv_q1_0(
        &self,
        weights: &[BlockQ1_0],
        activations: &[BlockQ8_0],
        output: &mut [f32],
        columns: usize,
        profile: Option<&mut ProfileStats>,
    ) {
        self.dispatch(
            TaskKind::Q1(SingleGemvJob {
                weights: weights.as_ptr() as usize,
                input: activations.as_ptr() as usize,
                output: output.as_mut_ptr() as usize,
                blocks: weights.len(),
                rows: output.len(),
                blocks_per_row: columns / 128,
            }),
            profile,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gemv_qkv_q8_0(
        &self,
        q_weights: &[BlockQ8_0],
        k_weights: &[BlockQ8_0],
        v_weights: &[BlockQ8_0],
        activations: &[f32],
        q_out: &mut [f32],
        k_out: &mut [f32],
        v_out: &mut [f32],
        columns: usize,
        profile: Option<&mut ProfileStats>,
    ) {
        self.dispatch(
            TaskKind::FusedQkv(FusedQkvJob {
                q_weights: q_weights.as_ptr() as usize,
                k_weights: k_weights.as_ptr() as usize,
                v_weights: v_weights.as_ptr() as usize,
                input: activations.as_ptr() as usize,
                q_out: q_out.as_mut_ptr() as usize,
                k_out: k_out.as_mut_ptr() as usize,
                v_out: v_out.as_mut_ptr() as usize,
                q_rows: q_out.len(),
                k_rows: k_out.len(),
                v_rows: v_out.len(),
                blocks_per_row: columns / 32,
            }),
            profile,
        );
    }

    pub fn gemv_fused_ffn_q8_0(
        &self,
        gate_weights: &[BlockQ8_0],
        up_weights: &[BlockQ8_0],
        activations: &[f32],
        output: &mut [f32],
        columns: usize,
        profile: Option<&mut ProfileStats>,
    ) {
        self.dispatch(
            TaskKind::FusedFfn(FusedFfnJob {
                gate_weights: gate_weights.as_ptr() as usize,
                up_weights: up_weights.as_ptr() as usize,
                input: activations.as_ptr() as usize,
                output: output.as_mut_ptr() as usize,
                rows: output.len(),
                blocks_per_row: columns / 32,
            }),
            profile,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_net_step(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        gate: &[f32],
        beta: &[f32],
        state: &mut [f32],
        output: &mut [f32],
        key_heads: usize,
        value_heads: usize,
        head_dim: usize,
        use_avx2: bool,
    ) {
        self.dispatch(
            TaskKind::GatedDelta(GatedDeltaJob {
                q: q.as_ptr() as usize,
                k: k.as_ptr() as usize,
                v: v.as_ptr() as usize,
                gate: gate.as_ptr() as usize,
                beta: beta.as_ptr() as usize,
                state: state.as_mut_ptr() as usize,
                output: output.as_mut_ptr() as usize,
                key_heads,
                value_heads,
                head_dim,
                use_avx2,
            }),
            None,
        );
    }

    fn dispatch(&self, kind: TaskKind, profile: Option<&mut ProfileStats>) {
        let is_profiling = profile.as_ref().is_some_and(|p| p.enabled);
        let task = Task {
            kind,
            profile: is_profiling,
        };

        unsafe {
            *self.shared.task.get() = task;
        }

        let started = if is_profiling {
            Some(Instant::now())
        } else {
            None
        };

        let target_gen = self.shared.generation.load(Ordering::Relaxed) + 1;
        self.shared.generation.store(target_gen, Ordering::Release);

        // Main thread executes chunk 0 directly!
        let main_start = if is_profiling {
            Some(Instant::now())
        } else {
            None
        };
        execute_task(task, 0, self.threads);
        if let Some(ms) = main_start {
            let ns = ms.elapsed().as_nanos() as u64;
            self.shared.worker_compute_ns[0].store(ns, Ordering::Relaxed);
        }
        self.shared.worker_done[0].store(target_gen, Ordering::Release);

        // Spin-wait for remaining workers (1..threads)
        for i in 1..self.threads {
            while self.shared.worker_done[i].load(Ordering::Acquire) != target_gen {
                core::hint::spin_loop();
            }
        }

        if let (Some(profile), Some(started)) = (profile, started) {
            let total_wall = started.elapsed();
            let max_compute_ns = (0..self.threads)
                .map(|i| self.shared.worker_compute_ns[i].load(Ordering::Relaxed))
                .max()
                .unwrap_or(0);
            let compute = Duration::from_nanos(max_compute_ns);
            let sync_overhead = total_wall.saturating_sub(compute);
            profile.pool_overhead += sync_overhead;
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        self.shared.generation.fetch_add(1, Ordering::Release);
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

fn worker_loop(worker_index: usize, workers: usize, shared: Arc<Shared>) {
    let mut local_gen = 0;
    let mut spins = 0u32;
    loop {
        // Fast spin-loop, backing off to yield if idle for a prolonged period
        while shared.generation.load(Ordering::Acquire) == local_gen {
            if shared.shutdown.load(Ordering::Relaxed) {
                return;
            }
            if spins < 2048 {
                core::hint::spin_loop();
                spins += 1;
            } else {
                thread::yield_now();
            }
        }
        spins = 0;

        local_gen = shared.generation.load(Ordering::Acquire);
        if shared.shutdown.load(Ordering::Relaxed) {
            return;
        }

        let task = unsafe { *shared.task.get() };

        let compute_start = if task.profile {
            Some(Instant::now())
        } else {
            None
        };

        execute_task(task, worker_index, workers);

        if let Some(compute_start) = compute_start {
            let elapsed_ns = compute_start.elapsed().as_nanos() as u64;
            shared.worker_compute_ns[worker_index].store(elapsed_ns, Ordering::Relaxed);
        }

        shared.worker_done[worker_index].store(local_gen, Ordering::Release);
    }
}

#[inline(always)]
fn execute_task(task: Task, worker_index: usize, workers: usize) {
    unsafe {
        match task.kind {
            TaskKind::None => {}
            TaskKind::Q8(job) => {
                let start = job.rows * worker_index / workers;
                let end = job.rows * (worker_index + 1) / workers;
                if start < end {
                    let weights =
                        std::slice::from_raw_parts(job.weights as *const BlockQ8_0, job.blocks);
                    let input =
                        std::slice::from_raw_parts(job.input as *const f32, job.blocks_per_row * 32);
                    let output = std::slice::from_raw_parts_mut(job.output as *mut f32, job.rows);
                    super::run_q8_rows(
                        &weights[start * job.blocks_per_row..end * job.blocks_per_row],
                        input,
                        &mut output[start..end],
                        job.blocks_per_row,
                    );
                }
            }
            TaskKind::Q4(job) => {
                let start = job.rows * worker_index / workers;
                let end = job.rows * (worker_index + 1) / workers;
                if start < end {
                    let weights =
                        std::slice::from_raw_parts(job.weights as *const BlockQ4_0, job.blocks);
                    let input =
                        std::slice::from_raw_parts(job.input as *const f32, job.blocks_per_row * 32);
                    let output = std::slice::from_raw_parts_mut(job.output as *mut f32, job.rows);
                    super::run_q4_rows(
                        &weights[start * job.blocks_per_row..end * job.blocks_per_row],
                        input,
                        &mut output[start..end],
                        job.blocks_per_row,
                    );
                }
            }
            TaskKind::PTQ1(job) => {
                let start = job.rows * worker_index / workers;
                let end = job.rows * (worker_index + 1) / workers;
                if start < end {
                    let weights =
                        std::slice::from_raw_parts(job.weights as *const crate::kernel::ternary::BlockPTQ1_0, job.blocks);
                    let input =
                        std::slice::from_raw_parts(job.input as *const f32, job.blocks_per_row * 128);
                    let output = std::slice::from_raw_parts_mut(job.output as *mut f32, job.rows);
                    crate::kernel::ternary::gemv_ptq1_0(
                        &weights[start * job.blocks_per_row..end * job.blocks_per_row],
                        input,
                        &mut output[start..end],
                        job.blocks_per_row,
                    );
                }
            }
            TaskKind::Q1(job) => {
                let start = job.rows * worker_index / workers;
                let end = job.rows * (worker_index + 1) / workers;
                if start < end {
                    let weights =
                        std::slice::from_raw_parts(job.weights as *const BlockQ1_0, job.blocks);
                    let input = std::slice::from_raw_parts(
                        job.input as *const BlockQ8_0,
                        job.blocks_per_row * 4,
                    );
                    let output = std::slice::from_raw_parts_mut(job.output as *mut f32, job.rows);
                    super::q1_0::gemv_q1_0_q8_0_rows(
                        &weights[start * job.blocks_per_row..end * job.blocks_per_row],
                        input,
                        &mut output[start..end],
                        job.blocks_per_row,
                    );
                }
            }
            TaskKind::GatedDelta(job) => {
                let head_start = job.value_heads * worker_index / workers;
                let head_end = job.value_heads * (worker_index + 1) / workers;
                if head_start < head_end {
                    let q = std::slice::from_raw_parts(
                        job.q as *const f32,
                        job.key_heads * job.head_dim,
                    );
                    let k = std::slice::from_raw_parts(
                        job.k as *const f32,
                        job.key_heads * job.head_dim,
                    );
                    let v = std::slice::from_raw_parts(
                        job.v as *const f32,
                        job.value_heads * job.head_dim,
                    );
                    let gate = std::slice::from_raw_parts(job.gate as *const f32, job.value_heads);
                    let beta = std::slice::from_raw_parts(job.beta as *const f32, job.value_heads);
                    let local_heads = head_end - head_start;
                    let state = std::slice::from_raw_parts_mut(
                        (job.state as *mut f32)
                            .add(head_start * job.head_dim * job.head_dim),
                        local_heads * job.head_dim * job.head_dim,
                    );
                    let output = std::slice::from_raw_parts_mut(
                        (job.output as *mut f32).add(head_start * job.head_dim),
                        local_heads * job.head_dim,
                    );
                    run_gated_delta_heads(
                        q,
                        k,
                        v,
                        gate,
                        beta,
                        state,
                        output,
                        job.key_heads,
                        job.head_dim,
                        head_start,
                        head_end,
                        job.use_avx2,
                    );
                }
            }
            TaskKind::FusedQkv(job) => {
                let total_rows = job.q_rows + job.k_rows + job.v_rows;
                let start = total_rows * worker_index / workers;
                let end = total_rows * (worker_index + 1) / workers;

                if start < end {
                    let input =
                        std::slice::from_raw_parts(job.input as *const f32, job.blocks_per_row * 32);

                    // Range 1: Q
                    let q_start = start.min(job.q_rows);
                    let q_end = end.min(job.q_rows);
                    if q_start < q_end {
                        let q_weights = std::slice::from_raw_parts(
                            job.q_weights as *const BlockQ8_0,
                            job.q_rows * job.blocks_per_row,
                        );
                        let q_out =
                            std::slice::from_raw_parts_mut(job.q_out as *mut f32, job.q_rows);
                        super::run_q8_rows(
                            &q_weights[q_start * job.blocks_per_row..q_end * job.blocks_per_row],
                            input,
                            &mut q_out[q_start..q_end],
                            job.blocks_per_row,
                        );
                    }

                    // Range 2: K
                    let k_offset = job.q_rows;
                    let k_limit = k_offset + job.k_rows;
                    let k_start = start.max(k_offset).min(k_limit);
                    let k_end = end.max(k_offset).min(k_limit);
                    if k_start < k_end {
                        let k_local_start = k_start - k_offset;
                        let k_local_end = k_end - k_offset;
                        let k_weights = std::slice::from_raw_parts(
                            job.k_weights as *const BlockQ8_0,
                            job.k_rows * job.blocks_per_row,
                        );
                        let k_out =
                            std::slice::from_raw_parts_mut(job.k_out as *mut f32, job.k_rows);
                        super::run_q8_rows(
                            &k_weights[k_local_start * job.blocks_per_row
                                ..k_local_end * job.blocks_per_row],
                            input,
                            &mut k_out[k_local_start..k_local_end],
                            job.blocks_per_row,
                        );
                    }

                    // Range 3: V
                    let v_offset = k_limit;
                    let v_limit = v_offset + job.v_rows;
                    let v_start = start.max(v_offset).min(v_limit);
                    let v_end = end.max(v_offset).min(v_limit);
                    if v_start < v_end {
                        let v_local_start = v_start - v_offset;
                        let v_local_end = v_end - v_offset;
                        let v_weights = std::slice::from_raw_parts(
                            job.v_weights as *const BlockQ8_0,
                            job.v_rows * job.blocks_per_row,
                        );
                        let v_out =
                            std::slice::from_raw_parts_mut(job.v_out as *mut f32, job.v_rows);
                        super::run_q8_rows(
                            &v_weights[v_local_start * job.blocks_per_row
                                ..v_local_end * job.blocks_per_row],
                            input,
                            &mut v_out[v_local_start..v_local_end],
                            job.blocks_per_row,
                        );
                    }
                }
            }
            TaskKind::FusedFfn(job) => {
                let start = job.rows * worker_index / workers;
                let end = job.rows * (worker_index + 1) / workers;
                if start < end {
                    let gate_weights = std::slice::from_raw_parts(
                        job.gate_weights as *const BlockQ8_0,
                        job.rows * job.blocks_per_row,
                    );
                    let up_weights = std::slice::from_raw_parts(
                        job.up_weights as *const BlockQ8_0,
                        job.rows * job.blocks_per_row,
                    );
                    let input =
                        std::slice::from_raw_parts(job.input as *const f32, job.blocks_per_row * 32);
                    let output = std::slice::from_raw_parts_mut(job.output as *mut f32, job.rows);
                    super::run_fused_ffn_q8_rows(
                        &gate_weights[start * job.blocks_per_row..end * job.blocks_per_row],
                        &up_weights[start * job.blocks_per_row..end * job.blocks_per_row],
                        input,
                        &mut output[start..end],
                        job.blocks_per_row,
                    );
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_gated_delta_heads(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    state: &mut [f32],
    output: &mut [f32],
    key_heads: usize,
    head_dim: usize,
    head_start: usize,
    head_end: usize,
    prefer_avx2: bool,
) {
    let scale = (head_dim as f32).sqrt().recip();
    let use_avx2 = prefer_avx2
        && std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("fma")
        && head_dim.is_multiple_of(8);
    for (local_head, value_head) in (head_start..head_end).enumerate() {
        let key_head = value_head % key_heads;
        let qh = &q[key_head * head_dim..(key_head + 1) * head_dim];
        let kh = &k[key_head * head_dim..(key_head + 1) * head_dim];
        let vh = &v[value_head * head_dim..(value_head + 1) * head_dim];
        let oh = &mut output[local_head * head_dim..(local_head + 1) * head_dim];
        let decay = gate[value_head].exp();
        let state_head = &mut state
            [local_head * head_dim * head_dim..(local_head + 1) * head_dim * head_dim];
        if use_avx2 {
            unsafe {
                gated_delta_head_avx2(qh, kh, vh, decay, beta[value_head], state_head, oh, scale);
            }
            continue;
        }
        for col in 0..head_dim {
            let column = &mut state_head[col * head_dim..(col + 1) * head_dim];
            let mut kv = 0.0f32;
            for i in 0..head_dim {
                kv = column[i].mul_add(kh[i], kv);
            }
            let delta = (vh[col] - decay * kv) * beta[value_head];
            let mut attn = 0.0f32;
            for i in 0..head_dim {
                column[i] = kh[i].mul_add(delta, decay * column[i]);
                attn = column[i].mul_add(qh[i], attn);
            }
            oh[col] = attn * scale;
        }
    }
}

/// # Safety
/// Caller must ensure AVX2/FMA support and lengths divisible by eight.
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn gated_delta_head_avx2(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    decay: f32,
    beta: f32,
    state: &mut [f32],
    output: &mut [f32],
    output_scale: f32,
) {
    use std::arch::x86_64::*;

    let n = q.len();
    let decay_v = _mm256_set1_ps(decay);
    for col in 0..n {
        let column = state.as_mut_ptr().add(col * n);
        let mut kv0 = _mm256_setzero_ps();
        let mut kv1 = _mm256_setzero_ps();
        for i in (0..n).step_by(16) {
            kv0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(column.add(i)),
                _mm256_loadu_ps(k.as_ptr().add(i)),
                kv0,
            );
            kv1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(column.add(i + 8)),
                _mm256_loadu_ps(k.as_ptr().add(i + 8)),
                kv1,
            );
        }
        let kv = hsum256(_mm256_add_ps(kv0, kv1));
        let delta_v = _mm256_set1_ps((v[col] - decay * kv) * beta);
        let mut out0 = _mm256_setzero_ps();
        let mut out1 = _mm256_setzero_ps();
        for i in (0..n).step_by(16) {
            let s0 = _mm256_loadu_ps(column.add(i));
            let s1 = _mm256_loadu_ps(column.add(i + 8));
            let new0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(k.as_ptr().add(i)),
                delta_v,
                _mm256_mul_ps(s0, decay_v),
            );
            let new1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(k.as_ptr().add(i + 8)),
                delta_v,
                _mm256_mul_ps(s1, decay_v),
            );
            _mm256_storeu_ps(column.add(i), new0);
            _mm256_storeu_ps(column.add(i + 8), new1);
            out0 = _mm256_fmadd_ps(new0, _mm256_loadu_ps(q.as_ptr().add(i)), out0);
            out1 = _mm256_fmadd_ps(new1, _mm256_loadu_ps(q.as_ptr().add(i + 8)), out1);
        }
        output[col] = hsum256(_mm256_add_ps(out0, out1)) * output_scale;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn hsum256(value: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let sum = _mm_add_ps(_mm256_castps256_ps128(value), _mm256_extractf128_ps(value, 1));
    let sum = _mm_hadd_ps(sum, sum);
    let sum = _mm_hadd_ps(sum, sum);
    _mm_cvtss_f32(sum)
}

#[cfg(target_os = "linux")]
pub fn pin_thread_to_core(worker_index: usize) {
    // Topology-aware pinning order:
    // Prioritize 4 P-cores (physical cores first: 0, 2, 4, 6, then their SMT siblings: 1, 3, 5, 7),
    // and only then spill over to E-cores (8, 9, 10, 11).
    let core_order: [usize; 12] = [0, 2, 4, 6, 1, 3, 5, 7, 8, 9, 10, 11];
    let core_id = if worker_index < core_order.len() {
        core_order[worker_index]
    } else {
        worker_index
    };

    let mut mask = [0u64; 16]; // supports up to 1024 cores
    if core_id < 1024 {
        mask[core_id / 64] |= 1u64 << (core_id % 64);
        unsafe {
            std::arch::asm!(
                "syscall",
                in("rax") 203, // SYS_sched_setaffinity
                in("rdi") 0,   // 0 = calling thread
                in("rsi") std::mem::size_of_val(&mask),
                in("rdx") mask.as_ptr(),
                out("rcx") _,
                out("r11") _,
                lateout("rax") _,
            );
        }
    }
}

#[cfg(not(target_os = "linux"))]
pub fn pin_thread_to_core(_core_id: usize) {}

#[cfg(test)]
mod tests {
    use super::ThreadPool;
    use crate::kernel::{fallback, q8_0::BlockQ8_0};

    #[test]
    fn static_workers_match_scalar_gemv() {
        let pool = ThreadPool::new(2).expect("pool");
        let weights = vec![
            BlockQ8_0 {
                d: 0x3c00,
                qs: [1; 32],
            };
            4
        ];
        let input = [0.5f32; 32];
        let mut parallel = [0.0f32; 4];
        let mut scalar = [0.0f32; 4];
        pool.gemv_q8_0(&weights, &input, &mut parallel, 32, None);
        fallback::gemv_q8_0(&weights, &input, &mut scalar, 1);
        assert_eq!(parallel, scalar);
    }

    #[test]
    fn fused_qkv_matches_separate_gemv() {
        let pool = ThreadPool::new(3).expect("pool");
        let q_weights = vec![
            BlockQ8_0 {
                d: 0x3c00,
                qs: [2; 32],
            };
            4
        ];
        let k_weights = vec![
            BlockQ8_0 {
                d: 0x3c00,
                qs: [3; 32],
            };
            2
        ];
        let v_weights = vec![
            BlockQ8_0 {
                d: 0x3c00,
                qs: [4; 32],
            };
            2
        ];
        let input = [0.25f32; 32];
        let mut q_out = [0.0f32; 4];
        let mut k_out = [0.0f32; 2];
        let mut v_out = [0.0f32; 2];

        pool.gemv_qkv_q8_0(
            &q_weights,
            &k_weights,
            &v_weights,
            &input,
            &mut q_out,
            &mut k_out,
            &mut v_out,
            32,
            None,
        );

        let mut ref_q = [0.0f32; 4];
        let mut ref_k = [0.0f32; 2];
        let mut ref_v = [0.0f32; 2];
        fallback::gemv_q8_0(&q_weights, &input, &mut ref_q, 1);
        fallback::gemv_q8_0(&k_weights, &input, &mut ref_k, 1);
        fallback::gemv_q8_0(&v_weights, &input, &mut ref_v, 1);

        assert_eq!(q_out, ref_q);
        assert_eq!(k_out, ref_k);
        assert_eq!(v_out, ref_v);
    }

    #[test]
    fn fused_ffn_matches_separate_gemv_and_swiglu() {
        let pool = ThreadPool::new(4).expect("pool");
        let gate_weights = vec![
            BlockQ8_0 {
                d: 0x3c00,
                qs: [1; 32],
            };
            4
        ];
        let up_weights = vec![
            BlockQ8_0 {
                d: 0x3c00,
                qs: [2; 32],
            };
            4
        ];
        let input = [0.125f32; 32];
        let mut fused_out = [0.0f32; 4];

        pool.gemv_fused_ffn_q8_0(&gate_weights, &up_weights, &input, &mut fused_out, 32, None);

        let mut ref_gate = [0.0f32; 4];
        let mut ref_up = [0.0f32; 4];
        fallback::gemv_q8_0(&gate_weights, &input, &mut ref_gate, 1);
        fallback::gemv_q8_0(&up_weights, &input, &mut ref_up, 1);
        crate::model::ops::swiglu(&mut ref_gate, &ref_up);

        for (a, b) in fused_out.iter().zip(ref_gate.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
