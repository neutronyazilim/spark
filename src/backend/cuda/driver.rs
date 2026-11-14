use std::ffi::{c_char, c_void, CStr, CString};

const RTLD_NOW: i32 = 2;

extern "C" {
    fn dlopen(filename: *const c_char, flags: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> i32;
}

pub struct CudaDriver {
    _handle: *mut c_void,
    pub cu_init: unsafe extern "C" fn(u32) -> i32,
    pub cu_device_get: unsafe extern "C" fn(*mut i32, i32) -> i32,
    pub cu_device_get_name: unsafe extern "C" fn(*mut c_char, i32, i32) -> i32,
    pub cu_device_total_mem: unsafe extern "C" fn(*mut usize, i32) -> i32,
    pub cu_ctx_create: unsafe extern "C" fn(*mut usize, u32, i32) -> i32,
    pub cu_ctx_destroy: unsafe extern "C" fn(usize) -> i32,
    pub cu_module_load_data: unsafe extern "C" fn(*mut usize, *const c_char) -> i32,
    pub cu_module_load_data_ex: unsafe extern "C" fn(
        *mut usize,
        *const c_void,
        u32,
        *mut u32,
        *mut *mut c_void,
    ) -> i32,
    pub cu_module_get_function: unsafe extern "C" fn(*mut usize, usize, *const c_char) -> i32,
    pub cu_mem_alloc: unsafe extern "C" fn(*mut usize, usize) -> i32,
    pub cu_mem_free: unsafe extern "C" fn(usize) -> i32,
    pub cu_memset_d8: unsafe extern "C" fn(usize, u8, usize) -> i32,
    pub cu_memcpy_htod: unsafe extern "C" fn(usize, *const c_void, usize) -> i32,
    pub cu_memcpy_dtoh: unsafe extern "C" fn(*mut c_void, usize, usize) -> i32,
    pub cu_mem_host_alloc: unsafe extern "C" fn(*mut c_void, usize, u32) -> i32,
    pub cu_mem_host_get_device_pointer:
        unsafe extern "C" fn(*mut usize, *mut c_void, u32) -> i32,
    pub cu_mem_free_host: unsafe extern "C" fn(*mut c_void) -> i32,
    pub cu_stream_create: unsafe extern "C" fn(*mut usize, u32) -> i32,
    pub cu_stream_destroy: unsafe extern "C" fn(usize) -> i32,
    pub cu_stream_synchronize: unsafe extern "C" fn(usize) -> i32,
    pub cu_stream_begin_capture: unsafe extern "C" fn(usize, i32) -> i32,
    pub cu_stream_end_capture: unsafe extern "C" fn(usize, *mut usize) -> i32,
    pub cu_graph_instantiate: unsafe extern "C" fn(*mut usize, usize, *mut c_void, *mut c_void, usize) -> i32,
    pub cu_graph_launch: unsafe extern "C" fn(usize, usize) -> i32,
    pub cu_graph_exec_destroy: unsafe extern "C" fn(usize) -> i32,
    pub cu_graph_destroy: unsafe extern "C" fn(usize) -> i32,
    pub cu_event_create: unsafe extern "C" fn(*mut usize, u32) -> i32,
    pub cu_event_record: unsafe extern "C" fn(usize, usize) -> i32,
    pub cu_event_elapsed_time: unsafe extern "C" fn(*mut f32, usize, usize) -> i32,
    pub cu_event_destroy: unsafe extern "C" fn(usize) -> i32,
    #[allow(clippy::type_complexity)]
    pub cu_launch_kernel: unsafe extern "C" fn(
        usize,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        usize,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> i32,
    pub cu_ctx_synchronize: unsafe extern "C" fn() -> i32,
}

unsafe impl Send for CudaDriver {}
unsafe impl Sync for CudaDriver {}

impl CudaDriver {
    #[allow(clippy::missing_transmute_annotations)]
    pub fn load() -> Option<Self> {
        let names = ["libcuda.so.1", "libcuda.so"];
        let mut handle = std::ptr::null_mut();

        for name in names {
            let c_name = CString::new(name).ok()?;
            let h = unsafe { dlopen(c_name.as_ptr(), RTLD_NOW) };
            if !h.is_null() {
                handle = h;
                break;
            }
        }

        if handle.is_null() {
            return None;
        }

        unsafe {
            #[allow(clippy::missing_transmute_annotations)]
            macro_rules! load_sym {
                ($sym:expr) => {{
                    let s_name = CString::new($sym).ok()?;
                    let ptr = dlsym(handle, s_name.as_ptr());
                    if ptr.is_null() {
                        dlclose(handle);
                        return None;
                    }
                    std::mem::transmute(ptr)
                }};
            }

            Some(Self {
                _handle: handle,
                cu_init: load_sym!("cuInit"),
                cu_device_get: load_sym!("cuDeviceGet"),
                cu_device_get_name: load_sym!("cuDeviceGetName"),
                cu_device_total_mem: load_sym!("cuDeviceTotalMem_v2"),
                cu_ctx_create: load_sym!("cuCtxCreate_v2"),
                cu_ctx_destroy: load_sym!("cuCtxDestroy_v2"),
                cu_module_load_data: load_sym!("cuModuleLoadData"),
                cu_module_load_data_ex: load_sym!("cuModuleLoadDataEx"),
                cu_module_get_function: load_sym!("cuModuleGetFunction"),
                cu_mem_alloc: load_sym!("cuMemAlloc_v2"),
                cu_mem_free: load_sym!("cuMemFree_v2"),
                cu_memset_d8: load_sym!("cuMemsetD8_v2"),
                cu_memcpy_htod: load_sym!("cuMemcpyHtoD_v2"),
                cu_memcpy_dtoh: load_sym!("cuMemcpyDtoH_v2"),
                cu_mem_host_alloc: load_sym!("cuMemHostAlloc"),
                cu_mem_host_get_device_pointer: load_sym!("cuMemHostGetDevicePointer_v2"),
                cu_mem_free_host: load_sym!("cuMemFreeHost"),
                cu_stream_create: load_sym!("cuStreamCreate"),
                cu_stream_destroy: load_sym!("cuStreamDestroy"),
                cu_stream_synchronize: load_sym!("cuStreamSynchronize"),
                cu_stream_begin_capture: load_sym!("cuStreamBeginCapture"),
                cu_stream_end_capture: load_sym!("cuStreamEndCapture"),
                cu_graph_instantiate: load_sym!("cuGraphInstantiate"),
                cu_graph_launch: load_sym!("cuGraphLaunch"),
                cu_graph_exec_destroy: load_sym!("cuGraphExecDestroy"),
                cu_graph_destroy: load_sym!("cuGraphDestroy"),
                cu_event_create: load_sym!("cuEventCreate"),
                cu_event_record: load_sym!("cuEventRecord"),
                cu_event_elapsed_time: load_sym!("cuEventElapsedTime"),
                cu_event_destroy: load_sym!("cuEventDestroy"),
                cu_launch_kernel: load_sym!("cuLaunchKernel"),
                cu_ctx_synchronize: load_sym!("cuCtxSynchronize"),
            })
        }
    }
}

const CU_MEMHOSTALLOC_DEVICEMAP: u32 = 0x02;
const CU_STREAM_NON_BLOCKING: u32 = 0x01;

/// Handle of the stream running the persistent keepalive kernel, or 0 when it
/// is not running.
const KEEPALIVE_IDLE: usize = 0;

pub struct CudaDevice {
    pub driver: CudaDriver,
    pub ctx: usize,
    pub name: String,
    pub total_mem_bytes: usize,
    pub fn_gemv_q8: usize,
    pub fn_gemv_q4: usize,
    pub fn_fused_qkv_q8: usize,
    pub fn_fused_qkv_q4: usize,
    pub fn_fused_ffn_q8: usize,
    pub fn_fused_ffn_q4: usize,
    pub fn_gemv_q8_v2: usize,
    pub fn_fused_qkv_q8_v2: usize,
    pub fn_fused_ffn_q8_v2: usize,
    pub fn_gemv_q8_v3: usize,
    pub fn_fused_ffn_q8_v3: usize,
    pub fn_fused_qkv_q8_v3: usize,
    use_q8_v3: bool,
    use_ffn_q8_v3: bool,
    pub fn_argmax_f32: usize,
    pub fn_top64_sample: usize,
    pub fn_gemv_ternary: usize,
    pub fn_gemv_q1_0: usize,
    pub fn_gemv_q1_0_v2: usize,
    pub fn_gemv_q1_0_v3: usize,
    pub fn_gemv_q1_0_bitplane: usize,
    pub fn_gemv_q1_0_u10: usize,
    pub fn_gemv_q1_0_prmt: usize,
    pub fn_gemv_q1_0_u10_tiled: usize,
    pub fn_gemv_q2_0: usize,
    pub fn_rmsnorm_f32: usize,
    pub fn_rmsnorm_f32_v2: usize,
    pub fn_add_rmsnorm_f32_v2: usize,
    pub fn_gated_delta_f32: usize,
    pub fn_add_f32: usize,
    pub fn_add_bias_f32: usize,
    pub fn_swiglu_f32: usize,
    pub fn_rope_store_f32: usize,
    pub fn_attn_scores_f32: usize,
    pub fn_attn_weighted_f32: usize,
    pub fn_quantize_q8_0_128: usize,
    pub fn_qwen35_q_gate_rope: usize,
    pub fn_qwen35_kv_rope_store: usize,
    pub fn_qwen35_attention_gate: usize,
    pub fn_qwen35_conv_silu: usize,
    pub fn_qwen35_qk_l2_norm: usize,
    pub fn_qwen35_decay_beta: usize,
    pub fn_qwen35_gated_rmsnorm: usize,
    pub fn_keepalive: usize,
    keepalive_host: *mut u32,
    keepalive_dev: usize,
    keepalive_stream: std::sync::atomic::AtomicUsize,
    /// Dedicated non-legacy stream for all real kernel launches. CUDA graph
    /// capture is not supported on the legacy default stream (error 900), so
    /// every launch and `synchronize()` go through this handle.
    pub stream: usize,
    use_rms_v2: bool,
    use_q1_v2: bool,
    use_q1_v3: bool,
    use_q1_bitplane: bool,
    use_q1_prmt: bool,
}

unsafe impl Send for CudaDevice {}
unsafe impl Sync for CudaDevice {}

impl CudaDevice {
    pub fn init() -> Option<Self> {
        if std::env::var_os("SPARK_NO_CUDA").is_some() {
            return None;
        }

        let driver = CudaDriver::load()?;
        // Diagnostics helper: CUDA init failure would otherwise degrade to the
        // CPU path silently, which is nearly impossible to notice from the
        // outside.
        macro_rules! bail {
            ($ctx:expr, $stage:expr, $code:expr) => {{
                eprintln!(
                    "\x1b[33m[spark] CUDA init failed at {} (error {}), using CPU\x1b[0m",
                    $stage, $code
                );
                (driver.cu_ctx_destroy)($ctx);
                return None;
            }};
        }
        unsafe {
            if (driver.cu_init)(0) != 0 {
                eprintln!("\x1b[33m[spark] cuInit failed, using CPU\x1b[0m");
                return None;
            }

            let mut dev = 0;
            if (driver.cu_device_get)(&mut dev, 0) != 0 {
                eprintln!("\x1b[33m[spark] cuDeviceGet failed, using CPU\x1b[0m");
                return None;
            }

            let mut name_buf = [0i8; 256];
            (driver.cu_device_get_name)(name_buf.as_mut_ptr(), 256, dev);
            let name = CStr::from_ptr(name_buf.as_ptr())
                .to_string_lossy()
                .into_owned();

            let mut total_mem_bytes = 0usize;
            (driver.cu_device_total_mem)(&mut total_mem_bytes, dev);

            let mut ctx = 0usize;
            if (driver.cu_ctx_create)(&mut ctx, 0, dev) != 0 {
                eprintln!("\x1b[33m[spark] cuCtxCreate failed, using CPU\x1b[0m");
                return None;
            }

            // Load embedded PTX module
            let ptx_src = include_str!("../../../kernels/gemv.ptx");
            let ptx_cstr = CString::new(ptx_src).ok()?;
            let mut module = 0usize;
            const CU_JIT_ERROR_LOG_BUFFER: u32 = 5;
            const CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: u32 = 6;
            let mut error_log = [0i8; 8192];
            let mut error_log_size = error_log.len();
            let mut options = [CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES];
            let mut option_values = [
                error_log.as_mut_ptr().cast::<c_void>(),
                (&mut error_log_size as *mut usize).cast::<c_void>(),
            ];
            let module_result = (driver.cu_module_load_data_ex)(
                &mut module,
                ptx_cstr.as_ptr().cast::<c_void>(),
                options.len() as u32,
                options.as_mut_ptr(),
                option_values.as_mut_ptr(),
            );
            if module_result != 0 {
                let log = CStr::from_ptr(error_log.as_ptr()).to_string_lossy();
                if !log.is_empty() {
                    eprintln!("\x1b[33m[spark] CUDA PTX JIT: {log}\x1b[0m");
                }
                bail!(ctx, "PTX module load", "invalid kernels/gemv.ptx");
            }

            let mut fn_gemv_q8 = 0usize;
            let n_gemv = CString::new("gemv_q8_0").ok()?;
            if (driver.cu_module_get_function)(&mut fn_gemv_q8, module, n_gemv.as_ptr()) != 0 {
                bail!(ctx, "kernel lookup gemv_q8_0", "missing from module");
            }

            let mut fn_gemv_q4 = 0usize;
            let n_gemv_q4 = CString::new("gemv_q4_0").ok()?;
            if (driver.cu_module_get_function)(&mut fn_gemv_q4, module, n_gemv_q4.as_ptr()) != 0 {
                bail!(ctx, "kernel lookup gemv_q4_0", "missing from module");
            }

            let mut fn_fused_qkv_q8 = 0usize;
            let n_qkv = CString::new("gemv_fused_qkv_q8_0").ok()?;
            if (driver.cu_module_get_function)(&mut fn_fused_qkv_q8, module, n_qkv.as_ptr()) != 0 {
                bail!(ctx, "kernel lookup gemv_fused_qkv_q8_0", "missing from module");
            }

            let mut fn_fused_qkv_q4 = 0usize;
            let n_qkv_q4 = CString::new("gemv_fused_qkv_q4_0").ok()?;
            if (driver.cu_module_get_function)(&mut fn_fused_qkv_q4, module, n_qkv_q4.as_ptr()) != 0 {
                bail!(ctx, "kernel lookup gemv_fused_qkv_q4_0", "missing from module");
            }

            let mut fn_fused_ffn_q8 = 0usize;
            let n_ffn = CString::new("gemv_fused_ffn_q8_0").ok()?;
            if (driver.cu_module_get_function)(&mut fn_fused_ffn_q8, module, n_ffn.as_ptr()) != 0 {
                bail!(ctx, "kernel lookup gemv_fused_ffn_q8_0", "missing from module");
            }

            let mut fn_fused_ffn_q4 = 0usize;
            let n_ffn_q4 = CString::new("gemv_fused_ffn_q4_0").ok()?;
            if (driver.cu_module_get_function)(&mut fn_fused_ffn_q4, module, n_ffn_q4.as_ptr()) != 0 {
                bail!(ctx, "kernel lookup gemv_fused_ffn_q4_0", "missing from module");
            }

            let mut fn_argmax_f32 = 0usize;
            let n_argmax = CString::new("argmax_f32").ok()?;
            if (driver.cu_module_get_function)(&mut fn_argmax_f32, module, n_argmax.as_ptr()) != 0 {
                bail!(ctx, "kernel lookup argmax_f32", "missing from module");
            }

            let mut fn_top64_sample = 0usize;
            let n_top64 = CString::new("top64_sample").ok()?;
            if (driver.cu_module_get_function)(&mut fn_top64_sample, module, n_top64.as_ptr()) != 0 {
                bail!(ctx, "kernel lookup top64_sample", "missing from module");
            }

            let mut fn_gemv_ternary = 0usize;
            let n_ternary = CString::new("gemv_ternary").ok()?;
            if (driver.cu_module_get_function)(&mut fn_gemv_ternary, module, n_ternary.as_ptr()) != 0 {
                bail!(ctx, "kernel lookup gemv_ternary", "missing from module");
            }

            let mut fn_gemv_q2_0 = 0usize;
            let n_q2 = CString::new("gemv_q2_0").ok()?;
            if (driver.cu_module_get_function)(&mut fn_gemv_q2_0, module, n_q2.as_ptr()) != 0 {
                bail!(ctx, "kernel lookup gemv_q2_0", "missing from module");
            }

            macro_rules! load_fn {
                ($name:expr) => {{
                    let n = CString::new($name).ok()?;
                    let mut handle = 0usize;
                    if (driver.cu_module_get_function)(&mut handle, module, n.as_ptr()) != 0 {
                        bail!(ctx, "kernel lookup", $name);
                    }
                    handle
                }};
            }

            let fn_gemv_q1_0 = load_fn!("gemv_q1_0");
            let fn_gemv_q1_0_v2 = load_fn!("gemv_q1_0_v2");
            let fn_gemv_q1_0_v3 = load_fn!("gemv_q1_0_v3");
            let fn_gemv_q1_0_bitplane = load_fn!("gemv_q1_0_bitplane");
            let fn_gemv_q1_0_u10 = load_fn!("gemv_q1_0_u10");
            let fn_gemv_q1_0_prmt = load_fn!("gemv_q1_0_prmt");
            let fn_gemv_q1_0_u10_tiled = load_fn!("gemv_q1_0_u10_tiled");
            let fn_rmsnorm_f32 = load_fn!("rmsnorm_f32");
            let fn_rmsnorm_f32_v2 = load_fn!("rmsnorm_f32_v2");
            let fn_add_rmsnorm_f32_v2 = load_fn!("add_rmsnorm_f32_v2");
            let fn_gated_delta_f32 = load_fn!("gated_delta_f32");
            let fn_add_f32 = load_fn!("add_f32");
            let fn_add_bias_f32 = load_fn!("add_bias_f32");
            let fn_swiglu_f32 = load_fn!("swiglu_f32");
            let fn_rope_store_f32 = load_fn!("rope_store_f32");
            let fn_attn_scores_f32 = load_fn!("attn_scores_f32");
            let fn_attn_weighted_f32 = load_fn!("attn_weighted_f32");
            let fn_quantize_q8_0_128 = load_fn!("quantize_q8_0_128");
            let fn_qwen35_q_gate_rope = load_fn!("qwen35_q_gate_rope");
            let fn_qwen35_kv_rope_store = load_fn!("qwen35_kv_rope_store");
            let fn_qwen35_attention_gate = load_fn!("qwen35_attention_gate");
            let fn_qwen35_conv_silu = load_fn!("qwen35_conv_silu");
            let fn_qwen35_qk_l2_norm = load_fn!("qwen35_qk_l2_norm");
            let fn_qwen35_decay_beta = load_fn!("qwen35_decay_beta");
            let fn_qwen35_gated_rmsnorm = load_fn!("qwen35_gated_rmsnorm");
            let fn_keepalive = load_fn!("keepalive");
            let fn_gemv_q8_v2 = load_fn!("gemv_q8_0_v2");
            let fn_fused_qkv_q8_v2 = load_fn!("gemv_fused_qkv_q8_0_v2");
            let fn_fused_ffn_q8_v2 = load_fn!("gemv_fused_ffn_q8_0_v2");
            let fn_gemv_q8_v3 = load_fn!("gemv_q8_0_v3");
            let fn_fused_ffn_q8_v3 = load_fn!("gemv_fused_ffn_q8_0_v3");
            let fn_fused_qkv_q8_v3 = load_fn!("gemv_fused_qkv_q8_0_v3");

            // Host-mapped flag for the keepalive kernel (zero-copy, polled by
            // the device each iteration).
            let mut keepalive_host: *mut c_void = std::ptr::null_mut();
            if (driver.cu_mem_host_alloc)(
                &mut keepalive_host as *mut *mut c_void as *mut c_void,
                4,
                CU_MEMHOSTALLOC_DEVICEMAP,
            ) != 0
            {
                (driver.cu_ctx_destroy)(ctx);
                return None;
            }
            let mut keepalive_dev = 0usize;
            if (driver.cu_mem_host_get_device_pointer)(&mut keepalive_dev, keepalive_host, 0) != 0 {
                (driver.cu_ctx_destroy)(ctx);
                return None;
            }
            *(keepalive_host as *mut u32) = 0;
            // Dedicated non-legacy stream for all launches + sync (graph
            // capture is unsupported on the legacy default stream).
            let mut stream = 0usize;
            {
                let res = (driver.cu_stream_create)(&mut stream, CU_STREAM_NON_BLOCKING);
                if res != 0 {
                    eprintln!("[spark] warning: cuStreamCreate failed ({res}); using legacy stream");
                    stream = 0;
                }
            }
            Some(Self {
                driver,
                ctx,
                name,
                total_mem_bytes,
                fn_gemv_q8,
                fn_gemv_q4,
                fn_fused_qkv_q8,
                fn_fused_qkv_q4,
                fn_fused_ffn_q8,
                fn_fused_ffn_q4,
                fn_gemv_q8_v2,
                fn_fused_qkv_q8_v2,
                fn_fused_ffn_q8_v2,
                fn_gemv_q8_v3,
                fn_fused_ffn_q8_v3,
                fn_fused_qkv_q8_v3,
                use_q8_v3: std::env::var_os("SPARK_NO_Q8_V3").is_none(),
                use_ffn_q8_v3: std::env::var_os("SPARK_Q8_FFN_V3").is_some(),
                fn_argmax_f32,
                fn_top64_sample,
                fn_gemv_ternary,
                fn_gemv_q1_0,
                fn_gemv_q1_0_v2,
                fn_gemv_q1_0_v3,
                fn_gemv_q1_0_bitplane,
                fn_gemv_q1_0_u10,
                fn_gemv_q1_0_prmt,
                fn_gemv_q1_0_u10_tiled,
                fn_gemv_q2_0,
                fn_rmsnorm_f32,
                fn_rmsnorm_f32_v2,
                fn_add_rmsnorm_f32_v2,
                fn_gated_delta_f32,
                fn_add_f32,
                fn_add_bias_f32,
                fn_swiglu_f32,
                fn_rope_store_f32,
                fn_attn_scores_f32,
                fn_attn_weighted_f32,
                fn_quantize_q8_0_128,
                fn_qwen35_q_gate_rope,
                fn_qwen35_kv_rope_store,
                fn_qwen35_attention_gate,
                fn_qwen35_conv_silu,
                fn_qwen35_qk_l2_norm,
                fn_qwen35_decay_beta,
                fn_qwen35_gated_rmsnorm,
                fn_keepalive,
                keepalive_host: keepalive_host as *mut u32,
                keepalive_dev,
                keepalive_stream: std::sync::atomic::AtomicUsize::new(KEEPALIVE_IDLE),
                stream,
                use_rms_v2: std::env::var_os("SPARK_NO_RMS_V2").is_none(),
                use_q1_v2: std::env::var_os("SPARK_NO_Q1_V2").is_none(),
                use_q1_v3: std::env::var_os("SPARK_Q1_V3").is_some(),
                use_q1_bitplane: std::env::var_os("SPARK_Q1_BITPLANE").is_some(),
                use_q1_prmt: std::env::var_os("SPARK_Q1_PRMT").is_some()
                    || (std::env::var_os("SPARK_NO_Q1_PRMT").is_none()
                        && std::env::var_os("SPARK_Q1_V3").is_none()
                        && std::env::var_os("SPARK_Q1_BITPLANE").is_none()),
            })
        }
    }

    pub fn alloc(&self, bytes: usize) -> Result<usize, String> {
        let mut ptr = 0usize;
        let res = unsafe { (self.driver.cu_mem_alloc)(&mut ptr, bytes) };
        if res != 0 {
            Err(format!("cuMemAlloc failed with error code {res}"))
        } else {
            Ok(ptr)
        }
    }

    /// Allocate host memory that is directly addressable by the GPU.
    pub fn alloc_mapped(&self, bytes: usize) -> Result<(*mut u8, usize), String> {
        let mut host: *mut c_void = std::ptr::null_mut();
        let res = unsafe {
            (self.driver.cu_mem_host_alloc)(
                (&mut host as *mut *mut c_void).cast::<c_void>(),
                bytes,
                CU_MEMHOSTALLOC_DEVICEMAP,
            )
        };
        if res != 0 {
            return Err(format!("cuMemHostAlloc failed with error code {res}"));
        }
        let mut device_ptr = 0usize;
        let res = unsafe {
            (self.driver.cu_mem_host_get_device_pointer)(&mut device_ptr, host, 0)
        };
        if res != 0 {
            unsafe { (self.driver.cu_mem_free_host)(host) };
            return Err(format!(
                "cuMemHostGetDevicePointer failed with error code {res}"
            ));
        }
        unsafe { std::ptr::write_bytes(host, 0, bytes) };
        Ok((host.cast::<u8>(), device_ptr))
    }

    pub fn free_mapped(&self, host: *mut u8) {
        if !host.is_null() {
            unsafe { (self.driver.cu_mem_free_host)(host.cast::<c_void>()) };
        }
    }

    pub fn free(&self, ptr: usize) {
        if ptr != 0 {
            unsafe {
                (self.driver.cu_mem_free)(ptr);
            }
        }
    }

    pub fn memset_zero(&self, ptr: usize, bytes: usize) -> Result<(), String> {
        let res = unsafe { (self.driver.cu_memset_d8)(ptr, 0, bytes) };
        if res != 0 {
            Err(format!("cuMemsetD8 failed with error code {res}"))
        } else {
            Ok(())
        }
    }

    pub fn copy_to_device(&self, dst: usize, src: &[u8]) -> Result<(), String> {
        let res = unsafe {
            (self.driver.cu_memcpy_htod)(dst, src.as_ptr().cast(), src.len())
        };
        if res != 0 {
            Err(format!("cuMemcpyHtoD failed with error code {res}"))
        } else {
            Ok(())
        }
    }

    pub fn copy_from_device(&self, dst: &mut [u8], src: usize) -> Result<(), String> {
        // cuMemcpyDtoH runs on the legacy stream and does NOT wait for kernels
        // launched on the device stream; order the two explicitly.
        self.synchronize()?;
        let res = unsafe {
            (self.driver.cu_memcpy_dtoh)(dst.as_mut_ptr().cast(), src, dst.len())
        };
        if res != 0 {
            Err(format!("cuMemcpyDtoH failed with error code {res}"))
        } else {
            Ok(())
        }
    }

    pub fn synchronize(&self) -> Result<(), String> {
        // Synchronize the stream that carries all real work (see CudaDevice::stream),
        // rather than the whole context: a context-wide sync would also wait for the
        // persistent keepalive kernel, which never exits by design.
        let res = unsafe { (self.driver.cu_stream_synchronize)(self.stream) };
        if res != 0 {
            Err(format!("cuStreamSynchronize failed with error code {res}"))
        } else {
            Ok(())
        }
    }

    pub fn event_create(&self) -> Result<usize, String> {
        let mut ev = 0usize;
        let res = unsafe { (self.driver.cu_event_create)(&mut ev, 0) };
        if res != 0 {
            Err(format!("cuEventCreate failed with error code {res}"))
        } else {
            Ok(ev)
        }
    }

    pub fn event_record(&self, ev: usize) -> Result<(), String> {
        let res = unsafe { (self.driver.cu_event_record)(ev, self.stream) };
        if res != 0 {
            Err(format!("cuEventRecord failed with error code {res}"))
        } else {
            Ok(())
        }
    }

    pub fn event_elapsed_ms(&self, start: usize, end: usize) -> Result<f32, String> {
        // cuEventElapsedTime does not block; wait for every recorded event first.
        self.synchronize()?;
        let mut ms = 0f32;
        let res = unsafe { (self.driver.cu_event_elapsed_time)(&mut ms, start, end) };
        if res != 0 {
            Err(format!("cuEventElapsedTime failed with error code {res}"))
        } else {
            Ok(ms)
        }
    }

    pub fn event_destroy(&self, ev: usize) {
        unsafe { (self.driver.cu_event_destroy)(ev) };
    }

    /// Begin stream capture on the device stream. Kernels launched while the
    /// capture is active are recorded into a graph instead of executing.
    pub fn begin_capture(&self) -> Result<(), String> {
        let res = unsafe { (self.driver.cu_stream_begin_capture)(self.stream, 0 /*CU_STREAM_CAPTURE_MODE_GLOBAL*/) };
        if res != 0 {
            Err(format!("cuStreamBeginCapture failed with error code {res}"))
        } else {
            Ok(())
        }
    }

    /// End stream capture; returns the CUgraph handle.
    pub fn end_capture(&self) -> Result<usize, String> {
        let mut graph = 0usize;
        let res = unsafe { (self.driver.cu_stream_end_capture)(self.stream, &mut graph) };
        if res != 0 {
            Err(format!("cuStreamEndCapture failed with error code {res}"))
        } else {
            Ok(graph)
        }
    }

    /// Instantiate an executable graph from a captured CUgraph.
    pub fn instantiate_graph(&self, graph: usize) -> Result<usize, String> {
        let mut exec = 0usize;
        let res = unsafe { (self.driver.cu_graph_instantiate)(&mut exec, graph, std::ptr::null_mut(), std::ptr::null_mut(), 0) };
        if res != 0 {
            Err(format!("cuGraphInstantiate failed with error code {res}"))
        } else {
            Ok(exec)
        }
    }

    /// Launch an instantiated graph on the device stream.
    pub fn graph_launch(&self, exec: usize) -> Result<(), String> {
        let res = unsafe { (self.driver.cu_graph_launch)(exec, self.stream) };
        if res != 0 {
            Err(format!("cuGraphLaunch failed with error code {res}"))
        } else {
            Ok(())
        }
    }

    pub fn destroy_graph_exec(&self, exec: usize) {
        unsafe { (self.driver.cu_graph_exec_destroy)(exec) };
    }

    pub fn destroy_graph(&self, graph: usize) {
        unsafe { (self.driver.cu_graph_destroy)(graph) };
    }

    /// Brief GEMV burn-in so the driver raises GPU clocks before inference.
    /// Token generation issues short GPU bursts separated by host gaps, which
    /// otherwise leaves the GPU parked in a low power state (the RTX 3050
    /// laptop drops to ~210 MHz and runs 7x slower without this).
    pub fn warmup(&self, millis: u128) {
        if std::env::var_os("SPARK_NO_WARMUP").is_some() {
            return;
        }
        // 2 MB scratch: reused as weights, input and output (all reads stay
        // in-bounds; values are garbage and irrelevant).
        let d_buf = match self.alloc(2 * 1024 * 1024) {
            Ok(ptr) => ptr,
            Err(_) => return,
        };
        let started = std::time::Instant::now();
        while started.elapsed().as_millis() < millis {
            for _ in 0..64 {
                // rows=180, cols=1024: 195 KB of weight reads within the buffer
                let _ = self.launch_gemv_q8(d_buf, d_buf, d_buf, 1024, 180);
            }
            let _ = self.synchronize();
        }
        let _ = self.synchronize();
        self.free(d_buf);
    }

    /// Launch the persistent keepalive kernel on its own non-blocking stream.
    ///
    /// Token generation is a stream of short GPU bursts separated by host
    /// gaps; on the RTX 3050 laptop the power manager otherwise parks the GPU
    /// at ~210 MHz between bursts (a ~7x slowdown). A light kernel spinning on
    /// a non-blocking stream keeps the GPU continuously busy — and therefore
    /// clocks boosted — without serializing against the real work, which runs
    /// on the legacy stream.
    ///
    /// Call once after all init-time allocations (cuMemFree may implicitly
    /// synchronize with the whole context, which would deadlock on a kernel
    /// that never exits). Idempotent.
    pub fn start_persistent_keepalive(&self) {
        use std::sync::atomic::Ordering;

        if self.keepalive_dev == 0 || std::env::var_os("SPARK_NO_KEEPALIVE").is_some() {
            return;
        }
        if self
            .keepalive_stream
            .compare_exchange(KEEPALIVE_IDLE, usize::MAX, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return; // already running
        }

        let mut stream = 0usize;
        if unsafe { (self.driver.cu_stream_create)(&mut stream, CU_STREAM_NON_BLOCKING) } != 0 {
            self.keepalive_stream.store(KEEPALIVE_IDLE, Ordering::SeqCst);
            return;
        }

        unsafe { std::ptr::write_volatile(self.keepalive_host, 1) };
        let mut args: [*mut c_void; 1] = [&self.keepalive_dev as *const usize as *mut c_void];
        let grid_x = if std::env::var_os("SPARK_LIGHT_KEEPALIVE").is_some() {
            1
        } else {
            16
        };
        let res = unsafe {
            (self.driver.cu_launch_kernel)(
                self.fn_keepalive,
                grid_x,
                1,
                1,
                64,
                1,
                1,
                0,
                stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if res != 0 {
            // Harmless: worst case the GPU declocks between tokens.
            unsafe { std::ptr::write_volatile(self.keepalive_host, 0) };
            unsafe { (self.driver.cu_stream_destroy)(stream) };
            self.keepalive_stream.store(KEEPALIVE_IDLE, Ordering::SeqCst);
            return;
        }
        self.keepalive_stream.store(stream, Ordering::SeqCst);
    }

    /// Stop the persistent keepalive kernel and reclaim its stream. Must be
    /// called before freeing device memory (cuMemFree may implicitly
    /// synchronize with the whole context, which would deadlock on the
    /// spinning kernel). Idempotent.
    pub fn stop_persistent_keepalive(&self) {
        use std::sync::atomic::Ordering;

        let stream = self.keepalive_stream.swap(KEEPALIVE_IDLE, Ordering::SeqCst);
        if stream == KEEPALIVE_IDLE || stream == usize::MAX {
            return;
        }
        unsafe { std::ptr::write_volatile(self.keepalive_host, 0) };
        let _ = unsafe { (self.driver.cu_stream_synchronize)(stream) };
        unsafe { (self.driver.cu_stream_destroy)(stream) };
    }

    pub fn launch_gemv_q8(
        &self,
        d_weights: usize,
        d_input: usize,
        d_output: usize,
        cols: usize,
        rows: usize,
    ) -> Result<(), String> {
        let cols_u32 = cols as u32;
        let rows_u32 = rows as u32;

        let mut args: [*mut c_void; 5] = [
            &d_weights as *const usize as *mut c_void,
            &d_input as *const usize as *mut c_void,
            &d_output as *const usize as *mut c_void,
            &cols_u32 as *const u32 as *mut c_void,
            &rows_u32 as *const u32 as *mut c_void,
        ];

        let warps_per_block = 4u32;
        let vectorized = cols.is_multiple_of(128) && cols >= 128;
        let two_rows = vectorized && self.use_q8_v3;
        let rows_per_block = warps_per_block * if two_rows { 2 } else { 1 };
        let grid_x = rows_u32.div_ceil(rows_per_block);
        let kernel = if two_rows {
            self.fn_gemv_q8_v3
        } else if vectorized {
            self.fn_gemv_q8_v2
        } else {
            self.fn_gemv_q8
        };

        let res = unsafe {
            (self.driver.cu_launch_kernel)(
                kernel,
                grid_x,
                1,
                1,
                32,
                warps_per_block,
                1,
                0,
                self.stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };

        if res != 0 {
            Err(format!("cuLaunchKernel gemv_q8_0 failed: {res}"))
        } else {
            Ok(())
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_gemv_q4(
        &self,
        d_weights: usize,
        d_input: usize,
        d_output: usize,
        cols: usize,
        rows: usize,
    ) -> Result<(), String> {
        let cols_u32 = cols as u32;
        let rows_u32 = rows as u32;

        let mut args: [*mut c_void; 5] = [
            &d_weights as *const usize as *mut c_void,
            &d_input as *const usize as *mut c_void,
            &d_output as *const usize as *mut c_void,
            &cols_u32 as *const u32 as *mut c_void,
            &rows_u32 as *const u32 as *mut c_void,
        ];

        let warps_per_block = 4u32;
        let rows_per_block = warps_per_block * 2;
        let grid_x = rows_u32.div_ceil(rows_per_block);

        let res = unsafe {
            (self.driver.cu_launch_kernel)(
                self.fn_gemv_q4,
                grid_x,
                1,
                1,
                32,
                warps_per_block,
                1,
                0,
                self.stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };

        if res != 0 {
            Err(format!("cuLaunchKernel gemv_q4_0 failed: {res}"))
        } else {
            Ok(())
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_fused_qkv_q8(
        &self,
        d_q_w: usize,
        d_k_w: usize,
        d_v_w: usize,
        d_input: usize,
        d_output: usize,
        cols: usize,
        q_rows: usize,
        k_rows: usize,
        v_rows: usize,
        d_bias: usize,
    ) -> Result<(), String> {
        let cols_u32 = cols as u32;
        let q_rows_u32 = q_rows as u32;
        let k_rows_u32 = k_rows as u32;
        let v_rows_u32 = v_rows as u32;
        let total_rows_u32 = q_rows_u32 + k_rows_u32 + v_rows_u32;

        let mut args: [*mut c_void; 10] = [
            &d_q_w as *const usize as *mut c_void,
            &d_k_w as *const usize as *mut c_void,
            &d_v_w as *const usize as *mut c_void,
            &d_input as *const usize as *mut c_void,
            &d_output as *const usize as *mut c_void,
            &cols_u32 as *const u32 as *mut c_void,
            &q_rows_u32 as *const u32 as *mut c_void,
            &k_rows_u32 as *const u32 as *mut c_void,
            &v_rows_u32 as *const u32 as *mut c_void,
            &d_bias as *const usize as *mut c_void,
        ];

        let warps_per_block = 4u32;
        let vectorized = cols.is_multiple_of(128) && cols >= 128;
        let two_rows = vectorized && self.use_q8_v3;
        let rows_per_block = warps_per_block * if two_rows { 2 } else { 1 };
        let grid_x = total_rows_u32.div_ceil(rows_per_block);
        let kernel = if two_rows {
            self.fn_fused_qkv_q8_v3
        } else if vectorized {
            self.fn_fused_qkv_q8_v2
        } else {
            self.fn_fused_qkv_q8
        };

        let res = unsafe {
            (self.driver.cu_launch_kernel)(
                kernel,
                grid_x,
                1,
                1,
                32,
                warps_per_block,
                1,
                0,
                self.stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };

        if res != 0 {
            Err(format!("cuLaunchKernel gemv_fused_qkv_q8 failed: {res}"))
        } else if d_bias != 0 && !two_rows {
            self.launch_add_bias_f32(d_output, d_bias, total_rows_u32 as usize)
        } else {
            Ok(())
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_fused_qkv_q4(
        &self,
        d_q_w: usize,
        d_k_w: usize,
        d_v_w: usize,
        d_input: usize,
        d_output: usize,
        cols: usize,
        q_rows: usize,
        k_rows: usize,
        v_rows: usize,
    ) -> Result<(), String> {
        let cols_u32 = cols as u32;
        let q_rows_u32 = q_rows as u32;
        let k_rows_u32 = k_rows as u32;
        let v_rows_u32 = v_rows as u32;
        let total_rows_u32 = q_rows_u32 + k_rows_u32 + v_rows_u32;

        let mut args: [*mut c_void; 9] = [
            &d_q_w as *const usize as *mut c_void,
            &d_k_w as *const usize as *mut c_void,
            &d_v_w as *const usize as *mut c_void,
            &d_input as *const usize as *mut c_void,
            &d_output as *const usize as *mut c_void,
            &cols_u32 as *const u32 as *mut c_void,
            &q_rows_u32 as *const u32 as *mut c_void,
            &k_rows_u32 as *const u32 as *mut c_void,
            &v_rows_u32 as *const u32 as *mut c_void,
        ];

        let warps_per_block = 4u32;
        let rows_per_block = warps_per_block * 2;
        let grid_x = total_rows_u32.div_ceil(rows_per_block);

        let res = unsafe {
            (self.driver.cu_launch_kernel)(
                self.fn_fused_qkv_q4,
                grid_x,
                1,
                1,
                32,
                warps_per_block,
                1,
                0,
                self.stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };

        if res != 0 {
            Err(format!("cuLaunchKernel gemv_fused_qkv_q4 failed: {res}"))
        } else {
            Ok(())
        }
    }

    pub fn launch_fused_ffn_q8(
        &self,
        d_gate: usize,
        d_up: usize,
        d_input: usize,
        d_output: usize,
        cols: usize,
        rows: usize,
    ) -> Result<(), String> {
        let cols_u32 = cols as u32;
        let rows_u32 = rows as u32;

        let mut args: [*mut c_void; 6] = [
            &d_gate as *const usize as *mut c_void,
            &d_up as *const usize as *mut c_void,
            &d_input as *const usize as *mut c_void,
            &d_output as *const usize as *mut c_void,
            &cols_u32 as *const u32 as *mut c_void,
            &rows_u32 as *const u32 as *mut c_void,
        ];

        let warps_per_block = 4u32;
        let vectorized = cols.is_multiple_of(128) && cols >= 128;
        let two_rows = vectorized && self.use_ffn_q8_v3;
        let rows_per_block = warps_per_block * if two_rows { 2 } else { 1 };
        let grid_x = rows_u32.div_ceil(rows_per_block);
        let kernel = if two_rows {
            self.fn_fused_ffn_q8_v3
        } else if vectorized {
            self.fn_fused_ffn_q8_v2
        } else {
            self.fn_fused_ffn_q8
        };

        let res = unsafe {
            (self.driver.cu_launch_kernel)(
                kernel,
                grid_x,
                1,
                1,
                32,
                warps_per_block,
                1,
                0,
                self.stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };

        if res != 0 {
            Err(format!("cuLaunchKernel gemv_fused_ffn_q8_0 failed: {res}"))
        } else {
            Ok(())
        }
    }

    pub fn launch_fused_ffn_q4(
        &self,
        d_gate: usize,
        d_up: usize,
        d_input: usize,
        d_output: usize,
        cols: usize,
        rows: usize,
    ) -> Result<(), String> {
        let cols_u32 = cols as u32;
        let rows_u32 = rows as u32;

        let mut args: [*mut c_void; 6] = [
            &d_gate as *const usize as *mut c_void,
            &d_up as *const usize as *mut c_void,
            &d_input as *const usize as *mut c_void,
            &d_output as *const usize as *mut c_void,
            &cols_u32 as *const u32 as *mut c_void,
            &rows_u32 as *const u32 as *mut c_void,
        ];

        let warps_per_block = 4u32;
        let rows_per_block = warps_per_block * 2;
        let grid_x = rows_u32.div_ceil(rows_per_block);

        let res = unsafe {
            (self.driver.cu_launch_kernel)(
                self.fn_fused_ffn_q4,
                grid_x,
                1,
                1,
                32,
                warps_per_block,
                1,
                0,
                self.stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };

        if res != 0 {
            Err(format!("cuLaunchKernel gemv_fused_ffn_q4_0 failed: {res}"))
        } else {
            Ok(())
        }
    }

    pub fn launch_argmax_f32(
        &self,
        d_logits: usize,
        d_out_token: usize,
        cols: usize,
    ) -> Result<(), String> {
        let cols_u32 = cols as u32;

        let mut args: [*mut c_void; 3] = [
            &d_logits as *const usize as *mut c_void,
            &d_out_token as *const usize as *mut c_void,
            &cols_u32 as *const u32 as *mut c_void,
        ];

        let res = unsafe {
            (self.driver.cu_launch_kernel)(
                self.fn_argmax_f32,
                1,
                1,
                1,
                32,
                1,
                1,
                0,
                self.stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };

        if res != 0 {
            Err(format!("cuLaunchKernel argmax_f32 failed: {res}"))
        } else {
            Ok(())
        }
    }

    pub fn launch_top64_sample(
        &self,
        d_logits: usize,
        d_out_tokens: usize,
        d_out_logits: usize,
        cols: usize,
        inv_temp: f32,
    ) -> Result<(), String> {
        let cols_u32 = cols as u32;

        let mut args: [*mut c_void; 5] = [
            &d_logits as *const usize as *mut c_void,
            &d_out_tokens as *const usize as *mut c_void,
            &d_out_logits as *const usize as *mut c_void,
            &cols_u32 as *const u32 as *mut c_void,
            &inv_temp as *const f32 as *mut c_void,
        ];

        let res = unsafe {
            (self.driver.cu_launch_kernel)(
                self.fn_top64_sample,
                64,
                1,
                1,
                32,
                1,
                1,
                0,
                self.stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };

        if res != 0 {
            Err(format!("cuLaunchKernel top64_sample failed: {res}"))
        } else {
            Ok(())
        }
    }

    pub fn launch_gemv_q1_0(
        &self,
        d_weights: usize,
        d_input: usize,
        d_output: usize,
        cols: usize,
        rows: usize,
    ) -> Result<(), String> {
        let cols_u32 = cols as u32;
        let rows_u32 = rows as u32;
        let mut args: [*mut c_void; 5] = [
            &d_weights as *const usize as *mut c_void,
            &d_input as *const usize as *mut c_void,
            &d_output as *const usize as *mut c_void,
            &cols_u32 as *const u32 as *mut c_void,
            &rows_u32 as *const u32 as *mut c_void,
        ];
        let warps_per_block = 4u32;
        let rows_per_warp = if self.use_q1_prmt || self.use_q1_bitplane {
            8
        } else if self.use_q1_v3 {
            4
        } else if self.use_q1_v2 {
            2
        } else {
            1
        };
        let rows_per_block = warps_per_block * rows_per_warp;
        let grid_x = rows_u32.div_ceil(rows_per_block);
        let res = unsafe {
            (self.driver.cu_launch_kernel)(
                if self.use_q1_prmt {
                    self.fn_gemv_q1_0_prmt
                } else if self.use_q1_bitplane {
                    self.fn_gemv_q1_0_bitplane
                } else if self.use_q1_v3 {
                    self.fn_gemv_q1_0_v3
                } else if self.use_q1_v2 {
                    self.fn_gemv_q1_0_v2
                } else {
                    self.fn_gemv_q1_0
                },
                grid_x,
                1,
                1,
                32,
                warps_per_block,
                1,
                0,
                self.stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if res != 0 {
            Err(format!("cuLaunchKernel gemv_q1_0 failed: {res}"))
        } else {
            Ok(())
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_gemv_q1_0_u10(
        &self,
        d_weights: usize,
        d_input: usize,
        d_output: usize,
        cols: usize,
        rows: usize,
        scale_min: f32,
        scale_step: f32,
    ) -> Result<(), String> {
        let cols_u32 = cols as u32;
        let rows_u32 = rows as u32;
        let mut args: [*mut c_void; 7] = [
            &d_weights as *const usize as *mut c_void,
            &d_input as *const usize as *mut c_void,
            &d_output as *const usize as *mut c_void,
            &cols_u32 as *const u32 as *mut c_void,
            &rows_u32 as *const u32 as *mut c_void,
            &scale_min as *const f32 as *mut c_void,
            &scale_step as *const f32 as *mut c_void,
        ];
        let warps_per_block = 4u32;
        let grid_x = rows_u32.div_ceil(warps_per_block * 2);
        let res = unsafe {
            (self.driver.cu_launch_kernel)(
                self.fn_gemv_q1_0_u10,
                grid_x,
                1,
                1,
                32,
                warps_per_block,
                1,
                0,
                self.stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if res != 0 {
            Err(format!("cuLaunchKernel gemv_q1_0_u10 failed: {res}"))
        } else {
            Ok(())
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_gemv_q1_0_u10_tiled(
        &self,
        d_weights: usize,
        d_input: usize,
        d_output: usize,
        cols: usize,
        rows: usize,
        scale_min: f32,
        scale_step: f32,
    ) -> Result<(), String> {
        let cols_u32 = cols as u32;
        let rows_u32 = rows as u32;
        let mut args: [*mut c_void; 7] = [
            &d_weights as *const usize as *mut c_void,
            &d_input as *const usize as *mut c_void,
            &d_output as *const usize as *mut c_void,
            &cols_u32 as *const u32 as *mut c_void,
            &rows_u32 as *const u32 as *mut c_void,
            &scale_min as *const f32 as *mut c_void,
            &scale_step as *const f32 as *mut c_void,
        ];
        let warps_per_block = 4u32;
        let grid_x = rows_u32.div_ceil(warps_per_block * 8);
        let res = unsafe {
            (self.driver.cu_launch_kernel)(
                self.fn_gemv_q1_0_u10_tiled,
                grid_x,
                1,
                1,
                32,
                warps_per_block,
                1,
                0,
                self.stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if res != 0 {
            Err(format!("cuLaunchKernel gemv_q1_0_u10_tiled failed: {res}"))
        } else {
            Ok(())
        }
    }

    pub fn q1_bitplane_enabled(&self) -> bool {
        self.use_q1_bitplane
    }

    pub fn q1_tiled_enabled(&self) -> bool {
        self.use_q1_bitplane || self.use_q1_prmt
    }

    pub fn launch_gemv_ternary(
        &self,
        d_weights: usize,
        d_input: usize,
        d_output: usize,
        cols: usize,
        rows: usize,
    ) -> Result<(), String> {
        let cols_u32 = cols as u32;
        let rows_u32 = rows as u32;

        let mut args: [*mut c_void; 5] = [
            &d_weights as *const usize as *mut c_void,
            &d_input as *const usize as *mut c_void,
            &d_output as *const usize as *mut c_void,
            &cols_u32 as *const u32 as *mut c_void,
            &rows_u32 as *const u32 as *mut c_void,
        ];

        let warps_per_block = 4u32;
        let grid_x = rows_u32.div_ceil(warps_per_block);

        let res = unsafe {
            (self.driver.cu_launch_kernel)(
                self.fn_gemv_ternary,
                grid_x,
                1,
                1,
                32,
                warps_per_block,
                1,
                0,
                self.stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };

        if res != 0 {
            Err(format!("cuLaunchKernel gemv_ternary failed: {res}"))
        } else {
            Ok(())
        }
    }

    pub fn launch_gemv_q2_0(
        &self,
        d_weights: usize,
        d_input: usize,
        d_output: usize,
        cols: usize,
        rows: usize,
    ) -> Result<(), String> {
        let cols_u32 = cols as u32;
        let rows_u32 = rows as u32;

        let mut args: [*mut c_void; 5] = [
            &d_weights as *const usize as *mut c_void,
            &d_input as *const usize as *mut c_void,
            &d_output as *const usize as *mut c_void,
            &cols_u32 as *const u32 as *mut c_void,
            &rows_u32 as *const u32 as *mut c_void,
        ];

        let warps_per_block = 4u32;
        let grid_x = rows_u32.div_ceil(warps_per_block);

        let res = unsafe {
            (self.driver.cu_launch_kernel)(
                self.fn_gemv_q2_0,
                grid_x,
                1,
                1,
                32,
                warps_per_block,
                1,
                0,
                self.stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };

        if res != 0 {
            Err(format!("cuLaunchKernel gemv_q2_0 failed: {res}"))
        } else {
            Ok(())
        }
    }

    fn launch_simple(&self, f: usize, grid_x: u32, args: &mut [*mut c_void], name: &str) -> Result<(), String> {
        let res = unsafe {
            (self.driver.cu_launch_kernel)(
                f,
                grid_x,
                1,
                1,
                256,
                1,
                1,
                0,
                self.stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if res != 0 {
            Err(format!("cuLaunchKernel {name} failed: {res}"))
        } else {
            Ok(())
        }
    }

    fn launch_warps(&self, f: usize, grid_x: u32, block_y: u32, args: &mut [*mut c_void], name: &str) -> Result<(), String> {
        let res = unsafe {
            (self.driver.cu_launch_kernel)(
                f,
                grid_x,
                1,
                1,
                32,
                block_y,
                1,
                0,
                self.stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if res != 0 {
            Err(format!("cuLaunchKernel {name} failed: {res}"))
        } else {
            Ok(())
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_gated_delta_f32(
        &self,
        d_state: usize,
        d_q: usize,
        d_k: usize,
        d_v: usize,
        d_decay: usize,
        d_beta: usize,
        d_output: usize,
        key_heads: usize,
        value_heads: usize,
        head_dim: usize,
    ) -> Result<(), String> {
        let key_heads_u32 = key_heads as u32;
        let value_heads_u32 = value_heads as u32;
        let head_dim_u32 = head_dim as u32;
        let scale = (head_dim as f32).sqrt().recip();
        let mut args: [*mut c_void; 11] = [
            &d_state as *const usize as *mut c_void,
            &d_q as *const usize as *mut c_void,
            &d_k as *const usize as *mut c_void,
            &d_v as *const usize as *mut c_void,
            &d_decay as *const usize as *mut c_void,
            &d_beta as *const usize as *mut c_void,
            &d_output as *const usize as *mut c_void,
            &key_heads_u32 as *const u32 as *mut c_void,
            &value_heads_u32 as *const u32 as *mut c_void,
            &head_dim_u32 as *const u32 as *mut c_void,
            &scale as *const f32 as *mut c_void,
        ];
        let res = unsafe {
            (self.driver.cu_launch_kernel)(
                self.fn_gated_delta_f32,
                value_heads_u32,
                1,
                1,
                32,
                4,
                1,
                0,
                self.stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if res != 0 {
            Err(format!("cuLaunchKernel gated_delta_f32 failed: {res}"))
        } else {
            Ok(())
        }
    }

    pub fn launch_rmsnorm_f32(
        &self,
        d_out: usize,
        d_in: usize,
        d_weight: usize,
        n: usize,
        epsilon: f32,
    ) -> Result<(), String> {
        let n_u32 = n as u32;
        let mut args: [*mut c_void; 5] = [
            &d_out as *const usize as *mut c_void,
            &d_in as *const usize as *mut c_void,
            &d_weight as *const usize as *mut c_void,
            &n_u32 as *const u32 as *mut c_void,
            &epsilon as *const f32 as *mut c_void,
        ];
        if self.use_rms_v2 {
            self.launch_simple(self.fn_rmsnorm_f32_v2, 1, &mut args, "rmsnorm_f32_v2")
        } else {
            self.launch_warps(self.fn_rmsnorm_f32, 1, 1, &mut args, "rmsnorm_f32")
        }
    }

    pub fn launch_add_f32(&self, d_dst: usize, d_src: usize, n: usize) -> Result<(), String> {
        let n_u32 = n as u32;
        let mut args: [*mut c_void; 3] = [
            &d_dst as *const usize as *mut c_void,
            &d_src as *const usize as *mut c_void,
            &n_u32 as *const u32 as *mut c_void,
        ];
        let grid_x = (n as u32).div_ceil(256);
        self.launch_simple(self.fn_add_f32, grid_x, &mut args, "add_f32")
    }

    pub fn launch_add_rmsnorm_f32(
        &self,
        d_out: usize,
        d_dst: usize,
        d_src: usize,
        d_weight: usize,
        n: usize,
        epsilon: f32,
    ) -> Result<(), String> {
        let n_u32 = n as u32;
        let mut args: [*mut c_void; 6] = [
            &d_out as *const usize as *mut c_void,
            &d_dst as *const usize as *mut c_void,
            &d_src as *const usize as *mut c_void,
            &d_weight as *const usize as *mut c_void,
            &n_u32 as *const u32 as *mut c_void,
            &epsilon as *const f32 as *mut c_void,
        ];
        self.launch_simple(
            self.fn_add_rmsnorm_f32_v2,
            1,
            &mut args,
            "add_rmsnorm_f32_v2",
        )
    }

    pub fn launch_add_bias_f32(&self, d_dst: usize, d_bias: usize, n: usize) -> Result<(), String> {
        let n_u32 = n as u32;
        let mut args: [*mut c_void; 3] = [
            &d_dst as *const usize as *mut c_void,
            &d_bias as *const usize as *mut c_void,
            &n_u32 as *const u32 as *mut c_void,
        ];
        let grid_x = (n as u32).div_ceil(256);
        self.launch_simple(self.fn_add_bias_f32, grid_x, &mut args, "add_bias_f32")
    }

    pub fn launch_swiglu_f32(&self, d_gate: usize, d_up: usize, n: usize) -> Result<(), String> {
        let n_u32 = n as u32;
        let mut args: [*mut c_void; 3] = [
            &d_gate as *const usize as *mut c_void,
            &d_up as *const usize as *mut c_void,
            &n_u32 as *const u32 as *mut c_void,
        ];
        let grid_x = (n as u32).div_ceil(256);
        self.launch_simple(self.fn_swiglu_f32, grid_x, &mut args, "swiglu_f32")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_rope_store_f32(
        &self,
        d_q: usize,
        d_k_src: usize,
        d_v_src: usize,
        d_kcache: usize,
        d_vcache: usize,
        d_cos_sin: usize,
        _pos: usize,
        d_pos: usize,
        n_q_pairs: usize,
        n_kv_pairs: usize,
        head_dim: usize,
        kv_stride: usize,
    ) -> Result<(), String> {
        let n_q_u32 = n_q_pairs as u32;
        let n_kv_u32 = n_kv_pairs as u32;
        let head_dim_u32 = head_dim as u32;
        let kv_stride_u32 = kv_stride as u32;
        let mut args: [*mut c_void; 11] = [
            &d_q as *const usize as *mut c_void,
            &d_k_src as *const usize as *mut c_void,
            &d_v_src as *const usize as *mut c_void,
            &d_kcache as *const usize as *mut c_void,
            &d_vcache as *const usize as *mut c_void,
            &d_cos_sin as *const usize as *mut c_void,
            &d_pos as *const usize as *mut c_void,
            &n_q_u32 as *const u32 as *mut c_void,
            &n_kv_u32 as *const u32 as *mut c_void,
            &head_dim_u32 as *const u32 as *mut c_void,
            &kv_stride_u32 as *const u32 as *mut c_void,
        ];
        let total = (n_q_pairs + n_kv_pairs + kv_stride) as u32;
        let grid_x = total.div_ceil(256);
        self.launch_simple(self.fn_rope_store_f32, grid_x, &mut args, "rope_store_f32")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_attn_scores_f32(
        &self,
        d_q: usize,
        d_kcache: usize,
        d_scores: usize,
        pos: usize,
        d_pos: usize,
        max_ctx: usize,
        n_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        scale: f32,
        grid_positions: usize,
    ) -> Result<(), String> {
        let max_ctx_u32 = max_ctx as u32;
        let n_heads_u32 = n_heads as u32;
        let kv_heads_u32 = kv_heads as u32;
        let head_dim_u32 = head_dim as u32;
        let mut args: [*mut c_void; 9] = [
            &d_q as *const usize as *mut c_void,
            &d_kcache as *const usize as *mut c_void,
            &d_scores as *const usize as *mut c_void,
            &d_pos as *const usize as *mut c_void,
            &max_ctx_u32 as *const u32 as *mut c_void,
            &n_heads_u32 as *const u32 as *mut c_void,
            &kv_heads_u32 as *const u32 as *mut c_void,
            &head_dim_u32 as *const u32 as *mut c_void,
            &scale as *const f32 as *mut c_void,
        ];
        // CUDA graph captures bake grid dimensions into the node. The engine
        // supplies a power-of-two position bucket and recaptures when it is
        // crossed; eager launches simply supply pos + 1.
        let covered_positions = grid_positions.max(pos + 1).min(max_ctx);
        let warps = (n_heads * covered_positions) as u32;
        let grid_x = warps.div_ceil(4);
        self.launch_warps(self.fn_attn_scores_f32, grid_x, 4, &mut args, "attn_scores_f32")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_attn_weighted_f32(
        &self,
        d_out: usize,
        d_vcache: usize,
        d_scores: usize,
        _pos: usize,
        d_pos: usize,
        max_ctx: usize,
        n_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<(), String> {
        let max_ctx_u32 = max_ctx as u32;
        let n_heads_u32 = n_heads as u32;
        let kv_heads_u32 = kv_heads as u32;
        let head_dim_u32 = head_dim as u32;
        let mut args: [*mut c_void; 8] = [
            &d_out as *const usize as *mut c_void,
            &d_vcache as *const usize as *mut c_void,
            &d_scores as *const usize as *mut c_void,
            &d_pos as *const usize as *mut c_void,
            &max_ctx_u32 as *const u32 as *mut c_void,
            &n_heads_u32 as *const u32 as *mut c_void,
            &kv_heads_u32 as *const u32 as *mut c_void,
            &head_dim_u32 as *const u32 as *mut c_void,
        ];
        let grid_x = (n_heads as u32).div_ceil(4);
        self.launch_warps(self.fn_attn_weighted_f32, grid_x, 4, &mut args, "attn_weighted_f32")
    }

    pub fn launch_quantize_q8_0_128(
        &self,
        d_input: usize,
        d_output: usize,
        n: usize,
    ) -> Result<(), String> {
        if !n.is_multiple_of(128) {
            return Err(format!("resident Q1 activation length {n} is not divisible by 128"));
        }
        let n_u32 = n as u32;
        let mut args: [*mut c_void; 3] = [
            &d_input as *const usize as *mut c_void,
            &d_output as *const usize as *mut c_void,
            &n_u32 as *const u32 as *mut c_void,
        ];
        self.launch_warps(
            self.fn_quantize_q8_0_128,
            (n / 32) as u32,
            1,
            &mut args,
            "quantize_q8_0_128",
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_qwen35_q_gate_rope(
        &self,
        d_raw: usize,
        d_q: usize,
        d_gate: usize,
        d_weight: usize,
        d_cos_sin: usize,
        heads: usize,
        head_dim: usize,
        rope_dim: usize,
        epsilon: f32,
    ) -> Result<(), String> {
        let heads_u32 = heads as u32;
        let head_dim_u32 = head_dim as u32;
        let rope_dim_u32 = rope_dim as u32;
        let mut args: [*mut c_void; 9] = [
            &d_raw as *const usize as *mut c_void,
            &d_q as *const usize as *mut c_void,
            &d_gate as *const usize as *mut c_void,
            &d_weight as *const usize as *mut c_void,
            &d_cos_sin as *const usize as *mut c_void,
            &heads_u32 as *const u32 as *mut c_void,
            &head_dim_u32 as *const u32 as *mut c_void,
            &rope_dim_u32 as *const u32 as *mut c_void,
            &epsilon as *const f32 as *mut c_void,
        ];
        self.launch_warps(
            self.fn_qwen35_q_gate_rope,
            heads_u32.div_ceil(4),
            4,
            &mut args,
            "qwen35_q_gate_rope",
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_qwen35_kv_rope_store(
        &self,
        d_k_src: usize,
        d_v_src: usize,
        d_k_cache: usize,
        d_v_cache: usize,
        d_weight: usize,
        d_cos_sin: usize,
        d_pos: usize,
        kv_heads: usize,
        head_dim: usize,
        rope_dim: usize,
        epsilon: f32,
    ) -> Result<(), String> {
        let kv_heads_u32 = kv_heads as u32;
        let head_dim_u32 = head_dim as u32;
        let rope_dim_u32 = rope_dim as u32;
        let mut args: [*mut c_void; 11] = [
            &d_k_src as *const usize as *mut c_void,
            &d_v_src as *const usize as *mut c_void,
            &d_k_cache as *const usize as *mut c_void,
            &d_v_cache as *const usize as *mut c_void,
            &d_weight as *const usize as *mut c_void,
            &d_cos_sin as *const usize as *mut c_void,
            &d_pos as *const usize as *mut c_void,
            &kv_heads_u32 as *const u32 as *mut c_void,
            &head_dim_u32 as *const u32 as *mut c_void,
            &rope_dim_u32 as *const u32 as *mut c_void,
            &epsilon as *const f32 as *mut c_void,
        ];
        self.launch_warps(
            self.fn_qwen35_kv_rope_store,
            kv_heads_u32.div_ceil(4),
            4,
            &mut args,
            "qwen35_kv_rope_store",
        )
    }

    pub fn launch_qwen35_attention_gate(
        &self,
        d_values: usize,
        d_gate: usize,
        n: usize,
    ) -> Result<(), String> {
        let n_u32 = n as u32;
        let mut args: [*mut c_void; 3] = [
            &d_values as *const usize as *mut c_void,
            &d_gate as *const usize as *mut c_void,
            &n_u32 as *const u32 as *mut c_void,
        ];
        self.launch_simple(
            self.fn_qwen35_attention_gate,
            n_u32.div_ceil(256),
            &mut args,
            "qwen35_attention_gate",
        )
    }

    pub fn launch_qwen35_conv_silu(
        &self,
        d_values: usize,
        d_weights: usize,
        d_state: usize,
        channels: usize,
        kernel: usize,
    ) -> Result<(), String> {
        let channels_u32 = channels as u32;
        let kernel_u32 = kernel as u32;
        let mut args: [*mut c_void; 5] = [
            &d_values as *const usize as *mut c_void,
            &d_weights as *const usize as *mut c_void,
            &d_state as *const usize as *mut c_void,
            &channels_u32 as *const u32 as *mut c_void,
            &kernel_u32 as *const u32 as *mut c_void,
        ];
        self.launch_simple(
            self.fn_qwen35_conv_silu,
            channels_u32.div_ceil(256),
            &mut args,
            "qwen35_conv_silu",
        )
    }

    pub fn launch_qwen35_qk_l2_norm(
        &self,
        d_q: usize,
        d_k: usize,
        heads: usize,
        head_dim: usize,
        epsilon: f32,
    ) -> Result<(), String> {
        let heads_u32 = heads as u32;
        let head_dim_u32 = head_dim as u32;
        let mut args: [*mut c_void; 5] = [
            &d_q as *const usize as *mut c_void,
            &d_k as *const usize as *mut c_void,
            &heads_u32 as *const u32 as *mut c_void,
            &head_dim_u32 as *const u32 as *mut c_void,
            &epsilon as *const f32 as *mut c_void,
        ];
        self.launch_warps(
            self.fn_qwen35_qk_l2_norm,
            heads_u32.div_ceil(4),
            4,
            &mut args,
            "qwen35_qk_l2_norm",
        )
    }

    pub fn launch_qwen35_decay_beta(
        &self,
        d_alpha: usize,
        d_beta: usize,
        d_dt_bias: usize,
        d_a: usize,
        n: usize,
    ) -> Result<(), String> {
        let n_u32 = n as u32;
        let mut args: [*mut c_void; 5] = [
            &d_alpha as *const usize as *mut c_void,
            &d_beta as *const usize as *mut c_void,
            &d_dt_bias as *const usize as *mut c_void,
            &d_a as *const usize as *mut c_void,
            &n_u32 as *const u32 as *mut c_void,
        ];
        self.launch_simple(self.fn_qwen35_decay_beta, 1, &mut args, "qwen35_decay_beta")
    }

    pub fn launch_qwen35_gated_rmsnorm(
        &self,
        d_values: usize,
        d_gate: usize,
        d_weight: usize,
        heads: usize,
        head_dim: usize,
        epsilon: f32,
    ) -> Result<(), String> {
        let heads_u32 = heads as u32;
        let head_dim_u32 = head_dim as u32;
        let mut args: [*mut c_void; 6] = [
            &d_values as *const usize as *mut c_void,
            &d_gate as *const usize as *mut c_void,
            &d_weight as *const usize as *mut c_void,
            &heads_u32 as *const u32 as *mut c_void,
            &head_dim_u32 as *const u32 as *mut c_void,
            &epsilon as *const f32 as *mut c_void,
        ];
        self.launch_warps(
            self.fn_qwen35_gated_rmsnorm,
            heads_u32.div_ceil(4),
            4,
            &mut args,
            "qwen35_gated_rmsnorm",
        )
    }
}

impl Drop for CudaDevice {
    fn drop(&mut self) {
        self.stop_persistent_keepalive();
        if self.ctx != 0 {
            unsafe {
                (self.driver.cu_ctx_destroy)(self.ctx);
            }
        }
        if !self.keepalive_host.is_null() {
            unsafe {
                (self.driver.cu_mem_free_host)(self.keepalive_host as *mut c_void);
            }
        }
    }
}
