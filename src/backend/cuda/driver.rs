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
    pub cu_module_get_function: unsafe extern "C" fn(*mut usize, usize, *const c_char) -> i32,
    pub cu_mem_alloc: unsafe extern "C" fn(*mut usize, usize) -> i32,
    pub cu_mem_free: unsafe extern "C" fn(usize) -> i32,
    pub cu_memcpy_htod: unsafe extern "C" fn(usize, *const c_void, usize) -> i32,
    pub cu_memcpy_dtoh: unsafe extern "C" fn(*mut c_void, usize, usize) -> i32,
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
                cu_module_get_function: load_sym!("cuModuleGetFunction"),
                cu_mem_alloc: load_sym!("cuMemAlloc_v2"),
                cu_mem_free: load_sym!("cuMemFree_v2"),
                cu_memcpy_htod: load_sym!("cuMemcpyHtoD_v2"),
                cu_memcpy_dtoh: load_sym!("cuMemcpyDtoH_v2"),
                cu_launch_kernel: load_sym!("cuLaunchKernel"),
                cu_ctx_synchronize: load_sym!("cuCtxSynchronize"),
            })
        }
    }
}

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
    pub fn_argmax_f32: usize,
    pub fn_top64_sample: usize,
    pub fn_gemv_ternary: usize,
    pub fn_gemv_q2_0: usize,
}

unsafe impl Send for CudaDevice {}
unsafe impl Sync for CudaDevice {}

impl CudaDevice {
    pub fn init() -> Option<Self> {
        if std::env::var_os("SPARK_NO_CUDA").is_some() {
            return None;
        }

        let driver = CudaDriver::load()?;
        unsafe {
            if (driver.cu_init)(0) != 0 {
                return None;
            }

            let mut dev = 0;
            if (driver.cu_device_get)(&mut dev, 0) != 0 {
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
                return None;
            }

            // Load embedded PTX module
            let ptx_src = include_str!("../../../kernels/gemv.ptx");
            let ptx_cstr = CString::new(ptx_src).ok()?;
            let mut module = 0usize;
            if (driver.cu_module_load_data)(&mut module, ptx_cstr.as_ptr()) != 0 {
                (driver.cu_ctx_destroy)(ctx);
                return None;
            }

            let mut fn_gemv_q8 = 0usize;
            let n_gemv = CString::new("gemv_q8_0").ok()?;
            if (driver.cu_module_get_function)(&mut fn_gemv_q8, module, n_gemv.as_ptr()) != 0 {
                (driver.cu_ctx_destroy)(ctx);
                return None;
            }

            let mut fn_gemv_q4 = 0usize;
            let n_gemv_q4 = CString::new("gemv_q4_0").ok()?;
            if (driver.cu_module_get_function)(&mut fn_gemv_q4, module, n_gemv_q4.as_ptr()) != 0 {
                (driver.cu_ctx_destroy)(ctx);
                return None;
            }

            let mut fn_fused_qkv_q8 = 0usize;
            let n_qkv = CString::new("gemv_fused_qkv_q8_0").ok()?;
            if (driver.cu_module_get_function)(&mut fn_fused_qkv_q8, module, n_qkv.as_ptr()) != 0 {
                (driver.cu_ctx_destroy)(ctx);
                return None;
            }

            let mut fn_fused_qkv_q4 = 0usize;
            let n_qkv_q4 = CString::new("gemv_fused_qkv_q4_0").ok()?;
            if (driver.cu_module_get_function)(&mut fn_fused_qkv_q4, module, n_qkv_q4.as_ptr()) != 0 {
                (driver.cu_ctx_destroy)(ctx);
                return None;
            }

            let mut fn_fused_ffn_q8 = 0usize;
            let n_ffn = CString::new("gemv_fused_ffn_q8_0").ok()?;
            if (driver.cu_module_get_function)(&mut fn_fused_ffn_q8, module, n_ffn.as_ptr()) != 0 {
                (driver.cu_ctx_destroy)(ctx);
                return None;
            }

            let mut fn_fused_ffn_q4 = 0usize;
            let n_ffn_q4 = CString::new("gemv_fused_ffn_q4_0").ok()?;
            if (driver.cu_module_get_function)(&mut fn_fused_ffn_q4, module, n_ffn_q4.as_ptr()) != 0 {
                (driver.cu_ctx_destroy)(ctx);
                return None;
            }

            let mut fn_argmax_f32 = 0usize;
            let n_argmax = CString::new("argmax_f32").ok()?;
            if (driver.cu_module_get_function)(&mut fn_argmax_f32, module, n_argmax.as_ptr()) != 0 {
                (driver.cu_ctx_destroy)(ctx);
                return None;
            }

            let mut fn_top64_sample = 0usize;
            let n_top64 = CString::new("top64_sample").ok()?;
            if (driver.cu_module_get_function)(&mut fn_top64_sample, module, n_top64.as_ptr()) != 0 {
                (driver.cu_ctx_destroy)(ctx);
                return None;
            }

            let mut fn_gemv_ternary = 0usize;
            let n_ternary = CString::new("gemv_ternary").ok()?;
            if (driver.cu_module_get_function)(&mut fn_gemv_ternary, module, n_ternary.as_ptr()) != 0 {
                (driver.cu_ctx_destroy)(ctx);
                return None;
            }

            let mut fn_gemv_q2_0 = 0usize;
            let n_q2 = CString::new("gemv_q2_0").ok()?;
            if (driver.cu_module_get_function)(&mut fn_gemv_q2_0, module, n_q2.as_ptr()) != 0 {
                (driver.cu_ctx_destroy)(ctx);
                return None;
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
                fn_argmax_f32,
                fn_top64_sample,
                fn_gemv_ternary,
                fn_gemv_q2_0,
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

    pub fn free(&self, ptr: usize) {
        if ptr != 0 {
            unsafe {
                (self.driver.cu_mem_free)(ptr);
            }
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
        let res = unsafe { (self.driver.cu_ctx_synchronize)() };
        if res != 0 {
            Err(format!("cuCtxSynchronize failed with error code {res}"))
        } else {
            Ok(())
        }
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
        let grid_x = rows_u32.div_ceil(warps_per_block);

        let res = unsafe {
            (self.driver.cu_launch_kernel)(
                self.fn_gemv_q8,
                grid_x,
                1,
                1,
                32,
                warps_per_block,
                1,
                0,
                0,
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
                0,
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
        let grid_x = total_rows_u32.div_ceil(warps_per_block);

        let res = unsafe {
            (self.driver.cu_launch_kernel)(
                self.fn_fused_qkv_q8,
                grid_x,
                1,
                1,
                32,
                warps_per_block,
                1,
                0,
                0,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };

        if res != 0 {
            Err(format!("cuLaunchKernel gemv_fused_qkv_q8 failed: {res}"))
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
                0,
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
        let grid_x = rows_u32.div_ceil(warps_per_block);

        let res = unsafe {
            (self.driver.cu_launch_kernel)(
                self.fn_fused_ffn_q8,
                grid_x,
                1,
                1,
                32,
                warps_per_block,
                1,
                0,
                0,
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
                0,
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
                0,
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
                0,
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
                0,
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
                0,
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
}

impl Drop for CudaDevice {
    fn drop(&mut self) {
        if self.ctx != 0 {
            unsafe {
                (self.driver.cu_ctx_destroy)(self.ctx);
            }
        }
    }
}
