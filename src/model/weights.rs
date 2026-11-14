use crate::{
    kernel::{
        gemv_q4_0, gemv_q4_0_parallel, gemv_q8_0, gemv_q8_0_parallel, pool::ThreadPool,
        q1_0::BlockQ1_0, q4_0::BlockQ4_0, q8_0::BlockQ8_0,
        ternary::{gemv_ternary, BlockQ2_0}, GemvError,
    },
    loader::gguf::{GgufError, GgufModel, TensorInfo},
    profile::ProfileStats,
};

const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_Q4_0: u32 = 2;
const GGML_TYPE_Q8_0: u32 = 8;
const GGML_TYPE_PTQ1_0: u32 = 143;
const GGML_TYPE_Q1_0: u32 = 41;

#[derive(Debug, Clone, Copy)]
pub struct Q1Matrix<'a> {
    pub bytes: &'a [u8],
    pub rows: usize,
    pub cols: usize,
}

impl Q1Matrix<'_> {
    pub fn blocks(&self) -> &[BlockQ1_0] {
        let count = self.rows * self.cols / 128;
        unsafe { std::slice::from_raw_parts(self.bytes.as_ptr().cast(), count) }
    }

    pub fn gemv_parallel(
        &self,
        input: &[f32],
        output: &mut [f32],
        pool: &ThreadPool,
        profile: Option<&mut ProfileStats>,
    ) -> Result<(), GemvError> {
        if input.len() != self.cols {
            return Err(GemvError::ActivationLength { expected: self.cols, actual: input.len() });
        }
        let rows = self.rows.min(output.len());
        let quantized = crate::kernel::q1_0::quantize_activations_q8_0(input);
        pool.gemv_q1_0(self.blocks(), &quantized, &mut output[..rows], self.cols, profile);
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Q8Matrix<'a> {
    pub bytes: &'a [u8],
    pub rows: usize,
    pub cols: usize,
}

impl Q8Matrix<'_> {
    pub fn blocks(&self) -> &[BlockQ8_0] {
        let count = self.rows * self.cols / 32;
        unsafe { std::slice::from_raw_parts(self.bytes.as_ptr().cast(), count) }
    }

    pub fn gemv(&self, input: &[f32], output: &mut [f32]) -> Result<(), GemvError> {
        gemv_q8_0(self.blocks(), input, output, self.cols)
    }

    pub fn gemv_parallel(
        &self,
        input: &[f32],
        output: &mut [f32],
        pool: &ThreadPool,
        profile: Option<&mut ProfileStats>,
    ) -> Result<(), GemvError> {
        gemv_q8_0_parallel(pool, self.blocks(), input, output, self.cols, profile)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Q4Matrix<'a> {
    pub bytes: &'a [u8],
    pub rows: usize,
    pub cols: usize,
}

impl Q4Matrix<'_> {
    pub fn blocks(&self) -> &[BlockQ4_0] {
        let count = self.rows * self.cols / 32;
        unsafe { std::slice::from_raw_parts(self.bytes.as_ptr().cast(), count) }
    }

    pub fn gemv(&self, input: &[f32], output: &mut [f32]) -> Result<(), GemvError> {
        gemv_q4_0(self.blocks(), input, output, self.cols)
    }

    pub fn gemv_parallel(
        &self,
        input: &[f32],
        output: &mut [f32],
        pool: &ThreadPool,
        profile: Option<&mut ProfileStats>,
    ) -> Result<(), GemvError> {
        gemv_q4_0_parallel(pool, self.blocks(), input, output, self.cols, profile)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TernaryMatrix<'a> {
    pub bytes: &'a [u8],
    pub rows: usize,
    pub cols: usize,
}

impl TernaryMatrix<'_> {
    pub fn blocks(&self) -> &[BlockQ2_0] {
        let count = self.rows * self.cols / 32;
        unsafe { std::slice::from_raw_parts(self.bytes.as_ptr().cast(), count) }
    }

    pub fn gemv(&self, input: &[f32], output: &mut [f32]) -> Result<(), GemvError> {
        let rows = self.rows.min(output.len());
        gemv_ternary(self.blocks(), input, &mut output[..rows], self.cols / 32);
        Ok(())
    }

    pub fn gemv_parallel(
        &self,
        input: &[f32],
        output: &mut [f32],
        _pool: &ThreadPool,
        _profile: Option<&mut ProfileStats>,
    ) -> Result<(), GemvError> {
        self.gemv(input, output)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PTQ1Matrix<'a> {
    pub bytes: &'a [u8],
    pub rows: usize,
    pub cols: usize,
}

impl PTQ1Matrix<'_> {
    pub fn blocks(&self) -> &[crate::kernel::ternary::BlockPTQ1_0] {
        let count = (self.rows * self.cols) / 128;
        unsafe { std::slice::from_raw_parts(self.bytes.as_ptr().cast(), count) }
    }

    pub fn gemv(&self, input: &[f32], output: &mut [f32]) -> Result<(), GemvError> {
        let rows = self.rows.min(output.len());
        crate::kernel::ternary::gemv_ptq1_0(self.blocks(), input, &mut output[..rows], self.cols / 128);
        Ok(())
    }

    pub fn gemv_parallel(
        &self,
        input: &[f32],
        output: &mut [f32],
        pool: &ThreadPool,
        profile: Option<&mut ProfileStats>,
    ) -> Result<(), GemvError> {
        let rows = self.rows.min(output.len());
        pool.gemv_ptq1_0(self.blocks(), input, &mut output[..rows], self.cols, profile);
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum QuantizedMatrix<'a> {
    Q1(Q1Matrix<'a>),
    Q8(Q8Matrix<'a>),
    Q4(Q4Matrix<'a>),
    Ternary(TernaryMatrix<'a>),
    PTQ1(PTQ1Matrix<'a>),
}

impl<'a> QuantizedMatrix<'a> {
    pub fn rows(&self) -> usize {
        match self {
            Self::Q1(m) => m.rows,
            Self::Q8(m) => m.rows,
            Self::Q4(m) => m.rows,
            Self::Ternary(m) => m.rows,
            Self::PTQ1(m) => m.rows,
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Self::Q1(m) => m.cols,
            Self::Q8(m) => m.cols,
            Self::Q4(m) => m.cols,
            Self::Ternary(m) => m.cols,
            Self::PTQ1(m) => m.cols,
        }
    }

    pub fn raw_bytes(&self) -> &'a [u8] {
        match self {
            Self::Q1(m) => m.bytes,
            Self::Q8(m) => m.bytes,
            Self::Q4(m) => m.bytes,
            Self::Ternary(m) => m.bytes,
            Self::PTQ1(m) => m.bytes,
        }
    }

    pub fn is_q4(&self) -> bool {
        matches!(self, Self::Q4(_))
    }

    pub fn is_q1(&self) -> bool {
        matches!(self, Self::Q1(_))
    }

    pub fn is_ternary(&self) -> bool {
        matches!(self, Self::Ternary(_) | Self::PTQ1(_))
    }

    pub fn gemv_parallel(
        &self,
        input: &[f32],
        output: &mut [f32],
        pool: &ThreadPool,
        profile: Option<&mut ProfileStats>,
    ) -> Result<(), GemvError> {
        match self {
            Self::Q1(m) => m.gemv_parallel(input, output, pool, profile),
            Self::Q8(m) => m.gemv_parallel(input, output, pool, profile),
            Self::Q4(m) => m.gemv_parallel(input, output, pool, profile),
            Self::Ternary(m) => m.gemv_parallel(input, output, pool, profile),
            Self::PTQ1(m) => m.gemv_parallel(input, output, pool, profile),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct F32Vector<'a> {
    bytes: &'a [u8],
    pub len: usize,
}

impl F32Vector<'_> {
    pub fn values(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.bytes.as_ptr().cast(), self.len) }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct F32Matrix<'a> {
    bytes: &'a [u8],
    pub rows: usize,
    pub cols: usize,
}

impl F32Matrix<'_> {
    pub fn values(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.bytes.as_ptr().cast(), self.rows * self.cols) }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Qwen35AttentionWeights<'a> {
    Full {
        q: QuantizedMatrix<'a>,
        k: QuantizedMatrix<'a>,
        v: QuantizedMatrix<'a>,
        output: QuantizedMatrix<'a>,
        q_norm: F32Vector<'a>,
        k_norm: F32Vector<'a>,
    },
    Recurrent {
        qkv: QuantizedMatrix<'a>,
        z: QuantizedMatrix<'a>,
        alpha: QuantizedMatrix<'a>,
        beta: QuantizedMatrix<'a>,
        conv1d: F32Matrix<'a>,
        dt_bias: F32Vector<'a>,
        a: F32Vector<'a>,
        state_norm: F32Vector<'a>,
        output: QuantizedMatrix<'a>,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct Qwen35LayerWeights<'a> {
    pub attn_norm: F32Vector<'a>,
    pub post_attention_norm: F32Vector<'a>,
    pub attention: Qwen35AttentionWeights<'a>,
    pub gate: QuantizedMatrix<'a>,
    pub up: QuantizedMatrix<'a>,
    pub down: QuantizedMatrix<'a>,
}

#[derive(Debug, Clone, Copy)]
pub struct LayerWeights<'a> {
    pub attn_norm: F32Vector<'a>,
    pub q: QuantizedMatrix<'a>,
    pub q_bias: F32Vector<'a>,
    pub k: QuantizedMatrix<'a>,
    pub k_bias: F32Vector<'a>,
    pub v: QuantizedMatrix<'a>,
    pub v_bias: F32Vector<'a>,
    pub o: QuantizedMatrix<'a>,
    pub ffn_norm: F32Vector<'a>,
    pub gate: QuantizedMatrix<'a>,
    pub up: QuantizedMatrix<'a>,
    pub down: QuantizedMatrix<'a>,
}

impl<'a> LayerWeights<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_qkv_parallel(
        &self,
        input: &[f32],
        q_out: &mut [f32],
        k_out: &mut [f32],
        v_out: &mut [f32],
        pool: &ThreadPool,
        profile: Option<&mut ProfileStats>,
    ) -> Result<(), GemvError> {
        if self.q.cols() != input.len() || self.k.cols() != input.len() || self.v.cols() != input.len() {
            return Err(GemvError::ActivationLength {
                expected: self.q.cols(),
                actual: input.len(),
            });
        }

        match (self.q, self.k, self.v) {
            (QuantizedMatrix::Q8(q_mat), QuantizedMatrix::Q8(k_mat), QuantizedMatrix::Q8(v_mat)) => {
                pool.gemv_qkv_q8_0(
                    q_mat.blocks(),
                    k_mat.blocks(),
                    v_mat.blocks(),
                    input,
                    q_out,
                    k_out,
                    v_out,
                    q_mat.cols,
                    profile,
                );
            }
            _ => {
                self.q.gemv_parallel(input, q_out, pool, None)?;
                self.k.gemv_parallel(input, k_out, pool, None)?;
                self.v.gemv_parallel(input, v_out, pool, None)?;
            }
        }
        Ok(())
    }

    pub fn gemv_fused_ffn_parallel(
        &self,
        input: &[f32],
        output: &mut [f32],
        pool: &ThreadPool,
        profile: Option<&mut ProfileStats>,
    ) -> Result<(), GemvError> {
        if self.gate.cols() != input.len() || self.up.cols() != input.len() {
            return Err(GemvError::ActivationLength {
                expected: self.gate.cols(),
                actual: input.len(),
            });
        }

        match (self.gate, self.up) {
            (QuantizedMatrix::Q8(gate_mat), QuantizedMatrix::Q8(up_mat)) => {
                pool.gemv_fused_ffn_q8_0(
                    gate_mat.blocks(),
                    up_mat.blocks(),
                    input,
                    output,
                    gate_mat.cols,
                    profile,
                );
            }
            (QuantizedMatrix::Q4(gate_mat), QuantizedMatrix::Q4(up_mat)) => {
                gate_mat.gemv_parallel(input, output, pool, None)?;
                let mut up_out = vec![0.0f32; output.len()];
                up_mat.gemv_parallel(input, &mut up_out, pool, None)?;
                crate::model::ops::swiglu(output, &up_out);
            }
            _ => {
                self.gate.gemv_parallel(input, output, pool, None)?;
                let mut up_out = vec![0.0f32; output.len()];
                self.up.gemv_parallel(input, &mut up_out, pool, None)?;
                crate::model::ops::swiglu(output, &up_out);
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct ModelWeights<'a> {
    pub token_embedding: QuantizedMatrix<'a>,
    pub output: QuantizedMatrix<'a>,
    pub final_norm: F32Vector<'a>,
    pub layers: Vec<LayerWeights<'a>>,
    pub qwen35_layers: Vec<Qwen35LayerWeights<'a>>,
}

impl<'a> ModelWeights<'a> {
    pub fn from_gguf(model: &'a GgufModel<'a>, layers: usize) -> Result<Self, GgufError> {
        let token_embedding = matrix(model, find_tensor(model, &["token_embd.weight"])?)?;
        let output = matrix(
            model,
            find_tensor(model, &["output.weight", "token_embd.weight"])?,
        )?;
        let final_norm = vector(model, find_tensor(model, &["output_norm.weight", "norm.weight"])?)?;

        if model
            .metadata_value("general.architecture")
            .and_then(|value| value.as_str())
            == Some("qwen35")
        {
            let mut qwen35_layers = Vec::with_capacity(layers);
            for layer in 0..layers {
                let prefix = format!("blk.{layer}.");
                let attn_norm = vector(
                    model,
                    required_prefixed(model, &prefix, &["attn_norm.weight"])?
                )?;
                let post_attention_norm = vector(
                    model,
                    required_prefixed(model, &prefix, &["post_attention_norm.weight"])?
                )?;

                let attention = if model.tensor(&format!("{prefix}ssm_out.weight")).is_some() {
                    Qwen35AttentionWeights::Recurrent {
                        qkv: matrix(model, required_prefixed(model, &prefix, &["attn_qkv.weight"])?)?,
                        z: matrix(model, required_prefixed(model, &prefix, &["attn_gate.weight"])?)?,
                        alpha: matrix(model, required_prefixed(model, &prefix, &["ssm_alpha.weight"])?)?,
                        beta: matrix(model, required_prefixed(model, &prefix, &["ssm_beta.weight"])?)?,
                        conv1d: f32_matrix(model, required_prefixed(model, &prefix, &["ssm_conv1d.weight"])?)?,
                        dt_bias: vector(model, required_prefixed(model, &prefix, &["ssm_dt.bias"])?)?,
                        a: vector(model, required_prefixed(model, &prefix, &["ssm_a"])?)?,
                        state_norm: vector(model, required_prefixed(model, &prefix, &["ssm_norm.weight"])?)?,
                        output: matrix(model, required_prefixed(model, &prefix, &["ssm_out.weight"])?)?,
                    }
                } else {
                    Qwen35AttentionWeights::Full {
                        q: matrix(model, required_prefixed(model, &prefix, &["attn_q.weight"])?)?,
                        k: matrix(model, required_prefixed(model, &prefix, &["attn_k.weight"])?)?,
                        v: matrix(model, required_prefixed(model, &prefix, &["attn_v.weight"])?)?,
                        output: matrix(model, required_prefixed(model, &prefix, &["attn_output.weight"])?)?,
                        q_norm: vector(model, required_prefixed(model, &prefix, &["attn_q_norm.weight"])?)?,
                        k_norm: vector(model, required_prefixed(model, &prefix, &["attn_k_norm.weight"])?)?,
                    }
                };

                qwen35_layers.push(Qwen35LayerWeights {
                    attn_norm,
                    post_attention_norm,
                    attention,
                    gate: matrix(model, required_prefixed(model, &prefix, &["ffn_gate.weight"])?)?,
                    up: matrix(model, required_prefixed(model, &prefix, &["ffn_up.weight"])?)?,
                    down: matrix(model, required_prefixed(model, &prefix, &["ffn_down.weight"])?)?,
                });
            }

            return Ok(Self {
                token_embedding,
                output,
                final_norm,
                layers: Vec::new(),
                qwen35_layers,
            });
        }

        let mut layer_weights = Vec::with_capacity(layers);
        for layer in 0..layers {
            let prefix = format!("blk.{layer}.");
            let attn_norm = vector(
                model,
                required_prefixed(model, &prefix, &["attn_norm.weight", "input_layernorm.weight"])?,
            )?;

            // Support either separate Q/K/V tensors or fused attn_qkv tensor
            let q = matrix(
                model,
                required_prefixed(model, &prefix, &["attn_q.weight", "attn_qkv.weight", "self_attn.q_proj.weight"])?,
            )?;
            let k = matrix(
                model,
                required_prefixed(model, &prefix, &["attn_k.weight", "attn_qkv.weight", "self_attn.k_proj.weight"])?,
            )?;
            let v = matrix(
                model,
                required_prefixed(model, &prefix, &["attn_v.weight", "attn_qkv.weight", "self_attn.v_proj.weight"])?,
            )?;

            let q_bias = optional_vector(model, &prefix, &["attn_q.bias", "self_attn.q_proj.bias"]);
            let k_bias = optional_vector(model, &prefix, &["attn_k.bias", "self_attn.k_proj.bias"]);
            let v_bias = optional_vector(model, &prefix, &["attn_v.bias", "self_attn.v_proj.bias"]);

            let o = matrix(
                model,
                required_prefixed(model, &prefix, &["attn_output.weight", "attn_out.weight", "self_attn.o_proj.weight", "ssm_out.weight"])?,
            )?;

            let ffn_norm = vector(
                model,
                required_prefixed(
                    model,
                    &prefix,
                    &["ffn_norm.weight", "post_attention_norm.weight", "post_attention_layernorm.weight"],
                )?,
            )?;

            let gate = matrix(
                model,
                required_prefixed(model, &prefix, &["ffn_gate.weight", "mlp.gate_proj.weight"])?,
            )?;
            let up = matrix(
                model,
                required_prefixed(model, &prefix, &["ffn_up.weight", "mlp.up_proj.weight"])?,
            )?;
            let down = matrix(
                model,
                required_prefixed(model, &prefix, &["ffn_down.weight", "mlp.down_proj.weight"])?,
            )?;

            layer_weights.push(LayerWeights {
                attn_norm,
                q,
                q_bias,
                k,
                k_bias,
                v,
                v_bias,
                o,
                ffn_norm,
                gate,
                up,
                down,
            });
        }
        Ok(Self {
            token_embedding,
            output,
            final_norm,
            layers: layer_weights,
            qwen35_layers: Vec::new(),
        })
    }
}

fn find_tensor<'a>(
    model: &'a GgufModel<'a>,
    names: &[&str],
) -> Result<&'a TensorInfo<'a>, GgufError> {
    names
        .iter()
        .find_map(|name| model.tensor(name))
        .ok_or_else(|| GgufError::InvalidFormat(format!("missing tensor {}", names[0])))
}

fn required_prefixed<'a>(
    model: &'a GgufModel<'a>,
    prefix: &str,
    suffixes: &[&str],
) -> Result<&'a TensorInfo<'a>, GgufError> {
    let mut names = Vec::with_capacity(suffixes.len());
    for s in suffixes {
        names.push(format!("{prefix}{s}"));
    }
    let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    find_tensor(model, &refs)
}

fn optional_vector<'a>(
    model: &'a GgufModel<'a>,
    prefix: &str,
    suffixes: &[&str],
) -> F32Vector<'a> {
    if let Ok(tensor) = required_prefixed(model, prefix, suffixes) {
        if let Ok(vec) = vector(model, tensor) {
            return vec;
        }
    }
    F32Vector { bytes: &[], len: 0 }
}

fn matrix<'a>(
    model: &'a GgufModel<'a>,
    tensor: &TensorInfo<'a>,
) -> Result<QuantizedMatrix<'a>, GgufError> {
    if tensor.dimensions.len() != 2 {
        return Err(GgufError::InvalidFormat(format!(
            "tensor {} must be a rank-2 matrix",
            tensor.name
        )));
    }
    let cols = usize::try_from(tensor.dimensions[0])
        .map_err(|_| GgufError::CountTooLarge("tensor dimension", tensor.dimensions[0]))?;
    let rows = usize::try_from(tensor.dimensions[1])
        .map_err(|_| GgufError::CountTooLarge("tensor dimension", tensor.dimensions[1]))?;
    if !cols.is_multiple_of(32) {
        return Err(GgufError::InvalidFormat(format!(
            "tensor {} columns are not a multiple of 32",
            tensor.name
        )));
    }

    match tensor.ggml_type {
        GGML_TYPE_Q1_0 => {
            if !cols.is_multiple_of(128) {
                return Err(GgufError::InvalidFormat(format!(
                    "tensor {} Q1_0 columns are not a multiple of 128",
                    tensor.name
                )));
            }
            let bytes = model.tensor_bytes(tensor, rows * cols / 128 * BlockQ1_0::BYTE_SIZE)?;
            Ok(QuantizedMatrix::Q1(Q1Matrix { bytes, rows, cols }))
        }
        GGML_TYPE_Q8_0 => {
            let bytes = model.tensor_bytes(tensor, rows * cols / 32 * BlockQ8_0::BYTE_SIZE)?;
            Ok(QuantizedMatrix::Q8(Q8Matrix { bytes, rows, cols }))
        }
        GGML_TYPE_Q4_0 => {
            let bytes = model.tensor_bytes(tensor, rows * cols / 32 * BlockQ4_0::BYTE_SIZE)?;
            Ok(QuantizedMatrix::Q4(Q4Matrix { bytes, rows, cols }))
        }
        // PTQ1_0 / Q2_0 / 2-bit ternary format
        10 | 20 | 31 => {
            let bytes = model.tensor_bytes(tensor, rows * cols / 32 * BlockQ2_0::BYTE_SIZE)?;
            Ok(QuantizedMatrix::Ternary(TernaryMatrix { bytes, rows, cols }))
        }
        GGML_TYPE_PTQ1_0 => {
            // 7 bytes per 32 elements (1.75 bits per weight)
            let bytes = model.tensor_bytes(tensor, (rows * cols * 7) / 32)?;
            Ok(QuantizedMatrix::PTQ1(PTQ1Matrix { bytes, rows, cols }))
        }
        other => Err(GgufError::InvalidFormat(format!(
            "tensor {} has unsupported ggml_type {}",
            tensor.name, other
        ))),
    }
}

fn vector<'a>(
    model: &'a GgufModel<'a>,
    tensor: &TensorInfo<'a>,
) -> Result<F32Vector<'a>, GgufError> {
    if tensor.ggml_type != GGML_TYPE_F32 || tensor.dimensions.len() != 1 {
        return Err(GgufError::InvalidFormat(format!(
            "tensor {} must be a rank-1 F32 vector",
            tensor.name
        )));
    }
    let len = usize::try_from(tensor.dimensions[0])
        .map_err(|_| GgufError::CountTooLarge("tensor dimension", tensor.dimensions[0]))?;
    let bytes = model.tensor_bytes(tensor, len * std::mem::size_of::<f32>())?;
    Ok(F32Vector { bytes, len })
}

fn f32_matrix<'a>(
    model: &'a GgufModel<'a>,
    tensor: &TensorInfo<'a>,
) -> Result<F32Matrix<'a>, GgufError> {
    if tensor.ggml_type != GGML_TYPE_F32 || tensor.dimensions.len() != 2 {
        return Err(GgufError::InvalidFormat(format!(
            "tensor {} must be a rank-2 F32 matrix",
            tensor.name
        )));
    }
    let cols = usize::try_from(tensor.dimensions[0])
        .map_err(|_| GgufError::CountTooLarge("tensor dimension", tensor.dimensions[0]))?;
    let rows = usize::try_from(tensor.dimensions[1])
        .map_err(|_| GgufError::CountTooLarge("tensor dimension", tensor.dimensions[1]))?;
    let bytes = model.tensor_bytes(tensor, rows * cols * std::mem::size_of::<f32>())?;
    Ok(F32Matrix { bytes, rows, cols })
}
