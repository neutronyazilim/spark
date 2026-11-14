use crate::loader::gguf::{GgufError, GgufModel};

#[derive(Debug, Clone, Copy)]
pub struct ModelConfig {
    pub is_qwen35: bool,
    pub hidden_dim: usize,
    pub layers: usize,
    pub intermediate_dim: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub context_len: usize,
    pub rope_base: f32,
    pub rms_epsilon: f32,
    pub has_hadamard: bool,
    pub rope_dimension_count: usize,
    pub full_attention_interval: usize,
    pub ssm_conv_kernel: usize,
    pub ssm_state_size: usize,
    pub ssm_group_count: usize,
    pub ssm_time_step_rank: usize,
    pub ssm_inner_size: usize,
}

impl ModelConfig {
    pub fn from_gguf(model: &GgufModel<'_>) -> Result<Self, GgufError> {
        let architecture = model
            .metadata_value("general.architecture")
            .and_then(|value| value.as_str())
            .unwrap_or("qwen2");

        let arch = architecture;
        let is_qwen35 = arch == "qwen35";
        let key = |suffix: &str| format!("{arch}.{suffix}");

        let hidden_dim = get_usize(model, &key("embedding_length"))
            .ok_or_else(|| GgufError::InvalidFormat(format!("missing {arch}.embedding_length")))?;

        let heads = get_usize(model, &key("attention.head_count"))
            .ok_or_else(|| GgufError::InvalidFormat(format!("missing {arch}.attention.head_count")))?;

        let head_dim = get_usize(model, &key("attention.key_length"))
            .or_else(|| hidden_dim.checked_div(heads))
            .ok_or_else(|| GgufError::InvalidFormat("invalid attention head dimensions".into()))?;

        let layers = get_usize(model, &key("block_count"))
            .ok_or_else(|| GgufError::InvalidFormat(format!("missing {arch}.block_count")))?;

        let intermediate_dim = get_usize(model, &key("feed_forward_length"))
            .unwrap_or_else(|| (hidden_dim * 8) / 3);

        let kv_heads = get_usize(model, &key("attention.head_count_kv")).unwrap_or(heads);

        let vocab_size = model
            .tensor("output.weight")
            .or_else(|| model.tensor("token_embd.weight"))
            .and_then(|tensor| tensor.dimensions.get(1))
            .and_then(|value| usize::try_from(*value).ok())
            .ok_or_else(|| GgufError::InvalidFormat("cannot determine vocabulary size".into()))?;

        let context_len = get_usize(model, &key("context_length"))
            .unwrap_or(2048)
            .min(2048); // Safe default for bounded zero-allocation KV cache; CLI can override

        let rope_base = get_f32(model, &key("rope.freq_base")).unwrap_or(1_000_000.0);

        let rms_epsilon = get_f32(model, &key("attention.layer_norm_rms_epsilon"))
            .or_else(|| get_f32(model, &key("attention.layer_norm_epsilon")))
            .unwrap_or(1e-6);

        let has_hadamard = model.metadata_value("prism.hadamard.version").is_some();
        let rope_dimension_count = get_usize(model, &key("rope.dimension_count"))
            .unwrap_or(head_dim);
        let full_attention_interval = get_usize(model, &key("full_attention_interval"))
            .unwrap_or(1);
        let ssm_conv_kernel = get_usize(model, &key("ssm.conv_kernel")).unwrap_or(0);
        let ssm_state_size = get_usize(model, &key("ssm.state_size")).unwrap_or(0);
        let ssm_group_count = get_usize(model, &key("ssm.group_count")).unwrap_or(0);
        let ssm_time_step_rank = get_usize(model, &key("ssm.time_step_rank")).unwrap_or(0);
        let ssm_inner_size = get_usize(model, &key("ssm.inner_size")).unwrap_or(0);

        Ok(Self {
            is_qwen35,
            hidden_dim,
            layers,
            intermediate_dim,
            heads,
            kv_heads,
            head_dim,
            vocab_size,
            context_len,
            rope_base,
            rms_epsilon,
            has_hadamard,
            rope_dimension_count,
            full_attention_interval,
            ssm_conv_kernel,
            ssm_state_size,
            ssm_group_count,
            ssm_time_step_rank,
            ssm_inner_size,
        })
    }
}

fn get_usize(model: &GgufModel<'_>, key: &str) -> Option<usize> {
    model
        .metadata_value(key)
        .and_then(|value| value.as_u64())
        .and_then(|value| usize::try_from(value).ok())
}

fn get_f32(model: &GgufModel<'_>, key: &str) -> Option<f32> {
    model
        .metadata_value(key)
        .and_then(|value| value.as_f32())
}
