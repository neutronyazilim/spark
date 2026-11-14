use crate::loader::gguf::{GgufError, GgufModel};

#[derive(Debug, Clone, Copy)]
pub struct ModelConfig {
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
}

impl ModelConfig {
    pub fn from_gguf(model: &GgufModel<'_>) -> Result<Self, GgufError> {
        let architecture = model
            .metadata_value("general.architecture")
            .and_then(|value| value.as_str())
            .unwrap_or("qwen2");

        let arch = architecture;
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

        Ok(Self {
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
