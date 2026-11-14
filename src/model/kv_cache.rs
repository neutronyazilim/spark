#[derive(Debug)]
pub struct KvCache {
    layers: usize,
    attention_slots: Vec<Option<usize>>,
    recurrent_slots: Vec<Option<usize>>,
    context_len: usize,
    kv_heads: usize,
    head_dim: usize,
    keys: Vec<f32>,
    values: Vec<f32>,
    conv_width: usize,
    state_width: usize,
    conv_states: Vec<f32>,
    recurrent_states: Vec<f32>,
}

impl KvCache {
    pub fn clear(&mut self) {
        self.keys.fill(0.0);
        self.values.fill(0.0);
        self.conv_states.fill(0.0);
        self.recurrent_states.fill(0.0);
    }

    pub fn new(
        layers: usize,
        context_len: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, &'static str> {
        let len = layers
            .checked_mul(context_len)
            .and_then(|value| value.checked_mul(kv_heads))
            .and_then(|value| value.checked_mul(head_dim))
            .ok_or("KV cache size overflow")?;
        Ok(Self {
            layers,
            attention_slots: (0..layers).map(Some).collect(),
            recurrent_slots: vec![None; layers],
            context_len,
            kv_heads,
            head_dim,
            keys: vec![0.0; len],
            values: vec![0.0; len],
            conv_width: 0,
            state_width: 0,
            conv_states: Vec::new(),
            recurrent_states: Vec::new(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_qwen35(
        layers: usize,
        context_len: usize,
        kv_heads: usize,
        head_dim: usize,
        full_attention_interval: usize,
        conv_kernel: usize,
        conv_channels: usize,
        value_heads: usize,
        value_head_dim: usize,
    ) -> Result<Self, &'static str> {
        if full_attention_interval == 0 || conv_kernel == 0 {
            return Err("invalid Qwen35 cache dimensions");
        }
        let mut attention_slots = vec![None; layers];
        let mut recurrent_slots = vec![None; layers];
        let mut n_attn = 0usize;
        let mut n_recurrent = 0usize;
        for layer in 0..layers {
            if (layer + 1) % full_attention_interval == 0 {
                attention_slots[layer] = Some(n_attn);
                n_attn += 1;
            } else {
                recurrent_slots[layer] = Some(n_recurrent);
                n_recurrent += 1;
            }
        }
        let kv_width = kv_heads.checked_mul(head_dim).ok_or("KV width overflow")?;
        let kv_len = n_attn
            .checked_mul(context_len)
            .and_then(|v| v.checked_mul(kv_width))
            .ok_or("KV cache size overflow")?;
        let conv_width = (conv_kernel - 1)
            .checked_mul(conv_channels)
            .ok_or("convolution cache size overflow")?;
        let state_width = value_heads
            .checked_mul(value_head_dim)
            .and_then(|v| v.checked_mul(value_head_dim))
            .ok_or("recurrent state size overflow")?;
        Ok(Self {
            layers,
            attention_slots,
            recurrent_slots,
            context_len,
            kv_heads,
            head_dim,
            keys: vec![0.0; kv_len],
            values: vec![0.0; kv_len],
            conv_width,
            state_width,
            conv_states: vec![0.0; n_recurrent * conv_width],
            recurrent_states: vec![0.0; n_recurrent * state_width],
        })
    }

    pub fn store(
        &mut self,
        layer: usize,
        position: usize,
        keys: &[f32],
        values: &[f32],
    ) -> Result<(), &'static str> {
        if layer >= self.layers || position >= self.context_len {
            return Err("KV cache index out of bounds");
        }
        if self.attention_slots[layer].is_none() {
            return Err("layer has no KV cache slot");
        }
        let width = self.kv_heads * self.head_dim;
        if keys.len() != width || values.len() != width {
            return Err("KV cache vector has invalid length");
        }
        let start = self.index(layer, position, 0, 0);
        self.keys[start..start + width].copy_from_slice(keys);
        self.values[start..start + width].copy_from_slice(values);
        Ok(())
    }

    pub fn key(&self, layer: usize, position: usize, head: usize) -> &[f32] {
        let start = self.index(layer, position, head, 0);
        &self.keys[start..start + self.head_dim]
    }

    pub fn value(&self, layer: usize, position: usize, head: usize) -> &[f32] {
        let start = self.index(layer, position, head, 0);
        &self.values[start..start + self.head_dim]
    }

    pub fn recurrent_state_mut(
        &mut self,
        layer: usize,
    ) -> Result<(&mut [f32], &mut [f32]), &'static str> {
        let slot = self
            .recurrent_slots
            .get(layer)
            .copied()
            .flatten()
            .ok_or("layer has no recurrent cache slot")?;
        let conv_start = slot * self.conv_width;
        let state_start = slot * self.state_width;
        Ok((
            &mut self.conv_states[conv_start..conv_start + self.conv_width],
            &mut self.recurrent_states[state_start..state_start + self.state_width],
        ))
    }

    fn index(&self, layer: usize, position: usize, head: usize, dimension: usize) -> usize {
        let slot = self.attention_slots[layer].expect("layer has no KV cache slot");
        (((slot * self.context_len + position) * self.kv_heads + head) * self.head_dim)
            + dimension
    }
}

#[cfg(test)]
mod tests {
    use super::KvCache;

    #[test]
    fn stores_and_reads_only_the_requested_position() {
        let mut cache = KvCache::new(1, 4, 2, 2).expect("cache");
        cache
            .store(0, 2, &[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0])
            .expect("store");
        assert_eq!(cache.key(0, 2, 0), &[1.0, 2.0]);
        assert_eq!(cache.key(0, 2, 1), &[3.0, 4.0]);
        assert_eq!(cache.value(0, 2, 0), &[5.0, 6.0]);
        assert_eq!(cache.value(0, 2, 1), &[7.0, 8.0]);
        assert_eq!(cache.key(0, 0, 0), &[0.0, 0.0]);
    }

    #[test]
    fn positions_and_layers_do_not_overwrite_each_other() {
        let mut cache = KvCache::new(2, 3, 1, 2).expect("cache");
        cache.store(0, 0, &[1.0, 2.0], &[3.0, 4.0]).expect("store");
        cache.store(0, 2, &[5.0, 6.0], &[7.0, 8.0]).expect("store");
        cache
            .store(1, 0, &[9.0, 10.0], &[11.0, 12.0])
            .expect("store");

        assert_eq!(cache.key(0, 0, 0), &[1.0, 2.0]);
        assert_eq!(cache.key(0, 2, 0), &[5.0, 6.0]);
        assert_eq!(cache.key(1, 0, 0), &[9.0, 10.0]);
        assert_eq!(cache.value(0, 2, 0), &[7.0, 8.0]);
    }

    #[test]
    fn qwen35_maps_attention_and_recurrent_layers_separately() {
        let mut cache = KvCache::new_qwen35(8, 4, 1, 2, 4, 4, 6, 2, 2).expect("cache");
        assert!(cache.store(0, 0, &[1.0, 2.0], &[3.0, 4.0]).is_err());
        cache.store(3, 1, &[5.0, 6.0], &[7.0, 8.0]).expect("KV store");
        assert_eq!(cache.key(3, 1, 0), &[5.0, 6.0]);

        let (conv0, state0) = cache.recurrent_state_mut(0).expect("recurrent layer");
        conv0[0] = 9.0;
        state0[0] = 10.0;
        let (conv1, state1) = cache.recurrent_state_mut(1).expect("next recurrent layer");
        assert_eq!(conv1[0], 0.0);
        assert_eq!(state1[0], 0.0);

        cache.clear();
        let (conv0, state0) = cache.recurrent_state_mut(0).expect("cleared state");
        assert_eq!(conv0[0], 0.0);
        assert_eq!(state0[0], 0.0);
    }
}
