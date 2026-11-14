#[derive(Debug)]
pub struct KvCache {
    layers: usize,
    context_len: usize,
    kv_heads: usize,
    head_dim: usize,
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl KvCache {
    pub fn clear(&mut self) {
        self.keys.fill(0.0);
        self.values.fill(0.0);
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
            context_len,
            kv_heads,
            head_dim,
            keys: vec![0.0; len],
            values: vec![0.0; len],
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

    fn index(&self, layer: usize, position: usize, head: usize, dimension: usize) -> usize {
        (((layer * self.context_len + position) * self.kv_heads + head) * self.head_dim) + dimension
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
}
