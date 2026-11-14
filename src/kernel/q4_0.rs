use super::f16::f16_to_f32;

pub const QK: usize = 32;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockQ4_0 {
    pub d: u16,
    pub qs: [u8; 16],
}

impl BlockQ4_0 {
    pub const BYTE_SIZE: usize = 18;

    #[inline]
    pub fn scale(self) -> f32 {
        f16_to_f32(self.d)
    }

    #[inline]
    pub fn quantized(self, index: usize) -> i32 {
        let packed = if index < 16 {
            self.qs[index] & 0x0f
        } else {
            self.qs[index - 16] >> 4
        };
        packed as i32 - 8
    }

    #[inline]
    pub fn value(self, index: usize) -> f32 {
        self.scale() * self.quantized(index) as f32
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let raw = bytes.get(..Self::BYTE_SIZE)?;
        let mut qs = [0u8; 16];
        qs.copy_from_slice(&raw[2..]);
        Some(Self {
            d: u16::from_le_bytes([raw[0], raw[1]]),
            qs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::BlockQ4_0;

    #[test]
    fn unpacks_low_and_high_nibbles_in_ggml_order() {
        let mut bytes = [0u8; BlockQ4_0::BYTE_SIZE];
        bytes[0] = 0x00;
        bytes[1] = 0x3c;
        bytes[2] = 0x01;
        bytes[3] = 0xef;
        let block = BlockQ4_0::from_bytes(&bytes).expect("complete block");
        assert_eq!(block.quantized(0), -7);
        assert_eq!(block.quantized(1), 7);
        assert_eq!(block.quantized(16), -8);
        assert_eq!(block.quantized(17), 6);
        assert_eq!(std::mem::size_of::<BlockQ4_0>(), BlockQ4_0::BYTE_SIZE);
    }
}
