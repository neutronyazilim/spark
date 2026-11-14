use super::f16::f16_to_f32;

pub const QK: usize = 32;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockQ8_0 {
    pub d: u16,
    pub qs: [i8; QK],
}

impl BlockQ8_0 {
    pub const BYTE_SIZE: usize = 34;

    #[inline]
    pub fn scale(self) -> f32 {
        f16_to_f32(self.d)
    }

    #[inline]
    pub fn value(self, index: usize) -> f32 {
        self.scale() * self.qs[index] as f32
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let raw = bytes.get(..Self::BYTE_SIZE)?;
        let mut qs = [0i8; QK];
        for (destination, source) in qs.iter_mut().zip(&raw[2..]) {
            *destination = *source as i8;
        }
        Some(Self {
            d: u16::from_le_bytes([raw[0], raw[1]]),
            qs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::BlockQ8_0;

    #[test]
    fn parses_standard_34_byte_block_without_alignment_assumptions() {
        let mut bytes = [0u8; BlockQ8_0::BYTE_SIZE];
        bytes[0] = 0x00;
        bytes[1] = 0x3c;
        bytes[2] = 0x80;
        bytes[33] = 127;
        let block = BlockQ8_0::from_bytes(&bytes).expect("complete block");
        assert_eq!(block.d, 0x3c00);
        assert_eq!(block.qs[0], -128);
        assert_eq!(block.qs[31], 127);
        assert_eq!(std::mem::size_of::<BlockQ8_0>(), BlockQ8_0::BYTE_SIZE);
    }
}
