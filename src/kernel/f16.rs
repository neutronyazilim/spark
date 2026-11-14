#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = (bits & 0x03ff) as u32;

    let value = if exponent == 0 {
        if fraction == 0 {
            sign
        } else {
            let mut fraction = fraction;
            let mut exponent = -14i32;
            while fraction & 0x0400 == 0 {
                fraction <<= 1;
                exponent -= 1;
            }
            fraction &= 0x03ff;
            sign | (((exponent + 127) as u32) << 23) | (fraction << 13)
        }
    } else if exponent == 0x1f {
        sign | 0x7f80_0000 | (fraction << 13)
    } else {
        sign | (((exponent as u32 + 112) << 23) | (fraction << 13))
    };

    f32::from_bits(value)
}

pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;

    if exponent == 0xff {
        return sign | if mantissa == 0 { 0x7c00 } else { 0x7e00 };
    }

    let half_exp = exponent - 127 + 15;
    if half_exp >= 31 {
        return sign | 0x7c00;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let mant = mantissa | 0x80_0000;
        let shift = (14 - half_exp) as u32;
        let mut half = mant >> shift;
        let remainder = mant & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        if remainder > halfway || (remainder == halfway && (half & 1) != 0) {
            half += 1;
        }
        return sign | half as u16;
    }

    let mut half = ((half_exp as u32) << 10) | (mantissa >> 13);
    let remainder = mantissa & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && (half & 1) != 0) {
        half += 1;
    }
    sign | half as u16
}

#[cfg(test)]
mod tests {
    use super::{f16_to_f32, f32_to_f16};

    #[test]
    fn converts_normals_subnormals_and_special_values() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!(f16_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn f32_encoder_round_trips_representable_values() {
        for (value, bits) in [(0.0, 0x0000), (1.0, 0x3c00), (-2.0, 0xc000), (0.5, 0x3800)] {
            assert_eq!(f32_to_f16(value), bits);
            assert_eq!(f16_to_f32(bits), value);
        }
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7c00);
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
    }
}
