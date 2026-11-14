use crate::loader::gguf::{GgufError, GgufModel};

pub const QWEN_IM_END_TOKEN_ID: u32 = 151_645;

#[derive(Debug)]
pub struct BpeTokenizer {
    vocab: Vec<String>,
    merges: Vec<(String, String)>,
    bos_token: Option<u32>,
    eos_token: Option<u32>,
    add_bos_token: bool,
    special_tokens: Vec<(String, u32)>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SegmentKind {
    None,
    Whitespace,
    Alphabetic,
    Numeric,
    Other,
}

impl SegmentKind {
    fn for_char(character: char) -> Self {
        if character.is_whitespace() {
            Self::Whitespace
        } else if character.is_alphabetic() {
            Self::Alphabetic
        } else if character.is_numeric() {
            Self::Numeric
        } else {
            Self::Other
        }
    }
}

impl BpeTokenizer {
    pub fn from_gguf(model: &GgufModel<'_>) -> Result<Self, GgufError> {
        let tokens = model
            .metadata_value("tokenizer.ggml.tokens")
            .and_then(|value| value.array_strings())
            .ok_or_else(|| GgufError::InvalidFormat("missing tokenizer vocabulary".into()))?;
        let merge_strings = model
            .metadata_value("tokenizer.ggml.merges")
            .and_then(|value| value.array_strings())
            .ok_or_else(|| GgufError::InvalidFormat("missing tokenizer merges".into()))?;
        let merges = merge_strings
            .into_iter()
            .filter_map(|merge| {
                let (left, right) = merge.split_once(' ')?;
                Some((left.to_owned(), right.to_owned()))
            })
            .collect();
        let vocab: Vec<String> = tokens.into_iter().map(str::to_owned).collect();
        let token_types = model
            .metadata_value("tokenizer.ggml.token_type")
            .and_then(|value| value.array_u32())
            .unwrap_or_default();
        let mut special_tokens: Vec<(String, u32)> = vocab
            .iter()
            .enumerate()
            .filter(|(index, token)| {
                matches!(token_types.get(*index), Some(3 | 4))
                    || (token.starts_with("<|") && token.ends_with("|>"))
                    || matches!(token.as_str(), "<think>" | "</think>")
            })
            .map(|(index, token)| (token.clone(), index as u32))
            .collect();
        special_tokens.sort_unstable_by_key(|token| std::cmp::Reverse(token.0.len()));
        Ok(Self {
            vocab,
            merges,
            bos_token: metadata_id(model, "tokenizer.ggml.bos_token_id"),
            eos_token: metadata_id(model, "tokenizer.ggml.eos_token_id"),
            add_bos_token: model
                .metadata_value("tokenizer.ggml.add_bos_token")
                .and_then(|value| value.as_bool())
                .unwrap_or(false),
            special_tokens,
        })
    }

    pub fn bos_token(&self) -> Option<u32> {
        self.bos_token
    }

    pub fn should_add_bos(&self) -> bool {
        self.add_bos_token && self.bos_token.is_some()
    }

    pub fn eos_token(&self) -> Option<u32> {
        self.eos_token
    }

    pub fn is_stop_token(&self, token: u32) -> bool {
        self.is_im_end_token(token) || self.eos_token == Some(token)
    }

    pub fn is_im_end_token(&self, token: u32) -> bool {
        token == QWEN_IM_END_TOKEN_ID || self.decode_token(token) == Some("<|im_end|>")
    }

    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let mut ids = Vec::new();
        if add_bos {
            if let Some(bos) = self.bos_token {
                ids.push(bos);
            }
        }

        let mut remaining = text;
        while !remaining.is_empty() {
            let next_special = self.special_tokens
                .iter()
                .filter_map(|(special, id)| {
                    remaining
                        .find(special)
                        .map(|index| (index, special.as_str(), *id))
                })
                .min_by_key(|(index, _, _)| *index);
            let Some((index, special, special_id)) = next_special else {
                ids.extend(self.encode_text(remaining));
                break;
            };
            if index != 0 {
                ids.extend(self.encode_text(&remaining[..index]));
            }
            ids.push(special_id);
            remaining = &remaining[index + special.len()..];
        }
        ids
    }

    fn encode_text(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut segment = String::new();
        let mut kind = SegmentKind::None;

        for character in text.chars() {
            let next_kind = SegmentKind::for_char(character);
            let attach_to_previous = !segment.is_empty()
                && segment.starts_with(' ')
                && segment.chars().count() == 1
                && next_kind != SegmentKind::Whitespace;
            if !segment.is_empty() && next_kind != kind && !attach_to_previous {
                ids.extend(self.encode_segment(&segment));
                segment.clear();
            }
            segment.push(character);
            kind = next_kind;
        }
        if !segment.is_empty() {
            ids.extend(self.encode_segment(&segment));
        }
        ids
    }

    fn encode_segment(&self, text: &str) -> Vec<u32> {
        let mut symbols: Vec<String> = text
            .as_bytes()
            .iter()
            .map(|&byte| byte_to_unicode(byte).to_string())
            .collect();
        let mut ids = Vec::with_capacity(symbols.len());
        while symbols.len() > 1 {
            let mut best: Option<(usize, usize)> = None;
            for (rank, (left, right)) in self.merges.iter().enumerate() {
                for index in 0..symbols.len() - 1 {
                    if symbols[index] == *left
                        && symbols[index + 1] == *right
                        && best.is_none_or(|(best_rank, _)| rank < best_rank)
                    {
                        best = Some((rank, index));
                    }
                }
            }
            let Some((_, index)) = best else { break };
            let merged = symbols[index].clone() + &symbols[index + 1];
            symbols.splice(index..=index + 1, [merged]);
        }
        for symbol in symbols {
            if let Some(id) = self.vocab.iter().position(|vocab| vocab == &symbol) {
                ids.push(id as u32);
            } else {
                for character in symbol.chars() {
                    if let Some(id) = self
                        .vocab
                        .iter()
                        .position(|vocab| vocab == &character.to_string())
                    {
                        ids.push(id as u32);
                    }
                }
            }
        }
        ids
    }

    pub fn decode_token(&self, token: u32) -> Option<&str> {
        self.vocab.get(token as usize).map(String::as_str)
    }

    pub fn decode_stream(&self, token: u32, pending: &mut Vec<u8>) -> String {
        let Some(piece) = self.decode_token(token) else {
            return String::new();
        };
        for character in piece.chars() {
            if let Some(byte) = unicode_to_byte(character) {
                pending.push(byte);
            } else {
                pending.extend_from_slice(character.to_string().as_bytes());
            }
        }
        match std::str::from_utf8(pending) {
            Ok(text) => {
                let result = text.to_owned();
                pending.clear();
                result
            }
            Err(error) => {
                let valid = error.valid_up_to();
                let result = String::from_utf8_lossy(&pending[..valid]).into_owned();
                pending.drain(..valid);
                result
            }
        }
    }
}

fn is_direct_byte(byte: u8) -> bool {
    matches!(byte, 33..=126 | 161..=172 | 174..=255)
}

fn byte_to_unicode(byte: u8) -> char {
    if is_direct_byte(byte) {
        return byte as char;
    }
    let omitted_before = (0..byte).filter(|&value| !is_direct_byte(value)).count();
    char::from_u32(256 + omitted_before as u32).expect("byte-level BPE code point")
}

fn unicode_to_byte(character: char) -> Option<u8> {
    let code = character as u32;
    if code <= u8::MAX as u32 && is_direct_byte(code as u8) {
        return Some(code as u8);
    }
    if !(256..=511).contains(&code) {
        return None;
    }
    let omitted_index = (code - 256) as usize;
    (0..=u8::MAX)
        .filter(|&value| !is_direct_byte(value))
        .nth(omitted_index)
}

fn metadata_id(model: &GgufModel<'_>, key: &str) -> Option<u32> {
    model
        .metadata_value(key)
        .and_then(|value| value.as_u64())
        .and_then(|value| u32::try_from(value).ok())
}

#[cfg(test)]
mod tests {
    use super::{byte_to_unicode, unicode_to_byte, BpeTokenizer, QWEN_IM_END_TOKEN_ID};

    #[test]
    fn merge_rank_prefers_earlier_rule() {
        let mut symbols = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let merges = [("a".to_string(), "b".to_string())];
        let mut merged = false;
        for (left, right) in merges {
            if symbols[0] == left && symbols[1] == right {
                symbols.splice(0..=1, [left + &right]);
                merged = true;
            }
        }
        assert!(merged);
        assert_eq!(symbols, ["ab".to_string(), "c".to_string()]);
    }

    #[test]
    fn byte_level_mapping_round_trips_utf8_bytes() {
        for byte in 0..=u8::MAX {
            assert_eq!(unicode_to_byte(byte_to_unicode(byte)), Some(byte));
        }
    }

    #[test]
    fn special_chat_tokens_are_kept_as_single_tokens() {
        let tokenizer = BpeTokenizer {
            vocab: vec![
                "<|im_start|>".into(),
                "<|im_end|>".into(),
                "u".into(),
                "s".into(),
                "e".into(),
                "r".into(),
                "\n".into(),
                "<think>".into(),
            ],
            merges: Vec::new(),
            bos_token: None,
            eos_token: None,
            add_bos_token: false,
            special_tokens: vec![
                ("<|im_start|>".into(), 0),
                ("<|im_end|>".into(), 1),
                ("<think>".into(), 7),
            ],
        };
        assert_eq!(tokenizer.encode("<|im_start|>user", false)[0], 0);
        assert_eq!(tokenizer.encode("<|im_end|>", false), vec![1]);
        assert_eq!(tokenizer.encode("<think>\n", false)[0], 7);
        assert!(tokenizer.is_stop_token(1));
    }

    #[test]
    fn qwen_im_end_id_is_always_a_stop_token() {
        let tokenizer = BpeTokenizer {
            vocab: Vec::new(),
            merges: Vec::new(),
            bos_token: None,
            eos_token: None,
            add_bos_token: false,
            special_tokens: Vec::new(),
        };
        assert!(tokenizer.is_stop_token(QWEN_IM_END_TOKEN_ID));
    }
}
