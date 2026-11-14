use std::{fmt, io};

const GGUF_MAGIC: u32 = 0x4655_4747;
const GGUF_VERSION_2: u32 = 2;
const GGUF_VERSION_3: u32 = 3;
const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug)]
pub enum GgufError {
    Io { context: String, source: io::Error },
    InvalidArgument(String),
    InvalidFormat(String),
    UnexpectedEof { at: usize, needed: usize },
    InvalidUtf8 { at: usize },
    UnsupportedVersion(u32),
    CountTooLarge(&'static str, u64),
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { context, source } => write!(f, "{context}: {source}"),
            Self::InvalidArgument(message) | Self::InvalidFormat(message) => f.write_str(message),
            Self::UnexpectedEof { at, needed } => {
                write!(
                    f,
                    "unexpected end of file at {at}, need {needed} more bytes"
                )
            }
            Self::InvalidUtf8 { at } => write!(f, "invalid UTF-8 string at offset {at}"),
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported GGUF version {version}; expected 2 or 3")
            }
            Self::CountTooLarge(name, value) => write!(f, "{name} is too large: {value}"),
        }
    }
}

impl std::error::Error for GgufError {}

#[derive(Debug, Clone, Copy)]
pub enum MetadataValue<'a> {
    Raw { type_code: u32, bytes: &'a [u8] },
}

impl<'a> MetadataValue<'a> {
    pub fn type_code(self) -> u32 {
        match self {
            Self::Raw { type_code, .. } => type_code,
        }
    }

    pub fn type_name(self) -> &'static str {
        match self.type_code() {
            0 => "uint8",
            1 => "int8",
            2 => "uint16",
            3 => "int16",
            4 => "uint32",
            5 => "int32",
            6 => "float32",
            7 => "bool",
            8 => "string",
            9 => "array",
            10 => "uint64",
            11 => "int64",
            12 => "float64",
            _ => "unknown",
        }
    }

    pub fn raw_bytes(&self) -> &[u8] {
        match self {
            Self::Raw { bytes, .. } => bytes,
        }
    }

    pub fn as_u64(self) -> Option<u64> {
        match self.type_code() {
            4 => Some(u32::from_le_bytes(self.raw_bytes().try_into().ok()?) as u64),
            10 => Some(u64::from_le_bytes(self.raw_bytes().try_into().ok()?)),
            _ => None,
        }
    }

    pub fn as_f32(self) -> Option<f32> {
        if self.type_code() == 6 {
            Some(f32::from_le_bytes(self.raw_bytes().try_into().ok()?))
        } else {
            None
        }
    }

    pub fn as_bool(self) -> Option<bool> {
        (self.type_code() == 7).then(|| self.raw_bytes().first().copied() == Some(1))
    }

    pub fn as_str(&self) -> Option<&'a str> {
        let (type_code, raw) = match self {
            Self::Raw { type_code, bytes } => (*type_code, *bytes),
        };
        if type_code != 8 || raw.len() < 8 {
            return None;
        }
        let length = usize::try_from(u64::from_le_bytes(raw[..8].try_into().ok()?)).ok()?;
        std::str::from_utf8(raw.get(8..8 + length)?).ok()
    }

    pub fn array_strings(&self) -> Option<Vec<&'a str>> {
        let (type_code, raw) = match self {
            Self::Raw { type_code, bytes } => (*type_code, *bytes),
        };
        if type_code != 9 || raw.len() < 12 {
            return None;
        }
        let element_type = u32::from_le_bytes(raw[..4].try_into().ok()?);
        if element_type != 8 {
            return None;
        }
        let count = usize::try_from(u64::from_le_bytes(raw[4..12].try_into().ok()?)).ok()?;
        let mut cursor: usize = 12;
        let mut result = Vec::with_capacity(count);
        for _ in 0..count {
            let end = cursor.checked_add(8)?;
            let length =
                usize::try_from(u64::from_le_bytes(raw.get(cursor..end)?.try_into().ok()?)).ok()?;
            cursor = end;
            let value_end = cursor.checked_add(length)?;
            result.push(std::str::from_utf8(raw.get(cursor..value_end)?).ok()?);
            cursor = value_end;
        }
        Some(result)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MetadataEntry<'a> {
    pub key: &'a str,
    pub value: MetadataValue<'a>,
}

#[derive(Debug)]
pub struct TensorInfo<'a> {
    pub name: &'a str,
    pub dimensions: Vec<u64>,
    pub ggml_type: u32,
    pub offset: u64,
}

#[derive(Debug)]
pub struct GgufModel<'a> {
    bytes: &'a [u8],
    version: u32,
    metadata: Vec<MetadataEntry<'a>>,
    tensors: Vec<TensorInfo<'a>>,
    tensor_data_offset: usize,
}

impl<'a> GgufModel<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self, GgufError> {
        let mut reader = Reader::new(bytes);
        let magic = reader.u32()?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidFormat(format!(
                "invalid GGUF magic: 0x{magic:08x}"
            )));
        }

        let version = reader.u32()?;
        if version != GGUF_VERSION_2 && version != GGUF_VERSION_3 {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let tensor_count = checked_count(reader.u64()?, "tensor count")?;
        let metadata_count = checked_count(reader.u64()?, "metadata count")?;

        let mut metadata = Vec::with_capacity(metadata_count);
        let mut alignment = DEFAULT_ALIGNMENT;
        for _ in 0..metadata_count {
            let key = reader.string()?;
            let type_code = reader.u32()?;
            let value_start = reader.position();
            reader.skip_value(type_code)?;
            let value = MetadataValue::Raw {
                type_code,
                bytes: &bytes[value_start..reader.position()],
            };
            if key == "general.alignment" && type_code == 4 {
                let raw = value.raw_bytes();
                if raw.len() == 4 {
                    alignment = u32::from_le_bytes(raw.try_into().unwrap()) as u64;
                }
            }
            metadata.push(MetadataEntry { key, value });
        }

        let mut tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = reader.string()?;
            let dimensions_count = checked_count(reader.u32()? as u64, "dimension count")?;
            let mut dimensions = Vec::with_capacity(dimensions_count);
            for _ in 0..dimensions_count {
                dimensions.push(reader.u64()?);
            }
            let ggml_type = reader.u32()?;
            let offset = reader.u64()?;
            tensors.push(TensorInfo {
                name,
                dimensions,
                ggml_type,
                offset,
            });
        }

        let data_start = align_up(reader.position() as u64, alignment)
            .ok_or_else(|| GgufError::InvalidFormat("tensor alignment overflow".into()))?;
        if data_start > bytes.len() as u64 {
            return Err(GgufError::UnexpectedEof {
                at: bytes.len(),
                needed: data_start as usize - bytes.len(),
            });
        }

        Ok(Self {
            bytes,
            version,
            metadata,
            tensors,
            tensor_data_offset: data_start as usize,
        })
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn metadata(&self) -> &[MetadataEntry<'a>] {
        &self.metadata
    }

    pub fn tensors(&self) -> &[TensorInfo<'a>] {
        &self.tensors
    }

    pub fn tensor_data_offset(&self) -> usize {
        self.tensor_data_offset
    }

    /// Returns a borrowed tensor byte range from the original mmap.
    pub fn tensor_bytes(
        &self,
        tensor: &TensorInfo<'a>,
        byte_len: usize,
    ) -> Result<&'a [u8], GgufError> {
        let start = self
            .tensor_data_offset
            .checked_add(
                usize::try_from(tensor.offset)
                    .map_err(|_| GgufError::CountTooLarge("tensor offset", tensor.offset))?,
            )
            .ok_or_else(|| GgufError::InvalidFormat("tensor offset overflow".into()))?;
        let end = start
            .checked_add(byte_len)
            .ok_or_else(|| GgufError::InvalidFormat("tensor byte range overflow".into()))?;
        self.bytes.get(start..end).ok_or(GgufError::UnexpectedEof {
            at: start,
            needed: end.saturating_sub(self.bytes.len()),
        })
    }

    pub fn metadata_value(&self, key: &str) -> Option<MetadataValue<'a>> {
        self.metadata
            .iter()
            .find(|entry| entry.key == key)
            .map(|entry| entry.value)
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo<'a>> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn position(&self) -> usize {
        self.position
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], GgufError> {
        let end = self
            .position
            .checked_add(len)
            .ok_or_else(|| GgufError::InvalidFormat("file offset overflow".into()))?;
        if end > self.bytes.len() {
            return Err(GgufError::UnexpectedEof {
                at: self.position,
                needed: end - self.bytes.len().min(self.position),
            });
        }
        let result = &self.bytes[self.position..end];
        self.position = end;
        Ok(result)
    }

    fn u32(&mut self) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<&'a str, GgufError> {
        let start = self.position;
        let len = checked_count(self.u64()?, "string length")?;
        let bytes = self.take(len)?;
        std::str::from_utf8(bytes).map_err(|_| GgufError::InvalidUtf8 { at: start })
    }

    fn skip_value(&mut self, type_code: u32) -> Result<(), GgufError> {
        match type_code {
            0 | 1 | 7 => self.take(1).map(|_| ()),
            2 | 3 => self.take(2).map(|_| ()),
            4..=6 => self.take(4).map(|_| ()),
            8 => self.string().map(|_| ()),
            9 => {
                let element_type = self.u32()?;
                let count = checked_count(self.u64()?, "array length")?;
                for _ in 0..count {
                    self.skip_value(element_type)?;
                }
                Ok(())
            }
            10..=12 => self.take(8).map(|_| ()),
            _ => Err(GgufError::InvalidFormat(format!(
                "unknown metadata type {type_code}"
            ))),
        }
    }
}

fn checked_count(value: u64, name: &'static str) -> Result<usize, GgufError> {
    usize::try_from(value).map_err(|_| GgufError::CountTooLarge(name, value))
}

fn align_up(value: u64, alignment: u64) -> Option<u64> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return None;
    }
    value
        .checked_add(alignment - 1)
        .map(|v| v & !(alignment - 1))
}

#[cfg(test)]
mod tests {
    use super::GgufModel;

    fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_string(bytes: &mut Vec<u8>, value: &str) {
        push_u64(bytes, value.len() as u64);
        bytes.extend_from_slice(value.as_bytes());
    }

    #[test]
    fn parses_v3_header_metadata_and_tensor() {
        let mut bytes = Vec::new();
        push_u32(&mut bytes, 0x4655_4747);
        push_u32(&mut bytes, 3);
        push_u64(&mut bytes, 1);
        push_u64(&mut bytes, 1);
        push_string(&mut bytes, "general.alignment");
        push_u32(&mut bytes, 4);
        push_u32(&mut bytes, 64);
        push_string(&mut bytes, "token_embd.weight");
        push_u32(&mut bytes, 2);
        push_u64(&mut bytes, 2);
        push_u64(&mut bytes, 3);
        push_u32(&mut bytes, 0);
        push_u64(&mut bytes, 0);
        bytes.resize(128, 0);

        let model = GgufModel::parse(&bytes).expect("valid GGUF");
        assert_eq!(model.version(), 3);
        assert_eq!(model.metadata()[0].key, "general.alignment");
        assert_eq!(model.tensors()[0].name, "token_embd.weight");
        assert_eq!(model.tensors()[0].dimensions, [2, 3]);
        assert_eq!(model.tensor_data_offset(), 128);
    }
}
