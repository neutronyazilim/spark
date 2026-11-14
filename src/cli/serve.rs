use std::{
    collections::HashMap,
    io::{self, Read, Write},
    net::{TcpListener, TcpStream},
    sync::atomic::{AtomicU64, Ordering},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use serde::Deserialize;
use serde_json::{json, Map, Value};

use super::runtime::InferenceSession;
use crate::{loader::gguf::GgufError, sampler::Sampler};

const MAX_HEADER_BYTES: usize = 64 * 1024;
const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;
const TOOL_INSTRUCTIONS: &str = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
const TOOL_SUFFIX: &str = "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nvalue_2\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question normally and do not mention function calls\n</IMPORTANT>";

static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub api_key: Option<String>,
    pub model_id: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    repetition_penalty: Option<f32>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    max_completion_tokens: Option<usize>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stream_options: Option<StreamOptions>,
    #[serde(default)]
    stop: Option<Value>,
    #[serde(default)]
    tools: Vec<Value>,
    #[serde(default)]
    enable_thinking: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatMessage {
    role: String,
    #[serde(default)]
    content: Value,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<Value>,
}

struct HttpRequest {
    method: String,
    path: String,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

#[derive(Debug)]
struct ApiError {
    status: u16,
    message: String,
    error_type: &'static str,
    code: &'static str,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: 400,
            message: message.into(),
            error_type: "invalid_request_error",
            code: "invalid_request",
        }
    }

    fn unauthorized() -> Self {
        Self {
            status: 401,
            message: "invalid API key".into(),
            error_type: "authentication_error",
            code: "invalid_api_key",
        }
    }

    fn not_found() -> Self {
        Self {
            status: 404,
            message: "route not found".into(),
            error_type: "invalid_request_error",
            code: "not_found",
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: 500,
            message: message.into(),
            error_type: "server_error",
            code: "inference_error",
        }
    }

    fn body(&self) -> Value {
        json!({
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": null,
                "code": self.code,
            }
        })
    }
}

struct Generation {
    text: String,
    prompt_tokens: usize,
    completion_tokens: usize,
    finish_reason: &'static str,
    elapsed_seconds: f64,
}

#[derive(Debug, PartialEq, Eq)]
struct ParsedToolCall {
    id: String,
    name: String,
    arguments: String,
}

pub fn run(session: &mut InferenceSession<'_>, config: ServerConfig) -> Result<(), GgufError> {
    let address = format!("{}:{}", config.host, config.port);
    let listener = TcpListener::bind(&address).map_err(|source| GgufError::Io {
        context: format!("cannot bind HTTP server to {address}"),
        source,
    })?;
    println!("\x1b[32m[spark] OpenAI-compatible API: http://{address}/v1\x1b[0m");
    println!(
        "\x1b[90m[spark] model: {} | context: {} tokens | requests are serialized\x1b[0m",
        config.model_id,
        session.context_len()
    );
    if config.api_key.is_none() {
        eprintln!("\x1b[33m[spark] warning: API authentication is disabled\x1b[0m");
    }

    for incoming in listener.incoming() {
        let mut stream = match incoming {
            Ok(stream) => stream,
            Err(error) => {
                eprintln!("[spark] accept error: {error}");
                continue;
            }
        };
        let _ = stream.set_nodelay(true);
        if let Err(error) = handle_connection(&mut stream, session, &config) {
            eprintln!("[spark] HTTP connection error: {error}");
        }
    }
    Ok(())
}

fn handle_connection(
    stream: &mut TcpStream,
    session: &mut InferenceSession<'_>,
    config: &ServerConfig,
) -> io::Result<()> {
    let request = match read_request(stream) {
        Ok(request) => request,
        Err(error) => {
            let api_error = ApiError::bad_request(error.to_string());
            return write_json(stream, api_error.status, &api_error.body());
        }
    };
    let path = request.path.split('?').next().unwrap_or(&request.path);

    if request.method == "OPTIONS" {
        return write_empty(stream, 204);
    }
    if path == "/health" || path == "/v1/health" {
        return write_json(stream, 200, &json!({ "status": "ok" }));
    }
    if let Err(error) = authorize(&request, config.api_key.as_deref()) {
        return write_json(stream, error.status, &error.body());
    }
    if request.method == "GET" && path == "/v1/models" {
        let created = unix_seconds();
        return write_json(
            stream,
            200,
            &json!({
                "object": "list",
                "data": [{
                    "id": config.model_id,
                    "object": "model",
                    "created": created,
                    "owned_by": "spark",
                }]
            }),
        );
    }
    if request.method == "GET" && path == format!("/v1/models/{}", config.model_id) {
        return write_json(
            stream,
            200,
            &json!({
                "id": config.model_id,
                "object": "model",
                "created": unix_seconds(),
                "owned_by": "spark",
            }),
        );
    }
    if request.method == "POST" && path == "/v1/chat/completions" {
        let parsed = serde_json::from_slice::<ChatCompletionRequest>(&request.body)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error));
        let parsed = match parsed {
            Ok(parsed) => parsed,
            Err(error) => {
                let api_error = ApiError::bad_request(format!("invalid JSON request: {error}"));
                return write_json(stream, api_error.status, &api_error.body());
            }
        };
        if let Err(error) = handle_chat_completion(stream, session, config, parsed) {
            return write_json(stream, error.status, &error.body());
        }
        return Ok(());
    }

    let error = ApiError::not_found();
    write_json(stream, error.status, &error.body())
}

fn authorize(request: &HttpRequest, expected: Option<&str>) -> Result<(), ApiError> {
    let Some(expected) = expected else {
        return Ok(());
    };
    let bearer = request
        .headers
        .get("authorization")
        .and_then(|value| value.strip_prefix("Bearer "));
    let provided = bearer.or_else(|| request.headers.get("x-api-key").map(String::as_str));
    if provided.is_some_and(|provided| constant_time_eq(provided.as_bytes(), expected.as_bytes())) {
        Ok(())
    } else {
        Err(ApiError::unauthorized())
    }
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right)
        .fold(0u8, |difference, (&a, &b)| difference | (a ^ b))
        == 0
}

fn handle_chat_completion(
    stream: &mut TcpStream,
    session: &mut InferenceSession<'_>,
    config: &ServerConfig,
    request: ChatCompletionRequest,
) -> Result<(), ApiError> {
    if request.messages.is_empty() {
        return Err(ApiError::bad_request("messages must not be empty"));
    }
    if let Some(model) = &request.model {
        if model != &config.model_id {
            return Err(ApiError::bad_request(format!(
                "unknown model {model}; expected {}",
                config.model_id
            )));
        }
    }

    let request_id = format!(
        "chatcmpl-spark-{}-{}",
        unix_seconds(),
        REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed)
    );
    let created = unix_seconds();
    let include_usage = request
        .stream_options
        .as_ref()
        .is_some_and(|options| options.include_usage);
    let has_tools = !request.tools.is_empty();

    if request.stream && !has_tools && !request.enable_thinking.unwrap_or(false) {
        write_sse_headers(stream).map_err(|error| ApiError::internal(error.to_string()))?;
        send_stream_role(stream, &request_id, created, &config.model_id)?;
        let generation = generate(session, config, &request, |text| {
            send_stream_content(stream, &request_id, created, &config.model_id, text)
        });
        let generation = match generation {
            Ok(generation) => generation,
            Err(error) => {
                let _ = send_sse(stream, &error.body());
                let _ = send_done(stream);
                return Ok(());
            }
        };
        send_stream_finish(
            stream,
            &request_id,
            created,
            &config.model_id,
            generation.finish_reason,
        )?;
        if include_usage {
            send_stream_usage(stream, &request_id, created, &config.model_id, &generation)?;
        }
        send_done(stream)?;
        log_generation(&request_id, &generation);
        return Ok(());
    }

    let generation = generate(session, config, &request, |_| Ok(()))?;
    let (reasoning, answer) = split_reasoning(&generation.text);
    let (visible_content, tool_calls) = parse_tool_calls(answer);
    let finish_reason = if tool_calls.is_empty() {
        generation.finish_reason
    } else {
        "tool_calls"
    };

    if request.stream {
        write_sse_headers(stream).map_err(|error| ApiError::internal(error.to_string()))?;
        send_stream_role(stream, &request_id, created, &config.model_id)?;
        if let Some(reasoning) = reasoning.filter(|value| !value.is_empty()) {
            send_stream_reasoning(stream, &request_id, created, &config.model_id, reasoning)?;
        }
        if !visible_content.is_empty() {
            send_stream_content(
                stream,
                &request_id,
                created,
                &config.model_id,
                visible_content,
            )?;
        }
        if !tool_calls.is_empty() {
            send_stream_tool_calls(stream, &request_id, created, &config.model_id, &tool_calls)?;
        }
        send_stream_finish(
            stream,
            &request_id,
            created,
            &config.model_id,
            finish_reason,
        )?;
        if include_usage {
            send_stream_usage(stream, &request_id, created, &config.model_id, &generation)?;
        }
        send_done(stream)?;
    } else {
        let mut message = Map::new();
        message.insert("role".into(), json!("assistant"));
        message.insert(
            "content".into(),
            if visible_content.is_empty() && !tool_calls.is_empty() {
                Value::Null
            } else {
                json!(visible_content)
            },
        );
        if let Some(reasoning) = reasoning.filter(|value| !value.is_empty()) {
            message.insert("reasoning_content".into(), json!(reasoning));
        }
        if !tool_calls.is_empty() {
            message.insert("tool_calls".into(), tool_calls_json(&tool_calls));
        }
        let response = json!({
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": config.model_id,
            "choices": [{
                "index": 0,
                "message": Value::Object(message),
                "finish_reason": finish_reason,
            }],
            "usage": usage_json(&generation),
        });
        write_json(stream, 200, &response)
            .map_err(|error| ApiError::internal(error.to_string()))?;
    }
    log_generation(&request_id, &generation);
    Ok(())
}

fn generate(
    session: &mut InferenceSession<'_>,
    config: &ServerConfig,
    request: &ChatCompletionRequest,
    mut on_text: impl FnMut(&str) -> Result<(), ApiError>,
) -> Result<Generation, ApiError> {
    let requested_max = request
        .max_completion_tokens
        .or(request.max_tokens)
        .unwrap_or(config.max_tokens)
        .max(1);
    let (tokens, compacted) = fit_prompt(session, request, requested_max)?;
    if compacted {
        eprintln!(
            "\x1b[33m[spark] compacted oversized API prompt to {} tokens (context {})\x1b[0m",
            tokens.len(),
            session.context_len()
        );
    }
    if tokens.is_empty() {
        return Err(ApiError::bad_request("messages produced an empty prompt"));
    }
    if tokens.len() >= session.context_len() {
        return Err(ApiError::bad_request(format!(
            "prompt has {} tokens but the active context limit is {}",
            tokens.len(),
            session.context_len()
        )));
    }

    session.reset();
    session
        .push_tokens(&tokens)
        .map_err(|error| ApiError::internal(error.to_string()))?;

    let available = session.context_len().saturating_sub(session.position);
    let limit = requested_max.min(available);
    let temperature = request.temperature.unwrap_or(config.temperature);
    let top_p = request.top_p.unwrap_or(config.top_p);
    let top_k = request.top_k.unwrap_or(config.top_k);
    let repetition_penalty = request
        .repetition_penalty
        .unwrap_or(config.repetition_penalty);
    let stop_sequences = parse_stop_sequences(request.stop.as_ref());
    let mut sampler = Sampler::new(temperature, top_p, top_k, repetition_penalty);
    let mut output = String::new();
    let mut pending = Vec::new();
    let mut completion_tokens = 0usize;
    let mut finish_reason = "length";
    let started = Instant::now();

    while completion_tokens < limit && session.position < session.context_len() {
        let (token, stop, _) = session
            .sample_and_forward(&mut sampler)
            .map_err(|error| ApiError::internal(error.to_string()))?;
        if stop {
            finish_reason = "stop";
            break;
        }
        completion_tokens += 1;
        let text = session.tokenizer.decode_stream(token, &mut pending);
        if !text.is_empty() {
            let previous_len = output.len();
            output.push_str(&text);
            if let Some((index, _)) = stop_sequences
                .iter()
                .filter_map(|stop| output.find(stop).map(|index| (index, stop)))
                .min_by_key(|(index, _)| *index)
            {
                if index > previous_len {
                    on_text(&output[previous_len..index])?;
                }
                output.truncate(index);
                finish_reason = "stop";
                break;
            }
            on_text(&output[previous_len..])?;
        }
    }
    if !pending.is_empty() {
        let trailing = String::from_utf8_lossy(&pending).into_owned();
        output.push_str(&trailing);
        on_text(&trailing)?;
    }

    Ok(Generation {
        text: output,
        prompt_tokens: tokens.len(),
        completion_tokens,
        finish_reason,
        elapsed_seconds: started.elapsed().as_secs_f64(),
    })
}

fn fit_prompt(
    session: &InferenceSession<'_>,
    request: &ChatCompletionRequest,
    requested_max: usize,
) -> Result<(Vec<u32>, bool), ApiError> {
    let is_qwen35 = session.transformer.config.is_qwen35;
    let enable_thinking = request.enable_thinking.unwrap_or(false);
    let add_bos = session.tokenizer.should_add_bos();
    let prompt = render_prompt(
        &request.messages,
        &request.tools,
        is_qwen35,
        enable_thinking,
    )?;
    let tokens = session.tokenizer.encode(&prompt, add_bos);
    let reserve = requested_max
        .min(256)
        .min(session.context_len().saturating_div(4).max(1));
    let budget = session.context_len().saturating_sub(reserve).max(1);
    if tokens.len() <= budget {
        return Ok((tokens, false));
    }
    if std::env::var_os("SPARK_NO_PROMPT_COMPACTION").is_some() {
        return Err(ApiError::bad_request(format!(
            "prompt has {} tokens but only {budget} are available after reserving output space",
            tokens.len()
        )));
    }

    let mut recent: Vec<ChatMessage> = request
        .messages
        .iter()
        .filter(|message| !matches!(message.role.as_str(), "system" | "developer"))
        .cloned()
        .collect();
    if recent.is_empty() {
        return Err(ApiError::bad_request("messages contain no user content"));
    }
    const COMPACT_SYSTEM: &str = "You are a concise AI coding agent. Follow the user's request. Use the provided tools when needed, inspect files before editing, and return valid tool arguments.";
    for tool_limit in [8usize, 6, 4, 2, 0] {
        let tools = compact_tools(&request.tools, tool_limit);
        loop {
            let prompt = render_prompt_with_system(
                &recent,
                &tools,
                is_qwen35,
                enable_thinking,
                Some(COMPACT_SYSTEM),
            )?;
            let tokens = session.tokenizer.encode(&prompt, add_bos);
            if tokens.len() <= budget {
                return Ok((tokens, true));
            }
            if recent.len() <= 1 {
                break;
            }
            recent.remove(0);
        }
        recent = request
            .messages
            .iter()
            .filter(|message| !matches!(message.role.as_str(), "system" | "developer"))
            .cloned()
            .collect();
    }
    Err(ApiError::bad_request(format!(
        "the latest message does not fit the {budget}-token input budget"
    )))
}

fn render_prompt(
    messages: &[ChatMessage],
    tools: &[Value],
    is_qwen35: bool,
    enable_thinking: bool,
) -> Result<String, ApiError> {
    render_prompt_with_system(messages, tools, is_qwen35, enable_thinking, None)
}

fn render_prompt_with_system(
    messages: &[ChatMessage],
    tools: &[Value],
    is_qwen35: bool,
    enable_thinking: bool,
    system_override: Option<&str>,
) -> Result<String, ApiError> {
    let system = system_override.map(str::to_owned).unwrap_or_else(|| {
        messages
            .iter()
            .filter(|message| matches!(message.role.as_str(), "system" | "developer"))
            .map(|message| content_text(&message.content))
            .filter(|content| !content.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n\n")
    });
    let mut prompt = String::new();
    if !tools.is_empty() {
        prompt.push_str("<|im_start|>system\n");
        prompt.push_str(TOOL_INSTRUCTIONS);
        for tool in tools {
            prompt.push('\n');
            prompt.push_str(
                &serde_json::to_string(tool)
                    .map_err(|error| ApiError::bad_request(error.to_string()))?,
            );
        }
        prompt.push_str(TOOL_SUFFIX);
        if !system.is_empty() {
            prompt.push_str("\n\n");
            prompt.push_str(&system);
        }
        prompt.push_str("<|im_end|>\n");
    } else if !system.is_empty() {
        prompt.push_str("<|im_start|>system\n");
        prompt.push_str(&system);
        prompt.push_str("<|im_end|>\n");
    }

    let mut index = 0usize;
    while index < messages.len() {
        let message = &messages[index];
        match message.role.as_str() {
            "system" | "developer" => {
                index += 1;
            }
            "assistant" => {
                prompt.push_str("<|im_start|>assistant\n");
                if let Some(reasoning) = message
                    .reasoning_content
                    .as_deref()
                    .filter(|value| !value.trim().is_empty())
                {
                    prompt.push_str("<think>\n");
                    prompt.push_str(reasoning.trim());
                    prompt.push_str("\n</think>\n\n");
                }
                prompt.push_str(content_text(&message.content).trim());
                for call in &message.tool_calls {
                    prompt.push_str(&render_prior_tool_call(call));
                }
                prompt.push_str("<|im_end|>\n");
                index += 1;
            }
            "tool" => {
                prompt.push_str("<|im_start|>user");
                while index < messages.len() && messages[index].role == "tool" {
                    prompt.push_str("\n<tool_response>\n");
                    prompt.push_str(content_text(&messages[index].content).trim());
                    prompt.push_str("\n</tool_response>");
                    index += 1;
                }
                prompt.push_str("<|im_end|>\n");
            }
            _ => {
                prompt.push_str("<|im_start|>user\n");
                prompt.push_str(content_text(&message.content).trim());
                prompt.push_str("<|im_end|>\n");
                index += 1;
            }
        }
    }
    prompt.push_str("<|im_start|>assistant\n");
    if is_qwen35 {
        if enable_thinking {
            prompt.push_str("<think>\n");
        } else {
            prompt.push_str("<think>\n\n</think>\n\n");
        }
    }
    Ok(prompt)
}

fn compact_tools(tools: &[Value], limit: usize) -> Vec<Value> {
    if limit == 0 || tools.is_empty() {
        return Vec::new();
    }
    const PRIORITY: &[&str] = &[
        "read",
        "grep",
        "glob",
        "patch",
        "shell",
        "question",
        "webfetch",
        "websearch",
    ];
    let mut selected = Vec::new();
    for preferred in PRIORITY {
        if selected.len() >= limit {
            break;
        }
        if let Some(tool) = tools
            .iter()
            .find(|tool| tool_name(tool) == Some(*preferred))
        {
            selected.push(compact_tool(tool));
        }
    }
    for tool in tools {
        if selected.len() >= limit {
            break;
        }
        let name = tool_name(tool);
        if selected.iter().any(|selected| tool_name(selected) == name) {
            continue;
        }
        selected.push(compact_tool(tool));
    }
    selected
}

fn tool_name(tool: &Value) -> Option<&str> {
    tool.get("function")
        .and_then(|function| function.get("name"))
        .and_then(Value::as_str)
        .or_else(|| tool.get("name").and_then(Value::as_str))
}

fn compact_tool(tool: &Value) -> Value {
    let function = tool.get("function").unwrap_or(tool);
    let mut compact = Map::new();
    if let Some(name) = function.get("name") {
        compact.insert("name".into(), name.clone());
    }
    if let Some(description) = function.get("description").and_then(Value::as_str) {
        let short: String = description.chars().take(120).collect();
        compact.insert("description".into(), Value::String(short));
    }
    if let Some(parameters) = function.get("parameters") {
        compact.insert("parameters".into(), compact_schema(parameters));
    }
    json!({ "type": "function", "function": Value::Object(compact) })
}

fn compact_schema(value: &Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.iter().map(compact_schema).collect()),
        Value::Object(object) => {
            let mut compact = Map::new();
            for (key, value) in object {
                if matches!(
                    key.as_str(),
                    "description" | "title" | "examples" | "default" | "$schema"
                ) {
                    continue;
                }
                compact.insert(key.clone(), compact_schema(value));
            }
            Value::Object(compact)
        }
        value => value.clone(),
    }
}

fn content_text(content: &Value) -> String {
    match content {
        Value::Null => String::new(),
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| {
                part.get("text")
                    .and_then(Value::as_str)
                    .or_else(|| part.as_str())
            })
            .collect::<Vec<_>>()
            .join(""),
        Value::Object(object) => object
            .get("text")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| Value::Object(object.clone()).to_string()),
        value => value.to_string(),
    }
}

fn render_prior_tool_call(call: &Value) -> String {
    let function = call.get("function").unwrap_or(call);
    let Some(name) = function.get("name").and_then(Value::as_str) else {
        return String::new();
    };
    let arguments = function
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let arguments = match arguments {
        Value::String(text) => serde_json::from_str(&text).unwrap_or(Value::String(text)),
        value => value,
    };
    let mut output = format!("\n<tool_call>\n<function={name}>\n");
    if let Value::Object(arguments) = arguments {
        for (name, value) in arguments {
            output.push_str(&format!("<parameter={name}>\n"));
            if let Some(value) = value.as_str() {
                output.push_str(value);
            } else {
                output.push_str(&value.to_string());
            }
            output.push_str("\n</parameter>\n");
        }
    }
    output.push_str("</function>\n</tool_call>");
    output
}

fn split_reasoning(output: &str) -> (Option<&str>, &str) {
    if let Some(index) = output.find("</think>") {
        let reasoning = output[..index].trim_start_matches("<think>").trim();
        let answer = output[index + "</think>".len()..].trim_start();
        (Some(reasoning), answer)
    } else {
        (None, output)
    }
}

fn parse_tool_calls(output: &str) -> (&str, Vec<ParsedToolCall>) {
    let Some(first_call) = output.find("<tool_call>") else {
        return (output, Vec::new());
    };
    let visible = output[..first_call].trim_end();
    let mut calls = Vec::new();
    let mut cursor = first_call;
    while let Some(relative_start) = output[cursor..].find("<tool_call>") {
        let start = cursor + relative_start + "<tool_call>".len();
        let Some(relative_end) = output[start..].find("</tool_call>") else {
            break;
        };
        let end = start + relative_end;
        let body = &output[start..end];
        if let Some(function_start) = body.find("<function=") {
            let name_start = function_start + "<function=".len();
            if let Some(name_end_relative) = body[name_start..].find('>') {
                let name_end = name_start + name_end_relative;
                let name = body[name_start..name_end].trim();
                if !name.is_empty() {
                    let function_body = &body[name_end + 1..];
                    let mut arguments = Map::new();
                    let mut parameter_cursor = 0usize;
                    while let Some(parameter_relative) =
                        function_body[parameter_cursor..].find("<parameter=")
                    {
                        let parameter_start =
                            parameter_cursor + parameter_relative + "<parameter=".len();
                        let Some(parameter_name_end_relative) =
                            function_body[parameter_start..].find('>')
                        else {
                            break;
                        };
                        let parameter_name_end = parameter_start + parameter_name_end_relative;
                        let parameter_name =
                            function_body[parameter_start..parameter_name_end].trim();
                        let value_start = parameter_name_end + 1;
                        let Some(value_end_relative) =
                            function_body[value_start..].find("</parameter>")
                        else {
                            break;
                        };
                        let value_end = value_start + value_end_relative;
                        let raw_value = function_body[value_start..value_end].trim();
                        let value = serde_json::from_str(raw_value)
                            .unwrap_or_else(|_| Value::String(raw_value.to_owned()));
                        if !parameter_name.is_empty() {
                            arguments.insert(parameter_name.to_owned(), value);
                        }
                        parameter_cursor = value_end + "</parameter>".len();
                    }
                    calls.push(ParsedToolCall {
                        id: format!(
                            "call_spark_{}",
                            REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed)
                        ),
                        name: name.to_owned(),
                        arguments: Value::Object(arguments).to_string(),
                    });
                }
            }
        }
        cursor = end + "</tool_call>".len();
    }
    (visible, calls)
}

fn parse_stop_sequences(stop: Option<&Value>) -> Vec<String> {
    match stop {
        Some(Value::String(value)) => vec![value.clone()],
        Some(Value::Array(values)) => values
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect(),
        _ => Vec::new(),
    }
}

fn tool_calls_json(calls: &[ParsedToolCall]) -> Value {
    Value::Array(
        calls
            .iter()
            .map(|call| {
                json!({
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments,
                    }
                })
            })
            .collect(),
    )
}

fn usage_json(generation: &Generation) -> Value {
    json!({
        "prompt_tokens": generation.prompt_tokens,
        "completion_tokens": generation.completion_tokens,
        "total_tokens": generation.prompt_tokens + generation.completion_tokens,
    })
}

fn send_stream_role(
    stream: &mut TcpStream,
    id: &str,
    created: u64,
    model: &str,
) -> Result<(), ApiError> {
    send_sse(
        stream,
        &json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": { "role": "assistant", "content": "" },
                "finish_reason": null,
            }]
        }),
    )
}

fn send_stream_content(
    stream: &mut TcpStream,
    id: &str,
    created: u64,
    model: &str,
    content: &str,
) -> Result<(), ApiError> {
    if content.is_empty() {
        return Ok(());
    }
    send_sse(
        stream,
        &json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": { "content": content },
                "finish_reason": null,
            }]
        }),
    )
}

fn send_stream_reasoning(
    stream: &mut TcpStream,
    id: &str,
    created: u64,
    model: &str,
    reasoning: &str,
) -> Result<(), ApiError> {
    send_sse(
        stream,
        &json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": { "reasoning_content": reasoning },
                "finish_reason": null,
            }]
        }),
    )
}

fn send_stream_tool_calls(
    stream: &mut TcpStream,
    id: &str,
    created: u64,
    model: &str,
    calls: &[ParsedToolCall],
) -> Result<(), ApiError> {
    let calls: Vec<Value> = calls
        .iter()
        .enumerate()
        .map(|(index, call)| {
            json!({
                "index": index,
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.arguments,
                }
            })
        })
        .collect();
    send_sse(
        stream,
        &json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": { "tool_calls": calls },
                "finish_reason": null,
            }]
        }),
    )
}

fn send_stream_finish(
    stream: &mut TcpStream,
    id: &str,
    created: u64,
    model: &str,
    finish_reason: &str,
) -> Result<(), ApiError> {
    send_sse(
        stream,
        &json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
            }]
        }),
    )
}

fn send_stream_usage(
    stream: &mut TcpStream,
    id: &str,
    created: u64,
    model: &str,
    generation: &Generation,
) -> Result<(), ApiError> {
    send_sse(
        stream,
        &json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [],
            "usage": usage_json(generation),
        }),
    )
}

fn send_sse(stream: &mut TcpStream, value: &Value) -> Result<(), ApiError> {
    let payload = format!("data: {value}\n\n");
    write_http_chunk(stream, payload.as_bytes())
        .map_err(|error| ApiError::internal(error.to_string()))
}

fn send_done(stream: &mut TcpStream) -> Result<(), ApiError> {
    write_http_chunk(stream, b"data: [DONE]\n\n")
        .and_then(|_| stream.write_all(b"0\r\n\r\n"))
        .and_then(|_| stream.flush())
        .map_err(|error| ApiError::internal(error.to_string()))
}

fn write_http_chunk(stream: &mut TcpStream, payload: &[u8]) -> io::Result<()> {
    write!(stream, "{:X}\r\n", payload.len())?;
    stream.write_all(payload)?;
    stream.write_all(b"\r\n")?;
    stream.flush()
}

fn log_generation(id: &str, generation: &Generation) {
    let speed = generation.completion_tokens as f64 / generation.elapsed_seconds.max(1e-9);
    eprintln!(
        "[spark] {id}: prompt={} completion={} {:.2} tok/s finish={}",
        generation.prompt_tokens, generation.completion_tokens, speed, generation.finish_reason
    );
}

fn read_request(stream: &mut TcpStream) -> io::Result<HttpRequest> {
    let mut bytes = Vec::with_capacity(4096);
    let header_end = loop {
        if bytes.len() >= MAX_HEADER_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "HTTP headers are too large",
            ));
        }
        let mut buffer = [0u8; 4096];
        let read = stream.read(&mut buffer)?;
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "connection closed before HTTP headers",
            ));
        }
        bytes.extend_from_slice(&buffer[..read]);
        if let Some(index) = find_bytes(&bytes, b"\r\n\r\n") {
            break index + 4;
        }
    };
    let header_text = std::str::from_utf8(&bytes[..header_end - 4])
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "HTTP headers are not UTF-8"))?;
    let mut lines = header_text.split("\r\n");
    let request_line = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing request line"))?;
    let mut request_parts = request_line.split_whitespace();
    let method = request_parts
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing HTTP method"))?
        .to_owned();
    let path = request_parts
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing HTTP path"))?
        .to_owned();
    let mut headers = HashMap::new();
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "malformed HTTP header",
            ));
        };
        headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_owned());
    }
    if headers
        .get("transfer-encoding")
        .is_some_and(|value| value.eq_ignore_ascii_case("chunked"))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "chunked request bodies are not supported",
        ));
    }
    let content_length = headers
        .get("content-length")
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid Content-Length"))
        })
        .transpose()?
        .unwrap_or(0);
    if content_length > MAX_BODY_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "HTTP request body is too large",
        ));
    }
    while bytes.len() < header_end + content_length {
        let remaining = header_end + content_length - bytes.len();
        let mut buffer = vec![0u8; remaining.min(8192)];
        let read = stream.read(&mut buffer)?;
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "connection closed before request body",
            ));
        }
        bytes.extend_from_slice(&buffer[..read]);
    }
    Ok(HttpRequest {
        method,
        path,
        headers,
        body: bytes[header_end..header_end + content_length].to_vec(),
    })
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn write_sse_headers(stream: &mut TcpStream) -> io::Result<()> {
    stream.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nTransfer-Encoding: chunked\r\nCache-Control: no-cache\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\nX-Accel-Buffering: no\r\n\r\n",
    )?;
    stream.flush()
}

fn write_json(stream: &mut TcpStream, status: u16, value: &Value) -> io::Result<()> {
    let body = serde_json::to_vec(value)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let reason = reason_phrase(status);
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: Authorization, Content-Type, X-API-Key\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n\r\n",
        body.len()
    )?;
    stream.write_all(&body)?;
    stream.flush()
}

fn write_empty(stream: &mut TcpStream, status: u16) -> io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status} {}\r\nContent-Length: 0\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: Authorization, Content-Type, X-API-Key\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n\r\n",
        reason_phrase(status)
    )?;
    stream.flush()
}

fn reason_phrase(status: u16) -> &'static str {
    match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "Error",
    }
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content: json!(content),
            reasoning_content: None,
            tool_calls: Vec::new(),
        }
    }

    #[test]
    fn renders_chatml_and_tools() {
        let messages = [
            message("system", "Be concise."),
            message("user", "List files."),
        ];
        let tools = [json!({
            "type": "function",
            "function": { "name": "list", "parameters": { "type": "object" } }
        })];
        let prompt = render_prompt(&messages, &tools, true, true).expect("prompt");
        assert!(prompt.contains("# Tools"));
        assert!(prompt.contains("\"name\":\"list\""));
        assert!(prompt.contains("Be concise."));
        assert!(prompt.contains("<|im_start|>user\nList files.<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n"));
    }

    #[test]
    fn parses_qwen_tool_calls() {
        let output = "I should inspect it.\n<tool_call>\n<function=read>\n<parameter=path>\nsrc/main.rs\n</parameter>\n<parameter=limit>\n100\n</parameter>\n</function>\n</tool_call>";
        let (visible, calls) = parse_tool_calls(output);
        assert_eq!(visible, "I should inspect it.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read");
        assert_eq!(calls[0].arguments, r#"{"limit":100,"path":"src/main.rs"}"#);
    }

    #[test]
    fn separates_reasoning_from_answer() {
        let (reasoning, answer) = split_reasoning("work it out\n</think>\n\nFour");
        assert_eq!(reasoning, Some("work it out"));
        assert_eq!(answer, "Four");
    }

    #[test]
    fn disabled_thinking_closes_block_before_generation() {
        let prompt = render_prompt(&[message("user", "Hi")], &[], true, false).expect("prompt");
        assert!(prompt.ends_with("<think>\n\n</think>\n\n"));
    }

    #[test]
    fn compacts_and_prioritizes_tool_schemas() {
        let tools = [
            json!({"type":"function","function":{"name":"other","description":"x","parameters":{"type":"object"}}}),
            json!({"type":"function","function":{"name":"read","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path"}}}}}),
        ];
        let compact = compact_tools(&tools, 1);
        assert_eq!(tool_name(&compact[0]), Some("read"));
        assert!(compact[0]["function"]["parameters"]["properties"]["path"]
            .get("description")
            .is_none());
    }

    #[test]
    fn constant_time_key_comparison_checks_length_and_content() {
        assert!(constant_time_eq(b"secret", b"secret"));
        assert!(!constant_time_eq(b"secret", b"other!"));
        assert!(!constant_time_eq(b"secret", b"secret-long"));
    }
}
