//! Tokenizer and prompt formatting.
//!
//! Prompts are rendered with the chat template embedded in the GGUF (Unsloth's port of
//! DeepSeek's `encoding.py`) through minijinja; if that fails a built-in formatter that covers
//! system / user / assistant / tool turns is used instead.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use serde_json::{Value, json};

pub const BOS: &str = "<｜begin▁of▁sentence｜>";
pub const EOS: &str = "<｜end▁of▁sentence｜>";
pub const THINK_END: &str = "</think>";
pub const DSML: &str = "｜DSML｜";

pub struct Tokenizer {
    pub inner: tokenizers::Tokenizer,
    pub stop_ids: Vec<u32>,
}

pub fn find_tokenizer(model: &Path) -> Option<PathBuf> {
    let mut dir = if model.is_dir() { Some(model) } else { model.parent() };
    for _ in 0..3 {
        let d = dir?;
        let p = d.join("tokenizer.json");
        if p.exists() {
            return Some(p);
        }
        dir = d.parent();
    }
    None
}

impl Tokenizer {
    pub fn load(path: &Path) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path).map_err(|e| anyhow!("loading {}: {e}", path.display()))?;
        let id = |s: &str| inner.token_to_id(s);
        let stop_ids: Vec<u32> = [EOS, "<｜User｜>", "<|EOT|>"].iter().filter_map(|s| id(s)).collect();
        Ok(Self { inner, stop_ids })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let e = self.inner.encode(text, false).map_err(|e| anyhow!("tokenize: {e}"))?;
        Ok(e.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.inner.decode(ids, false).unwrap_or_default()
    }
}

/// Incremental detokenizer: emits only complete UTF-8 text.
pub struct Detok {
    ids: Vec<u32>,
    emitted: usize,
}

impl Detok {
    pub fn new() -> Self {
        Self { ids: Vec::new(), emitted: 0 }
    }
    pub fn push(&mut self, tok: &Tokenizer, id: u32) -> String {
        self.ids.push(id);
        let s = tok.decode(&self.ids);
        if s.ends_with('\u{fffd}') || s.len() < self.emitted {
            return String::new();
        }
        let out = s[self.emitted..].to_string();
        self.emitted = s.len();
        out
    }
}

pub struct ChatTemplate {
    env: Option<minijinja::Environment<'static>>,
}

impl ChatTemplate {
    pub fn new(src: Option<&str>) -> Self {
        let env = src.and_then(|src| {
            let mut env = minijinja::Environment::new();
            env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
            env.add_function("raise_exception", |msg: String| -> Result<String, minijinja::Error> {
                Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg))
            });
            // Python json.dumps(ensure_ascii=False) formatting, as in DeepSeek's encoding.py
            env.add_filter("tojson", |v: minijinja::Value| -> Result<minijinja::Value, minijinja::Error> {
                let j: Value = serde_json::to_value(&v).map_err(|e| minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, e.to_string()))?;
                Ok(minijinja::Value::from_safe_string(py_json(&j)))
            });
            match env.add_template_owned("chat", src.to_string()) {
                Ok(()) => Some(env),
                Err(e) => {
                    tracing::warn!("chat template does not compile ({e}); using built-in formatter");
                    None
                }
            }
        });
        Self { env }
    }

    pub fn render(&self, messages: &Value, tools: Option<&Value>, thinking: bool, effort: Option<&Value>) -> Result<String> {
        // OpenAI clients send tool-call arguments as a JSON string; the template wants an object
        let mut messages = messages.clone();
        if let Some(arr) = messages.as_array_mut() {
            for m in arr {
                if let Some(calls) = m.get_mut("tool_calls").and_then(|c| c.as_array_mut()) {
                    for c in calls {
                        if let Some(a) = c.pointer_mut("/function/arguments") {
                            if let Some(parsed) = a.as_str().and_then(|s| serde_json::from_str::<Value>(s).ok()).filter(|v| v.is_object()) {
                                *a = parsed;
                            }
                        }
                    }
                }
            }
        }
        let messages = &messages;
        if let Some(env) = &self.env {
            let t = env.get_template("chat")?;
            let ctx = json!({
                "messages": messages,
                "tools": tools,
                "add_generation_prompt": true,
                "thinking_mode": if thinking { "thinking" } else { "chat" },
                "reasoning_effort": effort,
                "bos_token": BOS,
                "eos_token": EOS,
            });
            match t.render(minijinja::Value::from(minijinja::value::Serde(&ctx))) {
                Ok(s) => return Ok(s),
                Err(e) => tracing::warn!("chat template render failed ({e:#}); using built-in formatter"),
            }
        }
        builtin_render(messages, tools, thinking)
    }
}

/// `json.dumps(v, ensure_ascii=False)`: ", " and ": " separators, insertion key order.
pub fn py_json(v: &Value) -> String {
    match v {
        Value::Array(a) => format!("[{}]", a.iter().map(py_json).collect::<Vec<_>>().join(", ")),
        Value::Object(o) => format!(
            "{{{}}}",
            o.iter().map(|(k, v)| format!("{}: {}", Value::String(k.clone()), py_json(v))).collect::<Vec<_>>().join(", ")
        ),
        other => other.to_string(),
    }
}

fn content_text(c: &Value) -> String {
    match c {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()).map(|s| s.to_string()))
            .collect::<Vec<_>>()
            .join("\n\n"),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// Minimal re-implementation of the DeepSeek-V4.1 prompt format.
pub fn builtin_render(messages: &Value, tools: Option<&Value>, thinking: bool) -> Result<String> {
    let msgs = messages.as_array().context("messages must be an array")?;
    let mut out = String::from(BOS);
    let mut sys = String::new();
    if thinking {
        sys.push_str("Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n");
    }
    let mut start = 0;
    if let Some(m) = msgs.first() {
        if m.get("role").and_then(|r| r.as_str()) == Some("system") {
            sys.push_str(&content_text(m.get("content").unwrap_or(&Value::Null)));
            start = 1;
        }
    }
    if let Some(Value::Array(ts)) = tools {
        if !ts.is_empty() {
            sys.push_str("\n\n## Tools\n\nYou have access to a set of tools. Invoke them with a \"<｜DSML｜ calls>\" block:\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"$TOOL_NAME\">\n<｜DSML｜ parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>\n\n### Available Tool Schemas\n\n");
            for t in ts {
                let f = t.get("function").unwrap_or(t);
                sys.push_str(&f.to_string());
                sys.push('\n');
            }
        }
    }
    if !sys.is_empty() {
        out.push_str("<｜System｜>");
        out.push_str(&sys);
    }
    let n = msgs.len();
    for (i, m) in msgs.iter().enumerate().skip(start) {
        let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("user");
        let content = content_text(m.get("content").unwrap_or(&Value::Null));
        match role {
            "user" => {
                out.push_str("<｜User｜>");
                out.push_str(&content);
            }
            "tool" => {
                out.push_str("<｜User｜><tool_result>");
                out.push_str(&content);
                out.push_str("</tool_result>");
            }
            "system" => {
                out.push_str("<｜System｜>");
                out.push_str(&content);
            }
            _ => {
                out.push_str(&content);
                out.push_str(EOS);
            }
        }
        let next_is_assistant = msgs.get(i + 1).and_then(|m| m.get("role")).and_then(|r| r.as_str()) == Some("assistant");
        if role != "assistant" && (i + 1 == n || next_is_assistant) {
            out.push_str("<｜Assistant｜>");
            out.push_str(if thinking && i + 1 == n { "<think>" } else { "</think>" });
        }
    }
    Ok(out)
}

/// Split generated text into (reasoning, content).
pub fn split_reasoning(text: &str, thinking: bool) -> (Option<String>, String) {
    if !thinking {
        return (None, text.to_string());
    }
    match text.find(THINK_END) {
        Some(i) => (Some(text[..i].to_string()), text[i + THINK_END.len()..].trim_start_matches('\n').to_string()),
        None => (Some(text.to_string()), String::new()),
    }
}

/// Parse DSML tool calls out of the content. Returns (remaining content, tool_calls JSON array).
pub fn parse_tool_calls(content: &str) -> (String, Vec<Value>) {
    let open = format!("<{DSML} calls>");
    let Some(start) = content.find(&open) else { return (content.to_string(), Vec::new()) };
    let body = &content[start + open.len()..];
    let close = format!("</{DSML} calls>");
    let body = body.split(&close).next().unwrap_or(body);
    let mut calls = Vec::new();
    let inv_open = format!("<{DSML} invoke name=\"");
    let inv_close = format!("</{DSML} invoke>");
    let par_open = format!("<{DSML} parameter name=\"");
    let par_close = format!("</{DSML} parameter>");
    for (n, chunk) in body.split(&inv_open).skip(1).enumerate() {
        let Some(q) = chunk.find('"') else { continue };
        let name = &chunk[..q];
        let inner = chunk[q..].split(&inv_close).next().unwrap_or("");
        let mut args = serde_json::Map::new();
        for p in inner.split(&par_open).skip(1) {
            let Some(q) = p.find('"') else { continue };
            let pname = &p[..q];
            let is_str = p[q..].starts_with("\" string=\"true\"");
            let Some(gt) = p.find('>') else { continue };
            let val = p[gt + 1..].split(&par_close).next().unwrap_or("");
            let v = if is_str { Value::String(val.to_string()) } else { serde_json::from_str(val).unwrap_or(Value::String(val.to_string())) };
            args.insert(pname.to_string(), v);
        }
        calls.push(json!({
            "id": format!("call_{}_{n}", uuid::Uuid::new_v4().simple()),
            "type": "function",
            "function": { "name": name, "arguments": Value::Object(args).to_string() }
        }));
    }
    (content[..start].trim_end().to_string(), calls)
}
