//! OpenAI-compatible HTTP API: /v1/models, /v1/chat/completions, /v1/completions (streaming
//! and non-streaming), plus /health and /metrics.

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::chat::{ChatTemplate, parse_tool_calls, split_reasoning};
use crate::engine::{EngineHandle, Event, Request, SamplingParams};

pub struct AppState {
    pub engine: EngineHandle,
    pub template: ChatTemplate,
    pub default_thinking: bool,
}

type S = State<Arc<AppState>>;

fn err(code: StatusCode, msg: impl Into<String>) -> Response {
    (code, Json(json!({"error": {"message": msg.into(), "type": "invalid_request_error"}}))).into_response()
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/completions", post(completions))
        .route("/health", get(|| async { "ok" }))
        .route("/metrics", get(metrics))
        .with_state(state)
}

async fn models(State(s): S) -> Json<Value> {
    Json(json!({"object": "list", "data": [{
        "id": s.engine.model_id, "object": "model", "created": now(), "owned_by": "dsv4",
        "context_length": s.engine.max_ctx,
    }]}))
}

async fn metrics(State(s): S) -> Json<Value> {
    Json(s.engine.stats.snapshot_json())
}

fn sampling(body: &Value) -> SamplingParams {
    let mut p = SamplingParams::default();
    let fl = |k: &str| body.get(k).and_then(|v| v.as_f64()).map(|v| v as f32);
    if let Some(v) = fl("temperature") {
        p.temperature = v;
    }
    if let Some(v) = fl("top_p") {
        p.top_p = v;
    }
    if let Some(v) = fl("min_p") {
        p.min_p = v;
    }
    if let Some(v) = fl("repetition_penalty") {
        p.repetition_penalty = v;
    }
    if let Some(v) = fl("presence_penalty") {
        p.presence_penalty = v;
    }
    if let Some(v) = fl("frequency_penalty") {
        p.frequency_penalty = v;
    }
    if let Some(v) = body.get("top_k").and_then(|v| v.as_u64()) {
        p.top_k = v as usize;
    }
    if let Some(v) = body.get("seed").and_then(|v| v.as_u64()) {
        p.seed = Some(v);
    }
    if let Some(v) = body.get("max_completion_tokens").or(body.get("max_tokens")).and_then(|v| v.as_u64()) {
        p.max_tokens = v as usize;
    }
    if let Some(v) = body.get("ignore_eos").and_then(|v| v.as_bool()) {
        p.ignore_eos = v;
    }
    match body.get("stop") {
        Some(Value::String(s)) => p.stop = vec![s.clone()],
        Some(Value::Array(a)) => p.stop = a.iter().filter_map(|v| v.as_str().map(String::from)).collect(),
        _ => {}
    }
    p
}

/// Decide thinking mode from the many spellings clients use.
fn thinking_mode(body: &Value, default: bool) -> bool {
    if let Some(k) = body.get("chat_template_kwargs") {
        for key in ["thinking", "enable_thinking"] {
            if let Some(b) = k.get(key).and_then(|v| v.as_bool()) {
                return b;
            }
        }
        if let Some(m) = k.get("thinking_mode").and_then(|v| v.as_str()) {
            return m == "thinking";
        }
    }
    match body.get("thinking") {
        Some(Value::Bool(b)) => return *b,
        Some(Value::Object(o)) => return o.get("type").and_then(|t| t.as_str()) != Some("disabled"),
        _ => {}
    }
    if let Some(e) = body.get("reasoning_effort").and_then(|v| v.as_str()) {
        return e != "none";
    }
    default
}

struct Job {
    rx: mpsc::UnboundedReceiver<Event>,
    cancel: Arc<AtomicBool>,
}

fn start(s: &AppState, prompt: &str, params: SamplingParams) -> Result<(Job, usize), Response> {
    let ids = s.engine.tok.encode(prompt).map_err(|e| err(StatusCode::BAD_REQUEST, e.to_string()))?;
    if ids.len() + 1 >= s.engine.max_ctx {
        return Err(err(StatusCode::BAD_REQUEST, format!("prompt is {} tokens; context window is {}", ids.len(), s.engine.max_ctx)));
    }
    let n = ids.len();
    let (tx, rx) = mpsc::unbounded_channel();
    let cancel = Arc::new(AtomicBool::new(false));
    s.engine.submit(Request { prompt: ids, params, tx, cancel: cancel.clone() });
    Ok((Job { rx, cancel }, n))
}

struct CancelOnDrop(Arc<AtomicBool>);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

async fn chat(State(s): S, Json(body): Json<Value>) -> Response {
    let Some(messages) = body.get("messages") else { return err(StatusCode::BAD_REQUEST, "missing messages") };
    let thinking = thinking_mode(&body, s.default_thinking);
    let tools = body.get("tools").filter(|t| t.as_array().is_some_and(|a| !a.is_empty()));
    let effort = body.get("reasoning_effort").filter(|e| e.as_str() != Some("none"));
    let prompt = match s.template.render(messages, tools, thinking, effort) {
        Ok(p) => p,
        Err(e) => return err(StatusCode::BAD_REQUEST, format!("prompt formatting failed: {e:#}")),
    };
    let params = sampling(&body);
    let stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let include_usage = body.pointer("/stream_options/include_usage").and_then(|v| v.as_bool()).unwrap_or(false);
    let (mut job, _) = match start(&s, &prompt, params) {
        Ok(j) => j,
        Err(r) => return r,
    };
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    let model = s.engine.model_id.clone();
    let created = now();

    if !stream {
        let _guard = CancelOnDrop(job.cancel.clone());
        let mut text = String::new();
        let mut usage = json!({});
        let mut finish = "stop";
        while let Some(ev) = job.rx.recv().await {
            match ev {
                Event::Text(t) => text.push_str(&t),
                Event::Done { finish: f, prompt_tokens, completion_tokens, .. } => {
                    finish = f;
                    usage = json!({"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens});
                }
                Event::Error(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e),
                Event::Started { .. } => {}
            }
        }
        let (reasoning, content) = split_reasoning(&text, thinking);
        let (content, calls) = parse_tool_calls(&content);
        let mut msg = json!({"role": "assistant", "content": content});
        if let Some(r) = reasoning {
            msg["reasoning_content"] = json!(r);
        }
        if !calls.is_empty() {
            msg["tool_calls"] = json!(calls);
            finish = "tool_calls";
        }
        return Json(json!({
            "id": id, "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": usage,
        }))
        .into_response();
    }

    let sse = async_stream::stream! {
        let _guard = CancelOnDrop(job.cancel.clone());
        let chunk = |delta: Value, finish: Option<&str>| {
            json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
        };
        yield Ok::<_, Infallible>(SseEvent::default().data(chunk(json!({"role": "assistant", "content": ""}), None).to_string()));
        let mut in_reasoning = thinking;
        let mut buf = String::new();     // text not yet classified (could be a split "</think>")
        let mut content_all = String::new();
        let mut tool_mode = false;
        while let Some(ev) = job.rx.recv().await {
            match ev {
                Event::Text(t) => {
                    buf.push_str(&t);
                    if in_reasoning {
                        if let Some(i) = buf.find("</think>") {
                            let r = buf[..i].to_string();
                            if !r.is_empty() {
                                yield Ok(SseEvent::default().data(chunk(json!({"reasoning_content": r}), None).to_string()));
                            }
                            buf = buf[i + 8..].trim_start_matches('\n').to_string();
                            in_reasoning = false;
                        } else {
                            let keep = (1..8).rev().find(|&k| buf.ends_with(&"</think>"[..k])).unwrap_or(0);
                            let r: String = buf[..buf.len() - keep].to_string();
                            if !r.is_empty() {
                                yield Ok(SseEvent::default().data(chunk(json!({"reasoning_content": r}), None).to_string()));
                            }
                            buf = buf[buf.len() - keep..].to_string();
                            continue;
                        }
                    }
                    if !in_reasoning && !buf.is_empty() {
                        content_all.push_str(&buf);
                        if content_all.contains("<｜DSML｜ calls>") {
                            tool_mode = true;
                        }
                        if !tool_mode {
                            // hold back a possible partial "<｜DSML｜ calls>" opener
                            let opener = "<｜DSML｜ calls>";
                            let keep = (1..opener.len()).rev().filter(|&k| opener.is_char_boundary(k)).find(|&k| content_all.ends_with(&opener[..k])).unwrap_or(0);
                            let emit_upto = buf.len().saturating_sub(keep);
                            if emit_upto > 0 {
                                yield Ok(SseEvent::default().data(chunk(json!({"content": &buf[..emit_upto]}), None).to_string()));
                            }
                            buf = buf[emit_upto..].to_string();
                            content_all.truncate(content_all.len() - buf.len());
                        } else {
                            buf.clear();
                        }
                    }
                }
                Event::Done { finish, prompt_tokens, completion_tokens, .. } => {
                    let mut finish = finish;
                    if tool_mode {
                        let (_, calls) = parse_tool_calls(&content_all);
                        if !calls.is_empty() {
                            let tc: Vec<Value> = calls.into_iter().enumerate().map(|(i, mut c)| { c["index"] = json!(i); c }).collect();
                            yield Ok(SseEvent::default().data(chunk(json!({"tool_calls": tc}), None).to_string()));
                            finish = "tool_calls";
                        }
                    } else if !buf.is_empty() {
                        let key = if in_reasoning { "reasoning_content" } else { "content" };
                        yield Ok(SseEvent::default().data(chunk(json!({key: buf.clone()}), None).to_string()));
                    }
                    yield Ok(SseEvent::default().data(chunk(json!({}), Some(finish)).to_string()));
                    if include_usage {
                        let u = json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [],
                            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}});
                        yield Ok(SseEvent::default().data(u.to_string()));
                    }
                }
                Event::Error(e) => {
                    yield Ok(SseEvent::default().data(json!({"error": {"message": e}}).to_string()));
                }
                Event::Started { .. } => {}
            }
        }
        yield Ok(SseEvent::default().data("[DONE]"));
    };
    Sse::new(sse).keep_alive(KeepAlive::default()).into_response()
}

async fn completions(State(s): S, Json(body): Json<Value>) -> Response {
    let prompt = match body.get("prompt") {
        Some(Value::String(p)) => p.clone(),
        Some(Value::Array(a)) if a.len() == 1 => a[0].as_str().unwrap_or("").to_string(),
        _ => return err(StatusCode::BAD_REQUEST, "prompt must be a string"),
    };
    let params = sampling(&body);
    let stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let (mut job, _) = match start(&s, &prompt, params) {
        Ok(j) => j,
        Err(r) => return r,
    };
    let id = format!("cmpl-{}", uuid::Uuid::new_v4().simple());
    let model = s.engine.model_id.clone();
    let created = now();
    if !stream {
        let _guard = CancelOnDrop(job.cancel.clone());
        let mut text = String::new();
        let mut finish = "stop";
        let mut usage = json!({});
        while let Some(ev) = job.rx.recv().await {
            match ev {
                Event::Text(t) => text.push_str(&t),
                Event::Done { finish: f, prompt_tokens, completion_tokens, .. } => {
                    finish = f;
                    usage = json!({"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens});
                }
                Event::Error(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e),
                Event::Started { .. } => {}
            }
        }
        return Json(json!({"id": id, "object": "text_completion", "created": created, "model": model,
            "choices": [{"index": 0, "text": text, "finish_reason": finish, "logprobs": null}], "usage": usage}))
        .into_response();
    }
    let sse = async_stream::stream! {
        let _guard = CancelOnDrop(job.cancel.clone());
        while let Some(ev) = job.rx.recv().await {
            match ev {
                Event::Text(t) => {
                    yield Ok::<_, Infallible>(SseEvent::default().data(json!({"id": id, "object": "text_completion", "created": created, "model": model,
                        "choices": [{"index": 0, "text": t, "finish_reason": null}]}).to_string()));
                }
                Event::Done { finish, .. } => {
                    yield Ok(SseEvent::default().data(json!({"id": id, "object": "text_completion", "created": created, "model": model,
                        "choices": [{"index": 0, "text": "", "finish_reason": finish}]}).to_string()));
                }
                Event::Error(e) => yield Ok(SseEvent::default().data(json!({"error": {"message": e}}).to_string())),
                Event::Started { .. } => {}
            }
        }
        yield Ok(SseEvent::default().data("[DONE]"));
    };
    Sse::new(sse).keep_alive(KeepAlive::default()).into_response()
}
