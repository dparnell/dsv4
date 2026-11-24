//! The engine owns the model on a dedicated thread and serves generation requests one at a
//! time, reusing the KV state of the previous request when the new prompt shares its prefix.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::Result;
use crossbeam_channel::{Receiver, Sender, unbounded};
use tokio::sync::mpsc::UnboundedSender;

use crate::chat::{Detok, Tokenizer};
use crate::model::Model;
use crate::stats::{DecodeTime, Stats};

#[derive(Clone, Debug)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub seed: Option<u64>,
    pub max_tokens: usize,
    pub stop: Vec<String>,
    pub ignore_eos: bool,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            seed: None,
            max_tokens: 4096,
            stop: Vec::new(),
            ignore_eos: false,
        }
    }
}

pub enum Event {
    /// prompt was accepted; `cached` of its tokens are reused
    Started { cached: usize },
    Text(String),
    Done { finish: &'static str, prompt_tokens: usize, completion_tokens: usize, prefill_s: f64, decode_s: f64 },
    Error(String),
}

pub struct Request {
    pub prompt: Vec<u32>,
    pub params: SamplingParams,
    pub tx: UnboundedSender<Event>,
    pub cancel: Arc<AtomicBool>,
    /// for the request log
    pub endpoint: &'static str,
    /// set by `submit`
    pub id: u64,
}

#[derive(Clone)]
pub struct EngineHandle {
    tx: Sender<Request>,
    pub tok: Arc<Tokenizer>,
    pub stats: Arc<Stats>,
    pub model_id: String,
    pub max_ctx: usize,
}

impl EngineHandle {
    pub fn submit(&self, mut r: Request) {
        r.id = self.stats.log_request(r.endpoint, r.prompt.len());
        Stats::add(&self.stats.requests_queued, 1);
        let _ = self.tx.send(r);
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f32(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

pub fn sample(logits: &mut [f32], p: &SamplingParams, history: &[u32], rng: &mut impl FnMut() -> f32) -> u32 {
    if p.repetition_penalty != 1.0 || p.presence_penalty != 0.0 || p.frequency_penalty != 0.0 {
        let mut counts = std::collections::HashMap::<u32, u32>::new();
        for &t in history {
            *counts.entry(t).or_default() += 1;
        }
        for (&t, &c) in &counts {
            let l = &mut logits[t as usize];
            if p.repetition_penalty != 1.0 {
                *l = if *l > 0.0 { *l / p.repetition_penalty } else { *l * p.repetition_penalty };
            }
            *l -= p.presence_penalty + p.frequency_penalty * c as f32;
        }
    }
    if p.temperature <= 0.0 {
        return logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).unwrap_or(0);
    }
    let inv_t = 1.0 / p.temperature;
    let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut cand: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i as u32, ((l - mx) * inv_t).exp()))
        .filter(|&(_, pr)| pr > 0.0)
        .collect();
    if p.min_p > 0.0 {
        cand.retain(|&(_, pr)| pr >= p.min_p);
    }
    let k = if p.top_k > 0 { p.top_k.min(cand.len()) } else { cand.len() };
    let sort_needed = p.top_p < 1.0 || k < cand.len();
    if sort_needed {
        if k < cand.len() {
            cand.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
            cand.truncate(k);
        }
        cand.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        if p.top_p < 1.0 {
            let total: f32 = cand.iter().map(|c| c.1).sum();
            let mut acc = 0.0;
            let mut cut = cand.len();
            for (i, c) in cand.iter().enumerate() {
                acc += c.1 / total;
                if acc >= p.top_p {
                    cut = i + 1;
                    break;
                }
            }
            cand.truncate(cut);
        }
    }
    let total: f32 = cand.iter().map(|c| c.1).sum();
    let mut r = rng() * total;
    for &(t, pr) in &cand {
        r -= pr;
        if r <= 0.0 {
            return t;
        }
    }
    cand.last().map(|c| c.0).unwrap_or(0)
}

pub fn spawn(mut model: Model, tok: Arc<Tokenizer>, stats: Arc<Stats>, model_id: String) -> EngineHandle {
    let (tx, rx): (Sender<Request>, Receiver<Request>) = unbounded();
    let max_ctx = model.max_ctx;
    let st = stats.clone();
    let tk = tok.clone();
    std::thread::Builder::new()
        .name("engine".into())
        .spawn(move || {
            while let Ok(req) = rx.recv() {
                Stats::set(&st.requests_queued, rx.len() as u64);
                Stats::set(&st.requests_active, 1);
                if let Err(e) = run_request(&mut model, &tk, &st, &req) {
                    st.event(format!("request failed: {e:#}"));
                    st.update_request(req.id, |r| r.state = format!("failed: {e:#}"));
                    let _ = req.tx.send(Event::Error(format!("{e:#}")));
                    model.truncate(0);
                    model.drop_snapshot();
                }
                if let Some(p) = model.profile_out.clone() {
                    match model.experts.save_profile(&p) {
                        Ok(()) => st.event(format!("expert profile updated: {}", p.display())),
                        Err(e) => st.event(format!("could not write expert profile: {e:#}")),
                    }
                }
                Stats::set(&st.requests_active, 0);
                Stats::add(&st.requests_done, 1);
                st.set_phase("idle");
            }
        })
        .expect("spawn engine thread");
    EngineHandle { tx, tok, stats, model_id, max_ctx }
}

fn run_request(model: &mut Model, tok: &Tokenizer, st: &Stats, req: &Request) -> Result<()> {
    let p = &req.params;
    let prompt = &req.prompt;
    if prompt.is_empty() {
        anyhow::bail!("empty prompt");
    }
    if prompt.len() >= model.max_ctx {
        anyhow::bail!("prompt of {} tokens exceeds the context window ({})", prompt.len(), model.max_ctx);
    }
    // given up on while it was queued: a long prompt can take minutes
    if req.cancel.load(Ordering::Relaxed) || req.tx.is_closed() {
        st.update_request(req.id, |r| r.state = "cancelled".into());
        let _ = req.tx.send(Event::Done { finish: "cancelled", prompt_tokens: prompt.len(), completion_tokens: 0, prefill_s: 0.0, decode_s: 0.0 });
        return Ok(());
    }
    let reuse = model.prepare(prompt);
    st.update_request(req.id, |r| {
        r.state = "prefill".into();
        r.cached = reuse;
    });
    let _ = req.tx.send(Event::Started { cached: reuse });
    st.event(format!("request: {} prompt tokens ({} cached), max {} new", prompt.len(), reuse, p.max_tokens));

    st.set_phase(format!("prefill {} tokens", prompt.len() - reuse));
    st.push_text("\n\n── new request ──\n");
    let t0 = Instant::now();
    let mut logits = model.forward(&prompt[reuse..])?;
    // a follow-up turn usually diverges right after this prompt (e.g. dropped reasoning)
    model.take_snapshot();
    let prefill_s = t0.elapsed().as_secs_f64();
    let n_pre = (prompt.len() - reuse) as u64;
    Stats::add(&st.prompt_tokens, n_pre);
    Stats::add(&st.prefill_ns, t0.elapsed().as_nanos() as u64);
    Stats::set(&st.last_prefill_tps_milli, (n_pre as f64 / prefill_s.max(1e-9) * 1000.0) as u64);

    st.update_request(req.id, |r| {
        r.state = "generating".into();
        r.prefill_s = Some(prefill_s);
        // sampling the first token from these logits takes no time worth counting
        r.ttft_s = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).ok().map(|d| d.as_secs_f64() - r.at);
    });
    let seed = p.seed.unwrap_or_else(|| std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64);
    let mut rng = Rng(seed | 1);
    let mut detok = Detok::new();
    let mut generated: Vec<u32> = Vec::new();
    let mut text = String::new();
    let mut finish = "length";
    let max_new = p.max_tokens.min(model.max_ctx - prompt.len());
    st.set_phase("decoding");
    let t1 = Instant::now();
    let mut pending = String::new();
    for _ in 0..max_new {
        if req.cancel.load(Ordering::Relaxed) || req.tx.is_closed() {
            finish = "cancelled";
            break;
        }
        let ts = Instant::now();
        let id = sample(&mut logits, p, &generated, &mut || rng.f32());
        Stats::set(&st.t_sample_ns, ts.elapsed().as_nanos() as u64);
        if !p.ignore_eos && tok.stop_ids.contains(&id) {
            finish = "stop";
            break;
        }
        generated.push(id);
        st.update_request(req.id, |r| {
            r.generated = generated.len();
            r.tps = (generated.len() > 1).then(|| (generated.len() - 1) as f64 / t1.elapsed().as_secs_f64().max(1e-9));
        });
        let piece = detok.push(tok, id);
        st.push_text(&piece);
        if !piece.is_empty() {
            text.push_str(&piece);
            pending.push_str(&piece);
            // hold back text that could be the start of a stop string
            if let Some(s) = p.stop.iter().find(|s| text.contains(s.as_str())) {
                let cut = text.find(s.as_str()).unwrap();
                let already = text.len() - pending.len();
                if cut > already {
                    let _ = req.tx.send(Event::Text(text[already..cut].to_string()));
                }
                finish = "stop";
                break;
            }
            let hold = p.stop.iter().map(|s| (1..s.len()).rev().find(|&k| text.ends_with(&s[..k])).unwrap_or(0)).max().unwrap_or(0);
            if pending.len() > hold {
                let emit = pending[..pending.len() - hold].to_string();
                pending.drain(..pending.len() - hold);
                let _ = req.tx.send(Event::Text(emit));
            }
        }
        let td = Instant::now();
        let before = DecodeTime::read(st);
        logits = model.forward(&[id])?;
        let dt = td.elapsed();
        st.push_decode_time(DecodeTime::since(st, &before, dt.as_nanos() as u64));
        Stats::add(&st.gen_tokens, 1);
        Stats::add(&st.decode_ns, dt.as_nanos() as u64);
        Stats::set(&st.last_decode_ns, dt.as_nanos() as u64);
        {
            let mut h = st.tps_history.lock();
            h.push_back((1e9 / dt.as_nanos().max(1) as f64 * 100.0) as u64);
            while h.len() > 240 {
                h.pop_front();
            }
        }
    }
    if finish != "stop" && !pending.is_empty() {
        let _ = req.tx.send(Event::Text(pending));
    } else if finish == "stop" && !pending.is_empty() && p.stop.iter().all(|s| !text.contains(s.as_str())) {
        let _ = req.tx.send(Event::Text(pending));
    }
    let decode_s = t1.elapsed().as_secs_f64();
    st.update_request(req.id, |r| r.state = finish.into());
    st.event(format!(
        "done: {} tokens in {:.1}s ({:.2} tok/s), prefill {:.1} tok/s, finish={finish}",
        generated.len(),
        decode_s,
        generated.len() as f64 / decode_s.max(1e-9),
        n_pre as f64 / prefill_s.max(1e-9)
    ));
    let _ = req.tx.send(Event::Done { finish, prompt_tokens: prompt.len(), completion_tokens: generated.len(), prefill_s, decode_s });
    Ok(())
}
