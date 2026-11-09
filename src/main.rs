mod chat;
mod config;
mod cpu_moe;
mod download;
mod engine;
mod engram;
mod experts;
mod gguf;
mod gpu;
mod kernels;
mod model;
mod selftest;
mod server;
mod stats;
mod tui;
mod weights;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};

use crate::chat::{ChatTemplate, Tokenizer};
use crate::engine::{Event, Request, SamplingParams};
use crate::model::{Model, ModelOptions};
use crate::stats::Stats;

const DEFAULT_MODEL_DIR: &str = "F:/models/DeepSeek-V4.1-Flash-1017-GGUF";

#[derive(Parser)]
#[command(name = "dsv4", version, about = "DeepSeek-V4.1-Flash inference on wgpu with streamed MoE experts")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Args, Clone)]
struct ModelArgs {
    /// Model directory, quant folder, or first GGUF shard
    #[arg(long, short, default_value = DEFAULT_MODEL_DIR, env = "DSV4_MODEL")]
    model: PathBuf,
    /// Maximum context length (tokens)
    #[arg(long, default_value_t = 32768)]
    ctx: usize,
    /// Attention chunk size for prefill
    #[arg(long, default_value_t = 128)]
    chunk: usize,
    /// Prefill segment: tokens processed layer-major so experts are fetched once per segment
    #[arg(long, default_value_t = 1024)]
    segment: usize,
    /// VRAM for the routed-expert cache in GB (default: free VRAM minus a reserve)
    #[arg(long)]
    expert_vram_gb: Option<f64>,
    /// Host RAM for the routed-expert cache in GB (default: 75% of system RAM, leaving >= 16 GB,
    /// capped by the memory that is actually free at startup)
    #[arg(long)]
    ram_cache_gb: Option<f64>,
    /// GPU backend: vulkan | dx12 | metal (default: best available)
    #[arg(long)]
    backend: Option<String>,
    /// tokenizer.json path (default: next to the model)
    #[arg(long)]
    tokenizer: Option<PathBuf>,
    /// Run every routed expert on the GPU (uploading misses) instead of computing cold
    /// experts on the CPU from the RAM cache
    #[arg(long)]
    gpu_only_experts: bool,
}

#[derive(Args, Clone)]
struct GenArgs {
    #[arg(long, default_value_t = 1.0)]
    temperature: f32,
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,
    #[arg(long, default_value_t = 0)]
    top_k: usize,
    #[arg(long, default_value_t = 1024)]
    max_tokens: usize,
    #[arg(long)]
    seed: Option<u64>,
    /// Disable thinking mode (answer directly)
    #[arg(long)]
    no_think: bool,
}

#[derive(Subcommand)]
enum Cmd {
    /// Download GGUF weights (and the tokenizer) from Hugging Face
    Download {
        /// Quant folder in the repo
        #[arg(long, default_value = "UD-Q4_K_XL")]
        quant: String,
        /// Destination directory
        #[arg(long, default_value = DEFAULT_MODEL_DIR)]
        dir: PathBuf,
        #[arg(long, default_value = download::DEFAULT_REPO)]
        repo: String,
        /// Concurrent HTTP range connections
        #[arg(long, default_value_t = 16)]
        connections: usize,
        /// Only list the quants available in the repo
        #[arg(long)]
        list: bool,
    },
    /// Start the OpenAI-compatible HTTP server
    Serve {
        #[command(flatten)]
        model: ModelArgs,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8080)]
        port: u16,
        /// Show the live statistics dashboard
        #[arg(long)]
        tui: bool,
        /// Default to chat mode (no thinking) when the request does not say
        #[arg(long)]
        no_think: bool,
        /// Model id reported by /v1/models
        #[arg(long, default_value = "deepseek-v4.1-flash")]
        model_id: String,
    },
    /// Generate a completion for one prompt
    Run {
        #[command(flatten)]
        model: ModelArgs,
        #[command(flatten)]
        sampling: GenArgs,
        /// The user message (or raw text with --raw)
        #[arg(long, short)]
        prompt: String,
        /// Feed the prompt as-is, without the chat template
        #[arg(long)]
        raw: bool,
        /// Show the live statistics dashboard while generating
        #[arg(long)]
        tui: bool,
    },
    /// Interactive chat in the terminal
    Chat {
        #[command(flatten)]
        model: ModelArgs,
        #[command(flatten)]
        sampling: GenArgs,
        #[arg(long)]
        system: Option<String>,
    },
    /// Render a chat (OpenAI messages JSON file) with the model's template and print the prompt
    Prompt {
        #[arg(long, short, default_value = DEFAULT_MODEL_DIR)]
        model: PathBuf,
        /// JSON file: either a messages array or {"messages": [...], "tools": [...]}
        file: PathBuf,
        #[arg(long)]
        no_think: bool,
    },
    /// Check every GPU kernel against a CPU reference
    Selftest,
    /// Measure expert streaming throughput from disk
    BenchIo {
        #[arg(long, short, default_value = DEFAULT_MODEL_DIR)]
        model: PathBuf,
        /// Experts to read per run
        #[arg(long, default_value_t = 200)]
        n: usize,
    },
    /// Show GGUF metadata and tensor summary
    Info {
        #[arg(long, short, default_value = DEFAULT_MODEL_DIR)]
        model: PathBuf,
        /// List every tensor
        #[arg(long)]
        tensors: bool,
    },
}

fn total_ram() -> u64 {
    let Ok(mut s) = sysinfo::System::new() else { return 64 << 30 };
    s.refresh_memory();
    s.total_memory()
}

/// Memory we can actually take right now: the smaller of free physical RAM and (on Windows)
/// remaining commit charge, since every heap allocation counts against the commit limit.
fn available_ram() -> u64 {
    let phys = sysinfo::System::new()
        .map(|mut s| {
            s.refresh_memory();
            s.available_memory()
        })
        .unwrap_or(32 << 30);
    #[cfg(windows)]
    {
        #[repr(C)]
        struct MemoryStatusEx {
            len: u32,
            load: u32,
            total_phys: u64,
            avail_phys: u64,
            total_page_file: u64,
            avail_page_file: u64,
            total_virtual: u64,
            avail_virtual: u64,
            avail_ext: u64,
        }
        unsafe extern "system" {
            fn GlobalMemoryStatusEx(buf: *mut MemoryStatusEx) -> i32;
        }
        let mut m: MemoryStatusEx = unsafe { std::mem::zeroed() };
        m.len = std::mem::size_of::<MemoryStatusEx>() as u32;
        if unsafe { GlobalMemoryStatusEx(&mut m) } != 0 {
            return phys.min(m.avail_page_file);
        }
    }
    phys
}

/// Default size of the host expert cache: 75% of RAM (leaving 16 GB), but never more than what
/// is free now minus room for the rest of the process (~14 GB of loading buffers and tables).
fn default_ram_cache() -> u64 {
    let t = total_ram();
    let by_total = (t / 4 * 3).min(t.saturating_sub(16 << 30));
    let by_free = available_ram().saturating_sub(20 << 30);
    by_total.min(by_free).max(2 << 30)
}


struct Loaded {
    model: Model,
    tok: Arc<Tokenizer>,
    template: ChatTemplate,
}

fn load(args: &ModelArgs, stats: Arc<Stats>) -> Result<Loaded> {
    let first = gguf::resolve_model_path(&args.model)?;
    let tok_path = match &args.tokenizer {
        Some(p) => p.clone(),
        None => chat::find_tokenizer(&first).context("tokenizer.json not found next to the model; run `dsv4 download` or pass --tokenizer")?,
    };
    let tok = Arc::new(Tokenizer::load(&tok_path)?);
    let ram = args.ram_cache_gb.map(|g| (g * 1e9) as u64).unwrap_or_else(default_ram_cache);
    // The expert VRAM budget is decided after the dense weights are resident: ExpertStore is
    // created inside Model::load, so estimate free VRAM up-front from NVML minus the dense size.
    // None: measured once the dense weights and activations are resident
    let vram = args.expert_vram_gb.map(|g| (g * 1e9) as u64);
    let opt = ModelOptions { max_ctx: args.ctx, chunk: args.chunk, seg: args.segment, expert_vram_bytes: vram, expert_ram_bytes: ram, backend: args.backend.clone(), cpu_experts: !args.gpu_only_experts };
    let model = Model::load(&first, &opt, stats.clone())?;
    let template = ChatTemplate::new(model.g.str("tokenizer.chat_template"));
    Ok(Loaded { model, tok, template })
}

fn params(g: &GenArgs) -> SamplingParams {
    SamplingParams { temperature: g.temperature, top_p: g.top_p, top_k: g.top_k, max_tokens: g.max_tokens, seed: g.seed, ..Default::default() }
}

fn init_logging(to_file: bool) {
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,wgpu_core=warn,wgpu_hal=warn,naga=warn"));
    if to_file {
        if let Ok(f) = std::fs::File::create("dsv4.log") {
            let _ = tracing_subscriber::fmt().with_env_filter(filter).with_writer(std::sync::Mutex::new(f)).with_ansi(false).try_init();
            return;
        }
    }
    let _ = tracing_subscriber::fmt().with_env_filter(filter).with_writer(std::io::stderr).try_init();
}

/// Generate for a token prompt and stream text to stdout. Returns the produced text.
fn generate_stdout(engine: &engine::EngineHandle, prompt: Vec<u32>, p: SamplingParams, thinking: bool) -> String {
    generate_to(engine, prompt, p, thinking, true)
}

fn generate_to(engine: &engine::EngineHandle, prompt: Vec<u32>, p: SamplingParams, thinking: bool, echo: bool) -> String {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    engine.submit(Request { prompt, params: p, tx, cancel: Arc::new(AtomicBool::new(false)) });
    let mut out = String::new();
    let mut in_think = thinking;
    if thinking && echo {
        print!("\x1b[2m");
    }
    while let Some(ev) = rx.blocking_recv() {
        match ev {
            Event::Text(t) if !echo => out.push_str(&t),
            Event::Text(t) => {
                out.push_str(&t);
                if in_think && out.contains("</think>") {
                    in_think = false;
                    let tail = &out[out.find("</think>").unwrap() + 8..];
                    let pre = &t[..t.len().saturating_sub(tail.len() + 8).min(t.len())];
                    print!("{pre}\x1b[0m\n{}", tail.trim_start_matches('\n'));
                } else {
                    print!("{t}");
                }
                let _ = std::io::stdout().flush();
            }
            Event::Done { .. } if !echo => {}
            Event::Done { finish, prompt_tokens, completion_tokens, prefill_s, decode_s } => {
                eprintln!(
                    "\x1b[0m\n[{finish}] prompt {prompt_tokens} tok, prefill {:.1}s ({:.1} tok/s), generated {completion_tokens} tok in {:.1}s ({:.2} tok/s)",
                    prefill_s,
                    prompt_tokens as f64 / prefill_s.max(1e-9),
                    decode_s,
                    completion_tokens as f64 / decode_s.max(1e-9)
                );
            }
            Event::Error(e) => eprintln!("\x1b[0m\nerror: {e}"),
            Event::Started { .. } => {}
        }
    }
    out
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Download { quant, dir, repo, connections, list } => {
            init_logging(false);
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(download::run(&repo, &quant, &dir, connections, list))
        }
        Cmd::Info { model, tensors } => info(&model, tensors),
        Cmd::BenchIo { model, n } => {
            init_logging(false);
            let first = gguf::resolve_model_path(&model)?;
            let g = gguf::Gguf::open(&first)?;
            let cfg = config::Config::from_gguf(&g)?;
            let stats = Arc::new(Stats::default());
            let gpu = gpu::Gpu::new(stats.clone(), None)?;
            let store = experts::ExpertStore::new(&g, &gpu, &cfg, 1, 0, stats)?;
            for threads in [1, 4, 8, 16, 32] {
                let (gbps, ms) = store.bench_disk(n, threads)?;
                println!("{threads:>3} loaders: {gbps:>6.2} GB/s  ({ms:.2} ms/expert amortized, 18.8 MB each)");
            }
            // CPU expert compute from RAM, one token (decode case)
            let geom = &store.geom;
            let blobs: Vec<Vec<u8>> = (0..24).map(|i| vec![(i * 37 % 251) as u8; geom.expert_bytes as usize]).collect();
            let x = vec![0.01f32; cfg.dim];
            let t0 = std::time::Instant::now();
            for b in &blobs {
                let _ = cpu_moe::expert_forward(b, geom.inter, cfg.dim, geom.mat_bytes as usize, &[&x], &[1.0], 10.0);
            }
            let ms = t0.elapsed().as_secs_f64() * 1e3 / blobs.len() as f64;
            println!("CPU expert (1 token, avx2={}): {ms:.2} ms  ({:.1} GB/s effective)", cpu_moe::has_avx2(), geom.expert_bytes as f64 / ms / 1e6);
            use rayon::prelude::*;
            let t0 = std::time::Instant::now();
            for group in blobs.chunks(6) {
                group.par_iter().for_each(|b| {
                    let _ = cpu_moe::expert_forward(b, geom.inter, cfg.dim, geom.mat_bytes as usize, &[&x], &[1.0], 10.0);
                });
            }
            let ms = t0.elapsed().as_secs_f64() * 1e3 / blobs.len() as f64;
            println!("CPU experts, 6 concurrent: {ms:.2} ms/expert  ({:.1} GB/s effective)", geom.expert_bytes as f64 / ms / 1e6);
            Ok(())
        }
        Cmd::Prompt { model, file, no_think } => {
            let first = gguf::resolve_model_path(&model)?;
            let g = gguf::Gguf::open(&first)?;
            let t = ChatTemplate::new(g.str("tokenizer.chat_template"));
            let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&file)?)?;
            let (msgs, tools) = match &v {
                serde_json::Value::Array(_) => (v.clone(), None),
                _ => (v["messages"].clone(), v.get("tools").cloned()),
            };
            let thinking = v.get("thinking_mode").and_then(|m| m.as_str()).map(|m| m == "thinking").unwrap_or(!no_think);
            let text = t.render(&msgs, tools.as_ref(), thinking, v.get("reasoning_effort"))?;
            print!("{text}");
            if let Some(tp) = chat::find_tokenizer(&first) {
                let tok = Tokenizer::load(&tp)?;
                eprintln!("
[{} tokens]", tok.encode(&text)?.len());
            }
            Ok(())
        }
        Cmd::Selftest => {
            init_logging(false);
            selftest::run()
        }
        Cmd::Serve { model, host, port, tui, no_think, model_id } => {
            init_logging(tui);
            let stats = Arc::new(Stats::default());
            let listen = format!("{host}:{port}");
            let rt = tokio::runtime::Runtime::new()?;
            let st = stats.clone();
            let ma = model.clone();
            let mid = model_id.clone();
            let l2 = listen.clone();
            // load + serve in the background so the TUI can show progress
            let server = std::thread::spawn(move || -> Result<()> {
                let Loaded { model, tok, template } = load(&ma, st.clone())?;
                let engine = engine::spawn(model, tok, st.clone(), mid);
                let state = Arc::new(server::AppState { engine, template, default_thinking: !no_think });
                rt.block_on(async move {
                    let listener = tokio::net::TcpListener::bind(&l2).await?;
                    st.event(format!("listening on http://{l2}/v1"));
                    st.set_phase("idle");
                    axum::serve(listener, server::router(state)).await?;
                    anyhow::Ok(())
                })
            });
            if tui {
                tui::run(stats, listen)?;
                std::process::exit(0);
            }
            server.join().map_err(|_| anyhow::anyhow!("server thread panicked"))??;
            Ok(())
        }
        Cmd::Run { model, sampling, prompt, raw, tui } => {
            init_logging(tui);
            let stats = Arc::new(Stats::default());
            if tui {
                let st = stats.clone();
                std::thread::spawn(move || {
                    let _ = tui::run(st, "-".into());
                    std::process::exit(0);
                });
            }
            let Loaded { model, tok, template } = load(&model, stats.clone())?;
            let thinking = !sampling.no_think && !raw;
            let text = if raw { prompt.clone() } else { template.render(&serde_json::json!([{"role": "user", "content": prompt}]), None, thinking, None)? };
            let ids = tok.encode(&text)?;
            let engine = engine::spawn(model, tok, stats.clone(), "dsv4".into());
            generate_to(&engine, ids, params(&sampling), thinking, !tui);
            if !tui {
                print_breakdown(&stats);
            }
            if tui {
                std::thread::sleep(std::time::Duration::from_secs(3600 * 24));
            }
            Ok(())
        }
        Cmd::Chat { model, sampling, system } => {
            init_logging(false);
            let stats = Arc::new(Stats::default());
            let Loaded { model, tok, template } = load(&model, stats.clone())?;
            let engine = engine::spawn(model, tok.clone(), stats, "dsv4".into());
            let thinking = !sampling.no_think;
            let mut msgs: Vec<serde_json::Value> = Vec::new();
            if let Some(s) = system {
                msgs.push(serde_json::json!({"role": "system", "content": s}));
            }
            println!("dsv4 chat — empty line to quit, /reset to clear history");
            loop {
                print!("\n\x1b[1m> \x1b[0m");
                std::io::stdout().flush()?;
                let mut line = String::new();
                if std::io::stdin().read_line(&mut line)? == 0 || line.trim().is_empty() {
                    break;
                }
                if line.trim() == "/reset" {
                    msgs.retain(|m| m["role"] == "system");
                    continue;
                }
                msgs.push(serde_json::json!({"role": "user", "content": line.trim_end()}));
                let text = template.render(&serde_json::Value::Array(msgs.clone()), None, thinking, None)?;
                let out = generate_stdout(&engine, tok.encode(&text)?, params(&sampling), thinking);
                let (reasoning, content) = chat::split_reasoning(&out, thinking);
                let mut m = serde_json::json!({"role": "assistant", "content": content});
                if let Some(r) = reasoning {
                    m["reasoning_content"] = serde_json::json!(r);
                }
                msgs.push(m);
            }
            Ok(())
        }
    }
}

fn print_breakdown(st: &Stats) {
    let g = |c: &std::sync::atomic::AtomicU64| Stats::get(c);
    let s = |c: &std::sync::atomic::AtomicU64| Stats::get(c) as f64 / 1e9;
    eprintln!(
        "breakdown: gpu-wait {:.2}s  experts {:.2}s (gpu path {:.2}s, cpu path {:.2}s)  disk {:.2} GB in {} loads  ram hits {}  vram hits {}  cpu runs {}  uploads {} ({:.2} GB)  prefetches {}",
        s(&st.t_gpu_wait_ns),
        s(&st.t_moe_ns),
        s(&st.t_expert_ns),
        s(&st.t_cpu_moe_ns),
        g(&st.disk_bytes) as f64 / 1e9,
        g(&st.expert_disk_loads),
        g(&st.expert_ram_hits),
        g(&st.expert_vram_hits),
        g(&st.expert_cpu_runs),
        g(&st.expert_uploads),
        g(&st.upload_bytes) as f64 / 1e9,
        g(&st.expert_prefetches),
    );
}

fn info(model: &Path, tensors: bool) -> Result<()> {
    let first = gguf::resolve_model_path(model)?;
    let g = gguf::Gguf::open(&first)?;
    println!("{} shard(s), {:.1} GB", g.shards.len(), g.total_bytes() as f64 / 1e9);
    let mut keys: Vec<_> = g.kv.keys().collect();
    keys.sort();
    for k in keys {
        let v = &g.kv[k];
        let s = match v {
            gguf::Value::Arr(a) if a.len() > 16 => format!("[{} items]", a.len()),
            gguf::Value::Str(s) if s.len() > 120 => format!("\"{}...\" ({} chars)", &s[..s.char_indices().nth(100).map(|c| c.0).unwrap_or(s.len())], s.len()),
            other => format!("{other:?}"),
        };
        println!("  {k} = {s}");
    }
    let mut by_type: std::collections::BTreeMap<&str, (usize, u64)> = Default::default();
    for t in g.tensors.values() {
        let e = by_type.entry(gguf::type_name(t.ty)).or_default();
        e.0 += 1;
        e.1 += t.nbytes;
    }
    println!("{} tensors:", g.tensors.len());
    for (ty, (n, b)) in by_type {
        println!("  {ty:<6} {n:>5} tensors {:>9.2} GB", b as f64 / 1e9);
    }
    if tensors {
        for name in &g.tensor_order {
            let t = &g.tensors[name];
            println!("  {name:<45} {:<6} {:?}", gguf::type_name(t.ty), t.dims);
        }
    }
    let cfg = config::Config::from_gguf(&g)?;
    println!(
        "config: layers {} dim {} heads {}x{} (rope {}) q_lora {} window {} experts {}/{} inter {} hc {} engram layers {:?} kv sources {:?} index sources {:?}",
        cfg.n_layers,
        cfg.dim,
        cfg.n_heads,
        cfg.head_dim,
        cfg.rope_dim,
        cfg.q_lora_rank,
        cfg.window,
        cfg.n_active,
        cfg.n_experts,
        cfg.moe_inter,
        cfg.hc_mult,
        cfg.engram_layers,
        cfg.kv_source_layers,
        cfg.index_source_layers
    );
    Ok(())
}
