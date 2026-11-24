mod chat;
mod distill;
mod config;
mod cpu_moe;
mod download;
mod engine;
mod engram;
mod expert_io;
mod experts;
mod gguf;
mod gpu;
mod kernels;
mod model;
mod monitor;
mod pack;
mod quants;
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

const MODEL_FOLDER: &str = "DeepSeek-V4.1-Flash-1017-GGUF";

/// Where models are kept unless --model / --dir say otherwise: DSV4_MODELS_DIR, or the folder for
/// application data that belongs to this user and machine.
fn models_dir(var: impl Fn(&str) -> Option<PathBuf>, home: Option<PathBuf>) -> PathBuf {
    if let Some(d) = var("DSV4_MODELS_DIR").filter(|d| !d.as_os_str().is_empty()) {
        return d;
    }
    // the XDG specification says to ignore relative paths
    let abs = |name: &str| var(name).filter(|d| d.is_absolute());
    let data = if cfg!(windows) {
        abs("LOCALAPPDATA").or(home.map(|h| h.join("AppData").join("Local")))
    } else if cfg!(target_os = "macos") {
        home.map(|h| h.join("Library").join("Application Support"))
    } else {
        abs("XDG_DATA_HOME").or(home.map(|h| h.join(".local").join("share")))
    };
    data.map_or_else(|| "models".into(), |d| d.join("dsv4").join("models"))
}

fn default_model_dir() -> PathBuf {
    models_dir(|n| std::env::var_os(n).map(PathBuf::from), std::env::home_dir()).join(MODEL_FOLDER)
}

#[derive(Parser)]
#[command(name = "dsv4", version, about = "DeepSeek-V4.1-Flash inference on wgpu with streamed MoE experts")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Args, Clone)]
struct ModelArgs {
    /// Model directory, quant folder, or first GGUF shard
    #[arg(long, short, default_value_os_t = default_model_dir(), env = "DSV4_MODEL")]
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
    /// Expert pack written by `dsv4 repack` (default: experts.pack next to the shards, if there
    /// is one; "none": read the shards)
    #[arg(long)]
    expert_pack: Option<PathBuf>,
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
    /// Experiment: restrict routing to the N most-used experts of each layer (from
    /// --expert-profile) and pin them in VRAM, overflowing to RAM. N = 6 ignores the router's
    /// choice entirely (it only weights the fixed six).
    #[arg(long, requires = "expert_profile")]
    expert_subset: Option<usize>,
    /// Expert usage profile (JSON): chooses --expert-subset, and replaces the built-in ranking
    /// for warming the caches
    #[arg(long)]
    expert_profile: Option<PathBuf>,
    /// LOSSY: while decoding, pick experts as if those in VRAM or RAM scored LAMBDA higher
    #[arg(long, value_name = "LAMBDA", num_args = 0..=1, default_missing_value = "0.1", default_value_t = 0.0, hide_default_value = true)]
    route_prior: f32,
    /// LOSSY: while decoding, drop picks of weight rank R (0-based, 1-5) or lower that are not in
    /// VRAM or RAM
    #[arg(long, value_name = "R", num_args = 0..=1, default_missing_value = "5", default_value_t = 0, hide_default_value = true)]
    route_tail_skip: usize,
    /// Compute every layer for every prompt position, even where it cannot influence the output
    #[arg(long)]
    no_prompt_skip: bool,
    /// Do not read ahead likely experts of the next layer
    #[arg(long)]
    no_prefetch: bool,
    /// Do not fill the expert caches in the background after loading
    #[arg(long)]
    no_warm_up: bool,
    /// Add this run's expert selection counts to a profile (JSON, created if missing)
    #[arg(long)]
    save_expert_profile: Option<PathBuf>,
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
        /// Destination directory (set DSV4_MODELS_DIR to keep models elsewhere by default)
        #[arg(long, default_value_os_t = default_model_dir())]
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
        /// Do not serve the web UI (chat at /, status at /status)
        #[arg(long)]
        no_ui: bool,
        /// Default to chat mode (no thinking) when the request does not say
        #[arg(long)]
        no_think: bool,
        /// Model id reported by /v1/models
        #[arg(long, default_value = "deepseek-v4.1-flash")]
        model_id: String,
    },
    /// Experiment: record the routed inputs of one layer's experts while prefilling a corpus
    Capture {
        #[command(flatten)]
        model: ModelArgs,
        /// Layer whose expert inputs are recorded (later layers are skipped)
        #[arg(long)]
        layer: usize,
        /// Output file (records of [u16 expert][u16 slot][f32 weight][f32 x; dim])
        #[arg(long)]
        out: PathBuf,
        /// Stop after this many tokens
        #[arg(long, default_value_t = 49152)]
        tokens: usize,
        /// Tokens per prefill (each is a fresh context)
        #[arg(long, default_value_t = 1024)]
        doc_tokens: usize,
        /// Chunks taken from each file (for variety)
        #[arg(long, default_value_t = 2)]
        chunks_per_file: usize,
        /// Text file listing the corpus files, one path per line
        #[arg(long)]
        list: PathBuf,
    },
    /// Experiment: fit compressed replacements for one layer's experts on captured inputs and
    /// report their error on held-out samples
    DistillPilot {
        #[arg(long, short, default_value_os_t = default_model_dir(), env = "DSV4_MODEL")]
        model: PathBuf,
        #[arg(long)]
        layer: usize,
        /// Capture file from `dsv4 capture`
        #[arg(long)]
        capture: PathBuf,
        /// Compression ratios (bytes per expert vs original)
        #[arg(long, value_delimiter = ',', default_value = "2,4,8")]
        ratios: Vec<usize>,
        /// Methods: prune, prune+fit, lowrank, shared
        #[arg(long, value_delimiter = ',', default_value = "prune,prune+fit,lowrank,shared")]
        methods: Vec<String>,
        /// Write the replacement layer for METHOD:RATIO (use with DSV4_EXPERT_OVERRIDE)
        #[arg(long)]
        emit: Option<String>,
        #[arg(long)]
        emit_path: Option<PathBuf>,
        /// Analyse N experts spread evenly over the usage ranking (0 = all)
        #[arg(long, default_value_t = 0)]
        max_experts: usize,
    },
    /// Teacher-forced quality check: decode text token by token, report perplexity, speed and the
    /// distance to a saved reference run
    Eval {
        #[command(flatten)]
        model: ModelArgs,
        /// Text files; each is a fresh context
        #[arg(long, value_delimiter = ',')]
        text: Vec<PathBuf>,
        /// Tokens of each file processed as a prompt
        #[arg(long, default_value_t = 64)]
        prefill: usize,
        /// Tokens of each file then decoded one at a time
        #[arg(long, default_value_t = 64)]
        tokens: usize,
        /// Write the logits for later comparison
        #[arg(long)]
        save: Option<PathBuf>,
        /// Compare with logits written by --save
        #[arg(long)]
        base: Option<PathBuf>,
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
        #[arg(long, short, default_value_os_t = default_model_dir(), env = "DSV4_MODEL")]
        model: PathBuf,
        /// JSON file: either a messages array or {"messages": [...], "tools": [...]}
        file: PathBuf,
        #[arg(long)]
        no_think: bool,
    },
    /// Check every GPU kernel against a CPU reference
    Selftest,
    /// Measure how fast routed experts run on the CPU (random weights, no model needed)
    BenchCpu {
        #[arg(long, default_value_t = 5120)]
        dim: usize,
        /// Hidden size of one expert
        #[arg(long, default_value_t = 2304)]
        inter: usize,
        /// Types of the gate, up and down matrices, e.g. IQ2_XXS,IQ2_XXS,IQ2_S
        #[arg(long, default_value = "MXFP4,MXFP4,MXFP4")]
        types: String,
    },
    /// Copy the experts into one file where each is contiguous, which drives read faster
    /// (lossless; takes as much space as the experts). An interrupted run resumes.
    Repack {
        #[arg(long, short, default_value_os_t = default_model_dir(), env = "DSV4_MODEL")]
        model: PathBuf,
        /// Output file (default: experts.pack next to the shards, where it is found by itself).
        /// Needs --expert-pack when it is anywhere else, such as a faster drive
        #[arg(long)]
        out: Option<PathBuf>,
        /// Pack only the first N layers; the others are still read from the shards
        #[arg(long)]
        layers: Option<usize>,
    },
    /// Measure expert streaming throughput from disk
    BenchIo {
        #[arg(long, short, default_value_os_t = default_model_dir(), env = "DSV4_MODEL")]
        model: PathBuf,
        /// Experts to read per run
        #[arg(long, default_value_t = 200)]
        n: usize,
        /// Read only experts of the first N layers
        #[arg(long)]
        layers: Option<usize>,
        /// As for `run`
        #[arg(long)]
        expert_pack: Option<PathBuf>,
    },
    /// Show GGUF metadata and tensor summary
    Info {
        #[arg(long, short, default_value_os_t = default_model_dir(), env = "DSV4_MODEL")]
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


/// The expert pack to read: the one asked for, else one next to the model if it matches.
fn find_pack(asked: Option<&Path>, first: &Path, g: &gguf::Gguf, cfg: &config::Config) -> Result<Option<PathBuf>> {
    let path = match asked {
        Some(p) if p.as_os_str() == "none" => return Ok(None),
        Some(p) => p.to_path_buf(),
        None => pack::default_path(first),
    };
    if asked.is_none() && !path.exists() {
        return Ok(None);
    }
    let (geoms, _) = experts::layer_geoms(g, cfg)?;
    match pack::open(&path, g, &geoms, cfg.n_experts, 0) {
        Ok(_) => {
            tracing::info!("reading experts from {}", path.display());
            Ok(Some(path))
        }
        Err(e) if asked.is_none() => {
            tracing::warn!("not using the expert pack: {e:#}");
            Ok(None)
        }
        Err(e) => Err(e),
    }
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
    let counts = match &args.expert_profile {
        Some(p) => Some(experts::load_profile(p).with_context(|| format!("reading expert profile {}", p.display()))?),
        None => None,
    };
    let expert_pack = {
        let g = gguf::Gguf::open(&first)?;
        let cfg = config::Config::from_gguf(&g)?;
        find_pack(args.expert_pack.as_deref(), &first, &g, &cfg)?
    };
    let expert_subset = args.expert_subset.and_then(|n| counts.as_ref().map(|c| experts::top_experts(c, n)));
    let opt = ModelOptions {
        max_ctx: args.ctx,
        chunk: args.chunk,
        seg: args.segment,
        expert_vram_bytes: vram,
        expert_ram_bytes: ram,
        expert_pack,
        backend: args.backend.clone(),
        cpu_experts: !args.gpu_only_experts,
        expert_subset,
        expert_ranking: counts.as_ref().map(|c| experts::rank_counts(c)),
        route_prior: args.route_prior.max(0.0),
        route_tail_skip: args.route_tail_skip,
        prompt_skip: !args.no_prompt_skip,
        prefetch: !args.no_prefetch,
        warm_up: !args.no_warm_up,
    };
    if args.route_prior > 0.0 || args.route_tail_skip > 0 {
        stats.event(format!(
            "LOSSY routing is on (route-prior {}, route-tail-skip {}): output depends on which experts are cached",
            args.route_prior, args.route_tail_skip
        ));
    }
    let mut model = Model::load(&first, &opt, stats.clone())?;
    model.profile_out = args.save_expert_profile.clone();
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
    engine.submit(Request { prompt, params: p, tx, cancel: Arc::new(AtomicBool::new(false)), endpoint: "cli", id: 0 });
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
        Cmd::BenchIo { model, n, layers, expert_pack } => {
            init_logging(false);
            let first = gguf::resolve_model_path(&model)?;
            let g = gguf::Gguf::open(&first)?;
            let cfg = config::Config::from_gguf(&g)?;
            let stats = Arc::new(Stats::default());
            let gpu = gpu::Gpu::new(stats.clone(), None)?;
            let store = experts::ExpertStore::new(&g, &gpu, &cfg, 1, 0, None, find_pack(expert_pack.as_deref(), &first, &g, &cfg)?.as_deref(), stats)?;
            for threads in [1, 4, 8, 16, 32] {
                let (gbps, ms) = store.bench_disk(n, threads, layers.unwrap_or(usize::MAX))?;
                println!("{threads:>3} loaders: {gbps:>6.2} GB/s  ({ms:.2} ms/expert amortized, 18.8 MB each)");
            }
            Ok(())
        }
        Cmd::Repack { model, out, layers } => {
            init_logging(false);
            let first = gguf::resolve_model_path(&model)?;
            let g = gguf::Gguf::open(&first)?;
            let cfg = config::Config::from_gguf(&g)?;
            let (geoms, parts) = experts::layer_geoms(&g, &cfg)?;
            pack::repack(&g, &geoms, parts, layers.unwrap_or(cfg.n_layers), cfg.n_experts, &out.unwrap_or_else(|| pack::default_path(&first)))
        }
        Cmd::BenchCpu { dim, inter, types } => {
            let mut t = [0u32; 3];
            for (i, name) in types.split(',').enumerate().take(3) {
                t[i] = (0..64).find(|&t| gguf::type_name(t).eq_ignore_ascii_case(name.trim())).filter(|&t| t == gguf::GGML_MXFP4 || quants::is_low_bit(t)).with_context(|| format!("{name}: not an expert type"))?;
            }
            if t.contains(&0) {
                anyhow::bail!("--types takes three types");
            }
            cpu_moe::bench(dim, inter, t)
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
        Cmd::Serve { model, host, port, tui, no_ui, no_think, model_id } => {
            init_logging(tui);
            let stats = Arc::new(Stats::default());
            let listen = format!("{host}:{port}");
            let rt = tokio::runtime::Runtime::new()?;
            let st = stats.clone();
            let ma = model.clone();
            let mid = model_id.clone();
            let l2 = listen.clone();
            // serve while loading, so the status page can show progress
            let server = std::thread::spawn(move || -> Result<()> {
                let run = || -> Result<()> {
                    let listener = rt.block_on(tokio::net::TcpListener::bind(&l2)).with_context(|| format!("listening on {l2}"))?;
                    let state = Arc::new(server::AppState {
                        model_id: mid.clone(),
                        stats: st.clone(),
                        monitor: monitor::Monitor::start(st.clone()),
                        default_thinking: !no_think,
                        ready: Default::default(),
                    });
                    let app = server::router(state.clone(), !no_ui);
                    let task = rt.spawn(async move { axum::serve(listener, app).await });
                    if !no_ui {
                        st.event(format!("web UI at http://{l2}/ (chat) and http://{l2}/status"));
                    }
                    let Loaded { model, tok, template } = load(&ma, st.clone())?;
                    let engine = engine::spawn(model, tok, st.clone(), mid);
                    let _ = state.ready.set(server::Ready { engine, template });
                    st.event(format!("listening on http://{l2}/v1"));
                    st.set_phase("idle");
                    rt.block_on(task)??;
                    Ok(())
                };
                run().inspect_err(|e| {
                    st.set_phase("failed");
                    st.event(format!("{e:#}"));
                })
            });
            if tui {
                tui::run(stats, listen)?;
                std::process::exit(0);
            }
            server.join().map_err(|_| anyhow::anyhow!("server thread panicked"))??;
            Ok(())
        }
        Cmd::Capture { model, layer, out, tokens, doc_tokens, chunks_per_file, list } => {
            use std::io::Write;
            init_logging(false);
            let stats = Arc::new(Stats::default());
            let Loaded { mut model, tok, .. } = load(&model, stats.clone())?;
            let file = std::fs::File::create(&out).with_context(|| format!("creating {}", out.display()))?;
            model.capture = Some(model::Capture { layer, out: std::io::BufWriter::with_capacity(64 << 20, file), stop: true, records: 0 });
            let t0 = std::time::Instant::now();
            let mut total = 0usize;
            'files: for path in std::fs::read_to_string(&list)?.lines().map(str::trim).filter(|l| !l.is_empty()) {
                let Ok(text) = std::fs::read_to_string(path) else {
                    eprintln!("skip {path}: not readable as UTF-8");
                    continue;
                };
                let ids = tok.encode(&text)?;
                for chunk in ids.chunks(doc_tokens).take(chunks_per_file) {
                    if total >= tokens {
                        break 'files;
                    }
                    if chunk.len() < 64 {
                        continue;
                    }
                    model.truncate(0);
                    model.forward(chunk)?;
                    total += chunk.len();
                    let rec = model.capture.as_ref().unwrap().records;
                    eprintln!("{total:>6} tokens  {rec:>7} records  {:>6.0}s  {path}", t0.elapsed().as_secs_f64());
                }
            }
            let mut cap = model.capture.take().unwrap();
            cap.out.flush()?;
            eprintln!("captured {} records over {total} tokens of layer {layer} -> {}", cap.records, out.display());
            Ok(())
        }
        Cmd::DistillPilot { model, layer, capture, ratios, methods, emit, emit_path, max_experts } => {
            let first = gguf::resolve_model_path(&model)?;
            let g = gguf::Gguf::open(&first)?;
            let cfg = config::Config::from_gguf(&g)?;
            let methods = methods.iter().map(|m| distill::Method::parse(m)).collect::<Result<Vec<_>>>()?;
            let emit = match &emit {
                Some(s) => {
                    let (m, r) = s.split_once(':').context("--emit takes METHOD:RATIO")?;
                    let path = emit_path.as_deref().context("--emit needs --emit-path")?;
                    Some((distill::Method::parse(m)?, r.parse::<usize>()?, path))
                }
                None => None,
            };
            distill::pilot(&g, &cfg, &distill::PilotArgs { layer, capture: &capture, ratios, methods, emit, max_experts })
        }
        Cmd::Eval { model, text, prefill, tokens, save, base } => {
            init_logging(false);
            let stats = Arc::new(Stats::default());
            let Loaded { model, tok, .. } = load(&model, stats.clone())?;
            eval(model, &tok, &stats, &text, prefill, tokens, save.as_deref(), base.as_deref())
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

#[allow(clippy::too_many_arguments)]
fn eval(mut model: Model, tok: &Tokenizer, stats: &Stats, texts: &[PathBuf], prefill: usize, tokens: usize, save: Option<&Path>, base: Option<&Path>) -> Result<()> {
    use std::io::Read;
    let log_softmax = |l: &[f32]| -> Vec<f64> {
        let max = l.iter().copied().fold(f32::MIN, f32::max) as f64;
        let z = l.iter().map(|&x| (x as f64 - max).exp()).sum::<f64>().ln() + max;
        l.iter().map(|&x| x as f64 - z).collect()
    };
    let argmax = |l: &[f32]| (0..l.len()).max_by(|&a, &b| l[a].total_cmp(&l[b])).unwrap_or(0);
    let mut out = save.map(std::fs::File::create).transpose()?.map(std::io::BufWriter::new);
    let mut reference = base.map(std::fs::File::open).transpose()?.map(std::io::BufReader::new);
    let (mut n, mut nll, mut kld, mut same, mut worst) = (0usize, 0f64, 0f64, 0usize, 0f64);
    let (mut t_pre, mut t_dec, mut n_dec) = (0f64, 0f64, 0usize);
    for path in texts {
        let ids = tok.encode(&std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?)?;
        if ids.len() < prefill + tokens {
            anyhow::bail!("{} has only {} tokens", path.display(), ids.len());
        }
        model.truncate(0);
        let t0 = std::time::Instant::now();
        let mut logits = model.forward(&ids[..prefill])?;
        t_pre += t0.elapsed().as_secs_f64();
        for i in 0..tokens {
            let lp = log_softmax(&logits);
            nll -= lp[ids[prefill + i] as usize];
            n += 1;
            if let Some(f) = &mut out {
                f.write_all(bytemuck::cast_slice(&logits))?;
            }
            if let Some(f) = &mut reference {
                let mut r = vec![0f32; logits.len()];
                f.read_exact(bytemuck::cast_slice_mut(&mut r)).context("reference logits: too short for this run")?;
                let rp = log_softmax(&r);
                let d: f64 = rp.iter().zip(&lp).map(|(a, b)| a.exp() * (a - b)).sum();
                kld += d;
                worst = worst.max(d);
                same += (argmax(&r) == argmax(&logits)) as usize;
            }
            if i + 1 < tokens {
                let t0 = std::time::Instant::now();
                logits = model.forward(&ids[prefill + i..prefill + i + 1])?;
                t_dec += t0.elapsed().as_secs_f64();
                n_dec += 1;
            }
        }
    }
    if let Some(f) = &mut out {
        f.flush()?;
    }
    println!(
        "eval: {n} tokens  ppl {:.4}  prefill {:.1} s ({:.2} tok/s)  decode {:.2} tok/s",
        (nll / n as f64).exp(),
        t_pre,
        (prefill * texts.len()) as f64 / t_pre,
        n_dec as f64 / t_dec.max(1e-9)
    );
    if reference.is_some() {
        println!("vs reference: kld {:.5} (worst {:.4})  same top-1 {:.1}%", kld / n as f64, worst, 100.0 * same as f64 / n as f64);
    }
    print_breakdown(stats);
    Ok(())
}

fn print_breakdown(st: &Stats) {
    let g = |c: &std::sync::atomic::AtomicU64| Stats::get(c);
    let s = |c: &std::sync::atomic::AtomicU64| Stats::get(c) as f64 / 1e9;
    eprintln!(
        "breakdown: gpu-wait {:.2}s  experts {:.2}s (gpu path {:.2}s, cpu path {:.2}s, computing {:.2}s)  disk {:.2} GB in {} loads  ram hits {}  vram hits {}  cpu runs {}  uploads {} ({:.2} GB)  prefetches {} ({} used, {} cancelled)",
        s(&st.t_gpu_wait_ns),
        s(&st.t_moe_ns),
        s(&st.t_expert_ns),
        s(&st.t_cpu_moe_ns),
        s(&st.t_cpu_compute_ns),
        g(&st.disk_bytes) as f64 / 1e9,
        g(&st.expert_disk_loads),
        g(&st.expert_ram_hits),
        g(&st.expert_vram_hits),
        g(&st.expert_cpu_runs),
        g(&st.expert_uploads),
        g(&st.upload_bytes) as f64 / 1e9,
        g(&st.expert_prefetches),
        g(&st.expert_prefetch_hits),
        g(&st.expert_prefetch_cancels),
    );
    eprintln!(
        "routing: {} picks moved by the prior, {} dropped; {} prompt token-layers skipped",
        g(&st.route_moved),
        g(&st.route_dropped),
        g(&st.prompt_rows_skipped),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn models_are_kept_in_the_users_own_folder() {
        let home = std::env::temp_dir().join("home");
        let none = |_: &str| None;
        assert!(models_dir(none, Some(home.clone())).starts_with(&home));
        assert!(models_dir(none, Some(home.clone())).ends_with("dsv4/models"));
        assert_eq!(models_dir(none, None), PathBuf::from("models"));
        let set = |n: &str| (n == "DSV4_MODELS_DIR").then(|| PathBuf::from("big/drive"));
        assert_eq!(models_dir(set, Some(home.clone())), PathBuf::from("big/drive"));
        // empty and relative settings count as unset
        let empty = |_: &str| Some(PathBuf::new());
        assert_eq!(models_dir(empty, Some(home.clone())), models_dir(none, Some(home)));
    }
}
