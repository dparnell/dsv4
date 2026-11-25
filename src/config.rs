//! Model hyper-parameters, read from the GGUF metadata (`deepseek41.*`).

use anyhow::{Result, bail};

use crate::gguf::Gguf;

#[derive(Debug, Clone, Default)]
pub struct Config {
    pub n_layers: usize,
    pub dim: usize,
    pub vocab: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub q_lora_rank: usize,
    pub o_groups: usize,
    pub o_lora_rank: usize,
    pub window: usize,
    pub n_experts: usize,
    pub n_active: usize,
    pub moe_inter: usize,
    pub route_scale: f32,
    pub norm_eps: f32,
    pub swiglu_limit: f32,
    pub rope_theta: f32,
    pub compress_rope_theta: f32,
    pub rope_factor: f32,
    pub rope_orig_ctx: usize,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub compress_ratios: Vec<usize>,
    pub kv_source_layers: Vec<usize>,
    pub index_source_layers: Vec<usize>,
    pub index_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    pub candidate_source: Option<usize>,
    pub candidate_topk_blocks: usize,
    pub candidate_block: usize,
    pub hc_mult: usize,
    pub hc_iters: usize,
    pub hc_eps: f32,
    // engram
    pub engram_layers: Vec<usize>,
    pub engram_ngram: usize,
    pub engram_heads: usize,
    pub engram_head_dim: usize,
    pub engram_multipliers: Vec<i64>,
    pub engram_primes: Vec<i64>,
    pub engram_pad_token: u32,
    pub engram_token_map: Vec<i64>,
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let a = g.str("general.architecture").unwrap_or("");
        if a != "deepseek41" {
            bail!("unsupported architecture '{a}' (expected deepseek41)");
        }
        // Converters disagree about key names. Aliases after the first: llama.cpp forks, then
        // Hugging Face's names as ds4 keeps them.
        let key = |names: &[&str]| -> String {
            let full = |s: &&str| format!("deepseek41.{s}");
            names.iter().map(full).find(|k| g.get(k).is_some()).unwrap_or_else(|| full(&names[0]))
        };
        let u = |names: &[&str]| g.u64(&key(names)).map(|v| v as usize);
        let f = |names: &[&str]| g.f64(&key(names)).map(|v| v as f32);
        let ints = |names: &[&str]| g.arr_i64(&key(names));
        let arr = |names: &[&str]| -> Result<Vec<usize>> { Ok(ints(names)?.into_iter().map(|v| v as usize).collect()) };
        let n_layers = u(&["block_count", "num_hidden_layers"])?;
        let vocab = g.get("tokenizer.ggml.tokens").and_then(|v| v.as_arr()).map(|a| a.len()).unwrap_or(129280);
        let swiglu = g.arr_f64(&key(&["swiglu_clamp_exp"])).ok().and_then(|v| v.first().copied()).map(|v| v as f32);
        let swiglu = swiglu.or_else(|| f(&["swiglu_limit"]).ok()).unwrap_or(10.0);
        // without the lists of source layers, those are the ones that have the weights
        let with = |t: &str| (0..n_layers).filter(|l| g.has(&format!("blk.{l}.{t}.weight"))).collect::<Vec<_>>();
        let listed = arr(&["attention.kv_source_layers", "attention.kv_source_layer_ids", "kv_source_layer_ids"]).ok();
        let derived = listed.is_none();
        let kv_source_layers = listed.unwrap_or_else(|| with("attn_compressor_kv"));
        let index_source_layers = arr(&["attention.indexer.source_layers", "attention.index_source_layer_ids", "index_source_layer_ids"])
            .unwrap_or_else(|_| with("indexer.attn_q_b"));
        // files that lack those also lack the two-level search, which the model always has
        let mut candidate_source = u(&["attention.indexer.candidate_source_layer", "attention.candidate_source_layer_id", "candidate_source_layer_id"]).ok();
        let mut candidate_topk_blocks = u(&["attention.indexer.candidate_top_k_blocks", "attention.candidate_topk_blocks", "candidate_topk_blocks"]).unwrap_or(0);
        let mut candidate_block = u(&["attention.indexer.candidate_block_size", "attention.candidate_block_size", "candidate_block_size"]).unwrap_or(1);
        if derived && candidate_source.is_none() {
            (candidate_source, candidate_topk_blocks, candidate_block) = (kv_source_layers.iter().copied().max(), 2048, 8);
        }
        let engram_layers = arr(&["ple.layers", "engram.layer_ids", "engram.layers"]).unwrap_or_default();
        let int_list = |names: &[&str], tensor: &str| ints(names).ok().or_else(|| g.int_tensor(tensor)).unwrap_or_default();
        let engram_primes = int_list(&["ple.head_vocab_sizes", "engram.primes"], "engram_primes.weight");
        let (ngram, heads) = if engram_layers.is_empty() { (0, 0) } else { (4, 8) };
        Ok(Self {
            n_layers,
            dim: u(&["embedding_length", "hidden_size"])?,
            vocab,
            n_heads: u(&["attention.head_count", "num_attention_heads"])?,
            head_dim: u(&["attention.key_length", "head_dim"])?,
            rope_dim: u(&["rope.dimension_count", "qk_rope_head_dim"])?,
            q_lora_rank: u(&["attention.q_lora_rank", "q_lora_rank"])?,
            o_groups: u(&["attention.output_group_count", "o_groups"])?,
            o_lora_rank: u(&["attention.output_lora_rank", "o_lora_rank"])?,
            window: u(&["attention.sliding_window", "sliding_window"])?,
            n_experts: u(&["expert_count", "n_routed_experts"])?,
            n_active: u(&["expert_used_count", "num_experts_per_tok"])?,
            moe_inter: u(&["expert_feed_forward_length", "moe_intermediate_size"])?,
            route_scale: f(&["expert_weights_scale", "routed_scaling_factor"]).unwrap_or(1.5),
            norm_eps: f(&["attention.layer_norm_rms_epsilon", "rms_norm_eps"])?,
            swiglu_limit: swiglu,
            rope_theta: f(&["rope.freq_base", "rope_theta"])?,
            compress_rope_theta: f(&["attention.compress_rope_freq_base", "compress_rope_theta"])?,
            rope_factor: f(&["rope.scaling.factor", "rope_scaling.factor"]).unwrap_or(1.0),
            rope_orig_ctx: f(&["rope.scaling.original_context_length", "rope_scaling.original_max_position_embeddings"]).unwrap_or(0.0) as usize,
            beta_fast: f(&["rope.scaling.yarn_beta_fast", "rope_scaling.beta_fast"]).unwrap_or(32.0),
            beta_slow: f(&["rope.scaling.yarn_beta_slow", "rope_scaling.beta_slow"]).unwrap_or(1.0),
            compress_ratios: arr(&["attention.compress_ratios", "compress_ratios"])?,
            kv_source_layers,
            index_source_layers,
            index_heads: u(&["attention.indexer.head_count", "index_n_heads"])?,
            index_head_dim: u(&["attention.indexer.key_length", "index_head_dim"])?,
            index_topk: u(&["attention.indexer.top_k", "index_topk"])?,
            candidate_source,
            candidate_topk_blocks,
            candidate_block,
            hc_mult: u(&["hyper_connection.count", "hc_mult"])?,
            hc_iters: u(&["hyper_connection.sinkhorn_iterations", "hc_sinkhorn_iters"])?,
            hc_eps: f(&["hyper_connection.epsilon", "hc_eps"])?,
            engram_layers,
            engram_ngram: u(&["ple.ngram_size", "engram.max_ngram_size"]).unwrap_or(ngram),
            engram_heads: u(&["ple.heads_per_ngram", "engram.head_count", "engram.n_heads"]).unwrap_or(heads),
            engram_head_dim: u(&["embedding_length_per_layer_input", "engram.key_length", "engram.head_dim"]).unwrap_or(if heads > 0 { 256 } else { 0 }),
            engram_multipliers: int_list(&["ple.layer_multipliers", "engram.multipliers"], "engram_multipliers.weight"),
            engram_primes,
            engram_pad_token: u(&["engram.pad_token_id", "engram.pad_id"]).unwrap_or(2) as u32,
            engram_token_map: int_list(&["engram.token_map"], "engram_token_map.weight"),
        })
    }

    pub fn ratio(&self, layer: usize) -> usize {
        self.compress_ratios[layer]
    }

    /// The layer whose compressed KV / index keys this layer reads (the last kv source <= layer).
    pub fn kv_source_for(&self, layer: usize) -> Option<usize> {
        if self.ratio(layer) == 0 {
            return None;
        }
        self.kv_source_layers.iter().copied().filter(|&s| s <= layer).max()
    }

    pub fn is_kv_source(&self, layer: usize) -> bool {
        self.kv_source_layers.contains(&layer)
    }

    pub fn is_index_source(&self, layer: usize) -> bool {
        self.index_source_layers.contains(&layer)
    }

    /// First layer from which a position's output only matters to the positions shortly after it:
    /// the last compressed-KV source, when no later layer has Engram input. Everything past it
    /// sees earlier positions through that layer's compressed rows and the sliding window alone.
    pub fn skip_layer(&self) -> Option<usize> {
        let y = self.kv_source_layers.iter().copied().max()?;
        (y > 0 && (y..self.n_layers).all(|l| self.ratio(l) > 0) && self.engram_layers.iter().all(|&l| l < y)).then_some(y)
    }

    pub fn n_hash_cols(&self) -> usize {
        (self.engram_ngram - 1) * self.engram_heads
    }
}
