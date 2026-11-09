//! Model hyper-parameters, read from the GGUF metadata (`deepseek41.*`).

use anyhow::{Result, bail};

use crate::gguf::Gguf;

#[derive(Debug, Clone)]
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
    pub engram_offsets: Vec<i64>,
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
        let k = |s: &str| format!("deepseek41.{s}");
        let u = |s: &str| g.u64(&k(s)).map(|v| v as usize);
        let f = |s: &str| g.f64(&k(s)).map(|v| v as f32);
        let arr = |s: &str| -> Result<Vec<usize>> { Ok(g.arr_i64(&k(s))?.into_iter().map(|v| v as usize).collect()) };
        let n_layers = u("block_count")?;
        let vocab = g.get("tokenizer.ggml.tokens").and_then(|v| v.as_arr()).map(|a| a.len()).unwrap_or(129280);
        let swiglu = g.arr_f64(&k("swiglu_clamp_exp")).ok().and_then(|v| v.first().copied()).unwrap_or(10.0) as f32;
        let cand = g.u64(&k("attention.indexer.candidate_source_layer")).ok().map(|v| v as usize);
        Ok(Self {
            n_layers,
            dim: u("embedding_length")?,
            vocab,
            n_heads: u("attention.head_count")?,
            head_dim: u("attention.key_length")?,
            rope_dim: u("rope.dimension_count")?,
            q_lora_rank: u("attention.q_lora_rank")?,
            o_groups: u("attention.output_group_count")?,
            o_lora_rank: u("attention.output_lora_rank")?,
            window: u("attention.sliding_window")?,
            n_experts: u("expert_count")?,
            n_active: u("expert_used_count")?,
            moe_inter: u("expert_feed_forward_length")?,
            route_scale: f("expert_weights_scale").unwrap_or(1.5),
            norm_eps: f("attention.layer_norm_rms_epsilon")?,
            swiglu_limit: swiglu,
            rope_theta: f("rope.freq_base")?,
            compress_rope_theta: f("attention.compress_rope_freq_base")?,
            rope_factor: f("rope.scaling.factor").unwrap_or(1.0),
            rope_orig_ctx: u("rope.scaling.original_context_length").unwrap_or(0),
            beta_fast: f("rope.scaling.yarn_beta_fast").unwrap_or(32.0),
            beta_slow: f("rope.scaling.yarn_beta_slow").unwrap_or(1.0),
            compress_ratios: arr("attention.compress_ratios")?,
            kv_source_layers: arr("attention.kv_source_layers")?,
            index_source_layers: arr("attention.indexer.source_layers")?,
            index_heads: u("attention.indexer.head_count")?,
            index_head_dim: u("attention.indexer.key_length")?,
            index_topk: u("attention.indexer.top_k")?,
            candidate_source: cand,
            candidate_topk_blocks: u("attention.indexer.candidate_top_k_blocks").unwrap_or(0),
            candidate_block: u("attention.indexer.candidate_block_size").unwrap_or(1),
            hc_mult: u("hyper_connection.count")?,
            hc_iters: u("hyper_connection.sinkhorn_iterations")?,
            hc_eps: f("hyper_connection.epsilon")?,
            engram_layers: arr("ple.layers").unwrap_or_default(),
            engram_ngram: u("ple.ngram_size").unwrap_or(0),
            engram_heads: u("ple.heads_per_ngram").unwrap_or(0),
            engram_head_dim: u("embedding_length_per_layer_input").unwrap_or(0),
            engram_multipliers: g.arr_i64(&k("ple.layer_multipliers")).unwrap_or_default(),
            engram_offsets: g.arr_i64(&k("ple.head_offsets")).unwrap_or_default(),
            engram_primes: g.arr_i64(&k("ple.head_vocab_sizes")).unwrap_or_default(),
            engram_pad_token: u("engram.pad_token_id").unwrap_or(2) as u32,
            engram_token_map: g.arr_i64(&k("engram.token_map")).unwrap_or_default(),
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

    pub fn n_hash_cols(&self) -> usize {
        (self.engram_ngram - 1) * self.engram_heads
    }
}
