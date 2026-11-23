# dsv4 — DeepSeek-V4.1-Flash on consumer GPUs (Rust + wgpu)

`dsv4` runs [unsloth/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF)
(748B parameters, ~16B active) on a single consumer GPU by keeping only the dense part of the model in VRAM
and streaming the routed experts that each token actually uses.

* GPU compute through **wgpu** (Vulkan / DX12 / Metal), all kernels in WGSL
* **OpenAI-compatible HTTP API** (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, streaming SSE,
  `reasoning_content`, tool calls)
* **Built-in downloader** with parallel, resumable range requests
* Optional **TUI** with live tokens/s, memory, CPU/GPU utilisation, PCIe/disk transfer rates and expert-cache behaviour

## How it fits on consumer hardware

The UD-Q4_K_XL quant is 501 GB on disk, made of three very different parts:

| part | size | where it lives |
|---|---|---|
| attention, shared experts, routers, hyper-connections, head (Q8_0) | ~8.5 GB | **VRAM**, resident |
| 40 × 384 routed experts (MXFP4, 18.8 MB each) | 289 GB | **streamed**: VRAM slots → RAM cache → NVMe |
| Engram n-gram tables (MXFP8, 2 layers) | 203 GB | **memory-mapped**; ~6 KB read per token per layer |

Per token the router picks 6 of 384 experts in each of 40 layers (≈4.5 GB of expert weights), so expert
movement dominates. `dsv4` uses a three-tier expert store:

1. **VRAM slots** — fixed-size slots in a few slab buffers. Sized automatically from the free
   VRAM after the dense weights are resident (`--expert-vram-gb` to override).
2. **RAM tier** — experts already repacked into the GPU layout, with a byte budget
   (default 75% of system RAM, `--ram-cache-gb`).
3. **NVMe** — unbuffered positioned reads straight from the GGUF shards (no double caching in the OS page cache),
   one whole matrix per request. Reads a token is waiting for go before speculative ones.

Both caches replace the expert with the lowest decayed use count (half-life 16 tokens), starting from a ranking of
experts by routing frequency (built in, from [Basalt](https://github.com/jesdga95/basalt), MIT; or `--expert-profile`).
While the disk is idle after loading, the caches are filled in that order (`--no-warm-up` disables this).

On top of that:

* **Hybrid CPU/GPU experts.** During decode a cold expert is used by one token, so instead of pushing 18.8 MB
  over PCIe it is computed on the CPU (MXFP4×int8 dot products; AVX2, AVX-VNNI or AVX-512 VNNI) directly from the RAM tier, concurrently
  with the GPU computing the resident experts. Experts that keep getting picked, and are clearly
  hotter than the one they would replace, are then copied to VRAM in the background.
  (`--gpu-only-experts` disables this.)
* **Speculative prefetch.** After routing layer *l*, layer *l+1*'s router is applied to layer *l*'s FFN input on
  the CPU and the likely experts are read from disk in the background while the GPU runs attention. Reads
  that have not started when the real routing is known are dropped (`--no-prefetch` disables prefetching).
* **Layer-major prefill.** Prompts are processed in segments (`--segment`, default 1024 tokens): every token of a
  segment passes layer *l* before any passes layer *l+1*, so each routed expert is fetched at most once per layer
  per segment instead of once per token.
* **Prompt skip.** Past the last compressed-KV layer (20 of 40) a position only matters to those shortly after it, so
  early positions of long prompts stop there (`--no-prompt-skip` disables this). A later request that diverges from the
  prompt earlier than its last eighth (16-512 tokens) recomputes the new part of the previous prompt.
* **Prefix reuse.** The KV state of the previous request is kept; a new request that shares its prefix
  (multi-turn chat) only prefills the new tokens.
* The model's own KV-compression (128-token sliding window + 2:1 / 1:1 compressed KV with a sparse top-512
  indexer) keeps the KV cache tiny, so long contexts cost little VRAM.

### Lossy routing (off by default)

Both flags change the output of decoding, depending on which experts happen to be cached. Measured costs:
[experiments/RESULTS.md](experiments/RESULTS.md).

* `--route-tail-skip [R]` (default 5): drop picks of weight rank R or lower that would need the disk.
* `--route-prior [LAMBDA]` (default 0.1): pick experts as if those in memory scored LAMBDA higher.

### What speed to expect

Decode speed is bounded by how many expert bytes per token miss the faster tiers. On the development machine
(RTX 5080 16 GB with ~4 GB used by the desktop, 128 GB RAM, budget NVMe):

* experts from RAM on the CPU: ~0.5 ms each (6 in parallel, ~35 GB/s)
* experts from NVMe: ~25 ms each on cold data (a PNY CS2241 sustaining ~0.75 GB/s; check yours with `bench-io`)
* the whole model's experts (289 GB) do not fit in RAM, so the steady-state speed depends on how skewed expert
  usage is for your workload; the TUI shows the hit rates per tier and per layer.

### Smaller quants

Routed experts may also be IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, Q2_K or Q3_K (mixed per layer), and the
other weights any type Unsloth's UD-IQ1_S to UD-Q3_K_XL use, so those quants are meant to load; they have not been
run end to end yet. They need less disk and fit more experts in RAM, at a cost in quality (Basalt measured KLD 0.24
against BF16 for UD-IQ2_XXS and 0.28 for UD-IQ1_M). `download --quant UD-IQ2_XXS`, then pass that folder as `--model`.

* RAM and packs hold the compact GGUF bytes; the CPU decodes rows as it multiplies them. In VRAM they become
  4-bit codes plus scales (about 20 MB per expert), without further rounding.
* Dense weights that are not Q8_0 are requantized to Q8_0 when loaded.

## Quick start

```bash
cargo build --release
# 1. download the 4-bit quant (~501 GB) + tokenizer
target/release/dsv4 download --quant UD-Q4_K_XL --dir F:/models/DeepSeek-V4.1-Flash-1017-GGUF
# 2. check the GPU kernels on your adapter
target/release/dsv4 selftest
# 3. serve with the dashboard
target/release/dsv4 serve --model F:/models/DeepSeek-V4.1-Flash-1017-GGUF --tui
```

Then point any OpenAI client at `http://127.0.0.1:8080/v1`:

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "content-type: application/json" -d '{
  "model": "deepseek-v4.1-flash",
  "messages": [{"role": "user", "content": "Explain mixture-of-experts in two sentences."}],
  "stream": true
}'
```

## Commands

| command | purpose |
|---|---|
| `download` | fetch a quant (`--quant`, `--list` to see all) with `--connections` parallel ranges; resumable; set `HF_TOKEN` if needed |
| `serve` | OpenAI-compatible server (`--host`, `--port`, `--tui`, `--no-think`, `--model-id`) |
| `run -p "..."` | one completion to stdout (`--raw` skips the chat template, `--tui` shows the dashboard) |
| `chat` | interactive terminal chat with history (`/reset`) |
| `prompt file.json` | render an OpenAI messages file with the model's chat template |
| `info` | GGUF metadata, tensor types and model configuration |
| `selftest` | every WGSL kernel (and the CPU expert path) against a CPU reference |
| `eval` | teacher-forced perplexity and speed on text files; `--save` / `--base` compare two configurations |
| `repack` | copies the experts into one file (`experts.pack`, 289 GB for Q4_K_XL) with each expert contiguous, so a miss is one 19 MB read instead of three 6 MB ones; lossless, resumable, found by itself next to the model, `--expert-pack` elsewhere; `--layers N` for part of them |
| `bench-io` | expert streaming throughput from disk |
| `bench-cpu` | speed of routed experts on the CPU by thread count (no model needed); `DSV4_CPU_ISA=avx2\|avxvnni\|avx512vnni` picks the kernel |

Common model options: `--ctx` (context, default 32768), `--segment`, `--chunk`, `--expert-vram-gb`,
`--ram-cache-gb`, `--backend vulkan|dx12|metal`, `--gpu-only-experts`, `--no-prefetch`, `--no-warm-up`, `--tokenizer`.

## API notes

* **Thinking mode** is on by default (the model's native mode). Turn it off per request with
  `"thinking": false`, `"thinking": {"type": "disabled"}`, `"reasoning_effort": "none"` or
  `"chat_template_kwargs": {"thinking": false}`, or server-wide with `serve --no-think`. The reasoning is returned
  as `reasoning_content` (streamed as `delta.reasoning_content`).
* `reasoning_effort` accepts `low` / `medium` / `high` / `max` or 1–100.
* **Tools**: pass OpenAI `tools`; the model's DSML tool-call blocks are parsed back into `tool_calls`.
* Sampling: `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`, `presence_penalty`,
  `frequency_penalty`, `seed`, `max_tokens`, `stop`. DeepSeek recommends temperature 1.0, top_p 1.0.
* Requests are served one at a time (single sequence); others queue. `GET /metrics` returns the counters
  shown in the TUI as JSON.

Prompt formatting uses the chat template embedded in the GGUF (rendered with minijinja); it reproduces the
golden outputs of DeepSeek's reference `encoding.py` exactly.

## Layout

```
src/
  main.rs        CLI
  download.rs    resumable parallel Hugging Face downloader
  gguf.rs        multi-shard GGUF reader (mmap)
  config.rs      hyper-parameters from GGUF metadata
  gpu.rs         wgpu device, buffers, pipelines, bind-group cache, batches, readback
  kernels.rs     WGSL kernels (Q8_0/MXFP4 matmuls, sparse attention, indexer top-k, RoPE/YaRN,
                 hyper-connection Sinkhorn mixing, Engram gating, quantisation simulation, ...)
  weights.rs     dense weight loading / repacking
  experts.rs     three-tier routed-expert store (VRAM slots, RAM LRU, NVMe)
  pack.rs        expert packs (`repack`)
  quants.rs      decoders for ggml's low-bit formats
  cpu_moe.rs     MXFP4 expert execution on the CPU (AVX2 / VNNI kernels)
  engram.rs      n-gram hashing and MXFP8 row gathers
  model.rs       the forward pass (layer-major segments, chunked attention, hybrid MoE)
  engine.rs      request queue, prefix reuse, sampling
  chat.rs        tokenizer, chat template, reasoning / tool-call parsing
  server.rs      OpenAI-compatible HTTP API
  tui.rs         ratatui dashboard
  selftest.rs    kernel verification
```

## Model notes

The forward pass follows DeepSeek's reference `inference/model.py` for V4.1: hyper-connections with 4 residual
copies mixed by Sinkhorn-normalised matrices, low-rank Q and grouped low-rank output projections, MQA over a
512-dim latent with attention sinks, compressed KV (2:1 for layers 2–19, 1:1 from layer 20) shared across layer
groups, a two-level sparse indexer (candidate blocks from layer 20, top-512 positions), YaRN RoPE for compressed
layers, Engram n-gram embeddings at layers 1 and 14, and `sqrt(softplus)` routing with a selection bias. The KV
caches reproduce the reference's FP8 / FP4 quantisation of cached values. The vision tower and the DSpark
speculative drafter are not used.
