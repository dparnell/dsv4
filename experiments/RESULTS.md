# Pinned expert subsets (router restricted to a fixed set)

Profile: `expert_profile.json` from `calibrate.sh` (4 prompts x ~190 tokens). Per layer the
router used 266-320 of 384 experts; the top 6 / 16 / 48 cover 9-26% / 20-41% / 41-66% of picks.

Prompt: "Write a short paragraph explaining how a refrigerator works.", greedy, 200 tokens,
RTX 5080 (~2.9 GB VRAM free for experts), 128 GB RAM.

| Experts/layer | Pinned (VRAM / RAM) | Prefill tok/s | Decode tok/s | Output |
|---|---|---|---|---|
| all 384 (normal router, 43 GB RAM cache) | - | 0.8 | 0.83 | correct, fluent |
| 48 | 140 / 1780 (36 GB) | 8.3 | 3.14 | on topic but broken reasoning, loops |
| 16 | 140 / 500 (12 GB) | 16.1 | 3.81 | loops on "Need to include the reasoning" |
| 6 (router ignored) | 138 / 102 (4.5 GB) | 32.8 | 5.49 | degenerates to "a a a a ..." |

Conclusion: a static subset trades quality away fast; even 48/384 loses coherence. Speed is
bounded by RAM->GPU uploads and CPU experts since only ~140 experts fit in VRAM.

# Expert distillation pilot (layer 12) — stopped

Capture: `dsv4 capture --layer 12` over `corpus.txt` (65k tokens, 389,772 routed samples,
383/384 experts used). Fit with `dsv4 distill-pilot`, error = RMS relative output error on
held-out samples (every 5th), 4 most-used experts, after requantizing to MXFP4:

| Method | 2x | 4x | 8x |
|---|---|---|---|
| prune (drop hidden units) | 0.46 | 0.64 | 0.76 |
| prune + ridge refit of down | 0.48 | 0.61 | 0.69 |
| per-expert low-rank (activation-aware SVD) | 0.58 | 0.67 | 0.73 |
| layer-shared input/output subspaces | 0.44 | 0.60 | 0.70 |

Reference: the engine's int8-activation CPU path is 0.010. Sanity check at 1x: all methods 0.000.

Conclusion: the experts have little exploitable redundancy (contribution spread evenly over
hidden units; inputs not confined to a small subspace). Even 2x costs ~45% of each expert's
output, so the line of inquiry was stopped before the end-to-end layer swap.

# Expert tier rework (warm start, admission, I/O)

RTX 5080 with 26 expert slots, 36 GB RAM tier, PNY CS2241, 48 greedy tokens (22-token prompt),
two interleaved rounds; the drive's speed varies between runs.

| | Prefill s | Decode tok/s | Disk GB | Uploads GB | VRAM hits |
|---|---|---|---|---|---|
| before | 44.7 / 57.5 | 0.32 / 0.26 | 159 | 79 | 9 |
| after | 53.1 / 34.3 | 0.45 / 0.66 | 122 | 14 | 889 |
| after, `--no-prefetch` | 49.8 / 46.2 | 0.57 / 0.64 | 103 | 14 | 889 |

Idle server, 24 tokens (one run each): warm-up (1,914 experts in 34 s) 0.7 prefill / 0.72 decode
tok/s; `--no-warm-up` 0.4 / 0.51.

- Output with `--gpu-only-experts` (24 greedy tokens) is identical before and after.
- The old admission uploaded 3,741 experts for 9 VRAM hits.
- Trace replay (2 prompts, 110 steps, 1,800 experts in RAM): decode misses LRU 26.5%, decayed use
  count with half-life 16 24.5%; the ranking mainly helps prefill (4,500 experts: 1.00 -> 0.62).
- Prefetching reads ~18% more from disk with no clear speed gain on this drive.
- Cold random reads on this drive: ~0.75 GB/s regardless of queue depth (1 MB requests: 0.4).

# Lossy routing (`--route-prior`, `--route-tail-skip`)

`dsv4 eval`, 3 texts x (64 prompt + 60 teacher-forced tokens, 177 decode steps), 26 VRAM slots,
36 GB RAM tier. "Held" = in VRAM or RAM. Reference: the same build without the flags (a repeat of
it is bit-identical, so the quality noise floor is 0). Later runs (see "CPU experts") show that any change of
placement or kernel alone moves KLD by 0.012-0.025, so the KLDs below 0.03 here are not
distinguishable from that, and one later tail-skip 5 run got 0.75 tok/s against 0.65-0.76
without: treat its speedup as unconfirmed. One run each; the drive's speed varies, so
disk GB (prompts included) is the steadier measure of work.

| | Decode tok/s | Disk GB | PPL | KLD | Worst KLD | Same top-1 |
|---|---|---|---|---|---|---|
| off / repeat | 0.64 / 0.61 | 562 / 563 | 4.910 | 0 | 0 | 100% |
| tail-skip 5 | 1.01 | 491 | 4.895 | 0.0144 | 0.10 | 95.0% |
| prior 0.05 | 0.85 | 478 | 4.887 | 0.0185 | 0.18 | 95.6% |
| prior 0.1 | 0.78 | 428 | 5.039 | 0.0208 | 0.16 | 92.8% |
| prior 0.1 + tail-skip 5 | 1.11 | 410 | 4.956 | 0.0235 | 0.26 | 93.3% |
| tail-skip 3 | 1.80 | 369 | 5.325 | 0.0473 | 1.03 | 91.1% |

# Prompt skip

Only what the last `clamp(len/8, 16, 512)` positions need is computed past layer 20.

- Exact: 490-token prompt (segment 128, chunk 64) + 14 tokens, `--gpu-only-experts`: logits
  bit-identical with and without. On 1,100 tokens + 16 with the deterministic top-k (below), KLD is
  0.00000 with and without the skip.
- Deterministic top-k: the shader used to emit indices, and pick among tied scores, in scheduling
  order. Same 1,100-token test: old shader run to run KLD 0.009; new shader byte-identical logits,
  with 0.015 (worst 0.064, top-1 100%) between old and new, about the size of the old noise. Speed difference not
  measurable over drive variance.
- Work: 4,500 tokens + 32: disk 1,016 -> 757 GB (-25%), uploads 1,118 -> 870 GB. 1,100 tokens: -4%;
  no benefit for short prompts. Wall time there was 691 -> 803 s, but the drive averaged 1.4 vs
  0.9 GB/s in the two runs, so no speedup has been demonstrated.

# CPU experts

Ryzen 9 9950X, DDR5-3600 (~40 GB/s in practice), `bench-cpu`, ms per expert (18.8 MB):

| | Before (AVX2) | New AVX2 | New AVX-512 VNNI |
|---|---|---|---|
| 1 thread, 1 token | 0.97-1.04 | 1.00 | 0.74 |
| 32 threads, 1 expert | 0.73-0.78 | 0.73 | 0.66 (tasks of 128 rows; the rest used 64) |
| 32 threads, 6 experts | 0.53 | 0.56 | 0.55 |
| 32 threads, 6 experts x 4 tokens | 1.45 | 0.63 | 0.65 |

Single-token work is limited by DRAM. Sharing the weights between tokens is the real gain.

- End to end (2 texts x (64 + 60 tokens)): decode 0.76 / 0.76 before, 0.75 / 0.65 after: no gain.
  Computing is 23 s of ~275 s; the CPU path otherwise waits for the disk.
- Quality against `--gpu-only-experts` (42 slots, 30 GB RAM, 120 tokens): old AVX2 KLD 0.022
  (PPL 7.53), new AVX-512 0.024 (7.22), new AVX2 0.025 (7.69); GPU-only PPL 7.29. Equivalent
  within the sample; int8 activations cost ~0.02 either way.
- Repeats of one build differed by KLD 0.012-0.014 when free VRAM changed the slot count, since
  placement decides whether an expert is computed by the CPU or the GPU.

# Expert pack

- Random reads at 16 threads on F:, Q4_K_XL, in randomized order (GB/s, mean of 3): 2.1 MB 0.69,
  2.9 MB 0.88, 6.3 MB 0.96, 7.9 MB 1.09, 18.8 MB 1.40.
- `bench-io --layers 3`, GGUF vs a 3-layer pack, alternating, 16 loaders: 2.39 / 2.51 / 1.22 vs
  3.45 / 3.03 / 3.09 GB/s. The pack had just been written (drive cache), and repeated reads of 3
  layers flatter both. End-to-end speed with a full pack: not measured.
- Logits with and without the pack are bit-identical (gpu-only, 24 + 4 tokens).

# Low-bit experts

Not run end to end: no low-bit model was downloaded.

- Decoders against separate transcriptions of ggml's (random blocks), and those against real
  weights: first 256-2304 values of one tensor per type from Unsloth's files vs the same tensor in
  UD-Q4_K_XL, cosine IQ1_S 0.87, IQ2_XXS 0.94, IQ2_XS 0.95, Q2_K 0.95, IQ2_S 0.97, IQ3_XXS 0.98,
  Q3_K 0.99, IQ3_S 0.99, IQ4_NL 0.997, Q5_K 0.999, Q6_K 0.9998.
- GPU kernels: within 1.2e-6 of the output range of the f32 reference on random blocks.
- CPU, 32 threads, 6 experts (`bench-cpu --types`), ms per expert: MXFP4 0.56,
  IQ2_XXS+IQ2_S 0.42, IQ2_S+IQ3_XXS 0.46, IQ1_S+IQ2_XS 0.40, IQ1_S+Q2_K 0.87 (a single layer of
  UD-IQ2_XXS). One thread: 0.8 vs 4.4, so it needs a fair number of cores.
- UD-Q4_K_XL is unaffected (same perplexity on a 24 + 4 token gpu-only check).
- Simulated from a 300-step routing trace, LRU, 38 GB of RAM, decode: GB read per token
  Q4_K_XL 1.21, IQ3_XXS 0.78, Q2_K_XL 0.58, IQ2_XXS 0.39, IQ1_M 0.27, IQ1_S 0.20.

# Other quants and layouts

Not run with real weights: only headers and a few blocks of these files were fetched.

- Decoders vs the same tensors of UD-Q4_K_XL (first 2048 values): IQ1_M 0.84, IQ4_XS 0.997, Q4_K 0.998,
  Q5_K 0.9994, Q6_K 0.9999 (cosine).
- `engram_k * engram_q` of mrfakename, mxxm-t and antirez equals `engram_gate` exactly or to cosine 1.0;
  per-layer Engram rows match (Q4_K 0.997, MXFP4 0.994, Q6_K 0.9999, antirez byte for byte once reordered).
- Files of zeros with the real headers of 8 quants (mrfakename Q3_K_M and Q4_K_M, Adolphsson, antirez,
  ISTA-DASLab, AtomicChat, mxxm-t, peasantsmith) give the same configuration as UD-Q4_K_XL, load, and
  decode 3 tokens with CPU and with GPU experts.
- UD-Q4_K_XL, gpu-only, 64 + 64 tokens: ppl 1.7131 before, after, and with its metadata rewritten to per-layer
  Engram tables without source-layer lists.
- CPU, 32 threads, 6 experts, ms per expert: MXFP4 0.50, Q4_K+Q6_K 0.64, Q4_K+IQ4_XS 0.61,
  IQ1_M+IQ2_XS 0.47. Q3_K and Q2_K decode faster now: Q3_K+Q5_K 1.92 -> 0.53, IQ1_S+Q2_K 0.65 -> 0.36.

# Time outside the experts

Decode, Ryzen 9 9950X + RTX 5080, per token, from temporary timers and GPU timestamps:

- Was: read-ahead scoring on the CPU 13-17 ms, engram rows read before the first layer 7-33 ms,
  `hc_mix` 14 ms (one workgroup per call, so one GPU core), `mm_q8` 14 ms, submits 7-10 ms. A
  trivial readback is 0.1-0.2 ms, so round trips cost 4-7 ms of 43 ms.
- Now: read-ahead scored on the GPU, engram rows read in the background, `hc_mix` split into
  `hc_dot` (many workgroups) and the rest. All experts in memory (`--expert-subset 32`, 120 tokens,
  3 interleaved runs): decode 5.01 -> 5.70 tok/s, GPU wait (prefill included) 7.6 -> 5.0 s.
- Full model with 36 GB of RAM: no speedup shown (0.99, 0.88 -> 1.00, 0.77 tok/s; disk dominates).
  Read-ahead used 1898 and 1880 before, 1884 and 1853 after.
- `hc_mix` rounds differently. KLD against the old build, GPU-only experts: 0.0079 and 0.033 on two
  texts; the old build with only its summation order reversed: 0.0028 and 0.029. Routing near-ties
  amplify any rounding.

# NUMA-local experts

Rows of every expert divided among the nodes' memories by first touch; threads take rows of their
own node first.

- Two Xeon E5-2699 v4 (AVX2, Windows 10), `bench-cpu`, both sockets, ms per expert (two runs),
  unplaced -> NUMA-local; 99% of pages on the intended node (50% unplaced):
  - 22 threads: 1 expert 0.68 -> 0.44-0.45, 6 experts 0.39 -> 0.28-0.29, 6 x 4 tokens 0.72 -> 0.65-0.69
  - 44 threads: 1 expert 0.45-0.46 -> 0.33-0.34, 6 experts 0.37-0.38 -> 0.27-0.28,
    6 x 4 tokens 0.59-0.60 -> 0.48 (bandwidth 6 experts: 50 -> 68 GB/s)
  - 88 threads: 1 expert 0.53-0.54 -> 0.40-0.41, 6 experts 0.42-0.43 -> 0.31-0.33,
    6 x 4 tokens 0.56-0.70 -> 0.52-0.54
- Before any of this, 44 threads there: 1 expert 0.49, 6 experts 0.36.
- Locked task queues (first version), 44 threads: 1 expert 0.89 unplaced, 0.62 NUMA-local; 88 threads
  1.51 and 1.19. Counters replaced them; on the 9950X the two are the same.
- Server there (RTX 4070 Ti SUPER, 218 GB expert RAM, one prompt repeated, sampled), ms per token
  over the last 100, `3a9194b` -> `34a1736`: CPU experts 72 -> 61, disk wait 14 -> 14, GPU experts
  6 -> 6, rest of GPU 52 -> 53, other 11 -> 12. Requests 1 to 4: 3.42, 5.24, 6.13, 6.34 ->
  3.50, 5.72, 5.84, 6.48 tok/s; fifth, new only: 6.74. About 138 CPU expert runs per token, so
  0.44 ms each: less gain than `bench-cpu` shows (0.28-0.34); why is not known.
- One node (9950X), `bench-cpu` before/after: the same within noise. Windows with `DSV4_NUMA=2` and
  `3` (pretended nodes) and Linux (WSL, one real node, 2 pretended): the same within noise.
- Results are identical bit for bit however the rows are divided (unit test).
- macOS: `cpu.rs` type-checks for aarch64-apple-darwin; the whole program was not built there.

# Threads awake between layers

`bench-cpu --gap-ms 3`: a 3 ms pause before every call, ms per expert.

- Two Xeon E5-2699 v4 (Windows 10), 44 threads, NUMA-local, without -> with the pause: 1 expert
  0.33 -> 0.93, 6 experts 0.27 -> 0.35, about 0.5 ms per call; the High performance power plan
  changes nothing. Asleep -> awake, with the pause: 1 expert 0.89 -> 0.37, 3 experts 0.47 -> 0.26,
  6 experts 0.35 -> 0.25, 6 x 4 tokens 0.53 -> 0.42. 22 threads awake: 0.38, 0.29, 0.25, 0.56.
- Server there (same setup as above), ms per token over the last 100, `34a1736` -> `cbafa28`: CPU
  experts 61 -> 38, disk wait 14 -> 9, GPU experts 6 -> 4, rest of GPU 53 -> 52, other 12 -> 12.
  Requests 1 to 5: 3.50, 5.72, 5.84, 6.48, 6.74 -> 3.87, 6.67, 7.09, 7.70, 8.25 tok/s (sampled, so
  the answers differ).
- Linux in WSL2 (9950X), 16 threads, asleep -> awake: 1 expert 1.85 -> 0.53, 3 experts 0.83 -> 0.50,
  6 experts 0.64 -> 0.52.
- Windows 11 (9950X), 16 threads: 1 expert 0.63 -> 0.51, 3 experts 0.56 -> 0.52, 6 experts 0.57 -> 0.48.
  32 threads on its 32 logical processors: 0.70 -> 3.44, hence `auto`.
- Model there (`--threads 16 --expert-subset 32`, 120 tokens, 3 interleaved runs): no difference
  shown, 5.80, 6.19, 6.40 -> 6.18, 6.07, 6.13 tok/s.

# Low-bit experts on AVX2

UD-IQ2_XXS on the two Xeons above (RTX 4070 Ti SUPER; a user's run, sampled, repeated prompt, `7b964eb`):
5.0-5.2 tok/s from the first request to the seventh, where UD-Q4_K_XL goes from 3.9 to 8.25.

- All 15,360 experts (157 GB) are in RAM after warm-up and nothing waits for the disk, so only the
  VRAM share can still improve. The GPU stores 4-bit codes: 21.4 MB per slot for experts of 10.2 MB
  (MXFP4 18.8 MB), hence 285 slots, 39.7% hits.
- ms per token, UD-Q4_K_XL -> UD-IQ2_XXS: CPU experts 38 -> 94, disk wait 9 -> 0, GPU experts 4 -> 12,
  rest of GPU 52 -> 69, other 12 -> 13. About 143 CPU runs per token at 0.66 ms, against 0.28. Why the
  GPU part is slower was not looked into.
- Row decoders rewritten with AVX2 (same weights, unit test). `bench-cpu`, 9950X limited to AVX2, ms per
  expert, 1 thread / 16 threads with 6 experts, `7b964eb` -> now: IQ2_XXS+IQ2_S 4.6 / 0.59 -> 2.8 / 0.31,
  IQ2_XS 4.9 / 0.62 -> 3.1 / 0.39, IQ2_S+IQ3_XXS 4.7 / 0.61 -> 2.8 / 0.37, IQ3_S 5.4 / 0.68 -> 3.5 / 0.46,
  IQ1_S+Q2_K 3.9 / 0.50 -> 2.9 / 0.32, IQ1_M+IQ2_XS 5.2 / 0.68 -> 4.4 / 0.54, Q3_K+Q5_K 4.5 / 0.59 ->
  2.9 / 0.56, Q4_K+Q6_K 4.0 / 0.66 -> 2.3 / 0.62. MXFP4 (1.0 / 0.53) does not use them.
- Not yet run on the Xeons or with a real low-bit model.

# 2-bit experts in VRAM at their own size

IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S and Q2_K keep their bits in VRAM slots (before: 4-bit codes).

- Slot (largest expert) of UD-IQ2_XXS, from its headers: 21.4 -> 12.3 MB, so in 6.09 GB 285 -> about 495 slots.
- `bench-gpu`, RTX 5080, ms per expert of one token: MXFP4 0.17, 4-bit codes (Q3_K, what the 2-bit types used before)
  0.069, IQ2_XXS 0.055, IQ2_XS 0.058, IQ2_S 0.061, IQ3_S 0.060, IQ1_S 0.062, IQ1_M 0.065, Q2_K 0.055.
- Selftest: every type as gate/up and as down, one and eight tokens, within 1e-6 of the output range.
- UD-Q4_K_XL, gpu-only, 64 + 64 tokens, before and after: logits identical byte for byte (ppl 5.3016 on this text).
- Files of zeros with the real headers of antirez IQ2_XXS/Q2_K and ISTA-DASLab IQ2_XXS load and decode with GPU and
  CPU experts (10.6 and 10.4 MB slots).
- UD-IQ2_XXS with its real weights, two Xeons + RTX 4070 Ti SUPER (a user's runs, sampled, repeated prompt), `7b964eb`
  -> `3d2d747`: 5.0-5.2 -> 6.41, 6.65, 6.60, 6.55 tok/s; slots 285 -> 499; VRAM hits 39.7 -> 47.9%; CPU runs per
  token 143 -> 121, each 0.66 -> 0.48 ms. ms per token: CPU experts 94 -> 58, GPU experts 12 -> 16, rest of GPU
  69 -> 59, other 13 -> 12. The answers read well.
- `bench-gpu` there: IQ2_XXS+IQ2_S 0.120 ms, MXFP4 0.288 ms (before the next section).

# MXFP4 kernel

`bench-gpu`, RTX 5080, ms per expert of one token, one change at a time:

- as it was 0.167; exponent by bitcast instead of `ldexp` 0.165; the table of 16 values `const` 0.167, both 0.173;
  no table: values from bit arithmetic 0.062, with bitcast 0.058, from 4-bit fields of a constant, with bitcast
  **0.054** (used).
- UD-Q4_K_XL, gpu-only, 64 + 64 tokens: logits identical byte for byte before and after.
- Waiting for the GPU in that run with 0.5 GB of slots, 64 + 40 tokens, alternating: 2.5 and 3.3 s against 10.3 and
  6.4 s (an earlier pair with different slot counts: 10.4 against 6.4). Uploads dominate, so it varies. Decode speed
  with experts in VRAM was not measured.
- UD-Q4_K_XL on the two Xeons + RTX 4070 Ti SUPER (a user's run, sampled, repeated prompt), `bench-gpu` there
  MXFP4 0.288 -> 0.081 ms. `cbafa28` -> `fb7a5ca`: requests 3.87, 6.67, 7.09, 7.70, 8.25 -> 3.57, 5.89, 6.47, 7.85,
  7.85, 8.29 tok/s (last token 8.93); ms per token CPU experts 38 -> 40, disk wait 9 -> 7, GPU experts 4 -> 4, rest
  of GPU 52 -> 49, other 12 -> 11; VRAM hits 41.4%. About the same: GPU experts run while the CPU runs its own, and
  the CPU takes longer, so the faster kernel is mostly hidden. At most the 3 ms off "rest of GPU" is from it.

# Rest of GPU

What "rest of GPU" (waiting for the GPU between layers) is made of, RTX 5080, UD-Q4_K_XL, 60 tokens decoded:

- Per layer: 0.83 ms of waiting, 0.16 ms of it the round trip of an empty submit and read; the GPU work itself
  (timestamps around each dispatch, `DSV4_PROFILE`) 0.69 ms, of which the Q8 dense matrices (`mm_q8_1`, 8.7 per
  layer) 0.39 ms and some 20 small kernels of 3-130 us the rest.
- `mm_q8_1` had each lane read a 32-byte block, so 32 lanes read 4 bytes in every 32. Reading 16 bytes per lane in a
  row: `bench-gpu`, us (GB/s) 5120x5120 60.5 (460) -> 38.3 (727), 2304x5120 26.2 -> 20.3, 5120x2304 29.3 -> 18.9,
  1280x5120 18.9 -> 13.4, 32768x1280 83.3 -> 58.0. Two loads in flight per lane: no different.
- On the model: `mm_q8_1` 15.6 -> 11.6 ms per token, all GPU kernels 27.4 -> 24.1 ms.
- `eval --gpu-only-experts`, 64 + 64 tokens, against the old kernel: KLD 0.0014 (worst 0.012), top-1 100%; well
  below the 0.012-0.025 that rounding order has moved it before.
- Left per token: the router in f32 (`mm_f32_1`, 1.8 ms, about 340 GB/s), `hc_mix` 1.7, `attn_scores` 1.3,
  `hc_pre_norm` 1.1, `quant_sim` 1.0, `idx_scores` 0.9 and many dispatches of a few us; 40 round trips (6 ms).

# CPU experts inside a real decode

Two Xeon E5-2699 v4 + RTX 4070 Ti SUPER (Windows 10), UD-Q4_K_XL, `serve` with a 206 GB RAM tier warmed up, the same
prompt five times, greedy, 300 tokens each. `DSV4_CPU_TRACE=1` logs where the time of CPU expert calls goes.

- `bench-cpu` there now, 44 threads, NUMA-local, ms per expert, asleep / awake: 1 expert 0.34 / 0.31-0.32,
  3 experts 0.33 / 0.23-0.25, 6 experts 0.28 / 0.21-0.24, 6 x 4 tokens 0.47 / 0.36-0.43. `bench-gpu`: MXFP4 0.082 ms;
  `mm_q8_1` against `mm_q8_1_blocks` 5120x5120 114 / 71 us (the new kernel slower here, unlike the RTX 5080),
  2304x5120 32 / 41, 5120x2304 31 / 43, 1280x5120 22 / 29, 32768x1280 163 / 240.
- Server, `1b3b188`, ms per token over the last 100, requests 1 to 5: 3.00, 5.73, 5.25, 7.11, 8.21 tok/s; CPU experts
  41-44, about 142 CPU runs per token, so **0.30 ms each**, not 0.44: the 0.44 above was before threads were kept
  awake between layers. Disk wait 29-149 (the tier holds 206 of 289 GB), GPU experts 4.2-4.5, rest of GPU 52-75,
  other 13-15.
- Traced (tracing costs `bench-cpu` 1 expert 0.32 -> 0.34 ms), 60,000 calls: 3.31 experts per call, 1049 us per call,
  317 us per expert. By experts per call: 1: 507 us, 2: 667, 3: 934, 4-6: 1323 (`bench-cpu` 1: 353, 3: 825, 6: 1488).
  - Expert pages loaded at run time: 99.1% on the node that computes their rows (45.7 M pages checked); 2.1-2.6% of
    tasks taken from the other node's queue. Placement is not what is lost.
  - Threads join a phase 15-25 us after it starts (bench 13-15), the last 22-38 us after; the caller (73% of calls on
    node 1) waits ~50 us at the end of gate/up for the last threads. Waking is not what is lost either.
  - The CPU branch of `rayon::join` starts 50 us after the layer's experts do (the same thread as the GPU branch, after
    it, in 6% of layers).
  - Gate/up 588 us, down 287 us per call: about 10% more than `bench-cpu` for the same experts (other threads and the
    GPU's uploads share the memory).
  - Work on the caller alone: inputs quantized 66 us, between the phases 49, outputs gathered 26: **141 us of 1049**,
    twice what `bench-cpu` spends (31 / 33 / 20 for 3 experts). In decode every expert of a call has the same input,
    quantized again for each.
- Each input quantized once (all experts of a decode call share it); the hidden vectors quantized by the gate/up tasks
  themselves (tasks start at multiples of 32 rows); one-token outputs returned without a copy; the team's threads each
  on a core of their own (`DSV4_PIN_CORES=0`: the whole node, as before). Results identical bit for bit (the same
  arithmetic per 32-block; unit tests). `bench-cpu`, 44 threads awake, ms per expert, before -> after: 1 expert
  0.31-0.33 -> 0.29-0.31, 3 experts 0.23-0.25 -> 0.22-0.25, 6 experts 0.21-0.24 -> 0.21, 6 x 4 tokens 0.36-0.43 -> 0.35-0.37;
  pinned against not pinned in one sitting: 0.29 / 0.33, 0.22 / 0.25, 0.21 / 0.22, 0.37 / 0.39.
- Server, as above, traced, before -> after: 1049 -> 1001 us per call, **317 -> 299 us per expert**; inputs 66 -> 76 us,
  between 49 -> 27, after 26 -> 3; gate/up 588 -> 583, down 287 -> 283. CPU experts per token in requests 3 to 5:
  40.0, 39.2, 43.6 -> 38.5, 36.8, 37.5 ms. tok/s 6.77, 8.39, 7.30 -> 7.21, 7.82, 7.72 (the disk wait varies more).
- What is left on the caller is slow because everything it touches is cold after a phase has streamed 20-100 MB
  through the caches: quantizing one 5120-vector takes 3.3 us on an idle thread, 7-8 us there; setting up the tasks
  ~20 us.
- Smaller tasks against the threads that finish last: `bench-cpu` 44 threads with 3, 6 and 12 tasks per thread, ms per
  expert: 0.31 / 0.32 / 0.30, 0.24 / 0.24 / 0.26, 0.22 / 0.22 / 0.22, 0.33 / 0.39 / 0.38. No difference; kept at 3.
- Server with `--threads 40` (cores left over for the GPU, I/O and caller threads): 304 us per expert against 299 with
  44; gate/up 602, down 292. The phases are limited by memory, not by threads sharing cores.
- Experts that a call waited for from disk: 5% of calls (about 2.2 per token), all of one expert, 648 us against
  ~421 us for the other one-expert calls. One I/O thread wrote all of an expert, so the rows of the other node were
  in the wrong socket's cache.
- I/O threads now write each node's rows while bound to that node (bytes identical, unit test): disk-fresh calls
  648 -> 601 us, about 0.1 ms per token; the rest the same within noise (303 us per expert, gate/up 595, down 285).
- Left, per call of 3.3 experts (~1000 us): gate/up ~590 and down ~285 at 85-88 GB/s, which more threads do not raise;
  ~100 us on the caller alone, mostly cold-cache work setting up the tasks; ~35 us before all threads have joined.
- Tasks without a list: a counter per node over pieces worked out from the index (job, rows), writing through
  pointers into buffers the thread keeps between calls and does not clear; each token's output written by the down
  tasks straight into its own vector. Unit tests (now also a job of 12 tokens): bit for bit however rows are divided.
  `bench-cpu` traced, 44 threads awake, us per call: 1 expert 313 -> 278-282 (inputs 26 -> 8-10), 3 experts
  720 -> 627-682 (inputs 43 -> 10-15), 6 x 4 tokens 2504 -> 1868 (after the phases 219 -> 0); untraced ms per expert:
  1 expert 0.27-0.34, 3 experts 0.21-0.24, 6 experts 0.20-0.21, 6 x 4 tokens 0.33-0.37.
- Server, before -> after: 1020 -> 958 us per call, **303 -> 284 us per expert** (317 when this began); inputs 75 -> 26,
  between 28 -> 6; gate/up 595 -> 610, down 285 -> 310 (fresh output vectors are first touched there now); disk-fresh
  calls 601 -> 539. CPU experts per token in requests 3 to 5: 42.5, 39.1, 35.5 ms (disk wait 113, 47, 35).

# Rest of GPU on the RTX 4070 Ti SUPER: memory clock

Two Xeons + RTX 4070 Ti SUPER, UD-Q4_K_XL, server set up as in "CPU experts inside a real decode", `nvidia-smi`
sampled every 100 ms during requests 4 and 5.

- `DSV4_PROFILE` is no help here: with every dispatch waited for, the GPU idles and clocks down (`mm_q8_1` 965 us a
  call, against 53 us in `bench-gpu`). `bench-gpu` itself varies the same way: the first 5120x5120 `mm_q8_1` above
  (114 us) was taken while the clocks were still rising; repeated, 53-55 us against 66-69 for `mm_q8_1_blocks`, and
  the new kernel is faster at every shape here too (32768x1280 81-160 us, by clock).
- While decoding, the GPU is busy ~31% of the time, in short bursts between the CPU's experts, and the driver keeps it
  in P3 (memory 5001 MHz of 10501) for 44% of the samples, P0 for the rest; its kernels are memory-bound.
- Promotions (about 6 uploads of 18.8 MB a token) were the suspect: with them off (a temporary switch), no uploads,
  but rest of GPU 53-56 -> 65 ms, CPU experts 35 -> 47 (211 CPU runs a token instead of ~140), disk wait ~0, 7.7 tok/s
  either way; and P3 all the time. The uploads were keeping the memory clock up, not slowing the layers down.
- `keep_busy`: one workgroup of 32 doing arithmetic, about 0.5 ms (iterations calibrated when the model loads,
  `DSV4_GPU_BUSY_US`, 0: off), recorded with each layer's GPU experts while the CPU computes its own. P0 in 97-99% of
  the samples, 94 W against 69 W while decoding. Requests 3 to 5, before -> after: rest of GPU 57.7, 55.8, 52.8 ->
  45.5, 45.7, 47.7 ms; GPU experts 4.5 -> 4.6-4.8; CPU experts 36.1, 34.3, 35.5 -> 32.6, 32.7, 34.4; tok/s 6.85, 7.46,
  8.30 -> 7.60, 8.36, 7.85 (disk wait varies most: 55, 57, 36 -> 62, 46, 80).
- A first version with a submit of its own before the experts made GPU experts 4.5 -> 10.5 ms: a submit costs
  ~150 us, on the path to the CPU branch at every layer.

# Rest of GPU on the RTX 4070 Ti SUPER: encoding the commands

Same machine and server as above (UD-Q4_K_XL, `--ram-cache-gb 206`, 5 greedy requests of 300 tokens). `DSV4_GPU_SPAN`
(new) puts a GPU timestamp at the start of each layer's batch and one before its readback copies.

- Rest of GPU 48-53 ms a token, of which the batch waited for keeps the GPU busy 23-28 ms (more after long disk waits,
  when the clocks have dropped; 43 ms for the first request). About 24 ms a token was not GPU work.
- Not work queued ahead: with the GPU branch waiting for its own commands (a temporary switch), GPU experts 5.5 -> 16-19
  ms (the `keep_busy` tail, copies and experts that outlast the CPU's), rest of GPU unchanged (47-52).
- Not the wake-up: `bench-gpu`, a tiny kernel and a 4-byte read, 147-188 us median; polling the fence in a loop
  instead, 120-127 us (~2 ms a token; not pursued).
- It was encoding: timed inside the wait, finishing and submitting the command buffer took 42-44% of it, ~21 ms a token,
  0.5 ms a layer. `queue.submit` itself is cheap; `CommandEncoder::finish` turns the ~50 recorded dispatches into Vulkan
  commands (0.3 ms a layer with the experts in VRAM, 0.5 with the CPU's expert threads spinning).
- Prepared ahead: nothing in a layer's commands up to its routing readback depends on the layer before's results, so
  while the CPU computes a layer's experts, the GPU branch (which only waits for it) records and encodes the next
  layer's batch (the FFN output added to the stream, attention, router, shared expert, copies); once the CPU's outputs
  are written, it is submitted. Only for layers with CPU experts: with the experts in VRAM there is nothing to hide it
  behind (15.7 -> 14.4 tok/s), and nothing is gained. Same text, greedy, with and without.
- Promotions are no longer written ahead of the next layer: the expert is put in a staging buffer on the promoting
  thread and copied to its slot in pieces of 6 MB with the commands that run beside the CPU's experts (`keep_busy`
  still runs; a copy alone does not keep the clocks up: P3 in 40% of samples). Writing the pieces with
  `queue.write_buffer` on the GPU branch made GPU experts 5.5 -> 16 ms (the copy into wgpu's staging is on the caller).
- Requests 3 to 5, rest of GPU before -> after: 48.8, 53.1, 51.0 and 48.0, 48.9, 49.0 (two runs) -> 40.1, 39.7, 38.7 ms;
  GPU experts 5.5 -> 6.7-7.0; GPU busy 23.4-24.7 either way; tok/s 7.50, 7.03, 6.89 and 7.22, 7.62, 8.15 -> 7.23, 7.98,
  8.61 (disk wait 47-92 ms -> 40-79). Promotions copied all at once before the next layer instead (`DSV4_UPLOAD_PIECE_MB=0`),
  same build: 41.6, 40.9, 41.7 ms; the pieces gain 1.5-2 ms.
- Fence polled: the routing read waited for by `device.poll(Poll)` in a loop instead of `PollType::Wait` (sleeping in
  the driver). `bench-gpu` round trip 177 -> 123 us median. Live, requests 2 to 5, rest of GPU minus GPU busy (the time
  not covered by GPU work, steadier than either since disk waits move the clocks) 15.4, 15.4, 15.6, 15.3 -> 10.7, 12.4,
  10.4, 11.3 ms: ~4 ms a token, more than the round trips alone. Rest of GPU 38.8, 41.5, 33.6, 37.4; other 11.4-12.2 ->
  10.3-11.9; tok/s 7.02, 7.11, 9.13, 7.97 (disk wait 39-111 ms; at a similar disk wait, 38.6 vs 40.0 ms, 9.13 vs 8.61).
  Now the default (`DSV4_GPU_POLL=0`: sleep).

### Small kernels that left the GPU idle

Per token the dense weights are ~7.4 GB in the layers and 0.7 GB in the output head, ~14 ms at the 430-557 GB/s the Q8
kernel reaches in `bench-gpu`; the batches were busy 21-26 ms. The rest was kernels that use a few workgroups or wait
on themselves. `bench-gpu` now times them at one decode token (us per dispatch; a 16-float copy, the floor, is 3.2):

| kernel | before | after | what was slow |
|---|---|---|---|
| hc_dot | 10.9 | 4.9 | 40 workgroups for 25 sums; now 160 (6 rows of Wf each) |
| hc_mix | 26.4 | 6.7 | one thread ran the Sinkhorn iterations; now 16 lanes trading by shuffles, and the 40 partial sums loaded 8 at a time |
| hc_pre_norm | 9.9 | 6.8 | 256 threads for 5120 values; now 1024 |
| quant_sim | 13 (profile) | 3.5-4.3 | one thread per block of 32; now one per value, the block's maximum by shuffles |
| idx_scores, 128 positions | 127 (profile) | 8.2 | one workgroup, each thread 32 heads x 128 in a row; now a thread per (position, head) |
| attn_scores, 640 slots | 35.8 | 13.5 | each thread one running sum over 512; now vec4s, two sums, steps of 128 |
| attn_pv, 640 slots | 41.9 | 27.2 | 16-slot steps, one sum; now 64-slot steps, loads issued together, two sums |

Every sum in hc_dot, hc_mix and idx_scores is taken in the same order as before, quant_sim gives the same values; the
norm in hc_pre_norm and the attention sums are added in a different order (selftest: within 1.3e-6).

- GPU-only (`--expert-subset 6 --gpu-only-experts`, 120 tokens, two runs each): GPU busy 27.3-28.1 -> 22.6-22.9 ms a
  token, gpu-wait 52.4-53.2 -> 48.3-49.0, 16.35, 16.45 -> 17.86 tok/s (a second run 15.52, its expert time 2.4 s
  against 1.6).
- Live, requests 3 to 5 (hc kernels only / then quant_sim / then idx and attn scores): GPU busy 23.2-26.1 (before) ->
  20.8-23.3 -> 20.4-22.4 -> 18.6; rest of GPU 41.5, 33.6, 37.4 -> 35.3, 32.5, 32.3 -> 34.9, 33.8, 36.3 -> 33.8, 34.2,
  34.5. The time not covered by GPU work grew over the three runs (11.5-12.0 -> 13.4-13.9 -> 15.2-15.9), so live only
  part of the GPU time saved shows; not explained yet.
