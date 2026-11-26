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
