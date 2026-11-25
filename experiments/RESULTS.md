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
