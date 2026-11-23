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
