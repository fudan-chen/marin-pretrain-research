# Agent MoE Experiment: Embed Ablations
## User prompts

The thread covered five variants over multiple turns. Paraphrased:

1. > lets hop over to moe_may_pr, then follow agent.md to make a new branch. lets try having the embed follow its own lr decay schedule. instead of linear, have it only decay for the last 20%.
2. > now lets try another idea on another branch. remove the rmsnorm after the embed. instead, multiply the forward pass by a constant, that gives the embed the same norm that rms norm would give at init.
3. > try another without both the rms norm and without the embed gated norm
4. > lets also try just rms_norm, no gated norm on embed.
5. > then also try just gated norm on embed, with embed lr decay at 0.5 instead of 0.2.

## Plan

Five branches off `moe_may_pr`, each independent. Each launcher uses the May Recipe compute-optimal cell (`bs`, `steps`, `tokens` from the d=512/d=768 entries of `experiments/grug/moe/launch.py:_COMPUTE_OPT_CELLS`), MuonH on heuristic_v2, EP=1, v5p-8 us-east5-a interactive.

### A. `moe_may_embed_late_decay` (commits `4a50a5448`, `2348d941d`, `2ea5682db`)

`optimizer.py`: `GrugMoeMuonHConfig.embed_decay_frac: float | None`. When set, `token_embed` is split out of the `adam` LR group into a new `embed` group whose schedule holds peak constant for the first `1 - embed_decay_frac` of training, then linearly decays to 0. Other params keep the standard MuonH schedules.

Launchers use `embed_decay_frac=0.20` (constant for first 80%, decay in last 20%):
- `experiments/grug/moe/moe_may_compute_opt_d512_ep1_embed_late_decay.py`
- `experiments/grug/moe/moe_may_compute_opt_d768_ep1_embed_late_decay.py`

### B. `moe_may_embed_no_rms` (commit `7d1bef95b`)

`model.py`: `GrugModelConfig.embed_skip_rms_norm: bool = False`. When True, `embed_norm` (RMSNorm) is bypassed; the embedding is multiplied by `1/initializer_std` (the heuristic sets `initializer_std=0.5/sqrt(d)`, so the constant is `2*sqrt(d)` ≈ 45.25 at d=512, 55.43 at d=768) — unit-RMS at init, matching the RMSNorm output magnitude. `embed_gated_norm` still runs.
- `experiments/grug/moe/moe_may_compute_opt_d512_ep1_embed_no_rms.py`
- `experiments/grug/moe/moe_may_compute_opt_d768_ep1_embed_no_rms.py`

### C. `moe_may_embed_no_norms` (commit `9c41617c2`)

`model.py`: adds `embed_skip_gated_norm` alongside `embed_skip_rms_norm`. With both True, the embedding flows into block 0 with only the static `× 1/initializer_std` rescale — no per-token RMSNorm, no per-dim sigmoid gating.
- `experiments/grug/moe/moe_may_compute_opt_d512_ep1_embed_no_norms.py`
- `experiments/grug/moe/moe_may_compute_opt_d768_ep1_embed_no_norms.py`

### D. `moe_may_embed_only_rms` (commit `387d6fd99`)

`model.py`: only `embed_skip_gated_norm`. Keeps the post-embed RMSNorm; drops only the per-dim GatedNorm right after.
- `experiments/grug/moe/moe_may_compute_opt_d512_ep1_embed_only_rms.py`
- `experiments/grug/moe/moe_may_compute_opt_d768_ep1_embed_only_rms.py`

### E. `moe_may_embed_no_rms_late_decay50` (commit `5837395d5`)

Combines B + a longer high-LR phase for the embed: `embed_skip_rms_norm=True` + `embed_decay_frac=0.5` (embed LR holds peak constant for first 50% of training, decays linearly over last 50%). GatedNorm still runs.
- `experiments/grug/moe/moe_may_compute_opt_d512_ep1_embed_no_rms_late_decay50.py`
- `experiments/grug/moe/moe_may_compute_opt_d768_ep1_embed_no_rms_late_decay50.py`

## Gate 1 cells

Per `experiments/grug/moe/agent.md`: d=512 (3.82e17 FLOPs, bs=32, steps=10,980, 1.44 B tokens) and d=768 (2.81e18 FLOPs, bs=64, steps=16,875, 4.42 B tokens). v5p-8 EP=1, us-east5-a, interactive priority.

## Compare against

May Recipe compute-optimal baselines (from `experiments/grug/moe/README.md`):

| budget | dim | paloma macro | tok/s |
|---|---|---|---|
| 3.82e17 | d=512 | **3.5438** | 530,704 |
| 2.81e18 | d=768 | **3.2330** | 357,696 |

Note: baseline tok/s in the README is from the original v4-32 runs. All variants here ran on v5p-8 (us-east5-a) due to v4-32 us-central2 coscheduling issues seen during initial submission. Effective-speedup numbers will conflate algorithm and hardware unless an equal-throughput re-throughput-match is also computed; see initial result below.

## Early result

`embed_no_rms` d=512 finished:

| | loss | tok/s | hw |
|---|---|---|---|
| baseline (README) | 3.5438 | 530,704 | v4-32 |
| variant | 3.5355 | 333,601 (last-100-step avg) | v5p-8 |

Per agent.md formula:
- `A_bl = 87.725`
- `C_needed = 3.998e17` (baseline needs 1.047× the budget to match variant loss)
- Wall-clock effective speedup (mixed hw): **0.658**
- Equal-throughput speedup (same hw, isolates loss gain): **1.047**

Loss did improve (~0.0083 nats); wall-clock comparison is dominated by hardware (v5p-8 delivers ~63% of v4-32's tok/s here).

## Branches

- https://github.com/marin-community/marin/tree/moe_may_embed_late_decay
- https://github.com/marin-community/marin/tree/moe_may_embed_no_rms
- https://github.com/marin-community/marin/tree/moe_may_embed_no_norms
- https://github.com/marin-community/marin/tree/moe_may_embed_only_rms
- https://github.com/marin-community/marin/tree/moe_may_embed_no_rms_late_decay50


## S6442-01 2026-06-17T06:36:01Z
https://github.com/marin-community/marin/issues/6442#issuecomment-4726762768
🤖 Added a sixth variant for the norm-ablation series — moves from the **embed**-input norms to the **MLP**-input norm inside each block.

### F. \`moe_may_mlp_no_rms\` ([branch](https://github.com/marin-community/marin/tree/moe_may_mlp_no_rms))

\`model.py\`: \`GrugModelConfig.mlp_skip_rms_norm: bool = False\`. When True, the per-block pre-MLP RMSNorm (\`rms_mlp\`) is dropped so the MLP input is just \`mlp_gated_norm(x)\` (per-dim GatedNorm alone). Tests whether GatedNorm by itself is sufficient pre-MLP normalization.

Launchers:
- \`experiments/grug/moe/moe_may_compute_opt_d512_ep1_mlp_no_rms.py\`
- \`experiments/grug/moe/moe_may_compute_opt_d768_ep1_mlp_no_rms.py\`

Submitted on v5p-8 us-east5-a interactive:
- d=512: \`/larry/iris-run-job-20260617-063537\`
- d=768: \`/larry/iris-run-job-20260617-063546\`

## S6442-02 2026-06-17T06:39:40Z
https://github.com/marin-community/marin/issues/6442#issuecomment-4726787036
🤖 Equal-tps speedup table for variants A–E (all finished except E d=768 which crashed at ~90% budget). F (mlp_no_rms) is still running; will update when it lands.

Method: per the speedup formula in \`agent.md\`, refit the scaling-law coefficient \`A_bl\` through the baseline (loss, budget) point, invert to find \`C_needed\` for the variant loss, then report \`C_needed / budget\` (i.e., how much more compute the baseline would need to match the variant loss). Throughput is factored out, so this isolates the loss gain from any hardware/throughput differences.

Baselines: \`moe_may_compute_opt_d512_ep1\` (3.5422 @ 3.16e17 FLOPs), \`moe_may_compute_opt_d768_ep1\` (3.2273 @ 1.81e18 FLOPs).

| variant | d=512 loss | d=512 spdup | d=768 loss | d=768 spdup |
|---|---|---|---|---|
| baseline | 3.5422 | 1.000 | 3.2273 | 1.000 |
| **A** embed_late_decay | 3.5418 | 1.002 | 3.2231 | 1.028 |
| **B** embed_no_rms | 3.5355 | 1.037 | 3.2189 | **1.056** |
| **C** embed_no_norms | 3.5862 | **0.788** | 3.2592 | **0.814** |
| **D** embed_only_rms | 3.5343 | **1.044** | 3.2211 | 1.041 |
| **E** embed_no_rms_late_decay50 | 3.5353 | 1.038 | — | crashed @ ~90% |
| **F** mlp_no_rms | (running) | — | (running) | — |

Reads:
- **C (drop both norms) is clearly worse.** At d=768 the baseline only needs ~81% as much compute to match it — dropping the GatedNorm without anything in front of it hurts.
- **B and D are the winners**, but they cross over: at d=512 the RMSNorm-only path (D) is slightly better (1.044× vs 1.037×); at d=768 the GatedNorm-only path (B) takes the lead (1.056× vs 1.041×). Either norm alone helps; running both looks like mild overkill.
- **A (late-decay LR)** is small but real at d=768 (+2.8%); negligible at d=512.
- **E ≈ B at d=512** — the late-decay add-on doesn't move the needle on top of dropping the RMSNorm.

## S6442-03 2026-09-09T01:57:53Z
https://github.com/marin-community/marin/issues/6442#issuecomment-5594603859
This issue has been inactive for 83 days and is marked as stale.
If there is no further activity within 7 days, it will be automatically closed.
If you believe this issue is still relevant, please add a comment to keep it open.

## S6442-04 2026-09-16T02:08:39Z
https://github.com/marin-community/marin/issues/6442#issuecomment-5690947018
This issue has been automatically closed due to inactivity.
If you believe this issue is still relevant, please reopen it or create a new issue with updated information.
