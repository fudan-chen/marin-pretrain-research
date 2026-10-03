# Aug Hero Run LR Sweep
<!-- experiment-tldr:start -->
## Summary

This issue defines a 150-run learning-rate and token-budget sweep for the Aug/Hero MoE setup. It covers six model widths from d512 through d2048, five token budgets from 30x through 600x active parameters, and five LR multipliers around the May-Recipe MuonH recommendation. The goal is to see whether the recommended LR remains compute-optimal as both model size and training tokens scale, or whether the Aug/Hero recipe needs a systematic LR adjustment. Current status: the sweep is specified but no runs have been launched yet.

### Helpful links
- `experiments/grug/moe_hero_fsdp/sweep.md` on branch `aug_hero_run_ablations`
- #6711
<!-- experiment-tldr:end -->

## Description

Model sizes and architecture are fixed by the hero sweep table (`experiments/grug/moe_hero_fsdp/sweep.md`, branch `aug_hero_run_ablations`): d512, d768, d1024, d1280, d1536, d2048, each with its local/global KV split and explicit global-attention layers, a 1024-token sliding window, and sequence length 8192. The `60×` column in that table is the reference token budget; the other multipliers just scale the step count (batch is fixed per size).

## Goal

Map the compute-optimal learning rate relative to the May-Recipe heuristic across model size and token budget — i.e. whether the recommended LR holds, or needs a systematic multiplier, as we scale tokens (30×→600×) and width (d512→d2048).

## Sweep

- **Sizes (6):** d512, d768, d1024, d1280, d1536, d2048.
- **Token budgets (5):** 30×, 60×, 150×, 300×, 600× active params. 60× matches `sweep.md`; steps scale linearly.
- **LR multipliers (5):** 0.7, 0.85, 1.0, 1.2, 1.4 × the recommended MuonH LR.
- **Total: 6 × 5 × 5 = 150 runs.**

### Steps per (size × token budget)

| size | batch | 30× | 60× | 150× | 300× | 600× |
|------|------:|----:|----:|-----:|-----:|-----:|
| d512  |  32 | 2115 |  4230 | 10575 | 21150 |  42300 |
| d768  |  64 | 3172 |  6345 | 15862 | 31725 |  63450 |
| d1024 | 128 | 4170 |  8340 | 20850 | 41700 |  83400 |
| d1280 | 128 | 7547 | 15094 | 37734 | 75469 | 150938 |
| d1536 | 256 | 6238 | 12476 | 31191 | 62381 | 124762 |
| d2048 | 512 | 6221 | 12442 | 31106 | 62212 | 124425 |

### Recommended MuonH LR (1.0×) per (size × token budget)

The heuristic LR falls as tokens grow (`adam_lr ∝ tokens^-0.395`), so the base LR is per (size, token budget); each run multiplies its cell by {0.7, 0.85, 1.0, 1.2, 1.4}.

| size | 30× | 60× | 150× | 300× | 600× |
|------|----:|----:|-----:|-----:|-----:|
| d512  | 0.0202 | 0.0154 | 0.0107 | 0.0081 | 0.0062 |
| d768  | 0.0174 | 0.0133 | 0.0092 | 0.0070 | 0.0053 |
| d1024 | 0.0161 | 0.0123 | 0.0085 | 0.0065 | 0.0049 |
| d1280 | 0.0123 | 0.0094 | 0.0065 | 0.0050 | 0.0038 |
| d1536 | 0.0139 | 0.0106 | 0.0074 | 0.0056 | 0.0043 |
| d2048 | 0.0143 | 0.0109 | 0.0076 | 0.0058 | 0.0044 |

### Naming

`aug-hero-{size}-{mult}x-lr{lrmult}`, e.g. `aug-hero-d1024-150x-lr0.85`.

## Status

Defined; **no runs launched.**

## Links

- Sweep config: `experiments/grug/moe_hero_fsdp/sweep.md` (branch `aug_hero_run_ablations`)

Part of #6711


## S7856-01 2026-08-01T20:12:14Z
https://github.com/marin-community/marin/issues/7856#issuecomment-5153220108
@dlwh Converted the existing sweep TL;DR into the managed experiment summary block and labeled the issue `tldr`; no results were added because the issue still says no runs have launched.

## S7856-02 2026-08-05T02:39:57Z
https://github.com/marin-community/marin/issues/7856#issuecomment-5186902906
🤖 **Model config + step counts for the 6 sweep sizes.**

## Grug model config (6 sizes)

All sizes share: `vocab_size=128256` (marin_tokenizer, llama3-based), `head_dim=128`, `num_experts=128` top-`4` + `2` shared experts, `intermediate = hidden/2` (routed & shared), `max_seq_len=8192`, `sliding_window=512` (global attention every 4th layer + final layer), SConv on `(k,v,attn,mlp)`, MuonH optimizer (May-Recipe compute-scaled LR), `capacity_factor=1.0`, sigmoid-combine QB-routed MoE. KV heads follow ¼-local / ⅛-global of the query heads (floored, min 1).

| spec | d512 | d768 | d1024 | d1280 | d1536 | d2048 |
|---|---|---|---|---|---|---|
| hidden_dim | 512 | 768 | 1024 | 1280 | 1536 | 2048 |
| num_layers | 6 | 8 | 12 | 14 | 16 | 18 |
| num_heads (head_dim 128) | 4 | 6 | 8 | 10 | 12 | 16 |
| local_kv_heads | 1 | 1 | 2 | 2 | 3 | 4 |
| global_kv_heads | 1 | 1 | 1 | 1 | 1 | 2 |
| intermediate_dim | 256 | 384 | 512 | 640 | 768 | 1024 |
| batch_size (global) | 64 | 128 | 128 | 128 | 256 | 512 |
| nodes (GB200 ×4) | 1 | 1 | 1 | 2 | 4 | 4 |
| active params (excl embed/lm_head) | 18M | 54M | 145M | 262M | 434M | 868M |

## Step count by token budget

Token budget = *multiplier × active-params* (fixed tokens-per-active-param over-train ratio); `num_steps = tokens / (batch × 8192)`. Total training tokens in parentheses.

| size | active | 30× | 60× | 150× | 300× | 600× |
|---|---|---|---|---|---|---|
| d512 | 18M | 1,058 (0.6B) | 2,115 (1.1B) | 5,288 (2.8B) | 10,575 (5.5B) | 21,150 (11.1B) |
| d768 | 54M | 1,552 (1.6B) | 3,105 (3.3B) | 7,762 (8.1B) | 15,525 (16.3B) | 31,050 (32.6B) |
| d1024 | 145M | 4,162 (4.4B) | 8,325 (8.7B) | 20,812 (21.8B) | 41,625 (43.6B) | 83,250 (87.3B) |
| d1280 | 262M | 7,509 (7.9B) | 15,019 (15.7B) | 37,547 (39.4B) | 75,094 (78.7B) | 150,188 (157.5B) |
| d1536 | 434M | 6,210 (13.0B) | 12,420 (26.0B) | 31,050 (65.1B) | 62,100 (130.2B) | 124,200 (260.5B) |
| d2048 | 868M | 6,206 (26.0B) | 12,412 (52.1B) | 31,031 (130.2B) | 62,062 (260.3B) | 124,125 (520.6B) |


## S7856-03 2026-08-05T16:53:54Z
https://github.com/marin-community/marin/issues/7856#issuecomment-5194730613
🤖 **Aug hero LR sweep — results + LR scaling law.** Figures and analysis on branch [`grug-aug-lr-sweep-results`](https://github.com/marin-community/marin/tree/grug-aug-lr-sweep-results/experiments/grug/aug_lr_sweep_figs).

## Status: 122/150 cells at full budget

d512–d1536 mostly complete; the remaining ~28 are the largest/longest cells (d2048-300×/600×, d1536-600×), which crash-and-resume off temp checkpoints and are the long pole. Best paloma `macro_loss` per cell (metric used throughout):

| size | 30× | 60× | 150× | 300× | 600× |
|---|---|---|---|---|---|
| d512 | 3.844 | 3.666 | 3.502 | 3.409 | 3.336 |
| d768 | 3.467 | 3.325 | 3.185 | 3.104 | 3.048 |
| d1024 | 3.153 | 3.046 | 2.938 | 2.868 | 2.813 |
| d1280 | 2.994 | 2.899 | 2.802 | 2.740 | 2.692 |
| d1536 | 2.874 | 2.790 | 2.695 | 2.636 | (partial) |
| d2048 | 2.730 | 2.651 | (partial) | (partial) | (partial) |

## Per-cell LR optimum (log-quadratic fit)

Each cell's LR sweep fits a clean parabola in `log(LR)`; the optimum `lr*` **drifts up with token budget** — ~1.0× at 30× to **~1.2–1.4× at 300×/600×** — with a mild downward shift for wider models. The 1.0× heuristic is right for short runs but under-shoots the LR for over-trained budgets.

![per-cell](https://github.com/marin-community/marin/blob/grug-aug-lr-sweep-results/experiments/grug/aug_lr_sweep_figs/lr_sweep_percell_fit_paloma.png?raw=true)

## Learned optimal-LR power law

Fitting the per-cell `lr*` (× the May-Recipe heuristic LR) to `lr = a·tokens^b·hidden^c·batch^0.5` (batch exponent pinned):

```
lr = 34.3 · tokens^-0.346 · hidden^-0.345 · batch^0.5      R² = 0.978, median rel err 1.6%
```

![powerlaw](https://github.com/marin-community/marin/blob/grug-aug-lr-sweep-results/experiments/grug/aug_lr_sweep_figs/lr_powerlaw_fit.png?raw=true)

## Batch-controlled collapse

Dividing out `√batch` (and `hidden^c`) collapses **all 6 model sizes onto a single `tokens^-0.346` line**, R² = 0.979 — the optimal LR is a pure token power law once width and batch are controlled for.

![collapse](https://github.com/marin-community/marin/blob/grug-aug-lr-sweep-results/experiments/grug/aug_lr_sweep_figs/lr_batch_collapse.png?raw=true)

## vs. current heuristic, and at the d6144 / 15T target

muonh LR = `coeff·tokens^p·hidden^q·batch^0.5` (seq 8192 folded into coeff):

| | coeff | tokens exp | hidden exp |
|---|---|---|---|
| **ours (fit)** | 34.35 | **−0.346** | **−0.345** |
| **current heuristic** | 25.89 | −0.395 | −0.150 |

Two structural disagreements: ours decays **slower with tokens** (wants more LR at large budgets) and **much faster with width**. At **d6144 / 15T tokens** these nearly cancel — our law prescribes **~1.07× the heuristic's LR** (batch-independent; e.g. batch 4096 → heuristic 2.80e-3 vs ours 2.99e-3).

**Caveat:** that agreement is a coincidental cancellation, and d6144/15T is a large extrapolation — width is **3× beyond** the widest swept model (d2048) and 15T is well past the 600× budget. The `hidden^-0.345` term (the biggest disagreement) is the least-constrained; a d6144-or-wider point is needed to pin it before trusting either law at 15T.


## S7856-04 2026-08-05T16:56:28Z
https://github.com/marin-community/marin/issues/7856#issuecomment-5194756584
Conclusions:
Loss bug seems gone now that we are testing on B200 without the custom Liger cross entropy.
Only a 7% change for the new datamix on LR at the target size- seems we are fine here and can retune LR less frequently.
Will update the formula but expect minimal impact.

## S7856-05 2026-08-26T08:43:37Z
https://github.com/marin-community/marin/issues/7856#issuecomment-5422815053
🤖 Started the matched d512 constant-LR TPU extension.

The matrix keeps the completed #7856 d512 model, datakit mixture, batch 64, seed 0, 1% warmup, token budgets, peak-LR multipliers, and Paloma cadence. The only algorithmic change is linear decay to 5% of peak → constant post-warmup LR. All cells run on v4-8 in us-central2-b so the datakit TensorStores stay in-region.

The representative 30x/1.0x run passed the launch gate: loss fell from 11.782 at step 2 to 8.794 at step 20 and 5.685 at step 86; LR reached 0.028575256 at step 10 and stayed flat; dropped assignments and routing overflow remained zero. The full 25-cell StepRunner is now active with max concurrency 5. It is reusing the representative artifact lock and has not created a duplicate run.

Source: https://github.com/marin-community/marin/commit/4593a8f95
Representative W&B: https://wandb.ai/marin-community/marin_moe/runs/AUG-LRC-TPU-003-d512-30x-lr1
Research logbook: https://github.com/marin-community/marin/blob/codex/research-kaiyuew-7856-d512-constant-lr/.agents/logbooks/7856-d512-constant-lr-tpu.md

Confidence is high on the matched config and startup health; the scaling conclusion remains unknown until the 25 terminal Paloma evaluations complete. A 15-minute heartbeat is monitoring Iris, W&B, routing, and checkpoints. Next update will report terminal per-budget LR optima and the constant-LR loss-scaling fit against the original linear-decay baseline.

## S7856-06 2026-08-27T17:55:21Z
https://github.com/marin-community/marin/issues/7856#issuecomment-5443099203
🤖 The first 20 completed constant-LR runs selected the lowest tested multiplier (`0.7x`) at every finished budget (30x, 60x, 150x, and 300x). This makes `0.7x` an upper bound on the optimum, not an interior estimate.

I intentionally stopped the original matrix parent and its five active 600x children. Their partial W&B histories remain available, but this reproduction disables checkpointing, so those five runs cannot resume. The four completed `0.7x` cells will be reused and not retrained.

A shared log-quadratic fit with budget fixed effects estimates an optimum at `0.323x`; leave-one-budget-out fits span `0.304x–0.363x`. The per-budget extrapolations are `0.201x / 0.358x / 0.377x / 0.352x` for 30x through 300x. This is an extrapolation from a boundary-censored sweep, so the follow-up grid includes guardrails on both sides: `0.10x / 0.20x / 0.32x / 0.45x / 0.70x`.

The replacement parent is `/kaiyuew/issue-7856-d512-constant-lr-low`. It contains 21 new cells: four lower multipliers for 30x–300x and all five multipliers for 600x. The first five v4-8 children are running; all five have finite, advancing loss at steps 18–23, have reached their configured constant LRs, and report zero dropped assignments and zero routing overflow. No duplicate artifact was submitted.

- [Pinned launcher and experiment notes](https://github.com/marin-community/marin/tree/fdc6d312f5/experiments/grug/moe_hero_fsdp_constant_lr_tpu)
- [Fit result](https://github.com/marin-community/marin/blob/fdc6d312f5/scratch/20260827_constant_lr_range_fit.json)
- [Research logbook](https://github.com/marin-community/marin/blob/fdc6d312f5/.agents/logbooks/7856-d512-constant-lr-tpu.md)
- [W&B lower-LR group](https://wandb.ai/marin-community/marin_moe/groups/issue-7856-d512-constant-lr-low-tpu)

Next: combine the 21 new terminal Paloma values with the four reusable old `0.7x` values, refit the optimum per budget, and compare the selected constant-LR curve against the issue's linear-decay baseline.


## S7856-07 2026-08-29T09:41:51Z
https://github.com/marin-community/marin/issues/7856#issuecomment-5461607993
🤖 Final d512 constant-LR extension result

The bracketed lower-LR follow-up is complete: 21 new cells plus the four reused `0.70x` terminals cover 30x through 600x.

| budget | local optimum multiplier | local optimum LR | fitted constant-LR Paloma | linear-decay Paloma | delta |
|---:|---:|---:|---:|---:|---:|
| 30x | 0.42496 | 0.0121434 | 4.096464 | 3.844418 | +0.252046 |
| 60x | 0.34516 | 0.00750221 | 3.884609 | 3.666233 | +0.218376 |
| 150x | 0.28373 | 0.00429401 | 3.684398 | 3.501984 | +0.182413 |
| 300x | 0.27518 | 0.00316726 | 3.572887 | 3.409257 | +0.163630 |
| 600x | 0.25908 | 0.00226774 | 3.488184 | 3.335515 | +0.152668 |

The five-budget optimal-LR refit is:

`LR* = 8.213774e-3 * (tokens / 1B)^(-0.5563134)`

The log-space fit has `R² = 0.9921`; the exponent standard error is `0.0287`, the nominal OLS 95% interval is `[-0.6477, -0.4649]`, and leave-one-budget-out exponents span `[-0.5872, -0.5150]`. At 600x, the new curve predicts `2.1541e-3`. The independently bracketed local estimate is `2.2677e-3`, 5.3% above the new curve and 12.9% above the old 30x–300x extrapolation (`2.0088e-3`). The best directly tested 600x grid point is `0.32x` with Paloma `3.491543`; `0.259x` and Paloma `3.488184` are local quadratic estimates from the `0.20x / 0.32x / 0.45x` neighborhood.

The fitted best-loss laws are:

- constant LR: `L = 3.22436 + 0.68923 * (tokens / 1B)^(-0.39847)` (`R² = 0.999989`)
- linear decay: `L = 3.11098 + 0.58111 * (tokens / 1B)^(-0.39251)` (`R² = 0.999905`)

The exponents are essentially the same over these five budgets, so the constant-LR disadvantage is better described as a worse level/asymptote than as shallower scaling. It remains +0.153 Paloma at 600x. The loss-law fit uses only five budget optima and does not propagate the local quadratic uncertainty.

Operationally, the recovery parent `/kaiyuew/issue-7856-d512-constant-lr-low-006-021` succeeded with exit code 0 and zero failures. All 21 requested cells have terminal Paloma metrics and SUCCESS artifacts in `gs://marin-us-central2`; concurrency stayed at or below five, and no config or hardware drift was observed. The final 600x/0.45x cell recovered from 11 preemptions by restoring `step-20537`, then wrote non-temporary final checkpoint `step-21150`.

[W&B group](https://wandb.ai/marin-community/marin_moe/groups/issue-7856-d512-constant-lr-low-tpu/workspace)


## S7856-08 2026-09-11T17:20:02Z
https://github.com/marin-community/marin/issues/7856#issuecomment-5638088947
🤖 Submitted matched one-layer dense linear-decay LR sweeps for MuonH and SGD-H. Each sweep has 25 cells over 30x/60x/150x/300x/600x and 0.10x/0.20x/0.32x/0.45x/0.70x, with 1% warmup and linear decay to 5% of peak.

Iris parents:
- `/kaiyuew/issue-7856-d512-linear-decay-one-layer-dense-muonh-scaling`
- `/kaiyuew/issue-7856-d512-linear-decay-one-layer-dense-sgdh-scaling`

Training snapshot: `956ccdba5`. The first submission exposed a deterministic scheduler mismatch: the inherited v4 request asked for 200 GB while the live allocatable host limit is 192 GB. I cancelled both parents before W&B initialization, set child RAM explicitly to 190 GB, tested and pushed the fix, and resubmitted with all scientific parameters and identities unchanged. Both fixed parents are running, first-wave W&B identities exist, and sampled configs confirm one layer, d512, seed 0, `lr_schedule=linear`, warmup 0.01, and `min_lr_ratio=0.05`. All observed data and output paths are in `gs://marin-us-central2`. The existing one-layer sweep heartbeat now monitors these two runs and will perform per-budget boundary extension when needed.

## S7856-09 2026-09-11T17:50:49Z
https://github.com/marin-community/marin/issues/7856#issuecomment-5638539932
🤖 Added the one-layer dense SGD-MH condition (0.95 momentum + Nesterov before Hyperball; no Newton–Schulz) with matched constant and linear-decay 25-cell sweeps. Corrected all dense linear-decay launchers to decay to zero (`min_lr_ratio=0.0`): the earlier 5%-floor MuonH/SGD-H/SGD-MH trees were cancelled, and replacements use fresh `AUG-LIN0-*` artifact/run identities, `zero-tail` W&B groups, and artifact version `2026.09.11.1`. The corrected MuonH, SGD-H, and SGD-MH parents have each declared 25 cells with five v4-8 children pending for capacity; all output paths are in `gs://marin-us-central2`. Constant SGD-MH remains active. Source: `ad8a6d61e`; launch/logbook snapshot: `b38c6d3f5`.
