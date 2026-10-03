# Agent MoE Experiment: Alternating Dense/MoE Blocks
## User prompt

> lets make a new branch and try alternating full dense full MoE. On the dense layers, use intermediate_dim = 3x hidden_dim, on the full MoE, use pick 6 from 512, each with intermediate_dim hidden_dim/2. submit on v5p-8 us-east5-a.

## Plan

Branch: `moe_may_alternate_dense_moe` (off `moe_may_pr`). Commit `f2b0baf4b`.

`model.py`: `GrugModelConfig.alternate_dense_moe: bool` and `dense_intermediate_dim: int | None`. When `alternate_dense_moe=True`, even-indexed blocks are pure dense (a single `DenseMLP` with `dense_intermediate_dim`, defaulting to `3 * hidden_dim`); odd-indexed blocks are pure MoE (uses `num_experts`, `num_experts_per_token`, `intermediate_dim`; no shared expert regardless of `shared_expert_intermediate_dim`).

`Block` got a static `is_dense` flag and `mlp: MoEMLP | None`. Dense blocks return zero stats so the existing `routing_*_per_layer` stacking contract still holds. `DenseMLP` output is batch-resharded before the residual add (it flattens `(b, s) -> t` internally and can come back with the wrong sharding axis on the residual side).

Launchers set `num_experts=512`, `num_experts_per_token=6`, `intermediate_dim=hidden_dim/2` (heuristic default), `dense_intermediate_dim=3*hidden_dim`:
- `experiments/grug/moe/moe_may_compute_opt_d512_ep1_alternate_dense_moe.py`
- `experiments/grug/moe/moe_may_compute_opt_d768_ep1_alternate_dense_moe.py`

### Per-block layout

| block | d=512 (6 layers) | d=768 (8 layers) |
|---|---|---|
| 0, 2, 4 (, 6) | dense, intermediate_dim = 3×d = 1536 / 2304 | dense |
| 1, 3, 5 (, 7) | MoE, 512 experts, top-6, intermediate_dim = d/2 = 256 / 384, no shared | MoE |

Verified init produces alternating block types (3 dense + 3 MoE for d=512; 4 dense + 4 MoE for d=768).

## Gate 1 cells

Per `experiments/grug/moe/agent.md`: d=512 (3.82e17 FLOPs, bs=32, steps=10,980, 1.44 B tokens) and d=768 (2.81e18 FLOPs, bs=64, steps=16,875, 4.42 B tokens). v5p-8 EP=1, us-east5-a, interactive priority.

## Compare against

May Recipe compute-optimal baselines (from `experiments/grug/moe/README.md`):

| budget | dim | paloma macro | tok/s |
|---|---|---|---|
| 3.82e17 | d=512 | **3.5438** | 530,704 |
| 2.81e18 | d=768 | **3.2330** | 357,696 |

Baseline tok/s in the README is from v4-32 runs; variants here are on v5p-8. Effective-speedup numbers will conflate algorithm and hardware — also report an equal-throughput speedup when comparing.

A variant passes Gate 1 if its effective speedup > 1 at both scales.

## Submitted jobs

- d=512: `/larry/iris-run-job-20260616-210224`
- d=768: `/larry/iris-run-job-20260616-210232`

## Branch

https://github.com/marin-community/marin/tree/moe_may_alternate_dense_moe


## S6443-01 2026-08-27T00:49:37Z
https://github.com/marin-community/marin/issues/6443#issuecomment-5432883305
🤖 Results for the two Gate-1 runs, which were never posted here. Both finished; W&B under `marin_moe`.

| scale | run | paloma macro | May-Recipe baseline | tok/s |
|---|---|---:|---:|---:|
| d512 | [moe_may_compute_opt_d512_ep1_alternate_dense_moe](https://wandb.ai/marin-community/marin_moe/runs/moe_may_compute_opt_d512_ep1_alternate_dense_moe) | 3.5486 | 3.5438 (+0.0048) | 383,363 |
| d768 | [moe_may_compute_opt_d768_ep1_alternate_dense_moe](https://wandb.ai/marin-community/marin_moe/runs/moe_may_compute_opt_d768_ep1_alternate_dense_moe) | 3.2236 | 3.2330 (−0.0094) | 272,894 |

Quality is a wash: marginally worse at d512, marginally better at d768, both within eval noise. Alternating dense/MoE blocks gave no consistent quality gain over the compute-optimal May Recipe baseline at either scale.

Tokens per second (from `throughput/examples_per_second` × 4096 seq len):

| scale | alternating (v5p-8) | baseline (v4-32) |
|---|---:|---:|
| d512 | 383,363 | 530,704 |
| d768 | 272,894 | 357,696 |

The variant is slower in raw tok/s, but the baseline was measured on v4-32 while these ran on v5p-8, so the raw comparison conflates hardware with algorithm, as the plan noted. With quality neutral and no throughput advantage, the variant does not clear Gate 1 (effective speedup > 1 at both scales).


## S6443-02 2026-08-27T00:56:36Z
https://github.com/marin-community/marin/issues/6443#issuecomment-5432929722
Small scale iirc alternating is about 20% better MFU. but unclear yet how this scales up, both in terms of quality and MFU.
