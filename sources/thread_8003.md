# [grug] Aug hero iso-FLOP sweep (d512-d2048 x 1e18-3e20)
<!-- experiment-tldr:start -->
## Summary

This experiment is an Aug hero MoE iso-FLOP sweep across widths d512-d2048 and budgets 1e18-3e20 FLOPs to find the compute-optimal width and validate the LR law from #7856. The first substantial update reported 17 of 24 cells complete: the completed 1e18, 3e18, and 1e19 budgets form clean U-curves with approximate optima at d768, d768, and d1024 respectively, and an interim fit `L(C)=1.6+72.6*C^-0.0897`, `T*(C) proportional to C^0.542`. The sweep is not finished: two 3e19 cells diverged from an output projection collapse, several >=1e20 cells were still running or capacity-gated, and the high-budget law remains provisional until those cells and reruns close.

### Helpful links
- First results and interim scaling-law comment: https://github.com/marin-community/marin/issues/8003#issuecomment-5209786274
- Iso-FLOP overlay figure: https://raw.githubusercontent.com/marin-community/marin/2c603278b75d6655c6c8f404c89bb1b1eb7d93b4/experiments/grug/moe_hero_fsdp/isoflop_figs/isoflop_overlay.png
- Per-budget curves figure: https://raw.githubusercontent.com/marin-community/marin/2c603278b75d6655c6c8f404c89bb1b1eb7d93b4/experiments/grug/moe_hero_fsdp/isoflop_figs/isoflop_curves.png
<!-- experiment-tldr:end -->

## TL;DR

Iso-FLOP scaling sweep of the Aug hero MoE (128-expert / top-4 / 2-shared, SConv, hybrid-GQA, global-every-4, sliding-window 512) across 6 widths (d512–d2048) and 6 compute budgets (1e18, 3e18, 1e19, 3e19, 1e20, 3e20 FLOPs, **excluding embed/lm_head**), on H100. LR set by the power law fit from the LR sweep (#7856).

## Description

At each compute budget we run several model widths (tokens = budget / flops_per_token_excl), so the loss-vs-width curve at a fixed FLOP count reveals the compute-optimal width. Uses the same architecture as the Aug hero LR sweep (seq 8192, global-every-4, sliding-window 512) so results are comparable.

## Hypothesis or Goal

- Locate the compute-optimal model width at each budget (the U-curve minimum of loss vs size at fixed FLOPs).
- Validate the fitted optimal-LR law `lr = 34.3 · tokens^-0.346 · hidden^-0.345 · batch^0.5` (#7856) as the per-cell LR, out of the fitted range.

## Status

Launching the ~24-cell iso-FLOP band (skipping the degenerate under-/over-trained corners). Batch tied to model size (with per-budget/step-count adjustments); nodes by budget (1/1/1/2/4/8 ×8 H100); FLOPs computed excl embed/lm_head.

## Links

* Parent: #6711
* LR law + sweep: #7856
* Branch: `aug_hero_isoflop`

## Decision Log

## Conclusion

## S8003-01 2026-08-06T23:08:20Z
https://github.com/marin-community/marin/issues/8003#issuecomment-5209786274
🤖 First iso-FLOP results from the `aug_hero_isoflop` sweep (H100, group `aug-hero-isoflop` in `marin_moe`, seq_len 8192, refit LR heuristic from #7856).

## Compute-optimal frontier

At each FLOP budget, paloma macro loss vs tokens is a clean U-curve; the quadratic minimum gives the compute-optimal token count. Fits are solid for the three completed budgets (1e18 / 3e18 / 1e19); the minima (★) trace the compute-optimal frontier.

![iso-FLOP overlay](https://raw.githubusercontent.com/marin-community/marin/2c603278b75d6655c6c8f404c89bb1b1eb7d93b4/experiments/grug/moe_hero_fsdp/isoflop_figs/isoflop_overlay.png)

| budget (FLOPs) | compute-optimal tokens | loss at min | ~optimal width |
|---|---|---|---|
| 1e18 | 2.84e9 | 3.361 | d768 |
| 3e18 | 5.08e9 | 3.198 | d768 |
| 1e19 | 9.90e9 | 3.033 | d1024 |

## Full results (all cells)

paloma macro loss per cell; tokens = steps × batch × 8192.

| budget | width | tokens | paloma macro | status |
|---|---|---|---|---|
| 1e18 | d512 | 5.77e+09 | 3.399 | done |
| 1e18 | d768 | 2.33e+09 | **3.364** | done |
| 1e18 | d1024 | 9.23e+08 | 3.456 | done |
| 3e18 | d512 | 1.73e+10 | 3.298 | done |
| 3e18 | d768 | 7.00e+09 | **3.204** | done |
| 3e18 | d1024 | 2.77e+09 | 3.224 | done |
| 3e18 | d1280 | 1.57e+09 | 3.289 | done |
| 1e19 | d512 | 5.77e+10 | 3.226 | done |
| 1e19 | d768 | 2.33e+10 | 3.071 | done |
| 1e19 | d1024 | 9.23e+09 | **3.039** | done |
| 1e19 | d1280 | 5.23e+09 | 3.058 | done |
| 1e19 | d1536 | 3.29e+09 | 3.105 | done |
| 3e19 | d768 | 7.00e+10 | 2.991 | done |
| 3e19 | d1024 | 2.77e+10 | 11.761 | **diverged** |
| 3e19 | d1280 | 1.57e+10 | 11.762 | **diverged** |
| 3e19 | d1536 | 9.88e+09 | **2.916** | done |
| 1e20 | d1024 | 9.23e+10 | 2.815 | running 98% |
| 1e20 | d1280 | 5.23e+10 | **2.772** | done |
| 1e20 | d1536 | 3.29e+10 | 3.216 | running 21% |
| 1e20 | d2048 | 1.70e+10 | 3.309 | running 30% |
| 3e20 | d1024 | 2.77e+11 | 3.170 | running 6% |
| 3e20 | d1280 | — | — | not started |
| 3e20 | d1536 | — | — | not started |
| 3e20 | d2048 | — | — | not started |

(Bold = best converged loss at that budget. Running-cell losses are not final. `d1280`=2.772 at 1e20 is the study's best converged number so far.)

## Scaling law (L∞ pinned at 1.6, fit on the three converged minima)

```
L(C)  = 1.6 + 72.6 · C^-0.0897
T*(C) ∝ C^0.542   →   N*(C) ∝ C^0.458
```

**vs `experiments/grug/moe/README.md`** (May-recipe compute-optimal baseline, seq 4096: `L = 1.6 + 88.32·C^-0.0941`, `T*∝C^0.464`, `N*∝C^0.535`):

| C | ours (aug-hero) | README compute-opt | README v16 |
|---|---|---|---|
| 1e18 | 3.362 | 3.388 | 3.526 |
| 1e19 | 3.033 | 3.039 | 3.151 |
| 3e19 | 2.899 | 2.898 | 2.999 |
| 1e20 | 2.766 | 2.759 | 2.849 |

aug-hero sits **~on top of the current May compute-optimal baseline** (within ±0.025, crossing ~3e19), and ~0.11–0.16 below the old v16 law. The one structural difference: aug-hero's optimum is **more token-heavy** (T\*∝C^0.54 vs 0.46 → smaller models trained longer), likely partly the longer sequences (8192 vs 4096).

## Per-budget curves

![iso-FLOP per-budget curves](https://raw.githubusercontent.com/marin-community/marin/2c603278b75d6655c6c8f404c89bb1b1eb7d93b4/experiments/grug/moe_hero_fsdp/isoflop_figs/isoflop_curves.png)

## Status

17 / 24 cells done. Usable compute-optimal law across **1e18 → 1e19** now.
- **3e19**: d768 / d1536 clean, but **d1024 / d1280 diverged** — output_proj collapses to ~0 in one step under the MuonH hyperball renorm (a single non-finite update entry lands in the shared Frobenius denominator and zeroes the whole readout → logits uniform → loss pins at ln(V)=11.76). Diagnosis + LR-sensitivity reruns in progress.
- **1e20**: d1280 done (2.772, current best) + d1024 (98%) / d1536 / d2048 running.
- **3e20**: d1024 started (6%); d1280 / d1536 / d2048 not yet scheduled (8-node cells, capacity-gated).

Fits/frontier will close once the top budgets converge and the two diverged 3e19 cells are rerun; ≥1e20 predictions above are extrapolation until then.

Branch `aug_hero_isoflop`; figures + fit script under `experiments/grug/moe_hero_fsdp/isoflop_figs/` and `plot_isoflop.py`.


## S8003-02 2026-08-07T21:06:49Z
https://github.com/marin-community/marin/issues/8003#issuecomment-5222000328
@dlwh Added a managed experiment TL;DR capturing the interim iso-FLOP frontier, current blocker, and decisive result links.

## S8003-03 2026-08-13T18:30:23Z
https://github.com/marin-community/marin/issues/8003#issuecomment-5284819242
Remaining cells killed, we have since updated the architecture from 128 to 192 total experts. And also fixed the divergence issue caused by sharding numerics failure on H100.
