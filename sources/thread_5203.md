# Agent MoE Experiment: AdamH embed with init std=1.0 (RMSNorm no-op)
## TL;DR
Move token_embed to AdamH (lr_mult=1.0) and set embed init std to 1.0 so each row has norm ~sqrt(d). This makes the first RMSNorm a near-no-op at init. AdamH preserves this norm throughout training.

## User prompt
> follow agent.md and test moving the embeds to adamh and setting embed init std to 1 instead of 0.5/sqrt(d).

## Scope
- **Parent**: #4281
- **Branch**: `moe_adamh_embed_init`
- **Sweep file**: `experiments/grug/moe/adamh_embed_init_sweep.py`
- **Config**: `embed_initializer_std=1.0`, `embed_adamh_lr_mult=1.0`

## Gate 1 runs (2 total)
| Dim | Budget | Status |
|-----|--------|--------|
| 512 | 2.19e17 | pending |
| 768 | 1.70e18 | pending |

## Decision log
_empty_

## Conclusion
_pending_

## S5203-01 2026-04-28T00:21:11Z
https://github.com/marin-community/marin/issues/5203#issuecomment-4331424023
🤖 **Gate 1 Results: AdamH Embed with init std=1.0 (macro_loss)**

Both runs finished.

| Dim | Run | macro_loss | tps | tps ratio | Eff Speedup |
|-----|-----|-----------|-----|-----------|-------------|
| 512 | baseline | 3.8104 | 405,630 | — | — |
| 512 | adamh-embed-init1 | 3.7995 (−0.011) | 404,592 | 0.997 | 1.051 |
| 768 | baseline | 3.4339 | 273,532 | — | — |
| 768 | adamh-embed-init1 | 3.4252 (−0.009) | 273,613 | 1.000 | 1.052 |

Comparison with AdamH embed at default init (init std=0.5/sqrt(d)):

| Dim | Default init speedup | Init std=1.0 speedup |
|-----|---------------------|---------------------|
| 512 | 1.048 | 1.051 |
| 768 | 1.040 | 1.052 |

The larger init slightly improves d768 (1.052 vs 1.040) while d512 is similar. Making RMSNorm a near-no-op at init gives a small but consistent benefit, especially at d768.

**Verdict: PASS gate 1 (1.05× both scales).** Slightly better than default init at d768.

## S5203-02 2026-04-29T17:53:00Z
https://github.com/marin-community/marin/issues/5203#issuecomment-4346192244
🤖 **Gate 2 Results: AdamH embed with init std=1.0**

All 4 scales finished.

| Scale | Budget | BL loss | Var loss | BL tps | Var tps | Speedup |
|-------|--------|---------|----------|--------|---------|---------|
| d512 | 2.19e17 | 3.8104 | 3.7995 | 407,378 | 406,205 | 1.051 |
| d768 | 1.70e18 | 3.4339 | 3.4252 | 274,431 | 273,853 | 1.050 |
| d1024 | 9.00e18 | 3.1605 | 3.1567 | 176,935 | 177,712 | 1.031 |
| d1280 | 2.83e19 | 3.0065 | 3.0048 | 128,435 | 128,584 | 1.014 |

Scaling law (pinned α=0.0941):
- Baseline A=94.98, Variant A=94.66
- 1e21: baseline=2.6035, variant=2.6002
- 1e23: baseline=2.2506, variant=2.2484

**Gate 2: PASS** — speedup > 1 at all 4 scales and better projections at both 1e21 and 1e23. The improvement is strongest at small scale (~5%) and diminishes at larger scale (~1.4%), but remains positive throughout.

## S5203-03 2026-04-29T18:18:11Z
https://github.com/marin-community/marin/issues/5203#issuecomment-4346365771
std=1 is creating gradient norms below 0.001. Preferring the default AdamH approach over AdamH with init std=1.0.Gives identical performance, but has grad norms that don't become vanishingly small.

<img width="881" height="528" alt="Image" src="https://github.com/user-attachments/assets/3af94aae-a4a3-4598-a55a-a346403a32df" />
