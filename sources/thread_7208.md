# Agent MoE Experiment: Inkling relative position vs July Baseline
<!-- experiment-tldr:start -->
## Summary

This issue tests Inkling-style learned relative attention in the Grug MoE against the fixed July baseline from #6882 at the d512 and d768 Gate 1 cells. The first implementation was stopped immediately because an unfused scan-based attention path was orders of magnitude slower than the fused TPU Splash baseline, so the work shifted to a direct-distance TPU kernel aligned with the released Inkling implementation and covered by value/gradient parity tests. In the corrected rerun, d512 with the current Grug QK scale finished, while the Inkling net-`1/head_dim` QK-scale arm finished with essentially unchanged throughput but materially worse Paloma quality (+13.5% BPB, +0.45972 aggregate eval loss). Both d768 arms were initially stopped for sustained throughput violations around 189k-191k tok/s versus the 249,954 tok/s baseline, then were explicitly continued for diagnostic completion; their first matched step-1,000 comparison also showed a large QKS regression. Current recommendation: reject the net-`1/head_dim` QK-scale variant; treat the relative-position path as not promotable until the remaining d768 continuation data and any separate kernel/quality follow-up change that conclusion.

### Helpful links
- Baseline comparator: #6882
- Kernel bottleneck stop: https://github.com/marin-community/marin/issues/7208#issuecomment-4984492475
- Direct-distance parity and rerun reset: https://github.com/marin-community/marin/issues/7208#issuecomment-4987514360
- Final d512 Gate 1 result: https://github.com/marin-community/marin/issues/7208#issuecomment-4992545932
- d768 continuation status: https://github.com/marin-community/marin/issues/7208#issuecomment-4994301965
<!-- experiment-tldr:end -->

## Description

The variant in `experiments/grug/moe_relative_position/` replaces the baseline RoPE/NoPE positional scheme with a learned query-dependent relative attention bias. Gate 1 uses the July Baseline's exact 8192-token May-Recipe cells and compares final Paloma macro loss plus last-100-step throughput against #6882.

Initiating prompts:

> Find the new position embedding introduced in https://thinkingmachines.ai/news/introducing-inkling/ and then find the most up to date MoE baseline in the official marin library and replace rope with it

> Follow moe/agents.md and launch gate 1 experiment

> Use this baseline instead https://github.com/marin-community/marin/issues/6882

## Hypothesis or Goal

The learned input-dependent relative bias provides effective wall-clock speedup greater than 1 at both July Baseline Gate 1 cells: d512 at 3.82e17 FLOPs and d768 at 2.81e18 FLOPs.

## Status

Gate 1 is partially concluded and still collecting d768 continuation data by explicit user override. The corrected direct-distance, zero-`W_r`, Adam-routed rerun completed both d512 arms: current-scale d512 reached Paloma BPB 1.28971, macro loss 3.97874, aggregate eval loss 3.74715, and 295,603 tok/s; QKS net-`1/128` d512 reached Paloma BPB 1.46320, macro loss 4.42181, aggregate eval loss 4.20687, and 296,313 tok/s. The QKS d512 result is worse at matched compute with no throughput win. Both d768 arms showed sustained low throughput around 189k-191k tok/s, below the matched 249,954 tok/s baseline, and are being continued to final checkpoints for diagnostics rather than stopped by the original gate.

## Links

* Logbook: https://github.com/marin-community/marin/blob/codex/inkling-relative-position/.agents/logbooks/inkling-relative-position-gate1.md
* W&B project: https://wandb.ai/marin-community/dial_moe
* Baseline: #6882
* Launch snapshot: `a496bc044`
* Submission log: `a11b0d424`

## Decision Log

* 2026-07-15: Use #6882 rather than the newer README baseline, per user direction.
* 2026-07-15: Stop the initial scan-based attention run because the unfused path was far too slow to evaluate the embedding fairly.
* 2026-07-16: Switch the kernel API to direct distance order after checking the official Inkling implementation, then restart with fresh identities.
* 2026-07-16: Reject the net-`1/head_dim` QK-scale variant at d512; continue d768 despite low throughput only because the user requested diagnostic completion.

## Conclusion

The current evidence does not support promoting the Inkling net-`1/head_dim` QK-scale variant. The corrected d512 run is materially worse in quality at matched compute with no throughput gain, and the d768 continuations have already shown low throughput plus an early matched Paloma regression for QKS. Final d768 completion may refine diagnostics, but it would need to reverse both quality and throughput signals to change the recommendation.


## S7208-01 2026-07-15T19:13:10Z
https://github.com/marin-community/marin/issues/7208#issuecomment-4984378037
🤖 Update: Gate 1 submitted against the July Baseline

- Change: launched the Inkling-style relative-position variant at the exact #6882 d512 and d768 cells.
- Result delta: pending; both training jobs are running with zero failures and zero preemptions.
- Confidence: submission and task-level startup are confirmed. W&B run initialization and final metrics are still pending.
- Links:
  - Snapshot: https://github.com/marin-community/marin/commit/a496bc044
  - Logbook: https://github.com/marin-community/marin/blob/a11b0d424/.agents/logbooks/inkling-relative-position-gate1.md
  - Parent: https://iris.oa.dev/#/job/%2Fkaiyuew%2Finkling-relative-position-gate1-7208
  - d512: https://iris.oa.dev/#/job/%2Fkaiyuew%2Finkling-relative-position-gate1-7208%2Fgrug-train-MOE-RPE-001-d512
  - d768: https://iris.oa.dev/#/job/%2Fkaiyuew%2Finkling-relative-position-gate1-7208%2Fgrug-train-MOE-RPE-002-d768
  - W&B: https://wandb.ai/marin-community/dial_moe
- Next: wait for both runs to finish, pull final Paloma macro loss plus last-100-step throughput, and compute effective speedup against #6882.

The initial parent submission was rejected before job creation because its inferred non-preemptible constraint conflicted with the preemptible v5p-8 availability group. Retrying with explicit `--preemptible` submitted successfully; no training work was duplicated.


## S7208-02 2026-07-15T19:14:10Z
https://github.com/marin-community/marin/issues/7208#issuecomment-4984385565
🤖 Update: both Gate 1 training runs initialized

- d512: [MOE-RPE-001-d512](https://wandb.ai/marin-community/dial_moe/runs/MOE-RPE-001-d512) — running.
- d768: [MOE-RPE-002-d768](https://wandb.ai/marin-community/dial_moe/runs/MOE-RPE-002-d768) — running.
- Iris logs confirm accelerator initialization, cache loading, and entry into the training loop with zero job failures.
- Updated logbook: https://github.com/marin-community/marin/blob/fe4c8cea6/.agents/logbooks/inkling-relative-position-gate1.md

Final Paloma loss, last-100-step throughput, and effective speedup remain pending until both runs finish.


## S7208-03 2026-07-15T19:27:14Z
https://github.com/marin-community/marin/issues/7208#issuecomment-4984492475
🤖 Update: Gate 1 stopped after attention-kernel bottleneck

- Change: killed both Gate 1 children and the parent; Iris reports all three as `JOB_STATE_KILLED`.
- Result delta: d512 sustained 3,958 tokens/s at 33.115 s/step through step 14; d768 sustained 3,912 tokens/s at 67.005 s/step through step 4. The #6882 baselines are 352,609 and 249,954 tokens/s.
- Confidence: high that the generic nested block scans are the bottleneck; no conclusion about the position embedding itself. The comparator uses fused TPU Splash attention, while the trial used sequential `jax.lax.scan` blocks with unfused bias, masking, and online softmax.
- Links:
  - W&B d512: https://wandb.ai/marin-community/dial_moe/runs/MOE-RPE-001-d512
  - W&B d768: https://wandb.ai/marin-community/dial_moe/runs/MOE-RPE-002-d768
  - Logbook: `.agents/logbooks/inkling-relative-position-gate1.md`
- Next: retain the scan path as the numerical oracle, implement a fused TPU relative-position attention kernel, and require value/gradient parity plus a representative v5p microbenchmark before relaunch.


## S7208-04 2026-07-16T01:42:43Z
https://github.com/marin-community/marin/issues/7208#issuecomment-4987216155
🤖 Update: exact Inkling QK-scale relative-attention ablation is training

- Change: launched one additional d512/d768 pair with relative attention unchanged and the normalized-QK scale changed from the Grug control's `1.3 / sqrt(head_dim)` to Inkling's `1 / head_dim`. With head dimension 128, `qk_mult=0.0883883476` combines with the backend scale to give `1 / 128`; relative-position logits remain unscaled. No RoPE scaling arm was launched.
- Startup result: both children are running on their first attempts with zero failures or preemptions. Over matched 50-step windows, d512-QKS averaged 345,498 tok/s versus 348,274 for the current-scale control (-0.8%); d768-QKS averaged 231,328 versus 227,553 (+1.7%). These deltas are consistent with noise, as expected from changing only a scalar.
- Early quality signal: d512-QKS loss averaged 4.9393 versus 4.7412 over steps 464-513; d768-QKS averaged 8.9522 versus 8.9330 over steps 70-119. This is exploratory startup evidence, not a quality conclusion.
- Confidence: high that the implementation matches the released Inkling QK formula; final loss/effective-speedup conclusions remain pending scheduled evaluations.
- Links:
  - Parent: https://iris.oa.dev/#/job/%2Fkaiyuew%2Finkling-relative-position-qks-gate1-7208
  - d512 W&B: https://wandb.ai/marin-community/dial_moe/runs/MOE-RPE-QKS-001-d512
  - d768 W&B: https://wandb.ai/marin-community/dial_moe/runs/MOE-RPE-QKS-002-d768
  - Pinned Inkling implementation: https://github.com/huggingface/transformers/blob/dadf67c51a9b97d86e123bb967b9faf80a9582e5/src/transformers/models/inkling/modular_inkling.py
- Next: continue both cells, compare matched-step trajectories, then use final Paloma macro loss plus last-100-step throughput for the decision.


## S7208-05 2026-07-16T02:22:14Z
https://github.com/marin-community/marin/issues/7208#issuecomment-4987514360
🤖 Official-code parity against Transformers merge commit c375c5e8666aee864528aa52e8287b3ba82232bb found a table-order mismatch: the released Inkling implementation gathers relative column d directly, while the optimized TPU reverse tile layout had leaked through the Marin kernel API.

The API now accepts direct distance order and flips only at the Pallas boundary. On identical packed causal tensors, maximum forward error is 7.451e-9 for both JAX reference and interpreted Pallas; maximum Q/K/V gradient error is below 5.821e-11, and hidden-relative/table gradient errors are 1.281e-9 and 2.328e-9.

At user request, every nonterminal old MOE-RPE, QKS, and WRA job was stopped and the old heartbeat deleted. Fresh direct-order parents were submitted with new artifact/W&B identities, so no reverse-order checkpoint can be resumed:

- /kaiyuew/inkling-relative-position-direct-gate1-7208
- /kaiyuew/inkling-relative-position-direct-qks-gate1-7208
- /kaiyuew/inkling-relative-position-direct-wr-adam-gate1-7208

All three parents are running startup setup with zero failures/preemptions. Control and WRA keep qk_mult=1.3; QKS alone uses net 1/head_dim; WRA alone routes W_r to Adam.

## S7208-06 2026-07-16T02:45:57Z
https://github.com/marin-community/marin/issues/7208#issuecomment-4987652526
🤖 Final rerun configuration is submitted. I stopped the three superseded direct-order job trees and replaced them with only two fresh pairs: current Grug QK scaling and exact Inkling QK scaling. Both now use direct relative distances, zero-initialized W_r, and Adam for W_r; the relative table remains randomly initialized and Adam so W_r receives a first-step gradient. Other MuonH parameter groups are unchanged. All four children have the intended fresh identities and are pending v5p-8 capacity with zero failures. Focused tests pass (9), the full Grug contract passes (17 passed, 1 skipped), and changed-file lint/Pyrefly are clean.

## S7208-07 2026-07-16T11:22:47Z
https://github.com/marin-community/marin/issues/7208#issuecomment-4991242319
🤖 Gate 1 update: the current-scale d768 zero-W_r/Adam run was stopped at W&B step 102 by the throughput gate. About 30 consecutive samples stayed near 189k–191k tok/s; the final observed value was 190,674 tok/s, 76.3% of the matched 249,954 baseline and below the 199,963 minimum. Loss remained finite at 8.585. The d512 sibling remains healthy at step 2,741 with loss 3.7885 and 295,763 tok/s, above its 282,087 threshold, so it continues. Both QKS children remain capacity-pending with zero failures/preemptions.

## S7208-08 2026-07-16T11:51:42Z
https://github.com/marin-community/marin/issues/7208#issuecomment-4991486168
🤖 Gate 1 update: QKS d768 was also stopped by the throughput gate. Forty consecutive samples stayed near 187k–190k tok/s; the final observed point was step 439, loss 4.5716, and 189,405 tok/s, 75.8% of the matched 249,954 baseline and below the 199,963 minimum. Both d768 cells are now stopped. Current-scale d512 remains healthy at step 5,999 and 296,405 tok/s; QKS d512 is healthy at step 1,286 and 297,609 tok/s. Both d512 runs continue.

## S7208-09 2026-07-16T12:11:48Z
https://github.com/marin-community/marin/issues/7208#issuecomment-4991665376
🤖 First matched d512 Paloma result: at step 3,000, current-scale qk_mult=1.3 reaches Paloma BPB 1.43043 and aggregate eval loss 4.14920, while QKS net-1/128 reaches BPB 1.67951 and eval loss 4.80520. That is a +0.24907 (+17.4%) Paloma BPB regression for QKS at matched compute. Throughput is essentially unchanged (~296k vs ~298k tok/s), and both d512 runs remain above the gate, so both continue to the full 10,980-step horizon.

## S7208-10 2026-07-16T12:42:34Z
https://github.com/marin-community/marin/issues/7208#issuecomment-4991959926
🤖 Current-scale direct/zero-`W_r`/Adam d512 has completed successfully.

- Iris child: `succeeded`; W&B: `finished` at step 10,979/10,980
- Final train loss: 3.55470; throughput: 295,603 tok/s (above the 282,087 gate)
- Final Paloma: 1.28971 BPB; macro loss 3.97874; aggregate eval loss 3.74715
- Final step-10,980 checkpoint committed, and `metadata.json` was directly verified in GCS

The control parent reports `failed` only because its d768 sibling was intentionally killed for the sustained throughput violation, so I am not resubmitting it. QKS d512 remains healthy at step 6,919/10,980 with finite loss 3.6820 and 298,114 tok/s; it is continuing to the matched final horizon.

## S7208-11 2026-07-16T13:01:11Z
https://github.com/marin-community/marin/issues/7208#issuecomment-4992158569
🤖 QKS d512 recovered automatically from two worker-reconcile preemptions. Attempt 2 resumed from the step-6,408 temporary checkpoint and has now crossed the rollback boundary; W&B is fresh at step 7,240/10,980 with finite loss 3.9231 and 296,055 tok/s (above the 282,087 gate). There are zero task failures and no manual/helper restart.

At matched step 7,000, current-scale Paloma BPB is 1.38097 versus 1.53829 for QKS net-`1/128`, a +0.15733 (+11.4%) regression; aggregate eval loss is 4.00849 versus 4.41449. The gap has narrowed from +13.0% at step 6,000 but remains material. W&B config still verifies direct distance order, zero-initialized Adam-routed `W_r`, `R=16`, `E=1024`, and `qk_mult=1/sqrt(128)`.

## S7208-12 2026-07-16T13:37:07Z
https://github.com/marin-community/marin/issues/7208#issuecomment-4992545932
🤖 Final Gate 1 result: all four children are terminal, and both successful d512 checkpoints are verified.

QKS d512 finished after two automatic worker preemptions:
- Iris `succeeded`; W&B `finished` at step 10,979/10,980
- Train loss 3.72563; throughput 296,313 tok/s
- Final Paloma BPB 1.46320; macro loss 4.42181; aggregate eval loss 4.20687
- Step-10,980 checkpoint committed and `metadata.json` verified in GCS

Matched current-scale d512 finished at 1.28971 Paloma BPB and 3.74715 eval loss with 295,603 tok/s. QKS therefore ends +0.17348 BPB (+13.5%) and +0.45972 eval loss worse, with effectively unchanged throughput. The QKS BPB deficit remained positive at every scheduled matched evaluation (+17.2% at step 1k, narrowing to +10.6% at 9k, then ending +13.5%).

Both d768 cells remain intentionally killed for sustained throughput violations (190,674 and 189,405 tok/s), with terminal crashed W&B records. The two parent `failed` states are expected consequences of those stopped siblings; neither parent should be resubmitted. Gate 1 rejects the net-`1/128` QK scaling variant.

## S7208-13 2026-07-16T15:41:01Z
https://github.com/marin-community/marin/issues/7208#issuecomment-4993801551
🤖 Reopened d768 monitoring by explicit user request: both attention-scale d768 arms are being continued to completion despite the prior throughput gate. Low throughput is now recorded but will not stop a finite run.

Both d768 children are currently v5p capacity-pending with zero failures. The pair launchers also briefly enqueued duplicate d512 children because their completed checkpoints did not leave reusable artifact success records; those duplicates were terminated before allocation or training. A new d768-only recovery helper is now recorded for any subsequent retry.

Configurations remain unchanged: direct distance order, zero-initialized Adam-routed `W_r`, `R=16`, `E=1024`; control uses `qk_mult=1.3`, and QKS uses `qk_mult=1/sqrt(128)`. Both d768 cells will be monitored through final checkpoints and matched Paloma evaluations.

## S7208-14 2026-07-16T15:53:23Z
https://github.com/marin-community/marin/issues/7208#issuecomment-4993919785
🤖 Both d768 continuations are now running with exact identities verified and zero failures/preemptions.

- Current scale: restarted from scratch; step 265/16,875, finite loss 5.2647, 190,353 tok/s
- QKS net-`1/128`: resumed from temporary checkpoint step 231; currently step 494/16,875, finite loss 4.5470, 188,920 tok/s
- Both configs: direct distance order, zero-initialized Adam-routed `W_r`, `R=16`, `E=1024`; `qk_mult=1.3` versus `1/sqrt(128)`

Both duplicate d512 children remain killed before allocation. Throughput is being recorded but will not stop these continuations per the user override; monitoring continues through matched Paloma evaluations and final checkpoints.

## S7208-15 2026-07-16T16:34:57Z
https://github.com/marin-community/marin/issues/7208#issuecomment-4994301965
🤖 d768 continuation update: the first matched step-1,000 Paloma comparison landed. Current-scale control BPB is 1.423453 versus 1.693407 for the Inkling net-1/128 QK scale, an 18.96% regression for QKS; control/QKS eval losses are 4.130478/4.846550. Control advanced to step 1,077 with finite loss before a second infrastructure interruption (`worker reconcile failure threshold exceeded`) and Iris is automatically retrying attempt 2 with zero task failures, so no manual resubmit is appropriate. QKS remains healthy at step 2,064 with finite loss, 189,746 tokens/s, and temporary checkpoint step 1,850. Both d768 arms remain scheduled to finish; low throughput is recorded but is not a stop condition.

## S7208-16 2026-07-16T17:23:55Z
https://github.com/marin-community/marin/issues/7208#issuecomment-4994728669
@dlwh Added a managed experiment TL;DR block, refreshed the body status/conclusion from the latest Gate 1 comments, and applied the `tldr` label.

## S7208-17 2026-07-17T01:01:56Z
https://github.com/marin-community/marin/issues/7208#issuecomment-4997967809
🤖 QKS d768 has completed successfully. `MOE-RPE-DIR-ZRA-QKS-002-d768` finished at W&B global step 16,874 with finite train loss 3.625795, 191,484 tokens/s, final Paloma BPB 1.366413, macro BPB 1.488835, and eval loss 3.933637. The permanent `step-16875` checkpoint is committed; `metadata.json` reports step 16,875, `is_temporary: false`, timestamp `2026-07-17T00:28:09.169073`. The QKS pair parent is red only because the intentionally excluded d512 sibling was killed; the required d768 child succeeded with zero failures after four preemptions.

Control d768 remains healthy at step 15,093 with confirmed temporary checkpoint 14,995. Matched Paloma BPB gaps are +4.02% QKS at step 14,000 and +4.99% at step 15,000. Monitoring continues until control reaches step 16,875 and its final checkpoint/evaluation are verified.

## S7208-18 2026-07-17T01:50:54Z
https://github.com/marin-community/marin/issues/7208#issuecomment-4998211016
🤖 Both required d768 continuations are complete and fully verified.

- Control: child succeeded with zero failures (one preemption); W&B finished at global step 16,874 with train loss 3.693067 and 190,658 tok/s. Final Paloma BPB 1.305315, macro BPB 1.437967, macro loss 4.023966, eval loss 3.789962.
- QKS (net 1/128 QK scale): child succeeded with zero failures (four preemptions); W&B finished at global step 16,874 with train loss 3.625795 and 191,484 tok/s. Final Paloma BPB 1.366413, macro BPB 1.488835, macro loss 4.155726, eval loss 3.933637.
- Final matched result: QKS Paloma BPB is +4.68% versus control.
- Both permanent step-16,875 checkpoints and `metadata.json` were verified with `is_temporary: false`.

The pair parents are red only because the excluded d512 siblings were intentionally killed; both required d768 children themselves succeeded. Monitoring is complete.

## S7208-19 2026-07-17T23:19:20Z
https://github.com/marin-community/marin/issues/7208#issuecomment-5008441671
🤖 Launched one new d768 Gate-1 comparison that matches the released Inkling relative-attention parameterization on top of Grug: learned per-dimension Q/K RMSNorm gains (separate Q/K, shared across heads, initialized to 1, Adam), direct R=16/E=1024 relative bias, net 1/128 content scale, and untruncated N(0,0.02) Adam-routed initialization for both relative factors. Identity: parent `/kaiyuew/inkling-relative-position-direct-inkling-parameterized-gate1-7208`, sole child/W&B `MOE-RPE-DIR-INKP-002-d768`, group `MOE-RPE-DIR-INKP-gate1-issue-7208`. The corrected child is currently building with zero failures/preemptions. A first pre-allocation bundle was intentionally killed before W&B/training to preserve bfloat16 at the attention-kernel boundary; no duplicate or checkpoint state remains. Focused validation: 27 passed, one expected GPU-only skip.

## S7208-20 2026-07-18T01:11:01Z
https://github.com/marin-community/marin/issues/7208#issuecomment-5009089072
🤖 Correction and clean relaunch: the first INKP pair was intentionally stopped because W&B showed that the relative table and W_r inherited the width-scaled Grug model initializer (d512 0.0220971; d768 0.0180422) instead of released Inkling fixed 0.02. d512 had not logged a training step and d768 stopped at step 33, so neither v1 run is a valid result.

Fresh INKP2 d512 and d768 parents now each have exactly one expected TPU child running. W&B runtime configs confirm learned Q/K RMS gains, direct R=16/E=1024 relative bias, qk_mult=1/sqrt(128) for net 1/128, NORMAL W_r/table initialization with relative_position_initializer_std=0.02, and Adam routing for W_r. Both children had one automatically recovered setup preemption, failure_count=0; no manual recovery was used. They are still initializing and have not logged their first training metric yet.

- d512: https://wandb.ai/marin-community/dial_moe/runs/MOE-RPE-DIR-INKP2-001-d512
- d768: https://wandb.ai/marin-community/dial_moe/runs/MOE-RPE-DIR-INKP2-002-d768

## S7208-21 2026-07-18T01:23:10Z
https://github.com/marin-community/marin/issues/7208#issuecomment-5009151390
🤖 Clean INKP2 pair has reached healthy training with fresh, finite metrics and no runtime-error signals. Both exact parents and sole expected children are running with failure_count=0; each child had one setup preemption that Iris recovered automatically, and no manual restart was used.

- d512: step 928/10,980, train/loss 4.278149, 295,329 tok/s
- d768: step 269/16,875, train/loss 5.228224, 190,457 tok/s

Runtime configs still match the intended Inkling parameterization, including learned Q/K RMS gains, net 1/128 QK scaling, direct R=16/E=1024 relative bias, and fixed relative_position_initializer_std=0.02. Paloma evaluation is scheduled every 1,000 steps for both arms (batch 256, eight batches per dataset, current weights).

## S7208-22 2026-07-18T06:06:29Z
https://github.com/marin-community/marin/issues/7208#issuecomment-5010176425
🤖 Both corrected INKP2 arms automatically recovered from the v5p capacity wait and completed checkpoint catch-up with no manual restarts. d512 is freshly logging at step 1,931/10,980 with finite train loss 3.869757 and 296,450 tok/s; d768 is at step 630/16,875 with finite train loss 4.170486 and 189,711 tok/s. Both exact parents and sole same-width children are running with zero failures and no runtime-error signals. The scheduled Paloma evaluations remain active (d512 next at 2,000; d768 first at 1,000). Both manual restart counters remain zero.

## S7208-23 2026-07-18T07:24:59Z
https://github.com/marin-community/marin/issues/7208#issuecomment-5010401292
🤖 The corrected d512 INKP2 arm was preempted again near completion after reaching W&B step 9,570/10,980. Iris automatically resumed the exact same-width child from durable checkpoint step 8,533; live logs are already advancing with finite loss and no runtime-error signal. This is a zero-failure automatic recovery, so no manual resubmission was used and restart_count remains 0. d768 remains uninterrupted and healthy at step 3,749/16,875 with finite loss 3.644836 and 190,568 tok/s.

## S7208-24 2026-07-18T07:51:59Z
https://github.com/marin-community/marin/issues/7208#issuecomment-5010484653
🤖 Corrected INKP2 d512 is terminal-success. Iris parent and sole d512 child both succeeded with exit code 0 and zero failures; W&B `MOE-RPE-DIR-INKP2-001-d512` is finished at global step 10,979/10,980 with finite train loss 3.543131, throughput 297,665 tok/s, and 1,439,170,560 total tokens. Final Paloma is macro loss 3.967966, micro loss 3.737684, macro BPB 1.419132, and aggregate BPB 1.288077. The permanent step-10,980 checkpoint was directly verified in GCS; `metadata.json` reports step 10,980, timestamp `2026-07-18T07:45:40.365226`, and `is_temporary=false`.

The matched d512 result is the key signal: learned Q/K RMSNorm gains recover the parameter-free net-1/128 deficit (Paloma BPB 1.288077 vs 1.463196, -11.97%; macro loss 3.967966 vs 4.421813) and essentially tie the prior current-scale relative-attention control (BPB 1.289713; macro loss 3.978742) at similar throughput. It still trails the July baseline (BPB 1.136406; macro loss 3.566663) and is 16.08% slower (297,665 vs 354,706 tok/s).

The corrected d768 arm remains healthy and exact-identity at step 4,729/16,875 with finite loss 3.689411 and throughput 190,368 tok/s; Iris shows zero failures, no error signal, and the latest durable temporary checkpoint is step 4,633. No manual recovery has been used.

## S7208-25 2026-07-18T15:17:53Z
https://github.com/marin-community/marin/issues/7208#issuecomment-5011781107
🤖 Corrected INKP2 Gate 1 is complete at both widths.

d768 is terminal-success: Iris reports both the parent and sole child grug-train-MOE-RPE-DIR-INKP2-002-d768 as Succeeded, with zero failures, two automatically recovered preemptions, and no manual recovery. W&B MOE-RPE-DIR-INKP2-002-d768 finished at global step 16,874/16,875 with finite train loss 3.349397, throughput 190,718 tok/s, and 4,423,680,000 total tokens. Final Paloma is macro loss 3.658141, micro loss 3.428574, macro BPB 1.310132, aggregate BPB 1.179464, eval loss 3.425019, and eval BPB 1.163571.

The permanent checkpoint is verified at gs://marin-us-central1/users/root/grug/moe_relative_position_direct_inkling_parameterized_v2_gate1_d768/issue-7208-direct-inkling-parameterized-v2-dev/checkpoints/step-16875; metadata reports step 16,875, timestamp 2026-07-18T12:58:28.943516, and is_temporary=false. Runtime config exactly matches learned per-dimension Q/K RMSNorm gains initialized to one and Adam-routed, hidden/head dimensions 768/128, direct R=16/E=1024 relative bias, net 1/128 content scaling, independent untruncated N(0, 0.02) relative factors, and model initializer std 0.018042195912175808.

Matched d768 comparisons: learned gains improve final Paloma BPB by 13.68% and macro loss by 11.97% versus parameter-free QKS, and improve BPB by 9.64% and macro loss by 9.09% versus the prior current-scale relative control at essentially identical throughput. They still trail the July baseline by 15.75% BPB and 13.35% macro loss while running 23.59% slower.

Combined verdict: d512 essentially tied the prior current-scale control while recovering 11.97% BPB versus parameter-free QKS; d768 materially surpassed both relative-attention references. Inkling-style learned Q/K gains explain a substantial part of the earlier QKS quality gap, but do not close the gap to the July baseline.

## S7208-26 2026-07-19T05:57:57Z
https://github.com/marin-community/marin/issues/7208#issuecomment-5014625053
🤖 Update: real July-baseline RoPE + relative-attention Gate 1 launched

- Change: rebased the variant directly on `marin/july_baseline@52d8a9e`, preserved its GQA and half-RoPE policy (including long-layer RoPE disablement), and added the validated Inkling parameterization: learned Q/K RMSNorm gains, direct R=16/E=1024 relative bias, net 1/128 content scaling, independent untruncated N(0,0.02) W_r/table initialization, and Adam routing for the new parameters.
- Validation: 23 model/recipe/Grug contract tests, 4 Pallas value+gradient tests, and the full pre-commit/type gate pass.
- Launch: `/kaiyuew/july-baseline-rope-relative-attention-gate1-7208` now has exactly the d512 and d768 v5p-8 children. Both are pending normal demand-routed capacity with zero failures/preemptions; no sibling cells were launched.
- Fresh W&B group: `MOE-JULY-ROPE-RPE-INKP-gate1-issue-7208`
- Links:
  - Branch snapshot: https://github.com/WhenWen/marin/commit/87608af66
  - Logbook: https://github.com/WhenWen/marin/blob/87608af66/.agents/logbooks/inkling-rope-relative-attention-gate1.md
- Next: verify allocation, first finite losses, exact W&B configs, then babysit both cells through final checkpoints and matched Paloma comparisons.


## S7208-27 2026-07-19T06:27:11Z
https://github.com/marin-community/marin/issues/7208#issuecomment-5014700687
🤖 The first real-July-baseline RoPE+relative-attention Gate 1 parent reached both TPUs but both arms failed before step 1 with zero metrics. W&B output logs showed the same deterministic compatibility error: the ported Pallas output-shape helper accessed `manual_axis_type` on July's JAX 0.9.2 `ShapedArray`.

The compatibility fix is `e1303c0fd`; it removes that later-JAX-only metadata access. Kernel value/gradient tests and targeted Ruff/Black/Pyrefly checks pass. A corrected fresh-identity snapshot (`7473d0f07`) is now launched as:

- parent: `/kaiyuew/july-baseline-rope-relative-attention-v2-gate1-7208`
- d512: `MOE-JULY-ROPE-RPE-INKP2-001-d512`
- d768: `MOE-JULY-ROPE-RPE-INKP2-002-d768`

Exactly those two corrected children exist with no siblings; both are currently pending normal demand-routed v5p capacity with zero failures/preemptions. The original parent and original `INKP` W&B runs are terminal and will not be reused.


## S7208-28 2026-07-19T06:41:22Z
https://github.com/marin-community/marin/issues/7208#issuecomment-5014736458
🤖 Follow-up on the real-July-baseline RoPE+relative-attention Gate 1 startup:

- The `INKP2` d512 run passed the earlier JAX API fault, then exposed a July JAX 0.9.2 compile-time scoped VMEM limit: the default 512-token backward DQ tile requested 16.12 MB against v5p's 16.00 MB limit. It emitted no step/loss. The d768 sibling was stopped before completing the same unusable compile.
- Final recovery commit `2e1f5f9c3` explicitly sends the existing configured `attention_block_size=128` to every Pallas forward/backward tile. Model/data/batch/sequence length/optimizer/initialization/positional math are unchanged. Eight model/recipe tests, four kernel value/gradient tests, and the complete offline pre-commit/type suite pass.
- Final fresh identities are now launched under `/kaiyuew/july-baseline-rope-relative-attention-v3-gate1-7208`, with W&B runs `MOE-JULY-ROPE-RPE-INKP3-001-d512` and `MOE-JULY-ROPE-RPE-INKP3-002-d768`.

Exactly those two children materialized. Both workers were preempted once during startup and are currently retained by Iris as running/pending with zero failures while replacement demand-routed workers are acquired. This is the final configured recovery; it will be monitored in place without another relaunch.


## S7208-29 2026-07-19T06:52:30Z
https://github.com/marin-community/marin/issues/7208#issuecomment-5014765717
🤖 Final July-baseline RoPE + relative-attention Gate 1 recovery is now training at both widths. Both `INKP3` cells acquired replacement v5p-8 workers, passed TPU compilation with the explicit 128-token Pallas tiles, and have zero Iris failures.

- d512: step 233, finite train loss 5.7049, 211,660 tokens/s
- d768: step 12, finite train loss 11.7447, 128,225 tokens/s

The live W&B configs match the intended July half-RoPE/GQA plus Inkling parameterization (learned Q/K RMS gains, direct R=16/E=1024 relative term, normal std=0.02 table and W_r, net 1/128 content scale, Adam routing for new parameters). Monitoring continues through terminal checkpoints and matched Paloma evaluation; no further relaunch is permitted.

## S7208-30 2026-07-19T07:45:08Z
https://github.com/marin-community/marin/issues/7208#issuecomment-5014910889
🤖 Both final July half-RoPE + Inkling-relative-attention Gate 1 widths have now completed scheduled Paloma evaluations after recovering in place from the infrastructure preemptions. Iris remains at zero failures for both.

- d512: step 3660, finite train loss 3.7645, 211,582 tokens/s; Paloma macro improved 4.6831 @1k → 4.4868 @2k → 4.3565 @3k
- d768: step 1115, finite train loss 3.6962, 128,363 tokens/s; first Paloma macro 4.3271 @1k

Both also have advancing temporary recovery checkpoints with valid metadata (d512 step 2963, d768 step 954). These are trajectory values, not the final matched baseline comparisons; monitoring continues to terminal permanent checkpoints.

## S7208-31 2026-07-19T09:35:42Z
https://github.com/marin-community/marin/issues/7208#issuecomment-5015225009
🤖 d512 is terminal and fully verified for the final July-baseline half-RoPE + Inkling-relative-attention v3 run.

- Iris: succeeded, exit 0, 3 infrastructure preemptions, 0 failures
- W&B: finished at global step 10,979; final train loss 3.5351; Paloma macro loss 3.9639 (BPB 1.4173)
- Permanent checkpoint: `gs://marin-us-east5/grug/moe_july_rope_relative_position_inkling_gate1_v3_d512-e1a437/checkpoints/step-10980`; metadata step 10,980, `is_temporary=false`
- Matched July baseline: 3.5667, so RoPE+relattn is +0.3972 (+11.14%) worse
- Matched prior INKP2: 3.9680, so RoPE+relattn is -0.0041 (-0.10%) better, effectively tied
- Throughput: 211k tok/s, informational; 40.49% below July baseline and 29.09% below INKP2

d768 remains healthy and running at step 3,637; I will post the combined Gate 1 result after its exact terminal checkpoint and matched comparison are verified.

## S7208-32 2026-07-19T19:14:12Z
https://github.com/marin-community/marin/issues/7208#issuecomment-5017034085
🤖 Final Gate 1 result for the real-July-baseline half-RoPE + Inkling relative-attention cells (`INKP3`): both sole expected children and the parent succeeded with exit code 0 and zero application failures. W&B is finished and both exact permanent checkpoints are complete/non-temporary.

| width | final Paloma macro / BPB | throughput | vs real July baseline | vs prior INKP2 |
|---|---:|---:|---:|---:|
| d512 | 3.9639 / 1.4173 | 211,082 tok/s | +0.3972 (+11.14%) loss; -40.49% throughput | -0.0041 (-0.10%) loss; -29.09% throughput |
| d768 | 3.7001 / 1.3244 | 128,122 tok/s | +0.4728 (+14.65%) loss; -48.67% throughput | +0.0419 (+1.15%) loss; -32.82% throughput |

Final runs:
- d512: `MOE-JULY-ROPE-RPE-INKP3-001-d512`, W&B step 10979, train loss 3.5351, checkpoint `gs://marin-us-east5/grug/moe_july_rope_relative_position_inkling_gate1_v3_d512-e1a437/checkpoints/step-10980` (metadata timestamp `2026-07-19T09:33:20.369358`).
- d768: `MOE-JULY-ROPE-RPE-INKP3-002-d768`, W&B step 16874, train loss 3.3839, checkpoint `gs://marin-us-east5/grug/moe_july_rope_relative_position_inkling_gate1_v3_d768-b72fec/checkpoints/step-16875` (metadata timestamp `2026-07-19T19:09:47.039041`).

The final config identity matches the real July baseline plus retained GQA/half-RoPE (long layers still disable RoPE), PKO off, and the specified direct R16/E1024 Inkling relative-attention path with learned Q/K RMS gains, net 1/128 content scale, Adam-routed new parameters, and 128-token Pallas tiles.

Conclusion: keeping July half-RoPE while adding this relative-attention cell does not beat the real July baseline at either width and does not improve consistently over the prior no-RoPE INKP2 cell. Gate 1 does not support advancing this combined architecture unchanged. Leaving the issue open.

