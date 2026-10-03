# Experiment: Hero throughput and expert drops at 8K and 16K context
Agent-driven experiment issue: the runs and this record were produced by a Claude agent, supervised by @mcwitt.

## TL;DR

On one GB200 NVL72 rack, restoring the hero's step-180000 checkpoint at a fixed 4.19M tokens per step, 8K context trains at 297.0k tokens/s, 1.6% below 4K, and 16K at 292.5k tokens/s, 3.1% below 4K. The fraction of expert assignments dropped rises from 1.9e-4 at 4K to 8.3e-4 at 8K (4.5×) and 2.9e-3 at 16K (16×). The drops come from the receiving shard's per-chunk capacity; the ragged transport counts all of them as `sender_dropped` (see the correction comment). Each arm has three seeds; the seed-to-seed standard deviation is at most 0.03 MFU points and 4% to 7% of the drop fraction.

## Description

The hero plan in #8435 moves from 4K to 8K sequences at 50% of training (step ~195k) and to 65K at ~95%. These runs measure what the 8K switch costs in throughput and expert-capacity drops before it happens, with 16K as the next point and a 4K control from the same checkpoint.

Setup:

* Source: `main` at f38da1173de79f1eacd5f54bcfb88db6cb0a9f6e with no code changes. Launcher: `experiments/grug/moe_hero_ep/launch_diagnostics.py`.
* Checkpoint: `s3://marin-us-east-02a/marin/grug/hero-fa4sm100-nomask-step146k/2026.08.19.2/checkpoints/step-180000`, the newest permanent hero checkpoint on 2026-09-30. The current hero run id records that it started from step 146k.
* Hardware and parallelism: one rack (16 nodes × 4 GB200), EP64, the hero's ragged all-to-all transport and capacity factor 1.15.
* Every arm keeps 4,194,304 tokens per step (65,536 per GPU), so the learning rate from the hero's heuristic (`MoeHeuristic` in `experiments/grug/moe_hero_ep/heuristic.py`, which scales with tokens per step and total tokens) is the same in all arms. `--schedule-steps 390251` is the hero's full schedule length, so the restored step sits at the hero's position in the learning-rate schedule.
* `qk_mult`, the multiplier on the query-key attention scale, follows the #8435 extension rule `1.3 × (0.1 × ln(seq / 4096) + 1)`.
* Each run trains steps 180000 to 180099 with the production watch interval (10) and coordinated GC interval (100). There is no eval and no checkpoint write.
* The trainer seed sets the data shuffle, so seeds 0, 1, and 2 train on different batches.
* Statistics cover steps 180010 to 180099 (90 steps). Each run contributes its median MFU, median tokens/s, and mean drop fraction; the tables report mean ± standard deviation across the three seeds. Starting the window at step 180030 instead moves no run's median MFU by more than 0.03 points.

| Arm | `--seq-len` | `--batch-size` | `--qk-mult` |
| --- | --- | --- | --- |
| 4K control | 4096 | 1024 | 1.3 |
| 8K | 8192 | 512 | 1.39 |
| 16K | 16384 | 256 | 1.48 |

Command for one run (8K, seed 0):

```bash
python -m experiments.grug.moe_hero_ep.launch_diagnostics \
  --run-id mhep-ctx8k-s0-20260930 --seed 0 \
  --num-steps 180100 --schedule-steps 390251 \
  --batch-size 512 --seq-len 8192 --qk-mult 1.39 --gc-interval 100 \
  --restore-from s3://marin-us-east-02a/marin/grug/hero-fa4sm100-nomask-step146k/2026.08.19.2/checkpoints/step-180000 \
  --version dev --run
```

## Results

| Arm | tokens/s | vs 4K | step time | MFU | drop fraction | vs 4K |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | 301.9k ± 0.2k | | 13.89 s | 28.25 ± 0.02% | 1.86e-4 ± 0.08e-4 | 1× |
| 8K | 297.0k ± 0.2k | −1.6% | 14.12 s | 28.47 ± 0.02% | 8.29e-4 ± 0.39e-4 | 4.5× |
| 16K | 292.5k ± 0.3k | −3.1% | 14.34 s | 29.37 ± 0.03% | 2.94e-3 ± 0.20e-3 | 16× |

The logged `moe/receiver_drop_fraction` is zero in every run because the ragged transport hard-codes it to zero and counts receiver-capacity clipping as `sender_dropped`. Peak HBM was 103.1 GiB per GPU in every run. In every run, including the 4K control, the mean drop fraction over steps 180055 to 180099 was 0.75 to 1.00 times the mean over steps 180010 to 180054, so part of the measured drop fraction may be a transient after restore.

The live hero (11 racks, 4K) measured over steps 183310 to 183710 on 2026-09-30 had a median MFU of 26.47% and a mean drop fraction of 9.1e-5.

Caveats:

* MFU uses the analytic FLOP count, which charges more attention FLOPs per token at longer sequences (full causal attention on every fourth layer, a 2048-token window elsewhere). MFU therefore rises with sequence length while tokens/s falls. Compare arms by tokens/s.
* The one-rack 4K drop fraction is about 2× the 11-rack hero's. We have not tested the cause. The ratios between arms are the quantity to carry to the hero; an 8K hero drop fraction near 4e-4 is an extrapolation from that ratio, not a measurement.
* Each run covers 100 steps after the switch. Routing over thousands of steps at the new length is untested.
* The 8K and 16K arms change `qk_mult` together with sequence length.
* The diagnostic's peak learning rate comes from a one-rack token budget and is 0.69× the hero's. The diagnostic also leaves the attn_gate/router weight decay at 0 (the hero uses 0.02). Loss values are not comparable with the hero's.
* At 8K and 16K, XLA logged at compile time that rematerialization could only reduce its memory estimate to 193 GiB against a 161 GiB limit. The warning does not stop compilation. The runs trained normally with a runtime peak HBM of 103.1 GiB, the same as 4K, so the estimate overstated actual use.

## Links

* W&B runs (project `marin_moe`, group `moe-hero-ep`): 4K [s0](https://wandb.ai/marin-community/marin_moe/runs/mhep-ctx4k-s0-20260930), [s1](https://wandb.ai/marin-community/marin_moe/runs/mhep-ctx4k-s1-20260930), [s2](https://wandb.ai/marin-community/marin_moe/runs/mhep-ctx4k-s2-20260930); 8K [s0](https://wandb.ai/marin-community/marin_moe/runs/mhep-ctx8k-s0-20260930), [s1](https://wandb.ai/marin-community/marin_moe/runs/mhep-ctx8k-s1-20260930), [s2](https://wandb.ai/marin-community/marin_moe/runs/mhep-ctx8k-s2-20260930); 16K [s0](https://wandb.ai/marin-community/marin_moe/runs/mhep-ctx16k-s0-20260930), [s1](https://wandb.ai/marin-community/marin_moe/runs/mhep-ctx16k-s1-20260930), [s2](https://wandb.ai/marin-community/marin_moe/runs/mhep-ctx16k-s2-20260930)
* Hero plan: #8435. 262K context-parallel probe on the same hardware: #8754.

## Conclusion

At the planned 4K to 8K switch, expect about 1.6% fewer tokens per second and about 4.5× the 4K drop fraction, still below 0.1% of assignments on one rack. At 16K the costs are 3.1% and 16×.



## S9615-01 2026-10-01T01:22:53Z
https://github.com/marin-community/marin/issues/9615#issuecomment-5922813792
great! minimal speed difference. I am hoping the total grad norm can start arching downwards before we make the swap, but probably fine either way.

## S9615-02 2026-10-01T01:46:53Z
https://github.com/marin-community/marin/issues/9615#issuecomment-5923072768
I think swap to 16k context length will make the most sense, since its similar speed at 4k. Then when we want to do the first cooldown we can jump to 65k, another 4x gain. 4x jump per extension is on the more conservative side in literature. So I don't think we need to stick with 2x. Then another 4x to 262k at the very end of training.

On the 67B MoE we saw context extension 8k->65k at 20 percent into the run occur fine, with a grad spike in the first 10 steps but no loss spike. So it should be fine to swap when LR is still higher at 50 percent into the run.

The decision point during the swap is how we modify qk_mult. I think it is worth trying a couple options for 100 steps each w/ an eval at the end of each to empirically inform the decision. Deepseek empirically found their own optimal coef for MLA. The options are 1.3 × (0.1 × ln(4) + 1) -> 1.48, or 1.3 × (0.1 × ln(4) + 1)^2 -> 1.68, or we take the midpoint at 1.58. Then we with whichever shows the lowest paloma macro and uncheatable eval loss. 

It is a little surprising that the total grad norm hasn't started going down on the hero run at 47 percent in. Though it looks stable enough to not be concerning. - if we were to see substantial continual grad norm rise after swap during 100 step test, then we could consider holding off on context extension until LR is at lower position 60-80 percent into the run.

The main reason for doing the context length increase now instead of later would be to enable the earlier cooldown to have a longer window at higher context, which is particularly relevant because the current plan is to use that earlier cooldown for RL'ing experts for the first pass of the final model.

## S9615-03 2026-10-01T01:51:27Z
https://github.com/marin-community/marin/issues/9615#issuecomment-5923120526
When we change context length, we will have to be careful to make sure that every other schedule in the model still works correctly. The LR schedule, the weight decay schedule, the data mix schedule, the eval schedule, and the metrics around run_progress, and tokens_consumed.

## S9615-04 2026-10-01T20:45:53Z
https://github.com/marin-community/marin/issues/9615#issuecomment-5940217790
🤖 Context-switch inventory, independently audited across data/resume, schedules/model settings, and eval/metrics against `main` at `3daf065582`. This revises the earlier inventory: **constant tokens per update preserves the schedules, but does not preserve the exact data stream**. The list below states implementation requirements and deployment prerequisites. The implementation is in draft PR [#9639](https://github.com/marin-community/marin/pull/9639), rebuilt as three commits after independent xhigh code review and lint review; no review findings remain. Formatting/type checks pass, 135 targeted tests pass, and 2,901 selected tests pass. Twelve selected-suite failures and one focused checkpoint timeout reproduce on the base revision. Accelerator rehearsal and live cutover have not been performed.

**Schedules and model configuration**

- Keep the production update at **46,137,344 tokens**: batch 11264 at 4096, 5632 at 8192, or 2816 at 16384. Pass the actual context into both the model and optimizer heuristic. This preserves the 390251-step schedule, MuonH/Adam learning rates, beta2, epsilon, warmup/decay/cooldown fractions, and gate/router weight-decay annealing. Changing only batch or only the model silently changes these quantities. The weight-decay schedule reads the restored Adam count; restore the full optimizer state.
- Preserve the existing zero-based `run_progress = global_step / 390251` convention. Bounded diagnostics currently use their early stop as the denominator; fix them to use the full schedule. Grafana ETA depends on the existing convention. Watch, GC, eval, and checkpoint step cadences remain valid at constant tokens/update; checkpoint wall-clock cadence still follows elapsed time.
- Make context length, QK multiplier, and receiver capacity explicit launch settings. The current QK scalar applies to both local and global layers. The discussion proposes 16K trials at approximately 1.48, 1.58, and 1.68; choosing a winner requires evaluation. Global-only scaling would be an additional model change, not an existing option. Do not hardcode unmeasured QK/capacity choices into a phase table.
- Model and optimizer arrays have no batch/context-shaped leaves. Restore uses the new static model configuration with the old parameter/optimizer arrays and counters. A real save/resume rehearsal must still exercise the new compiled shapes, expert routing, and attention kernels. Local RoPE remains at the 2048 window; longer global attention changes FLOPs and memory.

**Data order and resume**

- The loader has no checkpointed cursor: it seeks by global step and batch schedule. Use a constant new sequence batch on relaunch. A historical sequence-batch schedule mixing 4K and 16K units does not represent token history correctly. Rebuild the datasets and loader at the new context length.
- Set `BlockShuffleConfig(io_block_size=2**20 // seq_len, window_blocks=512, perm_type="feistel")` for the Harrier mixture. It matches the old default at 4K and preserves the outer permutation of 1Mi-token blocks and complete 512Mi-token windows at longer contexts. **It does not preserve within-window order.** Keep the seed, component order, immutable token store, and shuffle policy fixed.
- The previous 17B-token estimate was a simulation result, not a general repeat bound. Each of 200 cells can have an in-progress 512Mi-token window: their combined window size is about 107.4B tokens. Per-cell mixture cursor shifts and truncated dataset tails are separate effects. The earlier naive-relaunch estimates (2.85T repeated and 1.6T unique tokens missed) have not been independently reproduced; use them as motivation, not a correctness guarantee.
- **Retain the 49152-sequence mixture blocks.** Shrinking them with context length changes floor-rounded per-cell weights and the remainder assigned to the largest cell. A calculation with the checked-in weights at step 195125 gives about 108.9B tokens of aggregate absolute complete-block cursor shift at 16K if the blocks are shrunk.
- With unchanged mixture blocks, the main/cooldown stage starts are 108000/312192 at 4K and 8K, 108096/312192 at 16K, and 108288/311808 at 64K. The previous inventory estimated 2.95B tokens of aggregate per-cell cursor displacement for the 4K→16K change with unchanged mixture blocks and the 108000→108096 historical boundary; that numerical estimate has not been independently reproduced. Require explicit acceptance when a continuation shifts these boundaries. Do not assert identical per-cell token blocks in the acceptance gate.

**Metrics and evaluation**

- `throughput/total_tokens` remains exact at the same completed step when tokens/update stays fixed. A one-rack rehearsal using fewer actual tokens/update has different data offsets and cumulative tokens even if its optimizer uses the production token budget.
- Cumulative `total_gflops` currently reprices all past updates using the new context. Supply a verified cumulative-FLOP baseline at the completed handoff step; add the new phase's cost after that boundary, retaining the baseline on retries. At boundary N, use the parent total after N completed updates (normally logged at global step N−1). Per-step MFU should use the new FLOP cost; compare speed using tokens/s and elapsed iteration time.
- Pin canonical `eval_dropless/*` evaluation to **4K packing** so inherited parent rows and the `eval_regressed` alert compare the same packing. Run 16K evaluation under a distinct prefix in the trial. Fixed packing does not isolate the effect of training context when QK also changes: record QK and use matched-QK controls when attributing gains.
- Run evaluation after the first resumed update, with deduplication when the periodic/final hook hits the same step. The old first-step hook fires only at global update 1, otherwise the first new-context evaluation can wait until the next 3000-step boundary. Dropless evaluation must emit the watchdog's evaluation lifecycle events. Rehearse train → eval → next train, including dropless memory, before deployment.

**Rehearsal and acceptance**

1. CPU regression checks: compare 4K/8K/16K schedules and optimizer updates from restored counters; verify constant token totals, fixed shuffle-window coverage, documented mixture boundaries, fixed eval packing, resumed eval cadence, and cumulative FLOPs across retry. Exact optimizer/schedule equality is appropriate; exact shuffled batches and equal loss are not.
2. Restore the actual handoff checkpoint on one rack: use a 4K/QK1.3 control and 16K arms at QK1.48,1.58,1.68, each for 2–3 seeds × 200 steps. Add 4K controls at matching QK values if attributing a gain specifically to training context. Use `schedule_steps=390251`, production optimizer tokens/update, and gate/router weight decay 0.02. The original diagnostics used a smaller optimizer budget and zero gate/router decay. Smaller execution batches still change gradient statistics and data position; this is a rehearsal, not a production-equivalent gradient comparison.
3. Measure loss offset/drift, grad norm, router/drop metrics, tokens/s, iteration time, and peak memory. Include a dropless evaluation at each required packing and a save/resume. Record replacement acceptance/rollback thresholds for loss drift, gradient spikes, expert drops, throughput, and memory from these arms before authorizing deployment. The earlier measured context-dependent loss offsets mean the normal same-objective paired-loss gate cannot be applied unchanged.
4. During the production trial, verify LR, Adam LR, progress, token totals, mixture stage/weights, and FLOP continuity at matching completed steps. Compare windowed loss/gradient/drop/throughput statistics with the rehearsal bands. Score parent and child checkpoints on fixed held-out Paloma and uncheatable sets at both packings; keep short-context retention and long-context quality separate.

**Handoff record and remaining decisions**

Update the run ID, complete durable checkpoint, W&B fork point, source SHA, context/batch/QK/capacity, eval policy, FLOP baseline, mixture-shift acceptance, and rollback command together. The trigger should validate the child tracker has the same settings on retries and record them in #8506. Check checkpoint metadata, region-local storage headroom, and retention before the cutover. An old binary without requested permanent saves needs a forced temporary checkpoint copied to a durable prefix before pruning; each full hero copy was estimated at 4.29 TB in the earlier inventory. The previous live step/ETA is stale and must be refreshed at deployment.

Open deployment decisions: final QK and capacity, explicit acceptance of the 16K data shift, empirical trial thresholds, and the handoff checkpoint/time. A 64K or 262K phase also needs its own batch/mesh/attention and memory validation. The changes described here do not establish those later deployments as ready.


## S9615-05 2026-10-01T21:55:41Z
https://github.com/marin-community/marin/issues/9615#issuecomment-5941450380
Testing a 100 step run of global only qk_mult update 1.3->1.68 is probably worth it too. On the 67B run global only did worse but it seems cheap enough to test. I think the tests should be on full 11 rack. I'd rather run evals under the same wandb field so our plots work. (evals should improve when they have longer context)

## S9615-06 2026-10-01T21:57:17Z
https://github.com/marin-community/marin/issues/9615#issuecomment-5941471694
Having a hard time understanding `estimated 2.95B tokens of aggregate per-cell cursor displacement`. Skipping some tokens is much preferred to repeating ones we just saw.

## S9615-07 2026-10-01T21:58:12Z
https://github.com/marin-community/marin/issues/9615#issuecomment-5941483912
Wonder if its possible to do a clean handoff point as long as our checkpoint lands on a step that is divisible by the right number.

## S9615-08 2026-10-01T21:58:21Z
https://github.com/marin-community/marin/issues/9615#issuecomment-5941486298
I believe thats what I did on the 67B.

## S9615-09 2026-10-02T20:01:46Z
https://github.com/marin-community/marin/issues/9615#issuecomment-5960464902
> Wonder if its possible to do a clean handoff point as long as our checkpoint lands on a step that is divisible by the right number.

@ClassicLarry I've been looking into this, and it seems pretty tricky to resume with a different context length without repeating or dropping tokens. It seems like there are at least 4 independent sources of potential misalignment:

1. Levanter currently splits sources into "shuffle blocks" of 256 sequences and then the blocks are permuted. When we change the sequence length, the block sizes also change and the permuted sequence differs. So even if we restart the run at a shuffle block boundary, we'll inevitably drop and repeat tokens. #9689 fixes this by setting the tokens per shuffle block to ~1M (the current value, with 4k context), independently of the context size.
2. In addition to the block-level permutation above, we also permute _sequences_ within 512-block windows. When we change the sequence length, we change the result of permutation. (Note, we can't just work around this by doing the handoff at a step divisible by some number, since each source is at a different point in its window, and the rate at which tokens are consumed from a source is determined by its mixture weight). This seems like the largest effect after (1) and also the hardest to work around.
3. The token stream is divided into "mixture blocks" of 49152 sequences. To avoid dropping/repeating tokens, we need to hand off at a mixture block boundary. For a given batch size, mixture block boundaries coincide with steps every `49152 / gcd(49152, B)` steps. At C=16k, B=2816, this is 192 steps.
4. The data mix has a stage start at 108k. But mixture stages must fall on steps that are also mixture block boundaries, which are multiples of 192 steps at batch size 2816 (what we'd use with 16k context). We could either move the stage start to 108096 or 107904; the former skips some tokens from sources whose weight decreased at the switch and repeats some tokens from sources whose weight increased; the latter vice-versa.

One option is to merge #9689 to fix (1), hand off at the 216k permanent checkpoint to avoid (3), and accept the drops/repeats due to (2) and (4). I'm a bit hesitant, though - it seems like this has the potential to drop/repeat 10s of billions of tokens.

Wondering if we should instead look into an adhoc patch, something like finishing each source's current window in the order the 4k run was using before switching to 16k? This seems like it could get quite messy, though, and I'm hesitant to pursue it unless we think it's really necessary.

@ravwojdyla @Helw150 curious for your thoughts here as well. How did we manage this for the 67B run?

## S9615-10 2026-10-02T21:31:13Z
https://github.com/marin-community/marin/issues/9615#issuecomment-5961793178
🤖 **Proposal: a clean context-length handoff** (replying to @mcwitt's four misalignment points above)

1. Fix shuffle blocks at ~1M tokens, so the level-1 block permutation is the same at every seq_len (#9689).
2. Give each data-mix stage its own `seq_len` and `batch_size`, e.g. `(216000, mix2, seq_len=16384, B=2816)`. A stage then only has to land on a mixture-block boundary in its own units, which fixes points 3 and 4.
3. Track each source's read position in **tokens** carried across stages, instead of in sequences of a single length.
4. On a `seq_len` change, advance each source to its next shuffle-window boundary (skip-to-window). Nothing repeats; only the unread rest of each source's current window is skipped, which fixes point 2. Using `harrier_mix_2026_08_18` at a 216k handoff, that is about **52B unseen tokens** across 200 sources, ≈0.26B each and ~0.6% of the remaining run.
5. Inside a stage, mixing, batching and BOS/EOS/segment handling are unchanged.

Finishing the partially read windows instead would serve leftover 4k slices glued into 16k sequences. Those need segment and loss masking at the joins, carry only 4k of context, and low-weight sources would keep serving them for thousands of steps after the switch. So skip-to-window looks like the better trade-off.


## S9615-11 2026-10-02T21:40:20Z
https://github.com/marin-community/marin/issues/9615#issuecomment-5961912438
Thanks for the thorough investigation! The 67B would have had some reshuffling on its context length change, I did not account for all levels of the shuffle there.

I am in favor of skipping ahead to each data bucket's next 512-block window. So at most a data bucket will skip 512x256x4096 = 500 million tokens. I cant think of a clean way around this.

## S9615-12 2026-10-02T21:51:40Z
https://github.com/marin-community/marin/issues/9615#issuecomment-5962055112
In the future a solution might be to not permute sequences in step 2, and instead only permute at max_seq_len, where max_seq_len is based on what we will eventually use in the training run.

We will face this challenge again when we go to 65k. So perhaps we should have the code permute sequences at 65k now, then chop up at 16k.

## S9615-13 2026-10-02T21:52:21Z
https://github.com/marin-community/marin/issues/9615#issuecomment-5962062667
Wondering what Rav and Will think here. Would defer to them on what we think will minimize impact.

## S9615-14 2026-10-03T03:23:25Z
https://github.com/marin-community/marin/issues/9615#issuecomment-5965033260
For the hero, we already shuffled the data globally in Datakit, so we could keep a fixed token-sized I/O block shuffle and drop the inner sequence shuffle. That'll make our shuffle logic context size independent for the rest of the run (and I agree skip the rest of the current block for everything here).

## S9615-15 2026-10-03T03:26:37Z
https://github.com/marin-community/marin/issues/9615#issuecomment-5965054947
> For the hero, we already shuffled the data globally in Datakit, so we could keep a fixed token-sized I/O block shuffle and drop the inner sequence shuffle. That'll make our shuffle logic context size independent for the rest of the run (and I agree skip the rest of the current block for everything here).

is the only benefit of the inner shuffle to decorrelate individual batches? So if we have one very long document it doesn't end up dominating a single batch? Would we be losing this if we dropped it, if datakit shuffle is only at a document level?

## S9615-16 2026-10-03T03:28:34Z
https://github.com/marin-community/marin/issues/9615#issuecomment-5965068812
Long term, we should add `seq_len` to `BatchSchedule`. For each shuffle window, shuffle at the largest context length used while consuming it, then split into smaller sequences as scheduled. With divisible lengths and transitions aligned to those larger units, this preserves data order across context changes without requiring a globally shuffled store.

## S9615-17 2026-10-03T03:31:11Z
https://github.com/marin-community/marin/issues/9615#issuecomment-5965086805
The inner shuffle was mainly intended to make it so that you didn't need to globally shuffle at write time and so you can test variance from different global shuffles w/o storing write copies.

## S9615-18 2026-10-03T03:59:25Z
https://github.com/marin-community/marin/issues/9615#issuecomment-5965318024
@ravwojdyla and I just chatted. I'm probs over-engineering above.

We should just skip the rest of each source's current shuffle window whenever we change `seq_len`. It's the lowest-risk option: it preserves the shuffle behavior that's been working so far, at a small cost (50B tokens worst case). Those tokens will come back around if we exhaust a bucket; otherwise, we'll use fresh tokens in their place.

## S9615-19 2026-10-03T04:01:31Z
https://github.com/marin-community/marin/issues/9615#issuecomment-5965332916
Ok- so we will repeat the same procedure when we go to 65k?
