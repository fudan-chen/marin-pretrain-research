# [levanter] Restore checkpoints that hold pinned-host state
A run with `offload_opt_state` and `FP32_PINNED_HOST` keeps its master and optimizer leaves on `pinned_host`, so those leaves are checkpointed with a pinned-host memory kind. Restoring failed: the deserialize path passed that sharding straight to tensorstore, which materializes each shard as a device buffer and then fails the memory-kind check while assembling the array (`memory kind 'pinned_host' ... buffer with memory_kind 'device'`). The run could write checkpoints it could not read back.

Deserialize each such leaf onto `device`, then move it to its target memory kind one leaf at a time so the sharded state never sits whole in device memory. Also preserve `memory_kind` when concretizing an abstract-mesh sharding — `eval_shape` produces that sharding for the restore template, and dropping the kind there would silently reload offloaded state into device memory.

Fixes #8441


## S8443-01 2026-08-19T06:47:35Z
https://github.com/marin-community/marin/pull/8443#issuecomment-5338547700
🤖 Validation of the pinned-host restore fix on real offloaded checkpoints. Summary: the fix restores offloaded (`FP32_PINNED_HOST` + `offload_opt_state`) checkpoints correctly. Two hero-scale issues surfaced during testing are orthogonal to this change.

## 1. Unit regression (CPU, 8 devices)
Replicated and sharded `pinned_host` leaves round-trip through `tree_serialize/deserialize_leaves_tensorstore` with matching values and `memory_kind == "pinned_host"`. Covered by the new test in `test_tensorstore_serialization.py`.

## 2. d2048 resume — full end-to-end success
Resumed the real `rav-ladder-d2048-v3` **step-15000** checkpoint (offloaded, pinned-host master/opt state) into a fresh run on 1 rack.

- Restore → eval at exactly step 15000 → trained 15000→15030 → **completed cleanly**. No memory-kind mismatch.
- Dropless paloma macro-loss at step 15000: **2.492**, vs the original run's **2.4904** at step 15059 (the original never evaled at exactly 15000). The resumed value sits just above the original's value 59 steps later, exactly as continued training predicts — consistent with a byte-correct restore. (The as-trained macro differs more, 2.680 vs 2.766, because that lossy eval is sensitive to the 11264→1024 batch change; the dropless eval is batch-invariant, so it's the right comparison.)

## 3. Hero d6144 save→restore (1 rack) — restore works; two orthogonal issues found

**Run 1 (save):** trained 10 steps, wrote a complete step-10 checkpoint (130 arrays + metadata, ~5 TB, 78 GiB/process of pinned-host state). Baseline dropless paloma macro-loss **7.12764** (high because only 10 steps in — absolute value irrelevant; it's the reference for the restored eval).

**Run 2 (restore):** the restore itself **succeeded** — `Loading checkpoint from …/step-10` → `Error check finished successfully` on every rank, **no memory-kind mismatch**. It then crashed at the *first post-restore training step*. Root cause is **not** this fix: the launch branch predated #8442, so it ran the default one-process-per-4-GPUs topology, which collides on per-process CUPTI/coordination at hero scale — one process died, cascading into the NCCL / `Fatal Python error: Aborted` abort. Adding `processes_per_task=HERO_GPUS_PER_NODE` (per #8442) cleared the training crash.

**Re-run with `processes_per_task=4`:** trained all 10 steps (crash fixed), but the **save** failed — one node died silently mid-save while writing its 78 GiB shard of the ~5 TB pinned-host state (host OOM signature; 4 processes/node each stage to host concurrently vs 1 with the original `pp=1` save, which wrote its checkpoint fine). Gang scheduling then bounced the siblings. This is a **save-side, hero-scale host-memory issue, independent of the restore path this PR changes**.

## Bottom line
The restore fix is validated: offloaded pinned-host checkpoints restore with correct values and continue training (d2048 to completion; the d6144 restore loads cleanly). The hero-scale problems found along the way — process topology (#8442) and save-side host memory under one-process-per-GPU — are separate from this change and are being tracked independently.


## S8443-02 2026-08-19T17:29:34Z
https://github.com/marin-community/marin/pull/8443#issuecomment-5345711727
🤖 The one-rack end-to-end checkpoint gate stopped before the checkpoint save.

- Configuration: d6144, 16 GB200 nodes, four JAX processes per node, 8 GiB staged host memory per process, and a 390,251-step schedule.
- Source: PR #8443 rebased onto `main` at `be0f70b16`.
- Result: every process failed before step one. `jit_train_step` received parameter 71 in memory space `device`, but the executable expected another memory space.
- No checkpoint save ran. The resume phase did not start.
- Iris reported the parent and child jobs as successful after the rank failures. The training logs contain no completed step.

[Iris job](https://iris.oa.dev/#/job/%2Frav%2Fmhep-d6144-e2e-stage8-save-171335-coord) | [incident record](https://echo.oa.dev/wiki/185)


## S8443-03 2026-08-19T17:42:18Z
https://github.com/marin-community/marin/pull/8443#issuecomment-5345846533
🤖 Agreed this is orthogonal to the change in this PR, and the incident record captures it well: `RuntimeProgramInputMismatch` on parameter 71 **from initialized state, before any save or restore**. This PR only touches the restore/deserialize path (`tensorstore_serialization.py` + its test), which a fresh-init train step never exercises.

A data point that should help narrow it: the one-rack d6144 save tests I ran on this branch (`mhep-hero-d6144-savetest-1rack`, one process per node, and `…-v2`, one process per GPU) both trained through step 10 and reached the checkpoint save with the same `offload_opt_state` + `FP32_PINNED_HOST` config — the pp=1 run wrote a complete step-10 checkpoint (130 arrays + metadata). Those ran on this branch's base `5342d7792f`, which is an ancestor of your `be0f70b16`, and there are **no changes to `experiments/grug/moe_hero_ep/train.py` or `optimizer.py` between the two**. So fresh d6144 training + save worked very recently with an unchanged hero train path; the param-71 memory-space mismatch looks like it entered upstream (levanter or a dependency) between those two mains — that's where a bisect would pay off.

The restore fix itself stays validated independently: the d2048 offloaded run resumed end-to-end (restore → eval → train → complete), and the d6144 restore loaded cleanly with no memory-kind mismatch. So this PR is safe to land on its own; the step-1 mismatch is a separate blocker for the full hero e2e gate.

+1 on fixing the multi-GPU wrapper so a local-rank failure surfaces as a failed Iris task — the "success after all ranks failed" behavior is what masked this.

