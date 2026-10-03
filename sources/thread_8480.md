# [levanter] Bound the host memory a hero resume holds
A d6144 hero resume was OOM-killed by the container runtime on every attempt, at an 850 GiB request
and again at 909 and 940. The restore itself always succeeded: all 176 tasks read step-164 in about
3 seconds each and logged `Loaded checkpoint`. Host memory then sat at 97 to 99 percent of whatever
the limit was, and the kernel killed the task.

Live sampling of the container placed the cost in the resident state rather than in a transient. A
rank held 148 GiB before it restored anything, then 230 GiB afterwards and stayed there, so four
ranks filled a 940 GiB node. A cold start of the same shape peaks at 175 GiB for each rank and
trains, which put 55 GiB for each rank of unexplained resident memory on the restore path.

Three changes remove it. A restore reads each local pinned-host shard in 1 GiB TensorStore chunks
instead of filling one buffer the size of the shard, and donates that buffer to `device_put` rather
than copying it. The Grug train entry point releases the initialized state before a complete restore,
so the initialized and restored states never coexist. Restore now also runs before data-cache
construction, and ranks inside a container take turns, so one rank's restore does not overlap
another's.

Fleet peak fell from 882 GiB to 735 GiB against a 940 GiB request, and the run resumed from step 164
and is training with no failures and no preemptions.

The retry budget is capped at 3 for each task and 5 cumulative. Iris retries without a backoff, so
the previous 1000 turned this deterministic failure into an unbounded loop that reran the gang every
25 minutes on 704 GPUs.

Measurements come from the finelog `iris.task` namespace and from `/sys/fs/cgroup/memory.stat` and
`/proc/<pid>/status` sampled inside the container every 10 seconds. Finelog samples every 30 seconds
and missed the fatal spike on every failed attempt.


## S8480-01 2026-08-20T07:47:57Z
https://github.com/marin-community/marin/pull/8480#issuecomment-5352974293
The last commit is the critical one. We will have to cleanup/minify this PR. The barrier, better ordering and more logs doesn't hurt tho.

CC: @ClassicLarry 

## S8480-02 2026-08-20T15:47:04Z
https://github.com/marin-community/marin/pull/8480#issuecomment-5358295214
Before we do a cleanup here, @ClassicLarry could you pls high-level/direction review only the last commit?

## S8480-03 2026-08-20T15:51:58Z
https://github.com/marin-community/marin/pull/8480#issuecomment-5358350578
last commit lgtm

## S8480-04 2026-08-20T18:40:18Z
https://github.com/marin-community/marin/pull/8480#issuecomment-5360184378
@ClassicLarry could you pls review full PR 🙏  (should be ready now)

## S8480-05 2026-08-20T18:47:35Z
https://github.com/marin-community/marin/pull/8480#issuecomment-5360262227
🤖 Local design review of this branch. Four items, ordered by risk. I did not run the hero path; the JAX claims come from reading `jax==0.11.0` as pinned in this checkout.

The core change looks right to me: reading each pinned-host shard straight into its target memory kind, and deleting `_device_shardings_for_load` / `_move_leaves_to_target_memory_kind`, is the correct fix, and `test_tensorstore_checkpoint_restores_mixed_memory_kinds_in_tree_order` covers it. The four items below are about what got added around it.

### 1. The restore barrier is on one of four exit paths

`experiments/grug/checkpointing.py:127` calls `sync_global_devices` only after `_load_candidate_state` returns. The other three exits skip it: empty search paths (line 110), `load_checkpoint=False` (line 113), and the not-found fallback (line 147).

Ranks can reach different exits. `_scan_checkpoint_root` catches any error while reading a candidate's `metadata.json` and drops that candidate for that rank alone (lines 81-83). One transient 5xx from the object store on one rank is enough: that rank loads an older checkpoint or starts from scratch and walks on to the data-loader build, while every other rank blocks in `sync_global_devices`. That call is `process_allgather` over all global devices (`jax/experimental/multihost_utils.py:162`), so it has no timeout. The result is 704 idle GPUs and no error line.

Before this change the same divergence produced a wrong-but-visible run.

`checkpointing.py` is shared by `grug/base`, `grug/moe`, `grug/moe_hero_fsdp` and `grug/moe_hero_ep`. All four get the barrier. Only `moe_hero_ep` gets the state release it supports.

Suggestion: move the barrier to the caller so every path reaches it, or drop it (see 2).

### 2. `/tmp/levanter-restore.lock` is a hidden global lock on every Levanter restore

`tensorstore_serialization.py:713-721`, taken at `:830` and `:878`. `tree_deserialize_leaves_tensorstore` is the restore path for all of Levanter (`checkpoint.py:886`), so the lock now covers single-process eval, HF export and the unit tests, with no config and no way to opt out.

* `open(path, "w")` raises `PermissionError` when the file exists under another UID. On a shared host with a persistent `/tmp`, the second user to restore a checkpoint cannot restore at all. There is no fallback.
* Two unrelated jobs on one host serialize against each other.
* A run without `offload_opt_state` has only device leaves, and the lock serializes local ranks through a restore that never had a host-memory problem.

The memory argument is also narrower than it looks. Restore is process-local: tensorstore read, `device_put`, `make_array_from_single_device_arrays`, with no cross-process communication. The only same-node overlap the lock removes is rank 0 starting the data-cache build while ranks 1-3 still restore, which is what the barrier in item 1 also targets. That is two mechanisms for one problem, each with its own failure mode.

### 3. The pinned-host restore is fully serial, and the lock multiplies it by the local rank count

`tensorstore_serialization.py:724-772`. `deserialize_non_device_leaves` walks leaves one at a time. `_deserialize_leaf_to_memory_kind` walks devices one at a time and calls `block_until_ready()` after each `device_put`. There is no I/O concurrency anywhere in that path. JAX's own loop declines that block and explains why at `tensorstore_impl.py:565-575` (15-20% cost); here it also holds up the next read.

This is the bulk of the hero checkpoint, not a corner case. `initial_state` (`moe_hero_ep/train.py:479-490`) puts `master_params` and the whole Adam `opt_state` on `pinned_host`, and grug leaves are raw `jax.Array`, so `_sharding_from_leaf` keeps `memory_kind` and every one of those leaves takes the serial path. `_local_restore_slot` then multiplies that by the ranks in the container.

Two smaller costs in the same function: `ts.open` runs per leaf without JAX's `_TS_CONTEXT`, so each open re-reads the OCDBT manifest and btree nodes with the cache pool disabled, and each open waits behind the previous leaf.

The description reports a peak-memory number and no restore duration, and quotes the old path at about 3 s per task. A measured restore time on the merged code would settle this. It matters more now that the retry budget is tighter.

`_HostStagingGate` (`tensorstore_serialization.py:221-259`) already bounds in-flight host bytes on the write side while letting work overlap. Reusing it, or an `asyncio.Semaphore` sized by `_RESTORE_CONCURRENT_GB`, gives the same bound with overlap, and is less code than the lock plus the serial loop.

### 4. Fresh starts initialize the full state twice, and the re-init trigger is a type sniff

`moe_hero_ep/train.py:735-761`. `_init_state` is materialized, converted to a `ShapeDtypeStruct` template, deleted, and then run again when nothing restored. On a fresh start with `load_checkpoint=None` that is two full inits of a d6144 state.

The trigger is `any(isinstance(leaf, jax.ShapeDtypeStruct) ...)`. It cannot separate "no checkpoint existed" from "a checkpoint existed and every candidate failed to load". The second case restarts a hero run at step 0 and then overwrites the checkpoint stream. The candidate loop that produces it is right there in `restore_grug_state_from_checkpoint`.

I checked the obvious simplification and it does not apply: `jax.jit(f).eval_shape(...)` returns `sharding=None` under Auto mode in jax 0.11, so materializing first to read `leaf.sharding` is reasonable. Discovery is separable from init, though. `restore_grug_state_from_checkpoint` already resolves candidates before it touches the state, so resolving the path first and releasing only when there is something to restore would remove the double init, the `gc.collect()` and the sniff.

`load_checkpoint_or_initialize` (`levanter/checkpoint.py:893-1000`) is the same load-or-init pattern. Its shardings come from the axis mapping, which grug does not use, so it is not a drop-in, but `init_and_merge` at line 956 is the merge step the sniff is standing in for.

