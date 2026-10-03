# [grug] Apply pending query bias during checkpoint evaluation
Apply a checkpoint's `pending_qb_betas` before Grug MoE evaluation callbacks run. A checkpoint-only evaluation at its stop step currently exposes the model parameters and EMA parameters without that pending update. The next normal training forward and the sampling restore path apply the update first, so evaluation can score a different routing state from the one the checkpoint would use on its next forward.

The observed score effect was small on two Hero checkpoints. It was still enough to make the first fixed-policy Paloma comparison semantically incomplete. The original `a1` results are superseded by corrected `a2` runs.

This issue is separate from the router-precision proposal in #8435. The omission occurs regardless of router dot precision. The measured impact comes from checkpoints produced by 20 steps on one B200 rack, so it does not establish the size of the effect for other checkpoints or long runs.

Requested behavior: apply the pending query-bias update to both the current and EMA model views used by evaluation hooks. Ordinary progress, logging, and checkpoint callbacks should continue to use the stored state without launching eager query-bias work.

An experimental fix and regression test exist on Romain's fork, but they have not been merged:

- Initial fix: [`2e392b6c0568e969425eb1bf97b02c853a678860`](https://github.com/yonromai/marin/commit/2e392b6c0568e969425eb1bf97b02c853a678860)
- Final scoped fix: [`a00cb77a491f4777a2c66edce54f47ff7b255c40`](https://github.com/yonromai/marin/commit/a00cb77a491f4777a2c66edce54f47ff7b255c40)
- Evaluation wrapper: [`train.py`](https://github.com/yonromai/marin/blob/a00cb77a491f4777a2c66edce54f47ff7b255c40/experiments/grug/moe_hero_ep/train.py#L679-L693) and [eval-hook application](https://github.com/yonromai/marin/blob/a00cb77a491f4777a2c66edce54f47ff7b255c40/experiments/grug/moe_hero_ep/train.py#L1188-L1195)
- Regression test for current and EMA model views: [`test_eval_callback_models_apply_pending_query_bias`](https://github.com/yonromai/marin/blob/a00cb77a491f4777a2c66edce54f47ff7b255c40/tests/test_moe_hero_ep.py#L361-L387)

Done when checkpoint evaluation applies the pending query-bias state to current and EMA models, the regression passes, and non-evaluation callbacks avoid this extra work.

<details><summary>Evidence and reproduction</summary>

### Observed behavior

The checkpoint stores `pending_qb_betas` separately from `params`. The training step applies those betas as centered router bias before the next forward, then stores newly computed betas for the following step. The evaluation callback model getters expose raw `params` and EMA parameters. A forced end-of-run callback at an already reached stop step can therefore evaluate the router bias embedded in the parameters and omit the pending update.

The problem appeared while comparing two step-126020 continuation checkpoints under current router arithmetic. Both checkpoints came from the same complete step-126000 native state:

`s3://marin-us-east-02a/marin/grug/hero-main-step121638/2026.08.19.2/checkpoints/step-126000`

[`e8460b35bd6b312b62799b12ae363b3981af3c83`](https://github.com/marin-community/marin/commit/e8460b35bd6b312b62799b12ae363b3981af3c83) is the observed production source identity for that checkpoint. It does not identify the experimental continuation code.

The experiment branch was based on [`1f9c387b6c452895ff58573c2b6117c5500303db`](https://github.com/marin-community/marin/commit/1f9c387b6c452895ff58573c2b6117c5500303db). The router arithmetic controls used by the experiment were added in [`f9178348f5237cb2cb591024511d4db693a3910c`](https://github.com/yonromai/marin/commit/f9178348f5237cb2cb591024511d4db693a3910c). The corrected evaluation behavior was introduced in [`2e392b6c0568e969425eb1bf97b02c853a678860`](https://github.com/yonromai/marin/commit/2e392b6c0568e969425eb1bf97b02c853a678860) and scoped to evaluation hooks in [`a00cb77a491f4777a2c66edce54f47ff7b255c40`](https://github.com/yonromai/marin/commit/a00cb77a491f4777a2c66edce54f47ff7b255c40).

The two continuations used seed 0, the restored data-loader state, and steps 126000 through 126019:

| Checkpoint | Continuation job | W&B | Checkpoint path |
| --- | --- | --- | --- |
| Current arithmetic | `/romain/hero-router-current-cont-coord-01a0c066-a1` | [run](https://wandb.ai/marin-community/marin_moe/runs/hero-router-current-cont-step126k-01a0c066-a1) | `s3://marin-us-east-02a/marin/users/romain/hero-training-router-01a0c066/checkpoints/current-a1/step-126020` |
| Preferred FP32 accumulation | `/romain/hero-router-pref-cont-coord-01a0c066-a1` | [run](https://wandb.ai/marin-community/marin_moe/runs/hero-router-pref-cont-step126k-01a0c066-a1) | `s3://marin-us-east-02a/marin/users/romain/hero-training-router-01a0c066/checkpoints/preferred-fp32-a1/step-126020` |

### Superseded and corrected evaluations

The `a1` jobs used raw callback model state and are superseded:

| Checkpoint | Iris job | W&B | Macro loss | Macro BPB |
| --- | --- | --- | ---: | ---: |
| Current | `/romain/hero-router-current-fixed-eval-coord-01a0c066-a1` | [superseded run](https://wandb.ai/marin-community/marin_moe/runs/hero-router-current-fixed-eval-step126020-01a0c066-a1) | 2.22020006 | 0.80684924 |
| Preferred-trained | `/romain/hero-router-pref-fixed-eval-coord-01a0c066-a1` | [superseded run](https://wandb.ai/marin-community/marin_moe/runs/hero-router-pref-fixed-eval-step126020-01a0c066-a1) | 2.22091150 | 0.80714315 |

The `a2` jobs applied each checkpoint's pending query-bias update before the same fixed-current-arithmetic Paloma evaluation:

| Checkpoint | Iris job | W&B | Macro loss | Macro BPB |
| --- | --- | --- | ---: | ---: |
| Current | `/romain/hero-router-current-fixed-eval-qb-coord-01a0c066-a2` | [corrected run](https://wandb.ai/marin-community/marin_moe/runs/hero-router-current-fixed-eval-qb-step126020-01a0c066-a2) | 2.22016478 | 0.80683833 |
| Preferred-trained | `/romain/hero-router-pref-fixed-eval-qb-coord-01a0c066-a2` | [corrected run](https://wandb.ai/marin-community/marin_moe/runs/hero-router-pref-fixed-eval-qb-step126020-01a0c066-a2) | 2.22088742 | 0.80713451 |

Applying the pending state changed macro loss by `-0.00003528` for the current checkpoint and `-0.00002408` for the preferred-trained checkpoint. The preferred-minus-current macro-loss delta changed from `+0.00071144` to `+0.00072265`. This correction does not change the router investigation's recommendation, but it makes the evaluation represent the complete checkpoint state.

Both corrected jobs completed all 16 tasks without failures or preemptions. Their extracted metrics are at `s3://marin-us-east-02a/marin/users/romain/hero-training-router-01a0c066/evidence/fixed-eval-with-pending-qb-a2.json`, SHA-256 `d13af487ee6b7f54644b9dcb5e3d7a37fdcc6e17420b7cf7476652877f8e9182`.

The final investigation report is at `s3://marin-us-east-02a/marin/users/romain/hero-training-router-01a0c066/evidence/hero-router-training-report.md`, SHA-256 `454aae5894a78c99752cff47301441c9a8d315b30d5d1ddf1410c069bba73f4c`. The review response that prompted and verified the correction is at `s3://marin-us-east-02a/marin/users/romain/hero-training-router-01a0c066/evidence/goal-review-response.md`, SHA-256 `a5bfff56802f87c14b6be057f4cbd9e6f86d298906071f28cbe77f532217a3d9`.

### Reproduction

Run from the experimental fork at [`a00cb77a491f4777a2c66edce54f47ff7b255c40`](https://github.com/yonromai/marin/commit/a00cb77a491f4777a2c66edce54f47ff7b255c40). Set `SOURCE` to either step-126020 checkpoint above. Because `num-steps` equals the restored state step, the optimization loop takes no step and the forced final callback runs Paloma:

```sh
python -m experiments.grug.moe_hero_ep.launch_diagnostics \
  --run-id <fixed-eval-run-id> \
  --dp-racks 1 --batch-size 1024 --optimizer-batch-size 11264 \
  --gate-router-weight-decay 0.02 --num-steps 126020 --schedule-steps 390251 \
  --initialize-from-checkpoint "${SOURCE}" --router-dot-precision current \
  --training-data mixture --eval-every 1 --version dev --run
```

The focused validation command was:

```sh
OPENBLAS_NUM_THREADS=1 nice -n 10 uv run --frozen --python 3.12 \
  pytest -n 0 --tb=short -q \
  tests/test_grug_hero_scaling_ladder.py tests/test_moe_hero_ep.py
```

Result: 71 passed, with one pre-existing JAX scatter future warning. `./infra/pre-commit.py --changed-files --fix` also passed.

</details>

