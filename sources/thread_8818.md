# Hero Run: Attn Gate Norm Growth
d6144 hero, `attn_gate` at step 42000 (bf16 forward params, restored via GPU deserialize). Per-head L2 = `||attn_gate[layer, :, head]||` over the 6144 hidden dim; 48 layers × 48 heads = 2304 heads. Whole-tensor L2 = 229.94.

**Per-head L2 over all heads:** min 0.87, median 2.46, mean 3.93, max 12.2 (right-skewed).

**Per layer:** concentrated in the early layers — mean per-head L2 ≈ 8–10 at layers 1–5, decaying to ≈ 1.5–2 by layer ~10 (layer 1 highest at 10.04, layer 33 lowest at 1.53).

Plots and the step-42k `attn_gate` npz attached below.


## S8818-01 2026-08-31T18:25:20Z
https://github.com/marin-community/marin/issues/8818#issuecomment-5482659608
<img width="931" height="265" alt="Image" src="https://github.com/user-attachments/assets/1e2c8841-5980-4700-861d-da7f3df281d4" />

1 percent into the run, the attn gate started exhibiting different behavior than the scaling ladder:
<img width="935" height="262" alt="Image" src="https://github.com/user-attachments/assets/14b14543-ec23-4c5a-a661-09d4a186d429" />

<img width="1875" height="687" alt="Image" src="https://github.com/user-attachments/assets/465a5c28-e731-49a7-ad6a-74ab1de843a4" />

<img width="1125" height="625" alt="Image" src="https://github.com/user-attachments/assets/3449ea2d-431b-4e46-b5f6-110792234f42" />

<img width="1800" height="1140" alt="Image" src="https://github.com/user-attachments/assets/a7e84e69-13f1-4fec-a3f3-3a8f9c9db9d9" />

<img width="1800" height="1140" alt="Image" src="https://github.com/user-attachments/assets/11eb7da1-fff2-4034-85fc-ea37d6468c7b" />

## S8818-02 2026-08-31T19:02:12Z
https://github.com/marin-community/marin/issues/8818#issuecomment-5483098276
[attn_gate_step42000.zip](https://github.com/user-attachments/files/31658787/attn_gate_step42000.zip)

## S8818-03 2026-08-31T20:33:31Z
https://github.com/marin-community/marin/issues/8818#issuecomment-5484183570
🤖 Vocab-wide layer-0 gate values (step 42k) — the gate-norm growth is the model gating attention **off**, not modulating it.

![layer-0 gate vocab](https://raw.githubusercontent.com/marin-community/marin/agent/hero-layer0-gate-analysis/experiments/grug/moe_hero_ep/analysis/hero_layer0_gate_vocab.png)

Evaluated over all 128,256 tokens (the gate is computed pre-attention from each token's own embedding, so this is the value every token induces in any context):

- 98.3% of the 6.15M token×head gates are shut (< 0.05); mean 0.013, median 0.007.
- Logits (`attn_in @ attn_gate`) have mean −6 and are 98.7% saturated, so the large learned gate norms (per-head L2 ~7–10) clamp the gate closed rather than modulate it.
- Only 546 / 128,256 tokens (0.43%) open any head; each head opens for ≤ 0.33% of the vocabulary (head 2 most active).

Plot + summary committed on branch `agent/hero-layer0-gate-analysis` under `experiments/grug/moe_hero_ep/analysis/`.


## S8818-04 2026-08-31T20:47:11Z
https://github.com/marin-community/marin/issues/8818#issuecomment-5484347752
This behavior seems concerning. Attention is basically inactive on layer0. This means the only layer0 data sharing mechanism between tokens is the sconv output of the MLP (sconv of attn would be zero), which means that all operations before that are redundant degrees of freedom. edit: attn is still active, just smaller scale. more details below.

We should break out grad norms by layer. It may be that grad norm on first 10 layers for attn is very small. 

Will take the checkpoint at step 42k, then log the 6144 length activation after every module for an arbitrary token in a sentence, to see where the contributions are coming from. 

Loss and evals all still look good... but perhaps the model doesn't 'need' the capacity yet of the first 10 layers, and will only see impact later in training. Want to make sure gradient is in a spot where if model wants to 'turn on' these layers, it can. 


## S8818-05 2026-08-31T22:21:32Z
https://github.com/marin-community/marin/issues/8818#issuecomment-5485531774
Projection is still ahead of schedule (somewhat driven by higher step count of run relative to scaling ladder)

<img width="1375" height="750" alt="Image" src="https://github.com/user-attachments/assets/474cfd70-13b9-4c24-9bcf-1dc8a0d149df" />

The first 20 layers in general are doing very little:
Here are two example sentences where the contribution of each submodule is plotted.
<img width="1872" height="1170" alt="Image" src="https://github.com/user-attachments/assets/ab87a142-45d4-4c6d-8976-ef9a0b909ab6" />

## S8818-06 2026-08-31T22:25:46Z
https://github.com/marin-community/marin/issues/8818#issuecomment-5485587906
repeated for sequences that require retrieval over 500 tokens back:

<img width="1950" height="676" alt="Image" src="https://github.com/user-attachments/assets/f142c0e8-81b1-4e17-a7c8-3b2dbf316f0f" />

## S8818-07 2026-08-31T23:14:00Z
https://github.com/marin-community/marin/issues/8818#issuecomment-5486042185
🤖 Layer-skip ablation on the step-42k checkpoint — dropless macro loss with whole transformer layers dropped from the forward (restored on one rack, scored with the hero's own dropless evaluator).

| ablation | layers run | dropless macro loss | Δ vs baseline |
|---|---|---:|---:|
| baseline | all 48 | 2.093 | — |
| skip first 10 | 10–47 | 6.584 | +4.49 (3.1×) |
| skip first 20 | 20–47 | 9.615 | +7.52 (4.6×) |
| skip last 10 | 0–37 | 5.455 | +3.36 (2.6×) |

Baseline reproduces the hero's logged `eval_dropless/macro_loss` at step 42k (2.094) to 0.001, so the standalone restore + reshard + dropless eval is faithful.

Every skip is catastrophic (2.6–4.6× the baseline), which cuts against the residual-decomposition read that early layers barely matter: their per-layer contribution norm is tiny (~0.01–1 vs the last-layer MLP's ~40), but removing them still breaks the loss. Small ‖Δx‖ is not removable — downstream layers were trained expecting those transformations. Two structural notes: more layers dropped is monotonically worse (skip-20 > skip-10), and skipping the first 10 hurts more than the last 10 (6.58 vs 5.45) despite the late layers carrying far larger contribution norms — the early layers build the foundation, the late layers refine it, and a missing foundation is less survivable.

W&B: `hero-12d8b6f0-skipeval-42k` (keys `skip_baseline/…`, `skip_first10/…`, `skip_first20/…`, `skip_last10/…`).


## S8818-08 2026-08-31T23:29:56Z
https://github.com/marin-community/marin/issues/8818#issuecomment-5486214819
<img width="1430" height="780" alt="Image" src="https://github.com/user-attachments/assets/131bf71e-9155-478c-8a44-6fc6023c2fd7" />

<img width="1950" height="780" alt="Image" src="https://github.com/user-attachments/assets/d4cf6b41-97d5-464d-8640-ffafec26a554" />

<img width="1950" height="780" alt="Image" src="https://github.com/user-attachments/assets/4b524c1b-c207-4ef0-8e60-a7552f66871f" />

The first 10 layers are still critical for accurate prediction. If we zoom into the first 20 layers, we can see that the layer contribution of the first 10 layers attn is still roughly around the scale of the starting embedding. So even though the scale of the contribution pales compared to the final 10 layers, its still meaningfully changing the representation of the token.

## S8818-09 2026-09-01T02:00:15Z
https://github.com/marin-community/marin/issues/8818#issuecomment-5487632300
🤖 Attention-only range knockout (dropless global macro loss, step-42000, 1 rack, 50 batches). Zeroes a contiguous range of layers' attention output before the residual add, keeping the MLPs. Compared against the whole-layer skips already posted here.

| Ablation | Global macro loss | Δ vs baseline |
|---|---|---|
| baseline | 2.093 | — |
| attn layers 0–9 zeroed (MLPs kept) | 2.798 | +0.705 |
| attn layers 38–47 zeroed (MLPs kept) | 2.436 | +0.343 |

Against the whole-layer skips (attention **and** MLP dropped):

| Range | Attn-only zeroed | Whole layer dropped |
|---|---|---|
| first 10 (0–9) | +0.705 | +4.49 (→6.58) |
| last 10 (38–47) | +0.343 | +3.36 (→5.45) |

- The MLPs carry the large majority of each range's contribution: removing only the attention in layers 0–9 costs +0.71, versus +4.49 for the whole block, so ~85% of that range's effect is the MLP path.
- Early attention matters more than late: zeroing 0–9 attention (+0.71) hurts about twice as much as 38–47 attention (+0.34), consistent with the single-layer knockout finding that no individual late-attention layer is critical.

W&B: `marin-community/marin_moe/hero-12d8b6f0-attnrangeeval-42k`


## S8818-10 2026-09-01T02:57:01Z
https://github.com/marin-community/marin/issues/8818#issuecomment-5488219562
Summarizing findings:
1. At hero_scale attention gate shows uncharacteristic increase in norm. 
2. The 48 heads of the attention gate show high cosine similarity on any given layer. Checking layer zero, 98 percent of the time the attention gate is driving attention down to under 5 percent of its initial output scale.
3. Contributions to the residual stream scale with layer depth. MLP tends to contribute about 2x more than attention. Layer zero contributes about 1/10 the norm of the initial embed. Layer 48 contributes about 10x the norm of the initial embed. 
4. However zeroing out attention on the first 10 layers causes a meaningful increase in loss. A greater increase (2x) than zeroing out attention on the last 10 layers. So attention in early layers is still contributing very meaningfully. (changes from attention in early layers influence what updates later layers make). This contribution should act as a 'tether' that prevents these layers from becoming useless. (the gradient would yell very loudly if attention gate turned them off completely). Main learning: output scale does not correspond to impact.

Suspicion is that the attn_gate parameter has been co-opted in early layers to function as an output scaling mechanism. The intention was for the input RMSNorm on the attn block to handle this, or the sconv on attn_out.

Given that the first 10 attn layers are still critical and actively getting used (just smaller scale outputs), not viewing as issue that requires immediate action. Things would prompt intervention:
1. Loss curve slows down and no longer beats projection.
2. Ablating removing early attention layers actually shows no impact on loss later on in training.
3. We get positive results from an ablation on 1 rack that resumes from checkpoint, when compared to baseline:

Ablation ideas:
1. Update attention gate from v = 2*sigmoid(x)*v to v = (0.05 + 2*sigmoid(x))*v. Give nonzero floor.
2. Instead of hardcoded 0.05, make it a learnable bias per head with very high beta1 and beta2. Init to 0.
3. Add a general output scalar on each attn output. Init to 1 with very high beta1 and beta2.
4. Add some regularization to attn gate weight (weight decay).
5. Add hardcoded bias that ramps per step from 0 to 0.05 over 1000 steps.

All of these ablation ideas would probably not be effective as-is. But testing on 1 rack would give signal and better intuition on how the system is behaving. Since attention gate runs through a sigmoid, it would never fully decay to zero. Since this dynamic only occurs on hero scale (48 layers), all ablations would need to occur there. 

## S8818-11 2026-09-01T04:03:05Z
https://github.com/marin-community/marin/issues/8818#issuecomment-5488709104
Main initial concern here was that gradient signal to the attention matrices would be getting killed by attn gate being off. But actually, the gradient signal is likely still reasonable, its just that model has adopted a behavior where output scale grows with depth (but early layers still impact prediction heavily through their impact on subsequent layers), and this depth scaling is managed partially by the attention gate, perhaps partially because matrices are under hyperball. Will be important to get per-layer gradient logging on Hero to track this more actively. (scan layers is bundling the gradient across all layers).

Traditionally the intent of residual highway is it gives direct gradient signal back from lm_head to every layer directly. However since we see output scale of every layer increase, the early layers will influence prediction very little through direct residual highway. Instead, their influence comes from how they influence subsequent layers. This seems like it would perhaps be less stable. Though training seems extremely stable so far? As long as gradient on attention matrices in initial layers looks ok, then biasing towards no action.

## S8818-12 2026-09-01T04:50:48Z
https://github.com/marin-community/marin/issues/8818#issuecomment-5489069497
🤖 Layer-0 headwise attention gate, per-vocab-token (step-42000). The gate is `2·sigmoid(attn_in @ attn_gate)`, one value per head (48 heads); at layer 0 `attn_in` is a deterministic function of the token embedding, so each token has a fixed 48-head gate vector. "On" = gate > 1 (i.e. `attn_in @ attn_gate > 0`).

Layer-0 attention is near-closed by default: mean gate over the whole vocab is **0.013**. Only **546 of 128,256 tokens open ≥1 head**, and just **93 open all 48**. The tokens that stay on fall into three buckets:

**1. Reserved/special tokens (128000–128255)** — the bulk. ~200 of the 256 special tokens open all 48 heads (opensum 96/96): `<|reserved_special_token_*|>` plus the structural specials (`<|begin_of_text|>`, `<|end_of_text|>`, `<|eot_id|>`).

**2. Malformed / byte-fragment real-vocab tokens** — the top non-special tokens are almost all junk:
- undecodable UTF-8 byte fragments that decode to `�` (ids 124, 125, 178–187), several at the full 96/96
- rare foreign-script subwords: Turkish (`ıldığında`, `ılmaz`, `uştur`, `ılmaktadır`), Cyrillic/Ukrainian (`уватися`, `илася`, `аракт`), Czech (`ujícím`), CJK spam strings (`菲律宾申博`, `갤로그로`)
- gibberish/spam: `eoqkrvldkf`, `.:.:.:.:.:.:`, `.**************`, `ЎыџN`

**3. Ordinary English text tokens** — essentially none; common words sit at gate ≈ 0.

Distribution of open-head counts (real-vocab tokens only, excluding the special block): 345 open ≥1 head, 32 open ≥12 heads.

## S8818-13 2026-09-01T05:47:42Z
https://github.com/marin-community/marin/issues/8818#issuecomment-5489500192
Hmm the MLP router follows a similar scaling trajectory, diverging from the scaling ladder.

<img width="926" height="518" alt="Image" src="https://github.com/user-attachments/assets/73741bef-1e5a-456c-b151-7283a9e9a263" />

A light weight decay on these two parameters would be relatively low risk and increase how learnable they are later on in training. 

The prior 67B-10T tokens had similar router growth and trained perfectly fine with zero loss spikes. However it didn't show this attn_gate behavior.

Feeling more optimistic on weight decay on these two parameters. Will run some experiments to collect more data. Smaller magnitude attn_gate will mean it can do less rescaling in early layers, but the rms norm weight should be able to pick up this behavior.

Low-opinionated form of WD would scale it via lr/peak_lr, so it doesn't impact end of training dynamics.



## S8818-14 2026-09-01T18:33:13Z
https://github.com/marin-community/marin/issues/8818#issuecomment-5498576154
🤖 Weight-decay ablation on the attn_gate / router weights (motivated by the router-norm growth vs small router-z-loss noted above). Decoupled AdamW-style wd applied only to the attn_gate and/or router weight leaves (router_bias, embeddings, norm gains untouched). H100 scaling-ladder rungs, dropless eval.

## d768 (16 H100, batch 1024, 11,420 steps — all finished)

Final dropless paloma macro loss (and overall macro across all eval tags):

| run | wd on gate+router | paloma macro | overall macro | Δ paloma vs base |
|---|---|---|---|---|
| baseline | 0 | 3.0158 | 2.8352 | — |
| wd 0.01 const | 0.01 | 3.0185 | 2.8372 | +0.0027 |
| wd 0.1 const | 0.1 | 3.0158 | 2.8348 | +0.0000 |
| wd 0.1 -> 0 linear | 0.1 decay | **3.0150** | **2.8344** | -0.0008 |
| wd 0.04, hold 15% then 0.0425 -> 0 | partial | 3.0182 | 2.8370 | +0.0024 |

Takeaway: at d768 the effect is negligible (all within ~0.003, ~0.1%, i.e. eval noise). The 0.1->0 linear decay is marginally best on both paloma and overall; the weaker/partial schedules (0.01 constant, 0.04-hold15) are marginally worse. No schedule hurt training, none clearly helped.

## d1024 (32 H100, batch 2048, 15,276 steps — still running)

Head-to-head at the latest common eval step (7,639 / 50%). wd here is the hold-15%-then-0.0425->0 schedule; the two runs differ only in whether it also decays the router weight.

| run | wd target | paloma macro @7639 | overall macro @7639 |
|---|---|---|---|
| d1024_baseline | none | 2.9778 | 2.8016 |
| gate only | attn_gate | 2.9801 | 2.8031 |
| gate + router | attn_gate + router | **2.9778** | **2.8012** |

Also within noise so far (~0.002 spread). gate+router edges out baseline; gate-only is marginally worse. Current steps: baseline 7.7k, gate 10.5k, gate+router 8.4k (offsets from a one-time cold kernel-compile stall); comparison is pinned to the last step all three share. Will update at completion.


## S8818-15 2026-09-01T18:34:42Z
https://github.com/marin-community/marin/issues/8818#issuecomment-5498595611
All runs within noise. But do constrain the norm somewhat. Impact on scale will probably be larger on hero since attn_gate scale has already grown so much.

## S8818-16 2026-09-02T04:09:37Z
https://github.com/marin-community/marin/issues/8818#issuecomment-5504239919
🤖 d1024 weight-decay ablation — final (all three finished at step 15,276; 32 H100, batch 2048; dropless eval). wd schedule on the targeted leaves: 0 for the first 15% of training, then the 0.05·(1−step/N) line (0.0425 at the 15% mark) decaying to 0. Same seed; differ only in the decay target.

| run | wd target | paloma macro | overall macro | Δ paloma vs base |
|---|---|---|---|---|
| d1024_baseline | none | 2.7799 | 2.5982 | — |
| gate only | attn_gate | 2.7797 | 2.5979 | -0.0002 |
| gate + router | attn_gate + router | **2.7777** | **2.5964** | -0.0022 |

Takeaway: consistent with d768 — the effect is small (within ~0.002). gate+router edges out baseline on both metrics; gate-only is indistinguishable from baseline. No schedule hurt training. So decaying the router weight (on top of the gate) gives the small, consistent improvement; the attention gate alone does essentially nothing.


## S8818-17 2026-09-02T06:19:32Z
https://github.com/marin-community/marin/issues/8818#issuecomment-5505370676
🤖 attn_gate logit scaling — dropless eval on the step-48000 hero checkpoint (d6144, 1 rack, expert-collapsed dropless mesh, 50 batches × 64 seq per paloma tag).

Applies a scalar multiplier to the headwise attn_gate logits before the sigmoid: `gate = 2·sigmoid(scale · (x @ attn_gate))`. `scale<1` shrinks the logits toward 0, relaxing every head's gate toward the neutral 1.0; `scale=0` removes the learned headwise gating entirely (all gates = 1.0).

| attn_gate logit scale | dropless macro loss |
|---|---|
| 1.0 (baseline) | 2.0875 |
| 0.8 | 2.0973 |
| 0.6 | 2.7852 |
| 0.4 | 8.4599 |
| 0.2 | 9.8986 |
| 0.0 | 9.3836 |

Shrinking the gate logits by 20% is nearly free (+0.010). The model degrades at 0.6 and collapses at 0.4 and below — the learned headwise gating is load-bearing, and its effect is concentrated in the top ~40% of the logit magnitude. (The 0.2 < 0.0 non-monotonicity is inside the already-collapsed regime.)

Caveat: the `paloma/macro_loss` sub-aggregate came back `nan` for every ablation including the unaltered baseline, so it's an eval-set aggregation artifact unrelated to the scaling; the table reports the overall dropless macro across all resolved eval tags.


## S8818-18 2026-09-02T06:23:48Z
https://github.com/marin-community/marin/issues/8818#issuecomment-5505409473
Model degrades heavily if attn_gate scale is dropped by 50%. It seems quite reliant on this parameter right now. Given loss curve and evals all look super stable, I dont think we are in a rush to collapse the norms quickly. 0.03 WD can still halve the norms over 9k steps, and bring everything to a reasonable level scale for later parts of training. So see very little downside to starting with a more conservative 0.03 WD, then ramping later if needed.

## S8818-19 2026-09-02T16:18:06Z
https://github.com/marin-community/marin/issues/8818#issuecomment-5512717644
One interesting data point here: 
"Pre-training (MAI-Base-1): AdamW with a constant weight decay of 0.1, with reduced weight decay on attention weights (0.01) and embedding weights (0.005) to limit regularization on parameters that benefit less from it."
Seems that a range of decay weights are used in practice.

## S8818-20 2026-09-03T05:19:24Z
https://github.com/marin-community/marin/issues/8818#issuecomment-5520889983
This might be related to what we found in our ultra-sparse MoE training (768 experts / top-8, 60B and 180B); the write-up is here: [https://alltoall.notion.site/save-lower-layer-moe-experts-llal](https://alltoall.notion.site/save-lower-layer-moe-experts-llal)

In our runs the routed experts of the **first few MoE layers stop learning** after a few thousand steps, while the shared expert stays healthy and loss / evals / load balance all look normal. We trained with AdamW, weight decay and the default ε, so it showed up as weight-norm collapse in those layers. With Muon, or with a very small ε, the norms look fine but the layers still stop learning (momentum in those layers drops ~3 orders of magnitude, and masking their routed experts barely changes downstream metrics) — unless we give them an extra informative signal during a critical period at the start of training (an aux LM loss on the first MoE layer for the first 2–4k steps). Alongside this we saw the early-layer attention stop learning too, which makes sense: if the MLP stops adding meaningful representation to the residual, there is nothing stable for attention to build on. In our runs the order was clearly MLP first, attention as a consequence. Your setup is very different from ours (MuonH with hyperball, latent MoE, headwise attention gates, QB, no dense layers), so we're only saying this might be the same thing. **We also don't know what happens at 18T tokens — our longest run in this blog is 2T, and there every collapsed layer except the first MoE layer had either recovered or was starting to by the end; the first one stayed dead.**

What you are seeing in layer 0 — attention gated shut for essentially every ordinary token, diverging from the ladder at ~1% into the run — looks like the attention side of the same thing. And in the per-layer watch from ra2a-evaldl-c3f51 (48k checkpoint) we also see the abnormally small gradients on the **layer-0/1 routed experts: w_gate/w_up/w_down at 2.4e-5 / 2.4e-5 / 6.0e-5 in layer 0, 200–430× below layers 5–19 and ~100× below the layer-0 shared expert,** while the shared experts and attention in the same layer are normal; layer 1 is ~10× low, layers 2–4 look fine. 

One cheap check would settle it: **routed-expert-only masking (probably  free ? ).**

<img width="1530" height="884" alt="Image" src="https://github.com/user-attachments/assets/1c9c5779-c290-407c-9ddd-aff8e59c5dd0" />


## S8818-21 2026-09-04T22:35:57Z
https://github.com/marin-community/marin/issues/8818#issuecomment-5547222362
> This might be related to what we found in our ultra-sparse MoE training (768 experts / top-8, 60B and 180B); the write-up is here: https://alltoall.notion.site/save-lower-layer-moe-experts-llal

Great blog! I think our early attention layers are still learning fine, just using a somewhat awkward mechanism (the attention gate) to manage their scale. But what you found- the dead gradient in first 2 layers, does probably indicate those routed experts are not contributing.

Updated grads:
<img width="1680" height="840" alt="Image" src="https://github.com/user-attachments/assets/17b8d949-9d6e-457a-8988-fe81664fcabc" />

I love your idea of starting training with a temporary aux loss. At this stage, I think interventions midrun are probably not worthwhile for those 2 layers. Doubling the number of routed parameters is equivalent to the gain of training on 20% more tokens, when measured near compute optimal. So losing 2 layers of routed experts, (4% of routed params), is probably very marginal impact on model quality (part of this is also because we have atypically large shared experts). But maybe after the pretrain run we could just remove them, or try some experiments on intermittent checkpoints for science purposes.

## S8818-22 2026-09-04T23:39:19Z
https://github.com/marin-community/marin/issues/8818#issuecomment-5547693371
Agree. We did something similar to our main model after the run finished😆, since we found this problem midway through training. This did not cause any fatal issues in our main run. 

Looking forward to more interesting findings!


