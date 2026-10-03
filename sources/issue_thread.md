# Issue 主帖
URL: https://github.com/marin-community/marin/issues/8435
Issue is to describe the Hero Run scaling ladder, config, and track its performance.

Tracking: https://wandb.ai/marin-community/marin_moe/reports/535B-A23B-18T-Token-Hero-Run-Scaling-Ladder--VmlldzoxNzc2MDM5Ng

Details on the full model spec is in the comment below.

### Why run a scaling ladder?

1. This lets us project the model performance and compare it to the scaling projections of our prior recipes. If our projection looks meaningfully worse, then we know we need to go back to the drawing board on some aspect of the model or data. This step can catch a lot of bugs!
2. We can look at how training dynamics change as we scale, in particular the gradient norms and token dropping. In our last run, the scaling ladder caught gradient norm growth to over 4 as we scaled out token horizon, which caused us to find a fix with logit z loss. Later ablations showed that without this fix, runs under certain conditions like high batch size would completely blow up mid-run.
3. We can project our evals across the entire trajectory of the run. This means that across the entire ~100 day run, we know exactly how we are doing compared to the projection. If at any point we fail to match the projection, it can prompt early investigation into training dynamics.
4. If something strange start to happen in the Hero run, we can reference the smaller scales runs to see if its actually normal. For instance, in prior scaling ladders we saw the grad norm grow heavily up to 40% of the run, then decrease as the LR decays. When we saw this on the hero run, having the small scale reference gave us confidence we didn't need to intervene, as long as the gradient followed the same pattern (steady growth for 40% of run).
5. It only costs 1% of total compute.
### What is our EP hardware implementation?
Described in detail here: https://storage.googleapis.com/marin-public/rav/moe-fixed-wave-a2a-384/2026.08.17/index.html.
We are hand-rolling our own EP implementation, props to @ravwojdyla, since we have not found a more performant variant on JAX XLA on GPU. 

### What is our long context extension plan?
Our last run was 8k seqlen pretrain, followed by 65k extension for 1T tokens, followed by 262k extension (to occur end of Aug). Prior to that run we have used 4k seqlen pretrain. For this run we are going back to 4k seqlen pretrain to start, to enable better expert balancing (twice as many sequences per batch). We wanted to minimize risk here because our EP token dropping implementation is more experimental. (our last run was dropless FSDP). Prior tests showed token dropping grow from ~7% to ~40% when extending from 4k to 65k seqlen. Our new pooled/wave EP drops about half as many tokens (3%) at 4k seqlen; however the token dropping is likely still excessively high at 65k seqlen.

We will spend 1-2 days to do an early cooldown at roughly 10-20 days into the run, to enable RL experiments, which will be our first full scale data point at how longer context influences dropping. A nice property of this test is that the cooldown will not contaminate or impact the main run, so we have some flexibility to collect data on how an extension approach impacts dropping. Our options will be:
1. If we have landed a Jax Mixture of Kittens or dropless ragged all-to-all, then we can swap to that for the cooldown. Prior small scale tests showed the model was robust from swapping from small amounts of dropping (5%) to dropless mid-run.
2. Otherwise, we can increase our capacity factor, and accept some amount of dropping.
3. If we find that neither of these options are suitable, the next option would be to introduce sequence level balancing mid-run (though forcing the experts to re-specialize mid-run will likely be worse than accepting some amount of dropping). 

Extending context will involve updating qk_mult according to mscale=X⋅ln(new_seqlen/old_seqlen)+1, testing coefficients near 0.1 (can quickly eval a range during extension on 1 rack). If the data from the first cooldown looks smooth, will plan on 4k->8k at 50 pct into run, then 8k->65k at about 95 pct, then a targeted 262k phase near end, based on data we collect from taking the 67B-A2B from 65k to 262k.

Have further derisked token dropping by staying at 2 shared experts, each at half width. 8 active experts are also at half width. However due to latent MoE 2x compression on active experts, we are effectively running at hidden_dim width neurons shared, and 2x hidden_dim neurons routed. Having 1/3 of our neurons come from shared expert will provide a stronger backbone in the event of high dropping. We see learning still progress with zero loss spikes even at 40 pct dropping.

### What if we see delays, beyond our buffer (hardware failure, MFU degradation)?
Up to roughly 25% of the token budget, the default approach will be to compensate by decreasing the token horizon of the run. Since we are using linear LR decay, we would update the decay rate to still reach 0.05 peak at whatever the new horizon point is, and update the datamix.


# C001 2026-08-19T00:15:31Z ClassicLarry
URL: https://github.com/marin-community/marin/issues/8435#issuecomment-5335872267
🤖 # Job Summary: Aug Hero Run — 535B-A23B MoE (d6144), 18T tokens

Full-scale EP64 MoE pretrain: d6144, **535.3B total / 22.76B active** params, **18.0T tokens** (~**2.70e24** train FLOPs, fwd+bwd), on GB200 with our hand-rolled pooled-wave expert-parallel all-to-all. Data is the two-phase Harrier mixture on the fuzzy-deduped datakit store. Below: model, optimizer, training/topology, the MFU optimizations, the scaling ladder, and outputs.

## 1. Model

### Hyperparameters
| | |
|---|---|
| hidden_dim | 6144 |
| num_layers | 48 |
| num_heads | 48 · head_dim 128 |
| num_kv_heads | 12 stored — **local 12 / global 6** (GQA 4:1 local, 8:1 global) |
| num_experts / top-k | **384 / 8** |
| intermediate_dim (expert FFN) | 3072 (= hidden/2) |
| latent_dim (LatentMoE) | 3072 (= hidden/2) |
| shared experts | 2 dense SwiGLU, width 3072 |
| vocab_size | 128,256 |
| max_seq_len | 4096 · sliding_window 2048 · global_every 4 |
| capacity_factor / transport cf | 1.15 / 1.15 · num_expert_waves 3 |
| qk_mult | 1.3 · rope θ=10000 (fused, half-RoPE) |
| sconv | sites k/attn/mlp, kernel 4 |
| init | 0.5/√hidden = 0.006379 (uniform, truncated normal) |
| active / total params | 22.76B / 535.3B |

### Features
All structural (always on) unless noted.
- **All-MoE**: every block's MLP is a QB-routed `MoEMLP`; no dense-only layers.
- **GQA, split local/global KV**: sliding-window (local) layers keep 12 KV heads, full-causal (global) layers 6; K/V are sliced/aligned per layer via `lax.cond`, so global layers pay less KV compute.
- **QB routing (histogram estimator)**: router takes top-(K+1) on biased logits; the (K+1)-th logit is the per-token threshold `alpha`, the first K are selected. Each expert's threshold `beta` is the **(1−K/E) upper quantile of the margins `score−alpha`**, read from a **10k-bin histogram over the live global `[min, max]` range** (a `pmin`/`pmax` this step sets the grid; a summed-histogram cumulative count gives the per-expert quantile). The router bias `−beta` is applied on the **next step** as a zero-mean stop-gradient offset — balances expert selection without touching gradients. The histogram estimator is a smoother global quantile than a per-device top-k.
- **Sigmoid combine weights**: combine weights are `sigmoid(unbiased selected logits)`, rescaled so the K weights sum to a fixed 2.5 (not softmax).
- **LatentMoE + learnable latent RMS-norm**: tokens are compressed `hidden→latent (3072)`, passed through a **learnable RMSNorm (`latent_norm`, per-dim gain)** to decouple the expert-input scale from the down-projection init, then dispatched to experts **entirely in latent space**, and projected `latent→hidden` after the weighted combine. The router still reads the full-width token. (Also halves the all-to-all payload — see §4.)
- **Shared experts**: 2 dense SwiGLU experts run on every token in parallel with the routed path; their output is added before the residual.
- **RMSNorm on embeddings and readout**: a learnable-gain [`RMSNorm`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/model.py#L594) is applied to the token embeddings ([`embed_norm`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/model.py#L1223)) and before the untied `lm_head` ([`final_norm`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/model.py#L1165)), in addition to the per-block `rms_attn`/`rms_mlp`; the GatedNorm below wraps each of these.
- **GatedNorm**: learnable `x·sigmoid(silu(x·w_down)·w_up)` gate (rank-128 bottleneck) applied after **every** RMSNorm (attn, mlp, embed, final).
- **XSA (Exclusive Self-Attention)**: per head, subtract the component of the attention output parallel to its own value vector: `z = y − (yᵀv/‖v‖²)v` — removes the copy-through component so each head learns new information rather than passing V unchanged.
- **Attention head gate**: each head's attention output is scaled by a learned `2·sigmoid(x·attn_gate)`, one scalar per head; `attn_gate` is zero-initialized (gate starts at 1).
- **RMS qk-norm**: Q and K are RMS-normed (non-parametric) per head before RoPE.
- **Partial (half) RoPE, fused**: rotary is applied to **only the first half of head_dim**; the second half is rope-free on every layer. Single-pass fused kernel (θ=10000).
- **Rope-free global layers**: every 4th layer + the last are full-causal and **rope-free** (NoPE, qk_mult-only); the others are sliding-window with RoPE. One per-layer boolean drives mask, KV-head count, and rope together.
- **ShortConv (SConv)**: depthwise causal 1-D conv (kernel 4), identity-init, packed-document segment-masked, at three sites — after K, on the attention-branch output, on the MoE-branch output.
- **Embeddings**: `token_embed` fully replicated + shard-local lookup; `lm_head` (`output_proj`) FSDP+model-sharded; **untied**.
- **Logit z-loss**: the final-logit logsumexp is penalized (`logsumexp_weight = z_loss_weight = 1e-4`) and **added to the training loss** (fused into the CE) — controls gradient-norm growth at long token horizons (the fix the prior scaling ladder surfaced). Distinct from the router z-loss, which is logging-only.
- **Monitoring** (logged, not in loss): per-layer routing entropy, load-balance loss, router z-loss, and sender/receiver token-drop fractions.

## 2. Optimizer — MuonH / AdamH / Adam (compute-scaled)

Three groups via `optax.multi_transform` + a path mask:
- **MuonH** — attention, expert-MLP, shared-expert, latent, and GatedNorm **matrices**. Newton–Schulz orthogonalization (bf16, `backend_steps=5`, quintic coeffs, nesterov momentum 0.95) → **Frobenius-hyperball** scale-invariant step (moves along the direction, projects back onto the param's Frobenius sphere; norms in fp32). Orthogonalization is distributed **intra-rack only** (never crosses DCN); 4D expert stacks are handled without gathering the matrix dims.
- **AdamH** — the **lm_head readout**: Adam moments give the direction, then the same norm-preserving hyperball projection. Gets the MuonH LR, not `adam_lr`.
- **Adam** — `token_embed`, router, router_bias, `attn_gate`, 1-D norm gains, and the tiny SConv kernels.

### Hyperparameters (@ 18T budget)
| | |
|---|---|
| MuonH LR | 0.003291 |
| Adam LR | 0.000759 (MuonH = 13/3 × Adam = 4.33×) |
| beta1 / beta2 | **0.90** / 0.95 |
| epsilon | 6.04e-15 (Adam) · muon_epsilon 1e-8 |
| momentum / nesterov | 0.95 / true |
| use_syrk | true (Blackwell SM100 symmetric GEMM for X·Xᵀ) |
| LR schedule | linear · warmup 1% · min_lr_ratio 0.05 |
| max_grad_norm | none (no global clip) |
| z_loss_weight | 1e-4 (logit z-loss, fused into CE) |

The MuonH/AdamH groups carry **no decoupled weight decay** — they regularize purely through the norm-preserving hyperball projection. LR / beta2 / epsilon are compute-scaled from the Aug hero refit (issues #7856 / #8003, R²=0.978, seq_len=8192): `adam_lr = 0.087571 · tokens^-0.3461 · hidden^-0.3448 · √(tokens_per_batch)`, `beta2 = clip(0.999^(tpb/131072), 0.95, 0.9999)`, `epsilon = 9.676e-18 · √(tokens/tpb)`.

## 3. Training & topology
- **Hardware**: 11× GB200 NVL72 racks. **EP64 within each rack** (expert axis = 16 nodes × 4 GPUs), **data-parallel replication across the 11 racks over DCN**. Mesh `(replica_dcn=11, data, expert=64, model)`; token/batch sharded over `(replica_dcn, data, expert)`.
- **batch_size** 11,264 seqs (1024 × 11 racks) · **seq_len** 4096 · **num_train_steps** 390,139 · **total_tokens 18.0T** · **791 tokens/active-param**.
- **compute_budget** ≈ 2.70e24 FLOPs (analytic, fwd+bwd, latent-corrected).
- **mixed_precision** `params=bf16, compute=bf16, output=bf16`, with **fp32 pinned-host master params** + **offloaded optimizer state** (see §4). Router path kept fp32 before top-k/softmax/QB stats.
- **data**: two-phase Harrier mixture (15T pretrain + 3.75T cooldown, phase-1 at 80% of steps) on the fuzzy-deduped datakit store, marin tokenizer; simulated epoching against the 18.75T reference with a ≤8-epoch cap (max cell exposure ~2.1). 4k-seq pretrain deliberately (2× sequences/batch for expert balancing; token-drop grows sharply at longer context).

## 4. MFU / performance optimizations
- **Hand-rolled fixed pooled-wave all-to-all EP** — every device packs sends into one fixed `[expert_shards, pool_capacity]` pool and receives into fixed per-wave `[local_experts, receiver_cap, H]` buffers; all sizes are compile-time constants so the `all_to_all` runs on fully static shapes. Chosen over `ragged_all_to_all` to avoid a metadata/offset collective and dynamic shapes; **expert IDs ride in-band** as packed header rows in the same activation collective (no separate metadata collective). [EP writeup](https://storage.googleapis.com/marin-public/rav/moe-fixed-wave-a2a-384/2026.08.17/index.html)
- **Static waves (3)** stripe each destination pool round-robin by rank, shrinking per-wave buffers; each wave is rematerialized to cap activation memory. **Round-robin receiver allocation** keeps overflow drops unbiased across senders.
- **LatentMoE halves EP transport bytes** — the all-to-all carries latent-width (3072) rows both directions instead of hidden (6144). FLOP accounting is corrected for this so MFU isn't overstated.
- **gpu_fa4_cute FA4 attention** — bf16 CuTe segmented flash attention with per-token key lower bounds (no materialized `[B,S,S]` mask, no THD compaction), real GQA, custom CUTLASS backward, grug-tuned 64×64 tiles for B200/head_dim-128. **Sliding-window vs full-causal is just a different lower-bound array feeding the same kernel** → cuts attention FLOPs on the ~3/4 local layers. Bounds are precomputed **outside** the layer scan.
- **Fused linear-softmax cross-entropy (bf16 backward)** — `implementation="xla_fast_bwd"`: streams the whole 65,536-token rank shard over `v_block=4096` vocab blocks with online logsumexp (never materializes the `[B,128k]` logits), and runs both **backward GEMMs in bf16 on tensor cores** (vs fp32/TF32) for ~2.3× on the hero CE shape; the logit z-loss is fused as `logsumexp_weight·lse²`. `v_block=4096` is the dominant MFU lever for the 128k vocab.
- **Fused depthwise causal ShortConv (Pallas)** — the SConv sites dispatch to a fused **Pallas Triton depthwise causal conv** kernel on GPU (forward and `dx` bit-identical to the pad-and-shift reference, `dw` as accurate; pad-and-shift fallback off-GPU). Shard-local (depthwise, no cross-channel/cross-shard term → no collectives); the channel dim is FSDP-sharded so its gradient reduce-scatters.
- **Muon symmetric-GEMM (QuACK SM100) Newton–Schulz** — the two `X·Xᵀ` products in each Newton–Schulz iteration go through a **QuACK symmetric CuTe GEMM** (`use_syrk`) that computes only one triangle of the symmetric output (`A = X·Xᵀ`, then `B = b·A + c·(A·Aᵀ)`) instead of a full dense matmul — roughly halves the matmul FLOPs per iteration on the batched 4D expert stacks, where orthogonalization is the bulk of the optimizer's compute.
- **Padded + resharded distributed Muon (intra-rack)** — the stacked leaves are orthogonalized in parallel across the rack instead of gathered to one device: 3D non-expert stacks **zero-pad the leading (layer) axis** so it divides the intra-rack shard count, then reshard the padded stack over the batch mesh axes (one matrix per shard, `vmap`'d, resliced back after); 4D expert stacks `[layers, experts, fan_in, fan_out]` reshard over the `expert` axis (or the largest batch-axis subset that divides `layers·experts`) and run under a `shard_map` **without ever gathering the `fan_in×fan_out` matrix dims**, then reshard back to the parameter's own spec. All of this is pinned **intra-rack only** — the reshards deliberately exclude the `replica_dcn` axis so Newton–Schulz never runs a collective over the slow inter-rack DCN link.
- **ArrayStacked + `jax.lax.scan`** — all 48 blocks are one compiled body scanned over the layer axis, cutting compile time and HBM; router/QB collectives are structured around the scan.
- **Heterogeneous global/local layers under one homogeneous scan** — `lax.scan` needs a single compiled body, so the ways global (full-causal, rope-free) and local (sliding-window) layers differ can't be Python branches — each rides in as a traced per-layer scalar and is selected *inside* the body:
  - **RoPE diff** — global layers run rope-free. Rather than branch, RoPE is **always computed** and then selected with `jnp.where` on a per-layer `disable_rope` scalar, so both layer kinds compile to the same body.
  - **GQA diff** — local layers keep 12 KV heads, global layers 6. K/V are stored at the max (12) width and sliced-then-`align_kv_heads`'d to the per-layer count via `lax.cond(is_global, …)`, so global layers pay less KV compute without changing the stacked parameter shape.
  - **Mask diff** — the full-causal and sliding-window FA4 bound arrays are precomputed **outside** the scan and the per-layer one is picked with `jnp.where` (the sliding window is a static mask field the body can't vary), so the window rides in as the selected bounds.
- **remat `recompute_all`** — the whole block is recomputed in backward (lowest memory); a `save_moe` mode is available to keep the MoE dispatch tensors and skip re-running the EP collectives.
- **Capacity-factor token dropping (1.15)** bounds expert compute and transport buffers regardless of routing imbalance; drops are counted/logged as sender/receiver fractions.
- **use_syrk** routes the optimizer's `X·Xᵀ` through QuACK's SM100 symmetric GEMM; Newton–Schulz runs in bf16 (halved all-gather bytes) and **intra-rack only**.
- **offload_opt_state + FP32_PINNED_HOST master** — optimizer state and fp32 master weights live in pinned host RAM, shuttled device↔host each step, to leave HBM for the pooled all-to-all buffers. (The pinned-host restore is fixed in [#8443](https://github.com/marin-community/marin/pull/8443) and memory-bounded in [#8480](https://github.com/marin-community/marin/pull/8480); the hero now writes an hourly temporary checkpoint and resumes from it after a preemption.)
- **XLA/runtime flags** — latency-hiding scheduler on, collective-overlap limit 4, CUDA async allocator, 192GB host-memory limit, PGLE off, GPU command buffers disabled.

## 5. Scaling ladder (why, and the rungs)
Runs at 5 widths on one iso-ratio recipe (**791 tokens/active-param**, same mixture + simulated epoching + per-cell epoch structure), ≈1% of total compute, to (a) project performance vs prior recipes and catch bugs, (b) watch grad-norm and token-drop dynamics across scale (the last ladder caught grad-norm growth → the logit z-loss fix), and (c) project evals across the ~100-day trajectory.

| rung | racks | batch | steps | tokens | active |
|---|---|---|---|---|---|
| d768 | 1 | 1024 | 11,420 | 48B | 61M |
| d1024 | 2 | 2048 | 15,276 | 128B | 162M |
| d1536 | 6 | 6144 | 15,128 | 381B | 481M |
| d2048 | 11 | 11,264 | 20,072 | 926B | 1.2B |
| **d6144 (hero)** | 11 | 11,264 | 390,139 | **18.0T** | 22.8B |

## 6. Code (at run commit [`12d8b6f`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be))

**Model** ([`model.py`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/model.py)) — [`GrugModelConfig`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/model.py#L160)
- All-MoE routed FFN: [`MoEMLP`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/model.py#L907) · shared experts [`DenseMLP`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/model.py#L637)
- QB routing (histogram estimator): threshold/beta [`_qb_beta_hist`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/model.py#L874), histogram build [`_histogram_from_expert_counts`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/model.py#L820), applied next-step as stop-grad bias
- Sigmoid combine + renorm-to-2.5: [`_ROUTING_RENORM_SUM`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/model.py#L58), [combine](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/model.py#L980)
- LatentMoE + learnable latent RMSNorm: [`latent_norm`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/model.py#L1046)
- RMSNorm on embeddings/readout: [`embed_norm`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/model.py#L1223), [`final_norm`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/model.py#L1203); [`RMSNorm`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/model.py#L594)
- GatedNorm: [`GatedNorm`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/model.py#L612)
- XSA (exclusive self-attention): [subtract V-parallel component](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/model.py#L573)
- Attention head gate (zero-init): [`attn_gate`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/model.py#L583)
- RMS qk-norm: [`rms_norm(q)`,`rms_norm(k)`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/model.py#L534)
- Partial (half) RoPE, fused: [`_rope`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/model.py#L553), [`rope_fused`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/model.py#L541); rope-free global layers via per-layer bounds [L1252](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/model.py#L1252)
- ShortConv: [`ShortConv`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/model.py#L420) → fused Pallas [`short_conv`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/model.py#L52)
- Attention (FA4-cute): [call](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/model.py#L572), [`fa4_cute_segment_bounds`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/model.py#L1252)
- Untied embeddings: [`token_embed`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/model.py#L1160), [`output_proj`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/model.py#L1163)
- Homogeneous global/local scan (rope diff + GQA diff): [`ArrayStacked` scan body](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/model.py#L1262), rope-free select via [`jnp.where(disable_rope)`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/model.py#L568), per-layer KV-head [`lax.cond(is_global)`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/model.py#L521), per-layer mask-bounds [select](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/model.py#L1268)

**Optimizer** — MuonH/AdamH/Adam
- Group mask (which param → which group): [`create_mask`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/optimizer.py#L195)
- MuonH transform + Frobenius hyperball: [`scale_with_grug_muonh`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/optimizer.py#L87), [`_scale_invariant_hyperball_updates`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/optimizer.py#L56); Newton–Schulz core [`_grug_scale_with_muon_hero`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/grugmuon_hero.py#L52); [`GrugMoeMuonHConfig`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/optimizer.py#L121)
- AdamH (lm_head readout): [`scale_by_adamh`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/adamh.py#L37)
- Compute-scaled LR/beta2/epsilon: [`MoeHeuristic`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/heuristic.py#L24), [formula](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/heuristic.py#L54), [`build_hero_configs`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/heuristic.py#L118)

**Training & topology** ([`train.py`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/train.py), [`launch_scaling_ladder.py`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/launch_scaling_ladder.py))
- Offloaded opt state + fp32 pinned-host master: [`_tree_to_memory_kind`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/train.py#L451), [`MasterParamMode`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/train.py#L81), shuttle [L483](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/train.py#L483)
- Logit z-loss (fused into CE): [`z_loss_weight`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/train.py#L117)
- Two-phase mixture + simulated epoching: [rescale](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/train.py#L200), [stage callback](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/train.py#L413)
- Dropless eval: [`build_tagged_evaluator`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/train.py#L287)
- EP64 mesh / DCN replication, ladder rungs, hourly temp checkpoints + resume: [`LADDER_RACKS`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/launch_scaling_ladder.py#L81), [`TOKENS_PER_ACTIVE_PARAM`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/launch_scaling_ladder.py#L87), [`CheckpointerConfig`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/launch_scaling_ladder.py#L248)

**MFU kernels** (levanter)
- Fixed pooled-wave EP all-to-all (static shapes, in-band expert IDs, 3 waves): [`MoEExpertMlp`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/model.py#L907), [config guard](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/model.py#L241)
- FA4-cute segmented flash attention (bf16, real GQA, sliding-window as a bounds array): [`fa4_cute_segment_bounds`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/model.py#L41)
- Fused linear-softmax cross-entropy (bf16 backward, streamed vocab blocks): [`fused_cross_entropy_loss`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss)
- Fused depthwise causal ShortConv (Pallas): [`short_conv`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/lib/levanter/src/levanter/kernels/pallas/short_conv)
- Muon symmetric-GEMM Newton–Schulz (QuACK SM100, `use_syrk`): [`_newtonschulz_batched_syrk`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/grugmuon_hero.py#L182) → [`quack_symmetric_gemm`](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/lib/levanter/src/levanter/grug/_moe/quack_symmetric_cute.py)
- Padded + resharded distributed Muon (intra-rack, no matrix-dim gather): [3D padded stack](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/grugmuon_hero.py#L298), [4D expert-stack distribution](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/grugmuon_hero.py#L211), [intra-rack axis selection](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/grugmuon_hero.py#L32)

## 7. Outputs (pending — run in progress)
- **Checkpoints**: `s3://marin-us-east-02a/marin/grug/<run>/…/checkpoints` (permanent every 6000 steps for the hero; output-only).
- **W&B**: pending.
- **Final evals** (`eval/paloma/macro_loss`, `eval_dropless/paloma/macro_loss`, `eval/paloma/c4_en/bpb`, `eval/uncheatable_eval/macro_loss`): pending.
- **Throughput** (mean over last 100 steps): pending. Reference from the d2048 rung: compute-step ~24M tok/s (MFU p50 ~13.6%), sustained ~17M tok/s (the gap is bursty cross-rack all-reduce over DCN, not evals/logging).




# C002 2026-08-19T01:16:46Z Helw150
URL: https://github.com/marin-community/marin/issues/8435#issuecomment-5336301526
## Raw Data

On the raw data side, our pool for this hero run is logged at this frozen commit of [our data provenance table on hugging face](https://huggingface.co/datasets/marin-community/token-counts/commit/3612ddc3350212e38ce8a50b99c3df7ea10b44a9). The raw (pre-dedupe) pool contains 25.6 trillion tokens from 292 sources (though >100 of those are a variety of rollout datasets courtesy of @penfever!).

## Deduplication & Decontamination

For this run, we removed 2.494T tokens through a combination of  fuzzy deduplication and n-gram based decontamination w.r.t many common benchmarks in LM-eval-harness + the Artificial Analysis Intelligence Index (~250k documents were removed) for a final count of 23.106T tokens. 

Fuzzy duplication used a combination of MinHash deduplication and heuristics among linked pairs within connected component clusters. Examples of duplicates can be found here: https://storage.googleapis.com/marin-public/rav/dedup-pair-browser/2026.08.18.2/index.html

## Bucketing

We then embedded all documents with [Harrier 0.6b](https://huggingface.co/microsoft/harrier-oss-v1-0.6b). 

For topic labels, documents were grouped into 40 semantic buckets by first performing K-means with 5000 clusters (using a sample of 2.56 Million documents) and then doing agglomerative clustering over those 500 clusters to get 40 buckets.

For quality labels, documents were labeled using GLM 5.2 for quality and these labels were distilled into a "Faster Transformer" classifier using both the harrier embedding and the token distribution of the document.

You can see samples from each bucket + distributions from source -> topic x document buckets here: https://storage.googleapis.com/marin-public/held/harrier-k40-cluster-overview/2026.08.18/index.html?revision=uniform-sampling



# C003 2026-08-19T04:56:01Z ClassicLarry
URL: https://github.com/marin-community/marin/issues/8435#issuecomment-5337709199
Scaling Ladder Wandb Report: https://wandb.ai/marin-community/marin_moe/reports/Hero-Run-Scaling-Ladder--VmlldzoxNzc2MDM5Ng


# C004 2026-08-19T15:47:34Z ClassicLarry
URL: https://github.com/marin-community/marin/issues/8435#issuecomment-5344533796
🤖 Scaling-ladder analysis for the **535B-A23B, 18T-token** hero. The ladder trains one uniform hero EP recipe at five widths; four rungs finished and the d2048 rung crashed at ~81%. Fitting a power law across the rungs (`L = 1.5 + A·C^-α`, compute excludes lm_head) and extrapolating to the hero's compute predicts a final **dropless paloma macro-loss ≈ 2.04**.

**Hero prediction by training percentile** — the four measured rungs plus the d6144 hero prediction at every 5% (dashed/hollow = extrapolated).
![hero prediction by percentile](https://raw.githubusercontent.com/marin-community/marin/d23e6e9c3673435fb82d83aa6c51a607d0da6009/experiments/grug/moe_hero_ep/scaling_ladder_figs/hero_prediction_by_percentile.png)

**Per-5% scaling-law fits (log-log)** — one power-law fit per training fraction; the crimson points trace the hero prediction out to hero-scale compute, ending at 2.039 @100%.
![scaling law fits](https://raw.githubusercontent.com/marin-community/marin/d23e6e9c3673435fb82d83aa6c51a607d0da6009/experiments/grug/moe_hero_ep/scaling_ladder_figs/ladder_scaling_laws.png)

**Method check** — a linear extrapolation from each finished rung's 60–80% window predicts its own 100% dropless macro-loss to within ~0.003–0.004 (consistently a hair optimistic, since the curve flattens under cosine decay).
![60-80 extrapolation vs actual](https://raw.githubusercontent.com/marin-community/marin/d23e6e9c3673435fb82d83aa6c51a607d0da6009/experiments/grug/moe_hero_ep/scaling_ladder_figs/ladder_extrapolation_60_80.png)

**Full-granularity training cross-entropy** — every step, with the datamix phase-2 boundary at 80% (a small loss bump is visible in every rung at that point).
![training cross-entropy](https://raw.githubusercontent.com/marin-community/marin/d23e6e9c3673435fb82d83aa6c51a607d0da6009/experiments/grug/moe_hero_ep/scaling_ladder_figs/ladder_train_ce_fullres.png)

Prior precedent for this kind of extrapolation — the 67B-A2B 10T run's preregistered (80%) and two retroactive (21%, 63%) scaling-law fits each predicted that run's target within ~0.6% (issue #6044):
![67B scaling fits](https://raw.githubusercontent.com/marin-community/marin/d23e6e9c3673435fb82d83aa6c51a607d0da6009/experiments/grug/moe_hero_ep/scaling_ladder_figs/scaling_fit_67b_prereg_vs_retro.png)

Full source to reproduce (pulls the W&B runs and regenerates the ladder figures): [`experiments/grug/moe_hero_ep/plot_scaling_ladder.py`](https://github.com/marin-community/marin/blob/d23e6e9c3673435fb82d83aa6c51a607d0da6009/experiments/grug/moe_hero_ep/plot_scaling_ladder.py).



# C005 2026-08-19T16:09:22Z ClassicLarry
URL: https://github.com/marin-community/marin/issues/8435#issuecomment-5344804457
<img width="1430" height="910" alt="Image" src="https://github.com/user-attachments/assets/5316f64c-ae45-48ea-a937-040787c82ed7" />

Comparing our scaling law to prior runs.

The main differences here is that the May recipe is at compute optimal, this run is at ~12x overtrained, and the 67B-A2B scaling law was at 100x overtrained. The 67B-A2B ran at 8k seqlen, and was therefore given a lower asympotote of 1.4 instead of 1.5. The May fit also inaccurately has an asympotote at 1.6 instead of 1.5. Main architecture change is we dropped from 4/256 level of sparsity to 4/192 level of sparsity (25% fewer total parameters, to fit on GB200 at this scale. Several other small features added. Overall, the scaling law looks reasonable and nothing looks off in the training dynamics or data.

Grad norms look stable over the run, peaking at 25% into the run. These runs ran at oversized batch sizes, which may be slightly suppressing the gradient. Expecting the grad norm to climb higher on the hero run, but still peak at the same ~25% into the run. This will be good metric to monitor for issues.

<img width="928" height="310" alt="Image" src="https://github.com/user-attachments/assets/c0a8552a-4da4-4d57-b4be-e5ec4afedf0d" />

Drop rate jumps to 10 pct at very start, then drops to 0, then climbs to ~4pct over the run. Slightly increasing as we scale. Predicting we wont have that 10 percent spike for the hero, because our warmup (1 percent) will cover a much larger number of steps since the hero is at a higher step count. I think most likely scenario is the hero runs around 2 percent dropping, but up to 8 percent seems plausible bsaed on the data at 4k seqlen.

<img width="933" height="290" alt="Image" src="https://github.com/user-attachments/assets/2aad2cb2-d7b2-4025-a662-1e006c6a4d27" />


# C006 2026-08-19T16:13:54Z ClassicLarry
URL: https://github.com/marin-community/marin/issues/8435#issuecomment-5344856737
Also noting- the d2048 scaling ladder rung failed at 81%. Chose not to resume to add more time to Hero Run. The dynamics look clear from the first 81 percent, and we can cleanly extrapolate the last 19%. From a practical perspective, we will not actually be running the end of the run at 4k seqlen, and the datamix may update midrun. So the utility of the scaling law is more around the first 50 percent of the run and ensuring the projection is reasonably within expectation (which it is).


# C007 2026-08-19T16:29:23Z ClassicLarry
URL: https://github.com/marin-community/marin/issues/8435#issuecomment-5345040115
From model perspective, main risk points during the run:

1. First 3%. Can we get through LR warmup and reach steady dynamics? If we see gradient run away/large issues here, we likely need to revisit initialization, or potentially initial batch size. Or increase capacity factor if the issue involves high dropping.
2. 30% mark. Gradient may peak, and we will see if gradient keeps climbing, or if its starts to cooldown. Indication that it keeps climbing indefinitely could indicate we need stronger logit z loss, lower token dropping, or revisit some of the norm constraints.
3. Context Extension. How will token dropping respond? See initial Issue desc.
4. Very low loss regime. If training plateaus or acts strange near end of run only, first things to check will be the numerics.

Health checklist:

1. Is total grad norm oscillating steadily? In our May 16B active run, the grad norm grew steadily to 1.5 and training was still very smooth. So I am less concerned with the absolute magnitude, and more concerned with the rate of change. Hyperball makes comparing absolute magnitude not apples to apples with AdamW runs.
2. Are all Hyberball constrained parameters actually respecting the constraint? In past we have seen bug where JAX does norm calculation incorrectly, due to some transient sharding issue. This has been updated to a more explicit calculation that seemed to resolve that issue, but issues of a similar nature may arise.
3. Is token dropping following similar trend as scaling ladder?
4. Is loss decreasing steadily?
5. Are evals decreasing steadily?
6. Is our router bias terms behaving steadily?
7. Does router entropy stay near max?
8. Is embed norm growth reasonable?
9. Is MFU holding constant?
10. Is LR schedule (both Adam and MuonH params) actually following the correct schedule?
11. Does the entire wandb config look accurate?
12. Are we hitting the projected eval targets at each 5 percent?


# C008 2026-08-20T15:19:25Z ClassicLarry
URL: https://github.com/marin-community/marin/issues/8435#issuecomment-5357943524
🤖 Parameter-norm health check on the hero run [`hero-12d8b6f0-dee637`](https://wandb.ai/marin-community/marin_moe/runs/hero-12d8b6f0-dee637) — every parameter's min / max / avg L2 norm across the run (`params/norm/<path>`), with its optimizer group from `create_mask` (`experiments/grug/moe_hero_ep/optimizer.py`).

| group | parameter | min | max | avg |
|---|---|--:|--:|--:|
| muonh | attn.w_q | 268 | 268 | 268 |
| muonh | attn.w_o | 268 | 268 | 268 |
| muonh | attn.w_k | 134 | 134 | 134 |
| muonh | attn.w_v | 134 | 134 | 134 |
| muonh | mlp.expert_mlp.w_gate | 2624 | 2624 | 2624 |
| muonh | mlp.expert_mlp.w_up | 2624 | 2624 | 2624 |
| muonh | mlp.expert_mlp.w_down | 2624 | 2624 | 2624 |
| muonh | mlp.w_latent_down | 189 | 189 | 189 |
| muonh | mlp.w_latent_up | 189 | 189 | 189 |
| muonh | shared.0.w_gate / w_up / w_down | 189 | 189 | 189 |
| muonh | shared.1.w_gate / w_up / w_down | 189 | 189 | 189 |
| muonh | attn_gated_norm.w_down / w_up | 38.75 | 38.75 | 38.75 |
| muonh | mlp_gated_norm.w_down / w_up | 38.75 | 38.75 | 38.75 |
| muonh | embed_gated_norm.w_down / w_up | 5.594 | 5.594 | 5.594 |
| muonh | final_gated_norm.w_down | 5.594 | 5.594 | 5.594 |
| muonh | final_gated_norm.w_up | 5.562 | 5.562 | 5.562 |
| adamh | output_proj (lm_head) | 177 | 177 | 177 |
| adam | token_embed | 177 | 197 | 182.4 |
| adam | mlp.router | 67 | 139 | 102 |
| adam | mlp.router_bias | 0 | 538.9 | 311.2 |
| adam | attn.attn_gate | 0 | 22.62 | 16.92 |
| adam | attn.sconv_k.weight | 270 | 272 | 271.3 |
| adam | mlp.latent_norm.weight | 384 | 410 | 394 |
| adam | rms_attn.weight | 544 | 544 | 544 |
| adam | rms_mlp.weight | 544 | 556 | 549.8 |
| adam | sconv_attn.weight | 544 | 544 | 544 |
| adam | sconv_mlp.weight | 544 | 564 | 553.5 |
| adam | embed_norm.weight | 74.5 | 78.5 | 76.7 |
| adam | final_norm.weight | 78.5 | 86.5 | 81.15 |

Group counts: **muonh 23, adamh 1, adam 12**. Norms are over the full 48-layer stacked tensors (one value per stacked parameter, covering all layers).

Read: healthy. The **muonh and adamh norms are pinned constant** — that is the Frobenius-hyperball projection holding those groups norm-preserving by design, so flat is the expected signature (drift there would be the flag). The **adam** group is the free set and evolves as expected. The two `min = 0` entries are intended init dynamics, not anomalies: `attn_gate` starts closed at zero and opens (→ 16.9 avg), and `router_bias` starts at 0 and grows as the QB stop-grad biases accumulate (→ 539). No exploding/collapsing norms and no NaN/inf.



# C009 2026-08-20T15:21:51Z ClassicLarry
URL: https://github.com/marin-community/marin/issues/8435#issuecomment-5357978745
rms attn has not moved, but that looks somewhat normal based on scaling ladder. 


# C010 2026-08-20T15:27:51Z ClassicLarry
URL: https://github.com/marin-community/marin/issues/8435#issuecomment-5358054499
Also to make it more clear- the pre-registered loss in this thread is under the hypothetical scenario where we stayed 4k seqlen for the entire run and kept the exact same datamix and token count. At a minimum the seqlen will be changing mid-run. And the datamix has already been updated on Aug 19th from what was used on the scaling ladder. Since the exact context length extension schedule will depend on how our EP kernels progress, it was not something we could realistically simulate via a scaling ladder. However the predicted trajectory will still be useful for every data point prior to context extension, and the old/new datamix should be reasonably close on paloma macro loss.


# C011 2026-08-20T21:15:16Z mansimov
URL: https://github.com/marin-community/marin/issues/8435#issuecomment-5361751142
nice work! @ClassicLarry do you mind linking the code snippets in the performance optimization section. might be useful for folks (even tho we use pytorch and fine-tuning open source trained models we can ask our agents to translate those jax code snippets)


# C012 2026-08-20T22:40:56Z ClassicLarry
URL: https://github.com/marin-community/marin/issues/8435#issuecomment-5362936645
> nice work! [@ClassicLarry](https://github.com/ClassicLarry) do you mind linking the code snippets in the performance optimization section. might be useful for folks (even tho we use pytorch and fine-tuning open source trained models we can ask our agents to translate those jax code snippets)

Good idea, done!


# C013 2026-08-22T06:22:27Z windsornguyen
URL: https://github.com/marin-community/marin/issues/8435#issuecomment-5378422117
The advertised Harrier store is not currently downloadable anonymously from the public CoreWeave endpoint:

- `ListObjectsV2` on `s3://marin-us-east-02a/marin/datakit/store_4d2e363d` returns `AccessDenied`.
- An unsigned `GetObject` for the deterministic artifact key `marin/datakit/store_4d2e363d/.artifact.json` returns HTTP 403 from `marin-us-east-02a.cwobject.com` (observed 2026-08-22 UTC).
- Exact known per-cell metadata keys fail the same way.

CoreWeave documents that a bucket policy can grant cross-organization or internet-readable access; the current behavior indicates that policy is absent or not effective. Is there an intended public access-key/download procedure, or can the bucket policy / a public mirror be published? The linked GCS composition report is accessible.


# C014 2026-08-24T04:48:10Z WhenWen
URL: https://github.com/marin-community/marin/issues/8435#issuecomment-5390867355
> First 3%. Can we get through LR warmup and reach steady dynamics? If we see gradient run away/large issues here, we likely need to revisit initialization, or potentially initial batch size. Or increase capacity factor if the issue involves high dropping.

Looks like this one is clear


# C015 2026-08-24T13:11:15Z MythosAd
URL: https://github.com/marin-community/marin/issues/8435#issuecomment-5395671408
Thanks for sharing this work.  
I have a question regarding the motivation behind LatentMoE. Since the training system uses GB200 NVL72 with high-bandwidth NVLink, I am curious why a latent representation is introduced before expert dispatch.  
Is LatentMoE mainly motivated by reducing communication overhead at larger scales (e.g., beyond a single NVLink domain), or does it also provide benefits beyond communication optimization, such as improving model capability?
Thanks!


# C016 2026-08-24T21:36:53Z dlwh
URL: https://github.com/marin-community/marin/issues/8435#issuecomment-5401686089
@windsornguyen we can share lots of metadata for data mix but we're unable to grant direct access to data. What are you looking for?


# C017 2026-08-25T06:53:00Z ClassicLarry
URL: https://github.com/marin-community/marin/issues/8435#issuecomment-5406648997
> Thanks for sharing this work. I have a question regarding the motivation behind LatentMoE. Since the training system uses GB200 NVL72 with high-bandwidth NVLink, I am curious why a latent representation is introduced before expert dispatch. Is LatentMoE mainly motivated by reducing communication overhead at larger scales (e.g., beyond a single NVLink domain), or does it also provide benefits beyond communication optimization, such as improving model capability? Thanks!

On its own, LatentMoE halves the comms cost. We found that putting a learnable RMSNorm on the latent projection improved model capability as well, so LatentMoE w/ norm performed better than no latent MoE. Without this learnable RMSNorm, we run into activation scaling issue and see 30 percent worse quality. We also found going from top 4 of 192 to top 8 of 384 (half sized) gave 15 percent quality gain. But it doubles comms cost. So one way to view latentMoE is it enables 4/192->8/384 without change to either comms or compute.


# C018 2026-08-26T15:28:41Z MythosAd
URL: https://github.com/marin-community/marin/issues/8435#issuecomment-5427499956
Thanks for the detailed reply! I have a few follow-up questions after looking through some of the previous Marin experiments.
1. Alternating dense/MoE vs shared experts.
I noticed #6443 already implemented an alternating dense/MoE architecture with no shared experts in the MoE blocks, which looks quite similar to the design later used in MAI-Thinking-1, but I couldn't find final results. Was that direction abandoned because of quality, systems efficiency, or simply because the runs weren't completed?
More broadly, do you think dense FFN layers can replace much of the functional role of shared experts? MAI's results suggest that dense layers can carry the common computation while the MoE layers specialize, while also eliminating all-to-all communication on half of the FFN layers. Since the current Hero model uses shared experts partly as a backbone against token dropping, would this design become more attractive once routing is fully dropless?
2. Relative position on local layers + NoPE on global layers.
I also saw the negative Inkling relative-position experiment in #7208. One slightly different variant I'm curious about is using learned relative position only on the sliding-window layers, while leaving global-attention layers completely NoPE. My intuition is that local layers can learn precise relative ordering, while global layers remain position-independent for better long-context extrapolation. Do you think the #7208 result argues against this variant as well, or would it be meaningfully different?
3. AdamH for token embeddings.
In #5203, moving token_embed to AdamH with unit-variance initialization seems to have worked across all four tested scales. However, the current Hero recipe puts token_embed back in Adam. Was that result later invalidated by the MuonH recipe / newer embedding ablations (#6442), or was there another reason it wasn't carried forward?


# C019 2026-08-27T01:23:04Z ClassicLarry
URL: https://github.com/marin-community/marin/issues/8435#issuecomment-5433154702
> Thanks for the detailed reply! I have a few follow-up questions after looking through some of the previous Marin experiments.

1. MAI Thinking makes their dense layers only have num_neurons = hidden_dim, which is roughly 3x smaller than normal. This to me a is a bit awkward at first glance because attn is still full capacity every layer. So the normal behavior of attention fetch info, mlp process it is a bit imbalanced (of course if one has data that an idea is good, that it seeming awkward doesn't matter, but it made me more hesitant to spend the time collecting that data instead of working on other things). At very small scale if I do an exact active:total replica (dense 3x width hidden instead of 1x), I get same quality and 20% better MFU than our baseline. But small scale alone is poor signal here, because dense models look much better at small scale in general. We never ended up doing large scale ablation here, since I was more worried about token dropping dynamics. Also the way we are using jax scan makes it a awkward to have different behavior per layer. I am less optimistic about alternating dense/moe long term, because with mature mixture-of-kittens kernels the MFU hit of MoE is probably not that high anyways.
2. Hmm I separately saw some early positive results from Inkling relative position embedding that gave lower loss. But our kernel was not quite fast enough to justify. And perhaps more importantly, the kernel was a bit buggy if I changed sliding window length. I think it should be good on global layers too, since its limited to just the most recent X (512 or 1024) positions. I am more bullish on relative position embedding + MLA. Since MLA has very janky approach to RoPE, which requires replicating 64 head dims across every head. Relative position embedding fixes this. Have to be a bit careful with ablations here because at very small scale RoPE on global looks much better than NoPE on global. (but this fades with scale)
3. So initial AdamH on embed looked good. But later I did learning rate sweep, and if LR was 40% too high, was seeing grad spikes up to 20. I think what was happening (guessing) was rare tokens were having their norm driven to near zero, and then getting massive grad when they finally ended up in the batch. (we have rms_norm on embed that will apply high growth multiplier to small vocab embeds) Dont want any features in the model that require perfectly optimal hypers. So reverted it.

<img width="874" height="552" alt="Image" src="https://github.com/user-attachments/assets/096aaa54-f17f-4e96-8a8e-a1a825f84270" />


# C020 2026-08-27T02:50:16Z MythosAd
URL: https://github.com/marin-community/marin/issues/8435#issuecomment-5433737400
Thanks, this is really helpful.

My motivation for alternating dense/MoE is less about the current MFU gap and more about separation of concerns: dense layers handle common computation, while MoE layers handle specialization. Keeping them as two clean execution patterns seems to leave more room for independent kernel, scheduling, and parallelism optimization, even if MoE kernels improve substantially.

For position encoding, my preference for learned relative PE on local layers + NoPE on global layers is mainly to avoid positional extrapolation entirely. With a fixed local window, the relative-position range never changes as context grows, while global attention stays purely content-addressed. I agree this probably needs larger-scale ablations to judge properly.



# C021 2026-08-28T23:14:07Z mudkjp
URL: https://github.com/marin-community/marin/issues/8435#issuecomment-5458731192
Is there a way to see sample model generations from the ongoing run?


# C022 2026-08-31T20:15:57Z ruisizhang123
URL: https://github.com/marin-community/marin/issues/8435#issuecomment-5483980316
> [@windsornguyen](https://github.com/windsornguyen) we can share lots of metadata for data mix but we're unable to grant direct access to data. What are you looking for?

Thank you for the info. I did notice that the data mix is selected by category & quality codeword in launch_datakit_moe_mix.py. To reproduce this, I will need the mapping from raw text to these codeword, which requires the k-means [checkpoint](https://github.com/marin-community/marin/blob/871f36b38184e659e471fa6d0e8e660f6170c177/experiments/datakit/cluster/domain/v0/train.py) and Fast transformer [checkpoint](https://github.com/marin-community/marin/issues/6739).

Wonder if you may kindly open-source them, or I might overlooked them somewhere in the repo : )


# C023 2026-08-31T23:08:11Z Helw150
URL: https://github.com/marin-community/marin/issues/8435#issuecomment-5485981757
> Wonder if you may kindly open-source them, or I might overlooked them somewhere in the repo : )

Great call! You can find weights for both for this version of the data mix in https://huggingface.co/marin-community/marin-data-mix-tools/tree/08.18.2026/08.18.2026. Inference code for both is in the repo itself!


# C024 2026-09-01T05:13:00Z ClassicLarry
URL: https://github.com/marin-community/marin/issues/8435#issuecomment-5489229549
> Is there a way to see sample model generations from the ongoing run?

Started here: https://github.com/marin-community/marin/issues/8827


# C025 2026-09-22T22:32:10Z yonromai
URL: https://github.com/marin-community/marin/issues/8435#issuecomment-5785310423
🤖 ## Finding and recommendation

Keep the ongoing Hero run on its current router arithmetic.

Using BF16 operands with FP32 accumulation and output reduced saved-input router score RMS error from `0.0074565` to `0.00002808` against an FP64 reference. It matched the reference top-8 order for all 324 valid saved entries; current arithmetic changed the order for 80 entries. On the restored step-126000 training batch, the arithmetic change altered top-8 order for `8.9853%` of routed tokens and changed the selected expert set for `2.6479%`.

The more accurate dot was about `1.5–2%` slower per training step in two short launch orders. Across all 30 paired post-compile steps, its median throughput ratio was `0.98245` and its median iteration-time ratio was `1.01436`. Peak HBM was unchanged at `115.5375 GiB` in the checkpoint-writing pair.

The 20-step paired continuation stayed finite and had almost identical mean training loss: preferred minus current was `-0.00000127`. The runs still followed different trajectories. Router gradients and pending query-bias state diverged, and mean sender drops increased by `5.55%` under the preferred arithmetic. Receiver drops remained zero. Expert-load summaries showed no notable systematic skew over this short run.

The corrected fixed-current-arithmetic Paloma evaluation applied each checkpoint's pending query-bias update. The preferred-trained checkpoint was higher by `0.00072265` macro loss and `0.00029618` macro BPB after 20 steps. Fifteen of 16 subset losses were higher. This difference is too small and too early to establish lasting quality harm.

The evidence shows a real numerical and routing-policy change with a small runtime cost and no demonstrated benefit for a mid-run transition. A fresh-run ablation remains reasonable.

## Limits

All full-model experiments used one 16-node B200 NVL72 rack. This preserves Hero's EP64 geometry, but uses one data-parallel replica and batch 1024 instead of the production run's 11 racks and global batch 11,264. It does not exercise cross-rack reductions. The query-bias histogram also sees 1024 sequences instead of 11,264.

The continuations cover only 20 optimizer steps. They can show immediate route and state divergence, but they do not establish long-term benefit or harm. The speed delta is also close to the one-rack noise floor, so `1.5–2%` is an approximate cost rather than a precise forecast for the 11-rack run.

<details><summary>Evidence and reproduction</summary>

### Source and code identities

- Observed production source: [`e8460b35bd6b312b62799b12ae363b3981af3c83`](https://github.com/marin-community/marin/commit/e8460b35bd6b312b62799b12ae363b3981af3c83)
- Complete native source checkpoint: `s3://marin-us-east-02a/marin/grug/hero-main-step121638/2026.08.19.2/checkpoints/step-126000`
- Investigation base: [`1f9c387b6c452895ff58573c2b6117c5500303db`](https://github.com/marin-community/marin/commit/1f9c387b6c452895ff58573c2b6117c5500303db)
- Experiment branch: [`yonromai/goal/hero-training-router-01a0c066`](https://github.com/yonromai/marin/tree/goal/hero-training-router-01a0c066)
- Final experiment commit: [`a00cb77a491f4777a2c66edce54f47ff7b255c40`](https://github.com/yonromai/marin/commit/a00cb77a491f4777a2c66edce54f47ff7b255c40)
- Producer commits: [`f9178348f5237cb2cb591024511d4db693a3910c`](https://github.com/yonromai/marin/commit/f9178348f5237cb2cb591024511d4db693a3910c), [`899efa2e8a3d04ef5c7cc7cd75e6abd18f885fd8`](https://github.com/yonromai/marin/commit/899efa2e8a3d04ef5c7cc7cd75e6abd18f885fd8), [`a1928638ca09fa26ec243ebd24754d1b3567d2b5`](https://github.com/yonromai/marin/commit/a1928638ca09fa26ec243ebd24754d1b3567d2b5), [`5b59fc491e5f25f3f39d338718b9d6fd72a565f8`](https://github.com/yonromai/marin/commit/5b59fc491e5f25f3f39d338718b9d6fd72a565f8), [`2e392b6c0568e969425eb1bf97b02c853a678860`](https://github.com/yonromai/marin/commit/2e392b6c0568e969425eb1bf97b02c853a678860), and [`a00cb77a491f4777a2c66edce54f47ff7b255c40`](https://github.com/yonromai/marin/commit/a00cb77a491f4777a2c66edce54f47ff7b255c40)

The branch is experimental. No PR was opened, and no live Hero setting or checkpoint was changed.

### Experiment map

| Experiment | Purpose | Iris job and evidence |
| --- | --- | --- |
| Saved-input B200 probe | Compare compiled forward and backward arithmetic at the production local shape | `/romain/hero-router-shape-b200-01a0c066-a7`; HLO bundle below |
| Immediate restored-batch comparison | Measure score and expert-selection changes before updating weights | `/romain/hero-router-pref-compare-coord-01a0c066-a2`; [W&B](https://wandb.ai/marin-community/marin_moe/runs/hero-router-pref-compare-step126k-01a0c066-a2) |
| Current 20-step continuation | Native-state control from step 126000 | `/romain/hero-router-current-cont-coord-01a0c066-a1`; [W&B](https://wandb.ai/marin-community/marin_moe/runs/hero-router-current-cont-step126k-01a0c066-a1); checkpoint `s3://marin-us-east-02a/marin/users/romain/hero-training-router-01a0c066/checkpoints/current-a1/step-126020` |
| Preferred 20-step continuation | Native-state candidate from the same checkpoint and batches | `/romain/hero-router-pref-cont-coord-01a0c066-a1`; [W&B](https://wandb.ai/marin-community/marin_moe/runs/hero-router-pref-cont-step126k-01a0c066-a1); checkpoint `s3://marin-us-east-02a/marin/users/romain/hero-training-router-01a0c066/checkpoints/preferred-fp32-a1/step-126020` |
| Reversed launch-order timing | Bound short-run timing variation | [Preferred W&B](https://wandb.ai/marin-community/marin_moe/runs/hero-router-pref-speed-rep-step126k-01a0c066-a1), [current W&B](https://wandb.ai/marin-community/marin_moe/runs/hero-router-current-speed-rep-step126k-01a0c066-a1) |
| Corrected current-checkpoint Paloma | Evaluate the complete learned state under current arithmetic | `/romain/hero-router-current-fixed-eval-qb-coord-01a0c066-a2`; [W&B](https://wandb.ai/marin-community/marin_moe/runs/hero-router-current-fixed-eval-qb-step126020-01a0c066-a2) |
| Corrected preferred-checkpoint Paloma | Same fixed policy and pending query-bias handling | `/romain/hero-router-pref-fixed-eval-qb-coord-01a0c066-a2`; [W&B](https://wandb.ai/marin-community/marin_moe/runs/hero-router-pref-fixed-eval-qb-step126020-01a0c066-a2) |

Both continuation jobs and both corrected Paloma jobs completed all 16 tasks without failures or preemptions.

The earlier fixed-evaluation `a1` runs omitted each checkpoint's pending query-bias update. They are superseded by the corrected `a2` runs above. The separate evaluation-path problem is tracked in [#9352](https://github.com/marin-community/marin/issues/9352).

### Saved-input accuracy

The fixture contains 324 valid layer-15 entries representing 32 byte-distinct activation vectors. The production-shape probe tiles the entries to 65,536 rows and preserves their captured frequency.

| Arithmetic | Score RMS error | Maximum error | Top-8 order versus FP64 reference | Isolated forward | Isolated backward |
| --- | ---: | ---: | --- | ---: | ---: |
| Current BF16 output, then FP32 cast | `0.0074565` | `0.0311123` | 80/324 entries changed | 14.07 ms | 14.43 ms |
| BF16 operands, FP32 accumulation/output | `0.00002808` | `0.00007003` | all 324 matched | 14.07 ms | 14.48 ms |
| FP32 operands | `0.00000416` | `0.00002373` | all 324 matched | 18.20 ms | 23.26 ms |

The isolated timing covers only the saved-input router dot and gradient. It is not a whole-step measurement.

### Continuation measurements

| Post-compile metric, 19 real batches | Current | Preferred FP32 accumulation | Preferred minus current |
| --- | ---: | ---: | ---: |
| Median train throughput | 275,523 tok/s | 271,399 tok/s | -1.50% by ratio of medians |
| Median full iteration | 16.3871 s | 16.5275 s | +0.86% by ratio of medians |
| Final cumulative peak HBM | 115.5375 GiB | 115.5375 GiB | 0 |
| Mean train loss | 1.24572355 | 1.24572228 | -0.00000127 |
| Mean sender drops | 684,900 | 722,882 | +37,983 (+5.55%) |
| Mean routing entropy | 5.945189 | 5.945129 | -0.000060 |
| Mean expert-count CV | 0.097451 | 0.098131 | +0.000680 |

The corrected Paloma macro losses were `2.22016478` for the current continuation and `2.22088742` for the preferred continuation. The runs used current router arithmetic for both checkpoints and applied each checkpoint's pending query-bias state. This common policy removes evaluation-time router rounding as a direct difference. It also evaluates the preferred-trained checkpoint under arithmetic different from its training policy, so the sign is not a clean estimate of long-run preferred-policy quality.

### Durable artifacts

- Final report: `s3://marin-us-east-02a/marin/users/romain/hero-training-router-01a0c066/evidence/hero-router-training-report.md`, SHA-256 `454aae5894a78c99752cff47301441c9a8d315b30d5d1ddf1410c069bba73f4c`
- Review response: `s3://marin-us-east-02a/marin/users/romain/hero-training-router-01a0c066/evidence/goal-review-response.md`, SHA-256 `a5bfff56802f87c14b6be057f4cbd9e6f86d298906071f28cbe77f532217a3d9`
- Exact fixture: `s3://marin-us-east-02a/marin/users/romain/hero-training-router-01a0c066/evidence/router-precision-fixture-all.npz`, SHA-256 `5ecf0fcedb325c31783c4e15c1f1a2e33a3395acd8b1439d5b52405e3d9ff659`
- B200 report and compiled forward/backward HLO: `s3://marin-us-east-02a/marin/users/romain/hero-training-router-01a0c066/evidence/router-shape-b200-a7-with-hlo.tar.zst`, SHA-256 `ae527a37127a356756adc508f212c546439c9138e77923f039fbda5d79929b6b`
- Probe provenance and exact invocation: `s3://marin-us-east-02a/marin/users/romain/hero-training-router-01a0c066/evidence/router-precision-provenance.json`, SHA-256 `e52dda974eca1b688658620c30a2562ab439fd39406c63e527673aebc72e4260`
- Expert-load summary: `s3://marin-us-east-02a/marin/users/romain/hero-training-router-01a0c066/evidence/expert-load-summary.json`, SHA-256 `859d66b149b4616ad3ac4de29542fc3dda5e979ed5d16a7cbd5cdbceabba068b`
- Corrected fixed-evaluation metrics: `s3://marin-us-east-02a/marin/users/romain/hero-training-router-01a0c066/evidence/fixed-eval-with-pending-qb-a2.json`, SHA-256 `d13af487ee6b7f54644b9dcb5e3d7a37fdcc6e17420b7cf7476652877f8e9182`

### Reproduction

Run from [`a00cb77a491f4777a2c66edce54f47ff7b255c40`](https://github.com/yonromai/marin/commit/a00cb77a491f4777a2c66edce54f47ff7b255c40). Download the exact fixture above as `router_precision_fixture_all.npz`, then run the compiled B200 probe:

```sh
python -m experiments.grug.moe_hero_ep.router_shape_probe \
  --fixture router_precision_fixture_all.npz --rows 65536 --dump-hlo
```

The full-model payloads used this Iris wrapper with unique job names and JAX ports:

```sh
marin-env uv run --frozen --python 3.12 iris --config lib/iris/config/marin.yaml job run \
  --no-wait --enable-extra-resources --target-cluster cw-us-east-08a \
  --priority interactive --cpu 2 --memory 8GB --disk 32GB --timeout 43200 \
  --job-name <job-name> -e IRIS_PORT_JAX <port> \
  -e XLA_PYTHON_CLIENT_MEM_FRACTION 0.75 \
  -e XLA_FLAGS '--xla_gpu_memory_limit_slop_factor=85' -- <payload>
```

Immediate comparison payload:

```sh
python -m experiments.grug.moe_hero_ep.launch_diagnostics \
  --run-id hero-router-pref-compare-step126k-01a0c066-a2 \
  --dp-racks 1 --batch-size 1024 --optimizer-batch-size 11264 \
  --gate-router-weight-decay 0.02 --num-steps 126001 --schedule-steps 390251 \
  --initialize-from-checkpoint s3://marin-us-east-02a/marin/grug/hero-main-step121638/2026.08.19.2/checkpoints/step-126000 \
  --router-dot-precision preferred_fp32 --router-compare-current \
  --training-data mixture --version dev --run
```

For the paired continuation, run once with `POLICY=current`, `RUN=current`, `CHECKPOINT=current-a1`, then with `POLICY=preferred_fp32`, `RUN=pref`, `CHECKPOINT=preferred-fp32-a1`:

```sh
python -m experiments.grug.moe_hero_ep.launch_diagnostics \
  --run-id hero-router-${RUN}-cont-step126k-01a0c066-a1 \
  --dp-racks 1 --batch-size 1024 --optimizer-batch-size 11264 \
  --gate-router-weight-decay 0.02 --num-steps 126020 --schedule-steps 390251 \
  --initialize-from-checkpoint s3://marin-us-east-02a/marin/grug/hero-main-step121638/2026.08.19.2/checkpoints/step-126000 \
  --router-dot-precision ${POLICY} --training-data mixture --save-checkpoints \
  --checkpoint-minutes 9999 \
  --checkpoint-path s3://marin-us-east-02a/marin/users/romain/hero-training-router-01a0c066/checkpoints/${CHECKPOINT} \
  --version dev --run
```

Focused validation at the final commit:

```sh
OPENBLAS_NUM_THREADS=1 nice -n 10 uv run --frozen --python 3.12 \
  pytest -n 0 --tb=short -q \
  tests/test_grug_hero_scaling_ladder.py tests/test_moe_hero_ep.py
```

Result: 71 passed, with one pre-existing JAX scatter future warning. `./infra/pre-commit.py --changed-files --fix` also passed.

</details>



# C026 2026-09-22T22:32:44Z Trangle
URL: https://github.com/marin-community/marin/issues/8435#issuecomment-5785318684
您好，邮件已收到。


# C027 2026-10-01T00:40:47Z mcwitt-agent
URL: https://github.com/marin-community/marin/issues/8435#issuecomment-5922369256
🤖 One-rack measurement ahead of the planned 4K to 8K switch, restoring the step-180000 checkpoint at the hero's per-rack load of 4.19M tokens per step, with 3 seeds × 100 steps per arm: 8K trains at 1.6% fewer tokens/s than 4K, and the fraction of expert assignments dropped rises 4.5×, from 1.9e-4 to 8.3e-4, all on the sender side. 16K costs 3.1% of tokens/s and 16× the drops. Setup, W&B runs, and caveats are in #9615. The main caveats: one-rack drop fractions at 4K are about 2× the 11-rack hero's, and each run covers only 100 steps after the switch.

