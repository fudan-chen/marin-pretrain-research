# 源码深读：训练状态、精度与切换时刻

上一章问“loss 的计算接口是否一致”，这一章问“下一步究竟从什么状态开始”。两者直接影响数据配比实验：相同的权重文件、step 和配方名称，不保证下一步更新相同。

源码仍固定为 `84869ae8c91ffe64e9f761c5bd714542eb1876e0`。新增七份来源覆盖 callback、checkpoint 测试、Hero 测试与推理导出。训练入口和模型沿用已有归档，不重复抓取。28 项本地探针见 [原函数控制流探针](scripts/probe_state.py)与[完整原值](analysis/train_state_probe.json)。它执行原训练步及恢复布局辅助函数，但替换 JIT、模型、梯度、optimizer、树操作、内存转移和存储探测；不能当作 JAX、Adam、GPU 或真实恢复的复现。

## 1. 哪份参数具有最终权威

六个公开 ladder 配置都选择 `FP32_PINNED_HOST`，device 参数策略为 BF16，optimizer 状态也 offload 到 host，EMA 关闭。[六份配置](analysis/train_state_probe.json)

固定源码在这一路径中先生成 FP32 master，再产生 BF16 副本，optimizer 从 master 初始化。每步将 optimizer 状态和 master 取到 device，用计算副本求梯度，将梯度转为 FP32后更新 master，再把 master cast 成 device 参数，最后 offload。这里“参数存在 host”与“参数精度为 FP32”是两个属性；迁移位置不会自动恢复被舍弃的精度。[初始化与训练步](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/experiments/grug/moe_hero_ep/train.py#L702)

“先更新 BF16 参数，再 cast 回 FP32”不能替代“在 FP32 master 上累积更新”。在 1 附近，BF16 相邻数间距比某些单步更新大，若每步都舍入，更新会消失；master 则保留小更新，直到累计变化足以反映在计算副本上。

探针采用确定的有限 FP32→BF16 最近偶数舍入模拟，梯度取 BF16 数值 0.01，SGD 步长 0.1，从权重 1 开始运行十步：

| 算术路径 | 十步后权威权重 | 十步后计算权重 | 证据范围 |
|---|---:|---:|---|
| 原训练步的 host master 分支 | 0.989989996 | 0.98828125 | 原控制流+替代依赖 |
| device 参数为 FP32 | 0.989989996 | 0.989989996 | 原控制流+替代依赖 |
| 每次更新后舍入到 BF16 的反例 | 1 | 1 | 人工构造的算术反例 |

第三行不是六个 ladder 的实际行为，也不是该源码推荐的 DEVICE 配置。DEVICE 分支本身使用 `mp.cast_to_param` 的结果，其精度要查参数策略，不能只凭枚举名称认为已经保证 FP32。[上游 master 累积测试](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/tests/test_moe_hero_ep.py#L1312)另有 JAX 测试；本机没有执行它。

**落地判断：**切换硬件、offload 或参数模式之前，先保存并核对 master 的初始化、更新和恢复证据。若小更新在参数上无法体现，增加高质量数据或调抽样比例并不能修复算术路径。

## 2. checkpoint 里的 params 不一定包含下一步将使用的路由偏置

Hero 的训练状态包含六个字段：`step`、`params`、`master_params`、`opt_state`、`ema_params`、`pending_qb_betas`。`pending_qb_betas` 不是冗余日志。每步开始把上一批估计的 beta 转成 router bias，本批计算又估计一份新 beta，保存为下一步待应用值。[训练状态](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/experiments/grug/moe_hero_ep/train.py#L671)

模型的 `apply_qb_betas` 将 beta 取负，并减去每层均值，然后**替换** router bias，不是累加。router 用加入 bias 的 logits 选择专家，用未加入 bias 的 logits 计算所选专家的组合权重。因此不能把它解释成“给专家输出乘一组质量权重”。[偏置应用](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/experiments/grug/moe_hero_ep/model.py#L1463)、[选择与组合](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/experiments/grug/moe_hero_ep/model.py#L999)

<svg viewBox="0 0 780 170" role="img" aria-label="本批使用上一批beta，本批估计的新beta保存到下一批才应用" style="width:100%;height:auto"><rect width="780" height="170" fill="#f3f6fa"/><g fill="#172b45" font-size="17" text-anchor="middle"><text x="120" y="45">上一步 pending beta</text><text x="120" y="74">转成当前 router bias</text><text x="390" y="45">本批前向、梯度、更新</text><text x="390" y="74">本批估计新的 beta</text><text x="665" y="45">保存新 pending beta</text><text x="665" y="74">下一步开始才应用</text><text x="390" y="140">checkpoint 同时保存参数与尚未应用的路由状态</text></g><g stroke="#547b9e" stroke-width="2"><path d="M230 55H275M510 55H550"/></g></svg>

原函数探针用 beta `[1,3]` 得到 centered bias `[1,-1]`，给 beta 全体加常数后得到相同 bias。两步原训练控制流确认：第一步结束时 params 保留本步已使用的 bias，新 beta 位于 pending；第二步求梯度前才应用它。完整模拟状态继续，结果与未中断两步相同；只保留参数和 step、把 pending 清零，则下一步 bias 不同。这个测试检验控制流，没有恢复真实 checkpoint。

用两个专家的近似打平 logits `[0.1,0.2]` 展示影响：未应用新 bias 时 top-1 为专家 1，应用 `[1,-1]` 后为专家 0。这是两专家 top-1 人工示例，不是 Hero 多专家 top-k 的实测翻转率，也没有测量任务影响。

**对数据顺序的意义：**切换配比后的首批，路由状态带着上一批的估计。这是训练算法状态的延续，不应在比较中任意清零。如果两组使用不同 pending 状态，再说“只是换了喂数顺序”就不成立。短窗口还可能包含路由对新分布的适应；不能直接将首批波动当作稳定配方优劣。

## 3. 训练内评估与导出模型，应用 pending 的时刻不同

固定训练入口把 `state.params` 交给普通 callback，把 EMA 或 params 交给 EMA 评估 getter，没有在这些 getter 中应用新的 pending beta。dropless 转换替换专家计算 backend，并不会在那一步更新 bias。[callback 接口](sources/state_2026_10_05/state_adapter.py)、[训练 getter 与 dropless 转换](sources/scale_2026_10_05/train_hero_ep.py)

推理消费者 `restore_weights` 则读取权威 master（存在时）、读取 pending beta，并在返回模型前应用它。导出随后转成 BF16，再写推理权重。[恢复消费者](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/experiments/grug/moe_hero_ep/weights.py)、[导出入口](sources/state_2026_10_05/export_vllm.py)

这形成两种可辨认的模型视图：

| 消费者 | 读取的视图 | 新 pending beta |
|---|---|---|
| 下一次训练更新 | 当前参数；更新时使用权威 master | 求梯度前应用 |
| 本步训练 callback 评估 | 已存 params；EMA 启用时另看 EMA | getter 不额外应用 |
| 固定 checkpoint 推理/导出 | master 优先，另读 pending | 返回模型前应用 |

这不是仅凭静态阅读就能宣布的线上 bug。它说明“同一个 checkpoint”仍需声明使用什么视图。前向评估与导出比较应同时控制 pending、EMA、计算 dtype、容量丢弃和 backend，实测输出与任务差异后再决定是否需要统一策略。本章没有测量两种视图在 Hero 上的误差大小。

## 4. 恢复布局不能只按新 run 的字段读取

若 checkpoint 同时保存 BF16 `params` 和 FP32 `master_params`，新 run 不使用独立 master，而加载模板只列出 `params`，可能恢复到舍入副本。固定源码用 manifest 判断 master 是否存在，将新 run 的 params 模板临时放到 master 字段，从存储中读取权威 FP32参数，再移回 params。[布局转换](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/experiments/grug/moe_hero_ep/train.py#L162)

原辅助函数探针覆盖四种组合：相同布局沿用模板；有 master→device 的转换选择 master 字段；没有 master→host master 则明确拒绝，不擅自从舍入参数“补回” master。上游还测试旧 `train_state/` 包装中的 master 标记，以及真实数组布局的迁移；本机只执行假 manifest 状态下的控制流。[上游布局测试](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/tests/test_moe_hero_ep.py#L490)

恢复后的 dtype 要从实际读取结果检查，模板名字不是 dtype 验收。模板的 sharding、memory kind 同样需要核对；上一章展示的“最新可读候选回退”也仍然适用。

## 5. 日志 step 0不是这份实现的初始化 loss

`GrugTrainState.step` 表示已完成更新数，callback 的 `StepInfo.step` 是它减一，`next_step` 才等于已完成数。首次评估 hook 在第一步更新后运行，记录为日志 step 0。写 checkpoint 与恢复 loader 使用 state step。[StepInfo](sources/state_2026_10_05/callback_core.py)、[首次评估与主循环](sources/scale_2026_10_05/train_hero_ep.py)

| 已完成更新数 / state.step | callback 日志 step | loader 恢复起点 |
|---:|---:|---:|
| 1 | 0 | 1 |
| 393 | 392 | 393 |
| 394 | 393 | 394 |

探针直接提取并执行原 `StepInfo` 属性的 lambda，确认这些标签关系。它没有读取真实 batch。研究顺序实验时，要先说明论文、日志、checkpoint 和 weight schedule 各自使用哪种 step；再计算 token 曝光。不能因为图上有 step 0，就拿它做未经训练的共同初始化基线。

该状态 dataclass 没有独立 RNG 字段。训练入口从 seed 重新构造 model/data key；loader 根据 `state.step` 重建读取起点。因此通用验收表里的“核对随机状态”在这里应落实为 seed/key 的派生、data_seed、索引、store 与 cursor 等证据，不是要求找到一个并不存在的 RNG tensor，也不能由其不存在宣布恢复有缺陷。

## 6. 应怎样验收数据配比切换

建议把下面的检查接到已有配比实验契约里。可复制的[状态切换验收模板](templates/state_switch_acceptance.json)保留实际checkpoint、master、pending、optimizer、第一批token与评估视图字段，三项对照均标为未执行，所有门槛仍未知。任何空项都应保留为未知，不能由曲线好看来补齐。

| 检查 | 应保留的原值 | 失败时怎样解释 |
|---|---|---|
| 权威参数身份 | master/params 布局、dtype、内容摘要、实际候选 | 算术或恢复状态不同，暂不归因于数据 |
| 路由续训身份 | 已存 bias、pending beta、下一步生效 bias | 额外改变了路由状态 |
| optimizer 连续性 | moment、计数与学习率实际值；当前源码的optimizer结构 | 权重相同也不保证更新相同 |
| 数据切换身份 | state step、日志标签、第一批 ID、权重阶段和曝光账本 | 可能是游标或边界错位 |
| 评估视图身份 | pending 是否应用、EMA、dtype、backend、drop策略 | 测到的可能不是同一模型视图 |
| 不中断/中断对照 | 同一状态与数据下的一步、若干步差值 | 恢复等价仍未验证 |

本轮没有新的配方收益结果。得到的是更严格的解释条件：**同状态续训需要恢复“能决定下一步”的状态，而不是只恢复可用于推理的权重。** 要研究配比和顺序，必须把这两种恢复目的分开。下一轮应继续检查 attention mask、segment 边界与打包 token，因为即使状态完全相同，样本能看到的上下文不同，也会改变所优化的任务。
