# 源码深读：相同配置为什么不等于相同训练

本章沿着 Hero EP 的训练、评估、融合交叉熵和恢复入口往下读。源码固定在 `84869ae8c91ffe64e9f761c5bd714542eb1876e0`，与上一章的公开配置审计使用同一版本。它能证明这个版本实现了什么，不能证明每个历史 run 都执行了它。上一章没有恢复出六个实验的实际训练代码 SHA；这项缺口仍然存在。本章不把源码注释中的硬件结果当成本机复现。

新增六份原始文件见 [来源清单与抓取时间](sources/source_manifest.json)，可执行探针见 [probe_contracts.py](scripts/probe_contracts.py)，46 项检查和所有替代依赖见 [结果 JSON](analysis/implementation_contract_probe.json)。探针只抽取原函数体：数组运算替换为 NumPy；环境变量、checkpoint 扫描、加载和屏障使用明确的假对象。没有执行模型、自动微分、训练内核或真实 checkpoint 恢复。

## 1. Loss 相同，梯度仍可能不同

Hero 的 `Transformer.next_token_loss` 显式选择 `xla_fast_bwd`。该实现保留前向，替换反向中的扫描、one-hot 和矩阵计算方式。源码注释报告，特定 GB200 形状上的梯度与 FP32 参考的相对 RMS 误差下降；这是作者在注释里报告的测量，不是本报告运行的实验，也不是所有形状的保证。[模型入口](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/experiments/grug/moe_hero_ep/model.py#L1382)

这里要区分三个东西：标量 loss、梯度向量和 optimizer 更新后的参数。前向相等只能检查第一个。浮点计算不满足实数运算的结合律，改变反向的累加顺序、分块或中间精度，就可能改变第二个，继而改变训练轨迹。尤其在 MoE 中，参数的微小变化还可能影响后续 token 的专家选择；这条放大链是需要实验验证的机制推断，不能凭源码断言已经发生。

<svg viewBox="0 0 760 170" role="img" aria-label="同一前向 loss 分成两条反向计算路径，梯度与后续轨迹需要独立验证" style="width:100%;height:auto"><rect width="760" height="170" fill="#f3f6fa"/><g fill="#172b45" font-size="17" text-anchor="middle"><text x="110" y="86">相同前向 loss</text><text x="385" y="42">旧反向累加路径</text><text x="385" y="133">新反向累加路径</text><text x="653" y="86">分别检查梯度</text><text x="653" y="111">与一步更新</text></g><g stroke="#547b9e" stroke-width="2" fill="none"><path d="M200 80L260 38H285M200 80L260 128H285M485 38H520L560 80M485 128H520L560 80"/></g></svg>

更容易被漏掉的是运行时覆盖：`LEVANTER_CE_XLA_FAST_BWD` 的优先级高于调用处的 `implementation`。调用处选择新反向，环境值 `0` 会强制旧反向；调用处选择旧反向，环境值 `1` 又会强制新反向。未设置时才使用调用处选择，库默认关闭新反向。拼错的值会报错。我们执行原始解析函数，覆盖了 18 个环境值与调用处组合；没有测量梯度误差。[覆盖逻辑](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/xla.py#L23)

上游测试包含前向逐位相等、梯度对照、环境覆盖和分块等价检查。这说明作者设置了相关验收项目；本机没有执行这些 JAX 测试，也不知道历史实验执行时它们是否通过。[上游测试](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py#L2478)

**对配比实验的影响：**更换配方的同时更换了反向实现，就同时改变了“喂什么”与“如何学习”。比较前应记录实际环境覆盖值、解析后的内核选择、分块、计算与参数 dtype，并从同一状态比较梯度和一步更新。不能用 loss 重合代替这些检查。

## 2. 不做 reduction，不代表还没有应用权重

最容易误读的是 `reduction="none"`。融合交叉熵 API 的 `_apply_reduction` 先执行 `loss * weight`，然后才判断要不要求和。所以返回的逐位置 loss 已经带权重。Grug 包装器保持这个结果；Hero 评估回调返回它，同时返回原始 `loss_weight`；`TaggedEvaluator` 又计算 `losses * weights`。这条调用链在小数权重下会出现平方权重。[API](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/api.py#L702)、[包装器](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/lib/levanter/src/levanter/grug/loss.py#L104)、[回调](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/experiments/grug/moe_hero_ep/train.py#L549)、[累计器](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/lib/levanter/src/levanter/eval.py#L550)

用三个位置说明，原始 NLL 为 `[2,6,100]`，权重为 `[1,0.5,0]`：

| 实际操作 | 分子 | 分母 | 平均值 |
|---|---:|---:|---:|
| 权重应用一次 | 2+3+0=5 | 1.5 | 3.333333 |
| 带权结果再乘原权重 | 2+1.5+0=3.5 | 1.5 | 2.333333 |

我们执行的是原 API 的 reduction 函数体，得到了第一行及带权逐位置结果，再用累计器公式计算第二行。这个例子证实了接口组合的风险；没有证实 Hero 的真实评估输入存在小数权重。若权重只取 0/1，`w²=w`，两种结果一致。这就是为什么全是普通 padding mask 的测试可能发现不了它。

**调整数据前先查清楚权重的含义。**抽样概率、token loss 权重和是否忽略 token 的 mask 是三个不同的接口。将质量分数直接写入 token loss 权重，会同时改变训练目标和这里的评估行为。质量分数也未必经过概率校准，不能看到“高质量”就直接当作数值权重。

修复方式取决于接口约定：或者回调返回未加权 NLL，由累计器统一加权；或者累计器接收已带权的分子与独立分母。不要在不知道其他调用方的情况下随手删掉一处乘法。验收输入必须包含 0/1、小数、全零 mask，并分别检查 CE 和 BPB；最后再用真实数据确认输入权重取值。

## 3. 指标叫 cross_entropy_loss，不保证它是纯 NLL

训练入口把 `z_loss_weight` 传到融合交叉熵，默认配置值为 `1e-4`；实际实验还要读各 run 配置。融合 API 把最终词表 logits 的 `logsumexp²` 惩罚加入逐位置 NLL，之后加权和 reduction。模型却把这个组合结果记录为 `train/cross_entropy_loss`。评估回调显式设置 `logsumexp_weight=None`，因此这条评估路径不包含该惩罚。[训练参数及调用](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/experiments/grug/moe_hero_ep/train.py#L278)、[融合目标](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/api.py#L969)

另一个名为 router z-loss 的量来自专家路由 logits。在这个 Hero 模型入口，它被记录为 `train/router/z_loss_logging_only`，没有加入返回的训练 loss。不能把“最终词表 z-loss 参与优化”与“路由 z-loss 仅监控”混为一谈。[模型中的监控与目标](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/experiments/grug/moe_hero_ep/model.py#L1421)

因此，报告中历史 `train/cross_entropy_loss` 数值应首先理解为原日志键，而不是无条件理解为纯 CE。没有绑定执行版本、系数和分项日志时，不能把某次变化完全归因于语言预测变好。训练与验证曲线之间的差也可能包含目标口径差异；本章没有重新归因既有历史曲线。

应记录 `nll`、`final_logit_z_penalty`、`objective_total` 三个分项，路由监控单独记录。把 logits 整体平移一个常数，softmax 概率和 NLL 不变，logsumexp 与 z 惩罚会改变；这是检验目标拆分的一个直接输入。实际梯度验收仍需要运行模型或融合内核。

## 4. 恢复到“最新”实际意味着最新可读候选

恢复函数根据 metadata 的 step 优先、timestamp 次优排序候选，再逐个尝试。遇到 `FileNotFoundError` 会继续更旧候选；其他异常不会由这一层笼统吞掉。加载成功后执行恢复屏障。因此填写一个搜索目录，不等于一定恢复了目录中 step 最大的那份。[恢复实现](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/experiments/grug/checkpointing.py#L52)

我们抽取原排序与恢复函数，用假扫描、加载和屏障测试了以下策略：

| 设置与输入 | 原函数行为 | 对实验的含义 |
|---|---|---|
| `load_checkpoint=False` | 不尝试加载 | 明确从初始化状态开始 |
| 可选恢复，无已发现 checkpoint | 返回初始状态 | 首次启动允许从头开始 |
| 强制恢复，但全部候选缺失 | 报错 | 不接受初始化代替恢复 |
| 可选恢复，已发现子目录 checkpoint 但全部不可读 | 报错 | 保护已有训练，避免误从 step 0 开始 |
| step 20 不可读、step 10 可读 | 恢复 step 10 | 恢复成功仍可能倒退 |
| step 10 时间较新，step 20 时间较旧 | 优先 step 20 | 文件时间不能代替训练进度 |

这份代码已经防范了“发现已有 checkpoint 却全部不可读，仍重新训练”的情况。不能把它描述成当前版本无条件静默重启。测试中的 I/O 都是假对象，没有验证对象存储一致性、多机同步或真实 tensor 完整性。

配比切换实验的验收对象应是**实际选中的候选路径、恢复 step 和完整状态内容**。只保存请求目录不足以证明两组从同一状态继续。若一组恢复较旧 step，数据游标和阶段切换可能随之变化；若允许部分恢复，还需检查哪些叶子沿用初始化值。没有 optimizer、RNG、EMA、router 更新状态和 master 参数的恢复证据，就不能称作严格的同状态续训。

## 5. 整数计数溢出也会制造“训练出问题”的假象

模型保留每层 int32 路由计数，由训练端在 host 上用 int64 汇总。源码解释这样做是为了避免大 batch 下跨层求和溢出，且 JAX 关闭 x64 时，直接请求设备 int64 不一定能得到期望的精度。[模型计数路径](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/experiments/grug/moe_hero_ep/model.py#L1431)

算术例子：batch=4096，序列=4096，top-k=8，每层最多对应 134,217,728 个 assignment，int32 可以容纳。48 层合计 6,442,450,944，int32 求和变成 −2,147,483,648，host int64 则保留正确总数。这里计算的是容量上界示例，不是实测丢弃数量。

若只看到负数或不满足“总丢弃=发送端丢弃+接收端丢弃”的日志，就贸然改容量、EP 或配比，会把计数问题当成模型问题。先检查每层值、总值 dtype、汇总发生位置和分母，再检查路由实现。这个版本采取的 per-layer 保留与 host 宽整数汇总，正是在修复监控语义。

## 6. 把这些发现变成训练前的验收规则

这些规则补充已有 18 条研究判断规则，不声称已经全部在 Hero 上实测通过。

| 规则 | 必须核对的对象 | 最小反例或验收 | 未通过时的判断 |
|---|---|---|---|
| C1 前向相等不代替更新等价 | 梯度、optimizer 更新、分块和精度 | 同一状态、同一 batch，对比梯度与一步参数变化 | 暂停将差异归因于配比 |
| C2 环境覆盖必须进入实验身份 | 实际环境值及解析后的内核 | unset、0、1、非法值 | 配置文件相同仍不足以配对 |
| C3 token 权重只应用约定次数 | callback 和 accumulator 两端 | 小数与全零 mask；CE/BPB 各自验收 | 先修口径，再看模型优劣 |
| C4 目标函数要按分项记录 | NLL、最终 logit 惩罚、路由监控 | logits 常数平移，核对分项变化 | 不解释训练/验证差值 |
| C5 恢复身份使用实际候选 | 实际路径、step、状态内容摘要 | 最新不可读、旧可读、全部不可读 | 标为回退恢复或失败恢复 |
| C6 计数先守恒，再诊断模型 | 每层与跨层 dtype、分子分母 | 大 batch 上界、sender/receiver 守恒 | 暂不改配比或容量 |
| C7 聚合必须对分批方式稳定 | 全局分子、分母与归一化 | 同一批样本换 eval batch size | 不根据可能依赖 batching 的排名选配方 |

C7 接续上一章 BPB 的 batching 反例。若实现 gradient accumulation，还应验证用各 microbatch 的有效 token 数归一化，而不是无条件平均它们的均值。例如一块 1 个有效 token、平均 loss 1，另一块 3 个、平均 loss 9，全局应为 7，简单平均是 5。本次读到的 Hero `_loss_and_grads` 和 `_make_train_step` 路径没有据此发现实际梯度累积 bug；这只是另一训练管线迁移时必须满足的条件。

建议按以下顺序执行，而不是先投入大规模配方搜索：

1. 固定源码与环境、实际 checkpoint、数据索引和 tokenizer 身份。
2. 验收逐位置 NLL、权重和有效 token/byte 分母，检查训练目标拆分。
3. 固定样本换分批、分片方式，验收累计口径；检查整数计数守恒。
4. 对候选内核做同状态梯度和一步更新检查，再看吞吐收益。
5. 验收恢复后的状态和第一批 token，确认切换时刻与曝光账本。
6. 上述检查通过后，从强基线开展配比与顺序实验，分别评估统计波动与任务退化。

**本轮结论：**数据优化的可信度，依赖于“输入、目标、更新、恢复、测量”五个接口都说清楚。源代码已经给出几处保护和验收设计，也暴露出需要确认适用输入的接口风险。下一步最有价值的证据是实际 run 的执行身份、真实 mask 取值、融合内核的梯度复现，以及恢复状态与第一批数据的联合核验；它们比再增加一张未经控制的 loss 对比图更能缩小原因范围。

## V49补充：全零mask的前向guard不能证明反向安全

本章原NumPy检查中的“全零mask返回0”只验证前向值。[真实JAX CPU审计](MASKED_NUMERICS_ZH.md)进一步运行原归约、reference CE与原backward scan函数体，得到前向0而梯度NaN的反例。原检查未被推翻，但其证明范围必须止于前向；默认公开配方的正分母条件也另行核对，尚未发现实际Hero全零分母事件。
