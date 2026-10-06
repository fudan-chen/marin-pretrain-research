# 配比切换后的loss变化：怎样逐步排查

假设从配比A切到B后，训练loss立刻下降，吞吐也提高。现在最需要回答的不是“B好多少”，而是下一步怎样区分三种情况：输入变容易、计算路径改变、固定评估上的能力改善。本章把已有源码发现合到同一条诊断流程，不新增训练结果。

贯穿流程的原则是：先验证曲线能否比较，再定位变化，最后做能力确认。诊断实验可以隔离机制，但其人工输入或短窗结果不能代替完整续训。

## 1. 先写清楚这次变化

用一行记录：什么时间、哪个run、从哪个实际状态、改了什么、首先出现哪个异常。

例如下面这句是合格的问题定义，而非结果：

> 在共同checkpoint上保留完整optimizer和pending路由状态，把代码桶从A改到B；检查固定程序执行评估是否改善，同时保留数学退步限制。

如果实际换了store、tokenizer、去重、质量评分、batch、optimizer或通信实现，就列出来。这样的运行仍可用于完整配方比较；不能在解释时删掉这些共同变化，称为单独配比收益。

## 2. 第一站：曲线是否表示同一个量

| 先问什么 | 从哪里拿证据 | 不通过时怎样处理 |
|---|---|---|
| loss到底含哪些项 | 执行版本、next_token_loss调用、实际loss_weight | 分开目标口径；不要把不同loss直接相减 |
| 分母是否相同 | scored targets、有效query、名义token、byte计数 | 用对应分母重算；保留原日志 |
| 比较的是同一模型视图吗 | stored/EMA/master、pending beta是否已应用、dtype/backend | 对齐同一视图重新评估 |
| batch/step标签是否对应同一进度 | state.step、下一loader index、实际token计数 | 以共同进度比较，单列恢复与重复样本 |

Hero的训练CE可能含输出logit z-loss，而router z-loss是日志项；前向相同也不保证反向相同。小数loss_weight的行为不能只用0/1 mask证明。[目标与梯度接口](IMPLEMENTATION_CONTRACTS_ZH.md)

BPB的聚合与评估分批也需要核查。某些不同分母下的平均会改变比较结果；不能把这种差异解释为某领域学习更快。[BPB口径](SCALE_TRANSFER_ZH.md)

本轮已经查到专用衰减字段默认零，所以“旧字段缺失、新字段0”先保持为声明差异，不再作为已知optimizer行为差异。更大的learning rate、beta2等差异仍须独立处理。[有效optimizer设置](OPTIMIZER_GROUPS_ZH.md)

## 3. 第二站：到底送进了什么数据

先核对内容身份，再解释领域比例。至少保存store/缓存产物摘要、tokenizer revision、处理政策与实际样本映射；只有bucket名称与配置权重不够。

下面这些变化都会影响train loss，却不能单独证明能力提高：

| 表象 | 一个可能的机制 | 最小区分证据 |
|---|---|---|
| train loss整体更低 | 易预测来源占比提高 | 同checkpoint分别评估A/B输入分布；另做固定确认评估 |
| 某质量桶效果突然更好 | 质量评分观察窗口、词表或校准规则改变 | 原文、实际评分产物、桶边界和评分模型身份 |
| 有效token或吞吐改变 | EOS/BOS、padding、文档拼接或loss mask变化 | 实际token序列、segment、query有效性与scored target mask |
| 曝光账本更“合理” | 去重豁免或稀疏属性缺失改变保留集合 | 实际样本ID、去重覆盖率、缺失状态和排除顺序 |
| 切换后来源比例偏离配置 | 整数块、边缘块与resume cursor改变实际计数 | 原加载器逐桶计数、next index、共同边缘输入 |

同checkpoint的交叉分布诊断只能说明输入分布差异，不能说明训练后的B模型更好。它是排除“换了更容易的题”这类解释的辅助材料，能力结论仍来自固定评估。

对应入口：[缓存身份](CACHE_PROVENANCE_ZH.md)、[评分](QUALITY_BUCKETS_ZH.md)、[去重](DEDUP_FILTERS_ZH.md)、[边界](DOCUMENT_BOUNDARIES_ZH.md)、[ShortConv与halo](SHORT_CONV_ZH.md)、[顺序账本](ORDER_GUIDE_ZH.md)。

## 4. 第三站：执行与更新是否改变

若同时改了kernel、transport或恢复方式，先在共同输入与完整起点上比较一步。这个起点包括compute/master、optimizer、pending beta、EMA以及实际输入key/loader位置。

| 诊断分支 | 需要比较 | 怎样解释差异 |
|---|---|---|
| 路由选择 | 本步bias、selected experts、alpha、margin和新的beta | 数据可通过显式路由状态改变下一步选择；不能只查梯度 |
| 容量接受 | 接受索引、每token/每层drop、丢弃权重、receiver/chunk需求 | 平均assignment drop不能替代受影响token或表示损伤 |
| 数值路径 | 前向、梯度、最终update、master差异与预声明容差 | loss相等只排除部分前向差异，不能批准反向语义 |
| optimizer分组 | 实际路径/shape/组/mask、有效LR与衰减、各状态count | 参数名或状态时钟变化可能改变更新规则 |
| 恢复接续 | 未中断与恢复后的共同下一输入、多步状态轨迹 | 读回成功只是恢复操作完成，不代表更新轨迹相同 |

HIST的共享margin范围会影响网格精度；TOPK的局部阈值平均具有分组敏感性。六份ladder配置声明HIST，不应把TOPK反例套到它们。配比改变路由统计是完整干预的一部分；冻结路由状态的短诊断用于隔离机制，需另外记录。[分位数与状态时序](QB_ESTIMATION_ZH.md)

丢弃专家分配后，routed combine未必重归一化，token也不会自动退出语言loss；更低的平均drop不能代填能力确认。[容量与训练目标](ROUTING_DROPS_ZH.md)

不要一边定位问题一边同时调整capacity、bins、weight decay和配比。先保留可以复查的起点与一次声明清楚的变化。若必须一起改才能恢复，记录为复合缓解，再用拆开的诊断查机制。

## 5. 怎样决定下一步

| 现有证据 | 当前允许的判断 | 下一步 |
|---|---|---|
| 指标或内容身份不一致 | 曲线不能用于这项归因 | 回到口径/样本核查；保留整体观测 |
| 共同输入下语义差异超过容差 | 这不是已确认的语义保持改动 | 定位并修复，或明确接受为新算法干预后重新设计 |
| 故障恢复、短窗通过 | 缓解在已测触发条件下有效 | 长窗稳定性、成本与语义验收 |
| 固定评估方向好，候选仍用于搜索 | 值得保留的候选 | 冻结候选与风险阈值，做独立确认 |
| 独立评估改善但关键任务退步 | 有取舍的配方 | 检查供体设计或任务目标，不用总均值盖过退步 |
| 独立确认、实际预算与风险限制均满足 | 在该规模与条件下可接受 | 放大验证和可回退切换，避免直接搬比例 |

这是一条决策路径，不是打分表；某一项“通过”不能抵消另一项缺失。它接到七阶段管线：身份与口径属于P0，候选属于P1，干预/状态属于P2，能力确认属于P3，顺序属于P4，放大属于P5，失败解释与下一问题属于P6。

## 6. 让每次诊断留下可交接的记录

复制[诊断记录](templates/loss_change_diagnosis.json)，先保存原问题和观察，按本章三站填证据。缺失字段保持null，未检查项保持not_checked。输出决定保持pending_evidence，直到记录中的证据真的支持改变它；填满表格不证明训练收益。

这份记录补充[训练变更记录](templates/training_change_review.json)，不会自动批准部署。它保留可能解释和推翻条件，避免把一个暂时说得通的机制写成已确定根因。


V61在归因之前增加一项检查：事件记录映射到实际评估时刻后，各改动列是否可区分？当前配比/执行两个标记完全相同。矩阵审计不替代反事实，但能提前阻止无依据的分项回归结论，见[MIX_TRAJECTORY](MIX_TRAJECTORY_ZH.md)。


## V72：预测不变，训练 loss 仍能变化多少

V71 分开配置权重、域样本与有效目标。这轮执行原 Transformer.next_token_loss 方法、原 Grug mean reducer、原 CPU reference CE，以及原 API 的输出 logsumexp 罚项语句。固定 hidden/provider 与 head，人工两域各一条序列；7 项检查通过，[输出值与梯度](analysis/loss_composition_cpu.json)可复查。没有执行 Transformer 前向或真实数据。

|控制，预测参数保持固定|有效目标权重质量 A/B|原路径纯 CE|
|---|---|---:|
|A 1个有效目标，B 3个|1 / 3|2.2985873|
|A 3个有效目标，B 1个|3 / 1|0.7985873|
|后一输入的 A 权重乘0.5|1.5 / 1|1.2485874|

本例所有位置的 logits 为 [2,-1]，A 的标签为0、B为1，故每域纯 CE 分别为0.04858735与3.04858735。把有效目标占比换向后，loss 下降1.5，没有任何参数更新。两个域的序列数始终各1，名义样本占比都是一半；目标质量与样本数量却不同。小数权重的分母是 sum(weight)，本例2.5，不是4个非零位置。将全部正权重统一乘7，mean 与 head 梯度在1e−6容差内保持；这只验证有限正分母的 CPU 小例，不能推广到下溢、全零分母或 GPU kernel。

### cross_entropy_loss 字段可能包含输出 z-loss

对 A3/B1 输入，纯 CE 为0.7985873；输出 logsumexp_weight=1e−4 时，原方法返回0.7990069，train/cross_entropy_loss 字段也等于这个含罚项的值。给两项 logits 都加10，softmax 概率不变，纯 CE 为0.7985878，与原值相差不足2e−6；含罚项的目标却变为0.8131046。head 梯度也随输出罚项改变。

为什么？原 API 先执行 `loss = loss + logsumexp_weight * (lse**2)`，再加权归约；原 model 方法把返回值直接赋给 cross_entropy_loss 及同名指标。输出 z-loss 限制 logit 的绝对尺度，所以并不对共同平移不变。router z-loss 是另一条日志项，本例 provider 把它设为零；不能用 router 指标替代输出罚项分解。这里实际执行了含罚项的原模型 loss 方法，但 backend dispatcher 明确被 CPU reference delegate 替代，未复现 xla_fast_bwd/GPU 数值。

### 配比决策应查看什么

训练平均 loss 至少受有效目标组成、loss_weight 与输出罚项影响。低 loss 可能来自更多易预测目标，也可能来自 logit 尺度约束；这些变化各有含义，不能仅凭训练曲线下降判断配比改善了模型能力。应同时保留域内纯 NLL、有效目标质量、输出 z-loss、明确固定分母的评估和能力保底项；按同一评估分布比较，而不是用新训练分布自己的 mean 判断收益。

这不意味着混合训练 loss 没有价值：它适合监测当前目标和异常。若要分析原因，可先冻结一批 logits/labels，重放旧新 mask/weight 来量化计算口径差，再在固定输入上比较实际模型变化。两项作用可能交互，需要标明基准预测与基准权重，不把其中一个分解叫成唯一因果贡献。

本轮只给出人工反例与接口事实，Hero 实际 per-domain targets、输出罚项贡献和配比效果仍未知。损失方法的 hidden/router 提供器、router summarizer、named_call 与 mesh 依赖均有显式替代；原 reference、原罚项语句和原归约实际执行，未修改上游。复现：`make loss-composition CPU_PYTHON=/tmp/marin-jax-cpu-072/bin/python`。
