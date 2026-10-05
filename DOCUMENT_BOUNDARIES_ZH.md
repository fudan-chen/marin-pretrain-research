# 源码深读：文档边界、可见上下文与有效目标

同样的 token 配比，可能因为文档长度、EOS、打包和 padding 产生不同的有效训练任务。本章把三个问题分开：某个 query 能看到哪些 key，某个位置预测的目标是否参与 loss，某个位置是否进入专家路由。它们有联系，但不是同一张 mask。

源码固定于 `84869ae8c91ffe64e9f761c5bd714542eb1876e0`。新增七份源文件，原函数探针见 [probe_boundaries.py](scripts/probe_boundaries.py)，17项本地检查及逐位置矩阵见[原值 JSON](analysis/boundary_probe.json)。函数体沿用原代码，NumPy、函数式 `.at.set`、关联扫描和轻量对象替代 JAX/Equinox；本机没有执行 attention 内核、反向或真实 token cache。七份源码通过 Python 3.12 语法检查。

## 1. 禁止跨文档 attention，并没有取消跨文档目标

六个 ladder 公开配置均启用 `block_cross_document_attention`。`GrugLmExample.causal` 在给定 EOS 时，以“前一位置为 EOS”生成新 segment：EOS 属于旧文档，它后面的 token 才属于新文档。attention 限制 query 和 key 的 segment 相同，再叠加 causal/window 条件。[样本构造](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/lib/levanter/src/levanter/data/text/examples.py#L113)

可是 loss 使用当前位置预测下一位置。构造器没有因为下一 token 属于另一个 segment，就把当前位置的 loss 权重设为零。原函数探针给出如下输入，其中99只是人工指定的EOS编号，并非声称是Marin tokenizer的真实ID：

| 位置 | 输入token | segment | 预测目标 | loss权重 |
|---:|---:|---:|---:|---:|
| 0 | 11 | 0 | 99 / EOS | 1 |
| 1 | 99 / EOS | 0 | 22 / 新文档首token | 1 |
| 2 | 22 | 1 | 23 | 1 |
| 3 | 23 | 1 | 无真实后继 | 0 |

位置2看不到文档0，但位置1仍用旧文档上下文预测新文档首token。关掉跨文档限制后，attention可见性变化，loss权重仍是`[1,1,1,0]`。所以“独立文档attention”与“完全独立文档似然目标”不能互换。[mask实现](sources/boundaries_2026_10_05/attention_core.py)

这并不自动构成bug：连续文本流可以有意保留这种边界目标；若cache插入BOS、EOS或其他标记，目标含义又会变化。**当前没有读取真实cache边界，不能宣称Hero实际优化了哪一种跨文档首token目标。** 配置中开启隔离也不足以证明每条真实样本含有正确EOS或segment。

对配比的直接启示是，短文档比例变化可能改变边界目标的数量，即使名义token比例相同。若要比较连续流与完全独立文档目标，应把边界目标作为一个明确干预，并重新核对有效target数；不能同时改mask、配比和预算，却把收益全部归到质量档。

## 2. Loss有效性与路由有效性为何不同

给定右侧padding的segment `[0,0,1,1,-1,-1]`，构造器用“下一位置segment非负”屏蔽预测padding的loss，再屏蔽序列末位。探针得到loss权重`[1,1,1,0,0,0]`，而路由有效性为`[真,真,真,真,假,假]`。

第四个真实token虽然没有可计分的下一个目标，它仍是真实输入位置。零loss权重也可能表示prompt、被忽略的目标或其他训练约定，不能直接拿来删除路由输入。[路由有效性函数](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/lib/levanter/src/levanter/grug/attention/_core.py#L145)

左侧padding还有另一种边界。segment `[-1,-1,0,0,1,1,-1]`，未额外给定loss权重时，构造器得到`[0,1,1,1,1,0,0]`：第二个padding输入位置的后继是真实token，所以单凭“后继非padding”规则仍可能给它loss权重。实际上游权重可能已经屏蔽了该位置；这个例子仅检验构造器，不证明真实训练存在这一输入。

如果任务要求排除所有padding query的目标，应检查完整上游权重链，明确是否还需要query-valid约束。不要凭一个零值token判断padding——真实token ID也可能为零，源码使用负segment表达padding。

## 3. 左侧padding为什么能破坏tile跳过优化

FA4/CuTe路径把packed causal attention压缩成每个query的key下界与valid标记，而不是生成整张平方矩阵。某个query的下界来自文档起点；局部attention再取窗口下界的较大值。窗口W包含自身和前面W−1个位置。[下界计算](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/lib/levanter/src/levanter/grug/attention/_fa4_cute.py#L80)

问题出在tile级跳过：若tile首个query是左侧padding，给它“下界=序列长度”的哨兵，可能让tile后面的真实query也跳过所有key tile。固定源码为无效query传递后面有效query的下界，同时保留valid=false；尾部padding继续用哨兵。

| segment | −1 | −1 | 0 | 0 | 1 | 1 | −1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原函数下界 | 2 | 2 | 2 | 2 | 4 | 4 | 7 |
| kernel valid | 假 | 假 | 真 | 真 | 真 | 真 | 假 |

**较小下界没有把padding变成真实token。** 下界负责安全跳过计算，valid才决定位置有效性。我们执行原下界函数，复现上表，并在真实query/key区域与dense mask比较一致；没有运行GPUtile优化。[上游左padding测试](sources/boundaries_2026_10_05/test_fa4.py)还包含真实GPU测试，本机未执行。

## 4. segment编号必须满足表示法的假设

Dense mask按segment编号相等判断可见性；压缩下界按相邻编号变化识别连续文档起点。若在同一序列中复用已结束的ID，如`[0,0,1,0]`，dense语义允许最后一个query看到前面的0号位置，压缩下界则把最后一个0视为新连续段。原函数探针发现两个真实query/key位置不一致。

这是假输入反例，**没有证明Hero生成了复用ID**。它说明自定义打包器应保证每个文档对应一个连续、不会被后续文档复用的编号段，或显式重新编号。有效输入上的mask对照通过，不能覆盖打包器不变量之外的输入。

## 5. Prompt权重也使用“预测位置”坐标

`prompt_length=3`、序列长度5时，原构造器的loss权重为`[0,0,1,1,0]`。第一个completion token位于输入索引3，但预测它的是索引2，所以从prompt_length−1开始计分。直接按输入token位置制作权重，会错过第一个completion目标，或多计一个prompt目标。

布尔dense mask的全假query行可以表明padding；加性mask则不能无歧义地表达query有效性，原路由辅助函数将其视为全有效。因此自定义mask必须同时检查实际attention输出与token-valid接口，不要只看一个带−∞的矩阵。[构造与validity原值](analysis/boundary_probe.json)

## 6. 怎样把这些检查接到配比实验

| 验收项 | 必须保留的证据 | 为什么影响配比结论 |
|---|---|---|
| EOS/BOS与segment | tokenizer、真实cache边界片段、ID与编号生成 | 决定跨文档可见性及首token目标 |
| 三种有效性 | attention mask、loss权重、路由valid逐位置对照 | 避免把零loss当padding或把padding当真实输入 |
| 编号不变量 | 连续段、ID复用、q/kv对应、padding范围 | 压缩表示可能有不同适用范围 |
| 有效曝光 | 名义token、真实query、计分target、边界target | 同样token预算可能对应不同训练目标数 |
| 输入边界测试 | 左/右padding、全padding、短文档、窗口、prompt长度 | 常规无padding样本无法覆盖边缘错误 |
| backend一致性 | 有效输入上的dense/压缩mask及前向/梯度 | mask正确仍不等于内核计算正确 |

当前读取到的`TokenSeqDataset`用cache扁平长度除以seq_len取整，再按固定长度取切片；显式packing则走另一条带segment的路径。不要把连续流切片和文档打包混成一种数据处理。不同component的format/pack、尾部余数及已有权重还需逐项追踪。[dataset路径](sources/boundaries_2026_10_05/datasets.py)

本轮的结论是：**数据质量和配比之外，预处理还定义了上下文边界与计分目标。** 解释loss前，先确定它到底在评价哪种任务。下一步需要把公开component的format/pack配置、tokenizer特殊标记与真实缓存样本对应起来；本轮没有把人工边界反例写成生产故障，也没有新的配方收益结果。

## 补充：attention之外还要检查ShortConv

六份ladder启用K/attention输出/MLP输出三处ShortConv。模型将segment IDs传入卷积，但卷积自己执行tap两端ID比较；context shard还需要同时交换激活与segment halo。非连续ID重用、identity初始化遮蔽、分片梯度舍入分别见[ShortConv深读](SHORT_CONV_ZH.md)。不能把attention隔离验收替代整个模型的文档隔离验收。
