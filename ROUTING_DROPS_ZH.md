# MoE丢弃：分配数量、权重质量与训练目标是三件事

本章补足主报告中“drop不是整条token删除”的源码依据，进一步解释为什么相同drop率可以有不同影响。归档六份容量/路由文件，沿用固定版本的model与train入口；11项原辅助函数检查见[结果](analysis/routing_probe.json)。没有运行GPU collective、模型梯度或真实批次，不重新归因历史曲线。

## 1. 分母先核对版本，再核对有效性

两个已经归档的训练版本使用不同分母：

| 源码版本 | moe/drop_fraction分母 | padding如何处理 |
|---|---|---|
| 12d8b6f0…，原[code_train.py](sources/code_train.py) | batch×sequence_length×top_k×layers | 这个函数未接收valid/skipped计数 |
| 84869ae8…，[train_hero_ep.py](sources/scale_2026_10_05/train_hero_ep.py) | 实际valid assignment总数 | 单列skipped，检查valid+skipped等于全部位置分配 |

人工输入只有四个位置，K=8，其中两个位置无效，丢弃四项有效分配。旧函数返回4/32=12.5%，新函数返回4/16=25%。丢弃数量没有变化，变化来自分母。这个例子含人工padding，不代表历史Hero曲线包含同样输入或已经发生这种跳变；真实执行SHA与输入尚未绑定。

这里的valid是用于路由的query有效性，不是loss_weight>0。一个能进入模型、却不计分的最后位置或prompt位置，仍可能生成专家分配。因此路由分配数、计分target数、名义token数应分别记录。[文档边界](DOCUMENT_BOUNDARIES_ZH.md)

原函数还检查total dropped=sender+receiver，并在host端以int64累加每层计数。数据守恒可以抓住接口错位，但不能证明路由策略最优或真实能力提高。

## 2. 同样丢弃八项，到底伤及多少token

六份ladder配置均为每token选8个专家、2个shared experts，并开启capacity overflow报告。专家分配丢弃统计的是token→expert边，而非token本身。

下面是8个token×8项分配的人工算例，两个方案都丢弃8/64=12.5%分配。

<div id="drop-pattern-placeholder"></div>

| 人工布局 | 受影响token | 全部routed分配丢弃的token | 原combine函数得到的routed输出 |
|---|---:|---:|---|
| 集中丢弃第一个token的8项 | 1/8 | 1/8 | 第一个0，其余2.5 |
| 每个token各丢弃1项 | 8/8 | 0/8 | 全部2.1875 |

每个存活专家输出在算例中人为设为1，原combine权重均为2.5/8。随后把一维输出人为当成二分类正确类logit，得到平均NLL约0.155672与0.106337。两者平均routed输出相同，但损伤分布与人工NLL不同。这只是展示非线性目标的反例，不是Hero的loss、专家输出或质量测量。

一般地，N个有效token、每个K项分配，共丢弃D项时，受影响token数在ceil(D/K)至min(D,N)之间；全部routed分配丢弃的token数在max(0,D−N(K−1))至floor(D/K)之间。这个边界只谈一层的计数；不能从48层汇总率反推出同一token跨层被丢弃多少次。探针对全部64种3×2人工布局检查了边界。

## 3. 丢弃权重有多大，也不能靠数量推断

[Hero路由入口](sources/contracts_2026_10_05/model.py)在带bias的logits上选择专家，但用未加bias的logits计算sigmoid combine权重，然后在dispatch前将K项权重归一到总和2.5。

[ragged combine](sources/routing_2026_10_05/ragged.py)按原权重求和；未返回的分配输出保持零，没有按剩余专家再次归一化。因此丢弃低权重与高权重分配，即使数量相同，失去的combine权重质量也不同。上述人工例均匀权重时，每个分配失去0.3125；真实权重不均匀时，计数更无法替代权重质量。

应另记录每token丢弃数、全部routed丢弃数、丢弃combine权重和及其分位数。记录权重质量仍不等于测出了输出损失：专家输出方向和大小也不同，梯度影响还需要独立检查。

不能用“returned专家输出等于零”直接判定分配被丢弃，合法专家输出也可能为零。应从实际接受/返回索引与capacity信息还原mask，再关联原selected_experts和combine_weights。

## 4. token没有自动退出语言loss

[Block调用](sources/contracts_2026_10_05/model.py)在routed输出之外加入shared专家输出，可能经过ShortConv，最后加回残差。即使某位置全部routed专家缺失，也不等于整层输出、该token表示或梯度全部为零。

`next_token_loss`接收原loss_weight，模型输出隐藏表示后计算语言目标；这条路径没有根据capacity drop自动把该位置的loss权重置零。受损表示仍可能参与预测与优化。不能把12.5%的assignment drop翻译成“训练数据少了12.5%”，也不能把计分target数减去assignment数量当成有效训练量。

这解释了为何通信实现改变drop后，同一批次loss也可能改变：模型实际走过的计算路径不同。是否提高了最终能力，仍需固定评估与长期训练结果。[历史同批对照与回退](REPORT_ZH.md)

## 5. capacity factor不是不会overflow的保证

[动态容量辅助函数](sources/routing_2026_10_05/common.py)依据assignment需求与capacity_factor计算logical capacity，并夹在minimum与静态physical capacity之间。ragged又把容量分到本地expert chunks；某个receiver或chunk的需求过大，其他位置有空闲也不保证可以借用。

[receiver裁剪](sources/routing_2026_10_05/ep_common.py)对分配计数按prefix截断。该路径优先较早expert，再优先该expert的较早sender，并非随机丢弃。人工两个sender、四个expert、每receiver容量4的输入，原函数得到：

```text
输入计数       接受计数
[4,0,0,4]     [4,0,0,0]
[0,4,4,0]     [0,0,4,0]
```

总共只接受8/16项；每个receiver未超容量，但expert选择的保留分布不均匀。这个例子执行原计数裁剪，没有执行all-to-all或真实专家计算，也不是本次训练实际矩阵。

排查时先看sender→receiver→chunk→expert的需求、接受量和剩余容量，再决定提高capacity、改变chunk或换transport。统一capacity_factor=1.15只说明系数相同；不同EP规模、波次或capacity层次仍可能导致不同接受路径。

## 6. 字段名与零值也需要证据范围

当前ragged路径把接收端per-chunk裁剪造成的丢弃汇入sender_dropped，receiver_dropped返回零。因此sender不是实际故障位置标签；receiver=0也不证明接收端没有容量压力。[实现](sources/routing_2026_10_05/ragged.py)

Hero的`report_capacity_overflow=False`分支给若干router摘要提供零计数，占位零不能证明真实drop为零。六份ladder配置开启了报告，但后续run仍需独立检查配置和执行路径，不能把本轮六份配置推广到所有日志。

routing_counts由dispatch前selected experts统计，表示有效token原本选择了谁，不是所有实际执行成功的专家负载。诊断前置路由均衡与诊断capacity裁剪后的接受负载，需要分别使用对应计数。

## 7. 怎样把drop检查接到数据配比

新配比可能改变token内容、序列内相关性与专家需求。平均drop下降可能只是把需求移到容量充裕的expert，而不是数据本身更有价值；平均drop相同也可能掩盖少量来源受到集中的损伤。

先在共同checkpoint上冻结token批次、token_valid、selected experts和combine权重，对不同transport/capacity路径做诊断；这一步用于隔离执行接受路径，不替代训练确认。再按来源、质量桶、文档长度与序列位置统计接受分布，观察是否与目标任务退步对应。样本身份与质量桶语义须先核对，不能从专家ID直接推断领域。

| 检查 | 保存什么 | 能支持什么 |
|---|---|---|
| 分母与报告状态 | 执行版本、valid/skipped/count守恒、是否开启报告 | 解释日志口径，拒绝把零占位当结果 |
| drop分布 | 每token/每层丢弃数、全丢弃比例、combine权重质量 | 区分集中与分散损伤，不能直接证明能力变化 |
| 接受路径 | sender/receiver/chunk/expert计数与索引映射 | 定位容量压力，不按字段名字认定根因 |
| 训练语义 | 同输入/state的输出、梯度、目标与数值容差 | 判断改动是否改变计算，不拿吞吐代填 |
| 最终决定 | 固定预算稳定性、独立评估、能力退步与回退记录 | 接受完整变更；拆开对照后才讨论单项收益 |

这些检查接到[训练变更评审](CHANGE_REVIEW_ZH.md)的“样本、更新、评估与预算”，不另增一套总分。原探针只支持计数和combine辅助接口；collective、自动微分、真实输入drop图与历史代码绑定均没有结果。
