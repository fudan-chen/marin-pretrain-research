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


## V79：active前缀不是假设，须由裁剪和传输offset一起保证

V78的portable专家只用sum(active_group_sizes)定义有效前缀，因此布局必须先把所有有效行连续放在前面。本轮取得同一固定PR head的ep_common.py，执行原_prefix_cap_counts、_clip_receiver_group_sizes、_expert_granular_a2a_params与_chunk_plans，使用真实JAX CPU数组计算元数据，再按原offset/size做host拷贝重放。三组人工控制、10项检查通过，[原参数向量与返回身份](analysis/receiver_layout_cpu.json)完整保留。没有执行all_gather、ragged_all_to_all、GPU或真实routing planner。

### 四种坐标如何接上

sender输入按全局expert排序，未裁剪的各group占用原起点。接收端裁剪按每个receiver独立进行，优先较早expert，再优先该expert的较早sender。dispatch使用**未裁剪起点、裁剪后的长度**读取各group前缀；receiver output offsets按expert-major、sender-major连续压实到前部。这样接受的行不需要先在sender压缩，也不会在receiver中间留下padding孔洞。

原_chunk_plans汇总每个本地expert的active数，再按chunk截取；physical_group_sizes只有最后一项增加chunk_capacity−sum(active)，其余项与active相同。因此padding只落在该chunk的最后一个expert段末尾。V78按总active数选择前缀，依赖的正是这个条件，不能将它推广到“每个expert段各自夹带padding”的其他布局。

return参数反向读取接收端压实区域，把有效前缀写回各sender原来的未裁剪起点。host重放逐项核对dispatch/return的send/recv size相互一致、写入无重叠、有效区恰为[0,total_active)、expert归属顺序正确；所有被接受的人工身份返回原位置，被丢弃位置保持未写NaN。这验证原元数据在本地模拟拷贝中的一致性，不证明真实collective发包、设备读写或非有限buffer安全。

### 总容量等于总需求，为什么仍丢5条

人工计数矩阵有两个sender、八个expert；每个receiver四个本地expert，分两chunk，每chunk两expert。sender0需求为[4,1,3,0,2,4,0,1]，sender1为[2,3,0,2,4,1,3,2]，总需求32。手动给每个receiver/chunk逻辑容量8、物理buffer10，四块总逻辑容量也是32，却只接受27。

<div id="receiver-layout-placeholder"></div>

[图原文件](assets/receiver_layout.svg)：需求10/5/11/6分别落在四个receiver/chunk，接受8/5/8/6；拥挤块丢2和3，另一侧剩余3和2不能跨块借用。灰色是未用逻辑容量，不是物理padding；每块物理buffer10又有单独的静态尾部。本图是人工需求下的原计划结果，不是Hero实际负载。

|控制|逻辑/物理容量，每块|总接受/丢弃|按sender接受量|
|---|---|---|---|
|低容量原顺序|3 / 5|12 / 20|8 / 4|
|较高容量原顺序|8 / 10|27 / 5|13 / 14|
|低容量、chunk内expert重编号|3 / 5|12 / 20|6 / 6|

本例逻辑与物理容量都直接给定，没有执行真实capacity_factor、minimum/maximum或动态chunk容量计划。它把本章此前的静态“局部容量不能借用”说明接到了原裁剪、offset与返回身份，不能据此推荐生产直接设成某个容量数值。

### 总drop相同，接受身份仍会改变

低容量原顺序中，按原语义expert0到7计，接受量为[3,0,3,0,3,0,3,0]。在每个chunk内部交换expert顺序，并将需求及人工身份同时对应重编号，需求、容量、接受总数12和丢弃总数20都不变；按原语义身份还原后，接受量变成[0,3,1,2,0,3,0,3]，sender接受份额也从8/4变成6/6。

这是前缀优先政策的可复现结果，不是随机drop，也不是等概率分配。expert编号/映射在无裁剪数学计算中可做对应重排，但裁剪规则显式依赖先后位置；不能只凭无drop公式判断编号重排、chunk划分或transport改造保持了实际接受语义。本轮没有执行完整模型去比较两种编号下的最终输出或训练收益。

### 数据配比研究要增加哪层记录

数据来源不能从expert ID直接推断。若要判断某种文档、语言或质量桶是否更受容量丢弃影响，需要把实际样本身份、selected assignments、sender/receiver/chunk归属、接受mask和最终loss位置连接起来。全局drop rate相同不足以证明各来源受到同样影响；反过来，这个人工编号反例也不能证明Hero存在域偏置。

对自己的改动，先绑定原group起点与接受长度，再验receiver压实、active/physical的最后padding关系、return镜像和有效身份；随后按来源及位置统计实际接受率，再做固定评估。提高容量、换chunk或换sender分布会同时影响HBM、通信和接受政策，只有整个路径的性能/稳定性与模型评估都完成，才能讨论配比收益。不要直接将assignment丢弃数从语言目标token分母扣除。

新增一份固定head helper源码，旧478份非bookkeeping来源保持原字节，当前来源480份。实际通信、生产接受mask、域偏置及GPU仍待验证。复现：`make receiver-layout-cpu CPU_PYTHON=/tmp/marin-jax-cpu-072/bin/python`。


## V80：被裁掉的专家，为什么仍能收到 router 梯度

V79验证了哪些assignment能进入专家计算。本节再沿权重路径追一步：QBRoutedMoE先对被选中的logit做sigmoid，再将它们归一化到routing_renorm_sum；容量接受mask稍后才把未接受权重置零。portable combine没有在裁剪后重新归一化。固定版本为b65be4c9550c5097f0a3add08933531a1c24d534，[原route](https://github.com/marin-community/marin/blob/b65be4c9550c5097f0a3add08933531a1c24d534/lib/levanter/src/levanter/grug/grug_moe.py)与[原mask及combine](https://github.com/marin-community/marin/blob/b65be4c9550c5097f0a3add08933531a1c24d534/lib/levanter/src/levanter/grug/_moe/ep_ragged_all_to_all.py)分别对应归档中的current_grug_moe.py与current_ep_ragged_all_to_all.py。这是该PR head的代码行为，不代表它已部署到Hero。

### 先把两个梯度分开

令已选专家的sigmoid分数为s_i，归一化常数为c，接受mask为a_i，专家输出为y_i。忽略dtype舍入的实数写法是：w_i=c·s_i/(Σ_selected s_j+ε)，out=Σ_selected a_i·w_i·y_i。分母仍包括被选中但未被接受的专家。

因此，“a_i=0，所以这个assignment的权重梯度为零”是对的；“所以这个logit的梯度也为零”却不成立。改变该logit会改变分母，从而改变其他已接受专家的权重。若损失对输出的cotangent为d，某个被丢弃但仍已选中的logit z_j，在本路径中的导数为−c·s_j(1−s_j)·Σ_i a_i s_i〈d,y_i〉/(Σ_selected s_i+ε)²。符号取决于cotangent与接受专家输出的内积，不是天然在惩罚拥挤专家。这里对固定选择、固定接受mask求导；没有对top-k或容量离散选择求导，也没有执行quantile-bias更新。

### 原源码 CPU 控制给出的数值

执行原moe_route block、原weights=jnp.where(accepted,combine_weights_local,0)语句和原portable combine，排序适配为JAX索引、reshard为identity，禁用Sonic分支。logit为[2,1,0]、top-2选中[0,1]，c=2.5，合成专家输出为3与1；只对输出之和求梯度，未执行语言损失、专家MLP、容量planner、通信或优化器。15项检查通过，[完整数值与替代依赖](analysis/post_clip_router_cpu.json)可复查。

两个专家全接受时，权重为[1.366123,1.133877]，输出5.232245。只接受第一个时，保留质量1.366123，输出4.098368；原权重梯度为[3,0]，router logit梯度却为[0.221577,−0.499913,0]。第二个assignment没有贡献专家输出，其已选logit仍通过归一化影响第一个。只接受第二个时，保留质量1.133877、输出1.133877，router梯度为[−0.073859,0.166638,0]。两种情况都丢掉一半assignment，却不是同样的输出扰动。

<div id="post-clip-router-placeholder"></div>

[图原文件](assets/post_clip_router.svg)：左图是保留权重之和，右图是合成输出对logit的梯度，不是训练loss或参数更新。全部丢弃时，这个routed输出和本探针router梯度都为零；完整模型仍可能有其他路径及辅助项，不能据此说整个token没有学习。将被丢弃专家输出改成NaN，四个控制的输出与router梯度均与有限值版本完全一致，这是portable mask路径的CPU结果，不覆盖真实未写设备buffer或Sonic kernel。

### “丢弃后补归一化”会改变什么

另设一个明确不同的比较函数：在原mask后，将剩余权重重新缩放到2.5。只接受第一个时，输出变成7.5，两个已选logit的本路径梯度均为零。原因是单个剩余专家的权重固定为2.5，与两者的分数无关。这个比较说明，补归一化同时改变前向幅度与router训练信号；它不是数值等价优化，本报告没有将其称为上游修复，也没有证明哪一种更有利于训练。

### 对配比、日志和变更验收的影响

按领域统计接受率之外，还应记录裁剪前权重总和、裁剪后保留权重总和、每token被接受专家数以及全丢弃比例。assignment drop rate相同，保留的权重质量仍可能不同；权重质量相同也不能保证输出或梯度相同，因为专家输出及cotangent不同。低精度下还需记录权重dtype、零值率、极小分数和epsilon主导样本；不应把2.5视为每个token都能精确达到的恒等式。

这些量是定位线索，不能直接作为配比奖励或把被丢assignment从目标token分母扣掉。正确的下一步是绑定样本/领域身份，比较相同checkpoint和相同样本下接受mask、保留质量、输出差异、router及专家梯度；再验证GPU实现和独立固定评估。真实Hero的这些记录仍缺失，本控制没有测得实际域偏置或配比收益。

验收规则：区分“assignment权重梯度”“已选logit梯度”“专家参数梯度”和“总模型梯度”；容量/分块/重编号改变后，同时比较接受身份和权重质量。若提出裁剪后归一化，应作为训练函数变更单独评审，不以drop减少或有限梯度替代模型收益。复现：`make post-clip-router-cpu CPU_PYTHON=/tmp/marin-jax-cpu-072/bin/python`。
