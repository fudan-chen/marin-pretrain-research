# Router精度：把数值更准与训练策略更好分开

本章回到#8435的C025，核对其公开实验fork代码，而不只转述评论结论。固定fork为`yonromai/marin@a00cb77a…`；它是实验分支，评论明确没有PR、没有改live Hero设置或checkpoint。本轮读取三份fork源码，没有下载真实fixture、HLO或native checkpoint，没有运行GPU或复现作者测量。[源码审计](analysis/router_precision_audit.json)

## 1. 三种dot写法分别改了哪里

[实验模型](sources/router_precision_2026_10_05/model.py)为router提供三个显式分支：

| 分支 | 输入 | dot调用与输出处理 | 应核查的对象 |
|---|---|---|---|
| current | 当前compute dtype的activation和router权重 | einsum后再astype(float32) | dot原输出是否已经舍入 |
| preferred_fp32 | 保留低精度输入 | einsum指定preferred_element_type=float32 | 编译器实际dot精度与输出dtype |
| fp32_operands | 先把两侧输入转float32 | einsum设置Precision.HIGHEST | 更高精度输入的代价与编译实现 |

`astype(float32)`放在dot后面，只能扩展已有结果的表示，不能恢复已经丢失的尾数。`preferred_element_type`在dot调用内声明期望的累加/结果类型；最终GPU算术还需HLO与实际误差验证，不能只凭一个Python关键字宣布已复现。

一个独立人工算例把两个FP32 score `[1.001,1.002]`按BF16输出舍入，二者都变成1；随后转回FP32仍是`[1,1]`。NumPy argmax采用首个最大值，因而人工选择从第二项变成第一项。这只展示信息丢失和离散选择敏感性；它不是原router dot，不证明实际JAX tie-break、fixture误差或真实专家变化率。

Marin主仓固定版本84869ae8的router dot也是“einsum后astype(float32)”。代码注释中的“router path保持FP32”不能代替对dot内部累加和输出dtype的检查。[当前固定模型](sources/contracts_2026_10_05/model.py)

## 2. 选择、顺序与combine权重是不同诊断量

fork在相同当前bias下比较候选与current的top-K：直接比较ID数组得到order change；先对ID排序再比较得到set change。两者不能混称“换了多少专家”。

| 变化 | 可能改变什么 | 单看这一项漏掉什么 |
|---|---|---|
| 只改变top-K顺序，集合未变 | 分配排列与执行路径；理想加权和对同一组配对排列不敏感 | 真实transport的接受/裁剪、舍入与索引配对仍需核查 |
| selected set改变 | 实际专家函数、梯度和下一步状态 | 不能由集合差异比例推算质量收益或损失 |
| 集合/顺序未变但score改变 | sigmoid combine、归一权重、概率与beta统计 | “路由ID相同”不等于整个更新相同 |

候选还会影响K+1阈值alpha、HIST网格和pending beta，因此一批次的细小数值差异可以进入下一批状态。评分误差更小、选择更接近高精度参照，说明算术更准确；它们没有直接证明原先训练出的模型应该中途改用这种策略。[路由分位数](QB_ESTIMATION_ZH.md)与[接受路径](ROUTING_DROPS_ZH.md)

## 3. 一rack诊断如何保留长期schedule

[launch_diagnostics](sources/router_precision_2026_10_05/launch_diagnostics.py)区分三项：实际训练batch、用于optimizer启发式配置的batch、完整schedule长度。短续训的停止step不等于重新从零跑一条短schedule。

C025复现命令实际batch=1024，optimizer_batch_size=11264，schedule_steps=390251，从126000续到126020。源码使用optimizer_batch_size和完整schedule_steps构造optimizer参数；停止位置由num_steps设置。这样可以避免把短实验误配置成一条微型warmup/decay，也避免单rack batch直接触发另一套峰值LR启发式。

但optimizer启发式使用生产batch，并不让真实梯度batch也变成生产batch。一次更新仍只看到一rack样本，HIST统计范围和跨rack归并条件仍不同。这个控制是必要的局部对齐，不是生产等价证明。[optimizer分组与时钟](OPTIMIZER_GROUPS_ZH.md)

在自己的短诊断中，应同时归档这三个数、解析后的有效LR/衰减和恢复count。否则“同checkpoint续20步”可能同时改变schedule和优化参数，无法把差异只归给router算术。

## 4. 现有评估究竟回答哪个问题

评论中的纠正评估应用各自pending beta，并固定使用current算术评价两份已续训checkpoint。它减少了评估算术本身的直接差异，但preferred-trained checkpoint被放到了不同于其训练的policy下。

下面的四格是待设计的交叉评估，不是已执行结果：

| 训练出来的checkpoint | current评估policy | preferred评估policy |
|---|---|---|
| current训练 | A：旧策略的匹配评估 | B：只切评估算术 |
| preferred训练 | C：切换后训练状态在旧policy下的评估 | D：新策略的匹配评估 |

已有评论主要提供A与C。A对B、C对D可以诊断各checkpoint的评估policy敏感性；A对C是在共同旧policy下的训练状态比较；A对D是完整策略比较，同时改变训练和评估policy，不能称为纯训练算术效应。四格都必须绑定同一评估数据、状态视图和分母。

fresh-run消融还需要另做共同初始化的新训练，而不是用四格评估替代。中途切换和从头采用新算术回答不同问题：旧权重、moments和beta已经适应旧策略。生产保持现状可以是有限证据下合理的决定，并不等于证明新算术没有价值。

作者观察、完整数字和限制已在[C025中文解读](ISSUE_8435_ZH.md)保留，本章不把20步有限结果外推为长期优劣。

## 5. 下一项最有区分力的检查

按问题选择下一步，不把所有实验同时执行：

| 问题 | 最小新增证据 | 所需资源与决定 |
|---|---|---|
| 当前源码是否真的发生低精度输出舍入 | 实际dtype、编译HLO、共同保存输入误差 | 对应GPU/runtime；确认算术机制，暂不判断能力 |
| checkpoint是否仅对评估policy敏感 | 两checkpoint×两policy，正确pending state | 可访问checkpoint和评估算力；完成四格诊断 |
| 生产中途切换是否可接受 | 共同完整状态、生产相关batch/mesh、稳定性、预算和独立评估 | 训练资源；按完整切换干预确认 |
| 从头采用新算术是否更好 | 共同初始化、固定预算、冻结评估与重复 | 新训练资源；回答fresh-run策略问题 |

本轮源码审计与人工舍入算例只能为这些检查提供实施依据，不代填结果。它接入[loss诊断](LOSS_TRIAGE_ZH.md)的数值与更新分支；真实GPU/HLO、fixture、训练与四格评估字段均保持未知。
