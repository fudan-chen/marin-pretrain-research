# 新MoE性能提案：省掉了什么，数值合同又改变了什么

10月7日重新读取完整公开API：#8435/#8506/#8870正文没有变化，评论仍为27/58/30条，没有新增、删除或正文修改。[当前核对](analysis/engineering_current_v56.json)。网页没有展开全部评论，因而本轮使用API完整页，并对issue.comments与实际条数逐项核对。旧事故翻译不需要补写一个不存在的新结论。

新的工程材料来自[#9708](https://github.com/marin-community/marin/pull/9708)拆出的[#9832](https://github.com/marin-community/marin/pull/9832)与[#9833](https://github.com/marin-community/marin/pull/9833)。本次API快照中三个PR都open且merged=false。它们是作者已测量的候选实现，不能称为生产已合并修复；非空merge_commit_sha也不能代替merged状态。两份子PR的全部文件列表分别为3和10项，没有漏下一页；各patch新增/删除行数也与API元数据一致。[19项独立源码审计](analysis/moe_proposal_source_audit.json)重算了issue比较与patch覆盖。完整修改覆盖不等于完整head依赖已可运行；本轮未构建PR checkout。[11份新增API文件来源](analysis/engineering_v56_acquisition.json)

## 1. 不填充传输缓冲区，需要证明每个消费者都不读未写行

#9832要消除的成本不是专家数学计算，而是大静态传输buffer上的额外zero fill、select和add。专家chunk写入互不重叠的行；原custom VJP按覆盖写处理这些更新，每层反向因此对[TK,H]大buffer做额外全量操作。

提案把“这一行只写一次、写前没有可微输入内容”的覆盖写，按添加写的转置处理，令输出cotangent继续传递。这里的条件很具体：被写入行上的output_init不能依赖需要求导的输入；后续chunk转发的operand也必须读取互不重叠的行。它不是一种可到处替换overwrite导数的技巧。

|动作与源码位置|减少或改变什么|成立条件与验收|
|---|---|---|
|传输buffer采用unwritten_buffer|省掉无人读取行的GPU初始化|所有consumer只读实际写入行；CPU fallback仍填充，不能用CPU成功代替GPU未初始化内存验收|
|buffer绑定loop-carried值|约束buffer在layer循环里的构造位置|避免无输入buffer被当成循环不变量提到循环外再复制；需核对lowering和内存计划|
|custom VJP改成add/forwarding结构|去掉覆盖写转置中的大buffer清零与累加|写入行与输入依赖、chunk范围满足合同；不能只比较前向|
|GPU token-major scatter|token行只读一次，写accepted的top-k槽位|接受槽位输出、所有输入/权重梯度一致；未接受槽位可以未写|
|portable ragged_dot入口和出口选择active行|非有限闲置行不进入有效输出或权重梯度|用选择屏蔽读入和cotangent；不能以0乘NaN代替|
|drop/padding槽位在combine与dispatch梯度中跳过|保证未写buffer不会被末端归约读取|强制出现drop和padding，再检查前向、drop计数、全部梯度及recompute|

作者新增GPU poison测试：buffer分别填0和NaN，强制drop/padding，检查所有结果有限且bitwise相同。测试还断言drop计数确实大于0，否则“没有差异”可能根本没碰到未写行。重计算必须随fill重新trace；旧trace闭包若仍保留之前的fill，会让测试看似覆盖NaN却没有覆盖recompute。这是测试设计的一部分，不是本报告已经运行该GPU测试。

[完整diff](sources/engineering_current_2026_10_07/pull_9832_files.json)里的专家接口也明确声明闲置行及其cotangent可为非有限值。性能优化因此扩大了允许输入的内部状态，consumer的active范围必须跟着成为接口合同。[既有非有限数值分析](MASKED_NUMERICS_ZH.md)已经说明乘零未必能隔离NaN；这里把它落实到真实待审MoE代码路径。

## 2. 把路由权重梯度搬到专家侧，代数等价还不够

EXACT模式保留专家输出y，计算combine权重w的梯度`dS=<dout,y>`。#9833新增EXPERT_SIDE：专家反向已有`dy=w×dout`和`dh=dy×W2ᵀ`，于是用`rowsum(dh×h)/w`重建同一个标量。反向返回每行一个float32 row-dot，而不是保留并返回完整y。这减少保存、重算和返回通信，同时改变保存的残差与remat图。

数学恒等式要求信息尚在。若w为0，或w×dout已在cotangent dtype中舍入为0，再除w无法找回丢失的权重梯度。diff把accepted且w非零的行定义为divisible，先选择安全分母，再选择最终梯度；drop/padding不给梯度。默认仍是EXACT，其他后端拒绝EXPERT_SIDE；候选Hero路径选EXPERT_SIDE，作者依据的是正sigmoid权重与bf16 cotangent，而非“所有浮点输入都严格等价”。

本报告执行diff中两条原除法/选择语句，row-dot由人工标量线性专家产生，y=3。它不是完整MoE复现。

|CPU控制|精确参考dS|原提案除法语句结果|结论范围|
|---|---:|---:|---|
|float32，w=0.25，dout=2|6|6|普通输入正向控制|
|已accepted，w=0，dout=2|6|0|zero-weight边界无法用除法恢复|
|float16，w=2⁻¹⁰，dout=2⁻²⁰|约2.8610e−6|0|正normal权重也可能丢失小cotangent乘积|
|相同小乘积，bf16|约2.8610e−6|约2.8610e−6|这个bf16控制保留信息，不证明所有bf16幅度都安全|

[7项CPU与源码核对](analysis/routing_gradient_proposal_cpu.json)保存原patch语句、输入、输出与源码SHA。作者的测试把EXPERT_SIDE边界分配从内部容差检查中排除，而EXACT额外检查这些边界；因此“所有测试通过”不能翻译成两种模式对所有输入等价。[完整diff](sources/engineering_current_2026_10_07/pull_9833_files.json)

## 3. 更少重计算为何可能更容易暴露通信与内存问题

省掉完整y后，候选carry-offload策略改为保存up projection之前的routed output。作者报告跨48层约18 GiB额外保存，峰值HBM增加约20 GiB；编译调度预算太紧会重新计算本来想保存的output，使优化收益消失。提案因此调整内存fraction/slop，并把ragged collective overlap limit收紧到1。作者记录多collective并发曾造成不同梯度或挂起；这不等于本报告已复现该故障，也不代表此前#8870的所有hang同属这个机制。

#9708的整体候选还包含prefetch、carry-offload copy与共享GEMM调度。共享专家计算变快，可能暴露原先被它掩盖的通信；一行单独测量为负，不能据此判断删掉后整个组合一定更快。应在相同最终图上做移除对照，检查compute/transport时间线，而不是把每行Gain相加。

作者整包测量在一个GB200 NVL72 rack的64 GPU上，从step180000、seed0运行100步，报告约13.90→12.59秒/步、MFU约28.24%→31.18%。这是单rack候选窗口；不代替11rack生产稳定性、自己保存后恢复或长程能力评估。100步loss差与重复baseline差处于相近量级，也不是已经完成统计等价检验。[作者测量条件](https://github.com/marin-community/marin/pull/9708)

## 4. 对自己的代码评审，先确定哪一层合同改变了

这两个提案适合提炼为三类评审动作：

1. **未写内存合同。** 列出每个buffer的write/read集合，证明分块互斥，覆盖drop/padding、前向/反向/recompute；以NaN poison检查真正读到的路径，并确认每种fill对应新的trace。
2. **梯度数值合同。** 保留EXACT基线，覆盖zero权重、normal但小乘积、dtype下溢和非有限inactive行；分别统计dS及其他梯度，记录允许的近似输入域。forward近似相同不能代替router梯度一致。
3. **调度与部署合同。** 固定checkpoint、批次、dtype、执行SHA、wheel、mesh和内存设置；在最终组合上做移除对照，随后另验长窗口稳定性、自己保存后恢复和跨rack行为。短窗提速与合并/部署是不同证据。

数据配比改变专家负载、drop与梯度幅度，可能改变这些合同被触发的频率；本轮没有测此关联。配比干预与kernel/gradient模式更换应分别记录，联合变更只能先评价整个组合。不能拿新kernel的效率收益去补写某个数据桶的独立学习收益。

当前建议是：继续跟踪提案，保留EXACT及旧buffer路径作对照；没有本地GPU/生产验收依据，不在报告中批准部署。原issue事故索引、旧曲线与翻译保持原日期。复现CPU语句控制可运行`make routing-proposal CPU_PYTHON=/你的环境/bin/python`，依赖使用[既有CPU环境](requirements-cpu-numerics.txt)。
