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


## V57：怎样读逐项优化表，而不把条件收益当独立收益

[新的性能图SVG](assets/moe_performance_attribution.svg)与[PNG](assets/moe_performance_attribution.png)来自既有PR快照：18个表格条目中12个有秒/步测量，其余6个没有独立步时，不能当作0收益样本。图A保留有测量的累计代码序列；图B复算端点；图C是机制示意，不是device trace。[原值与6项核对](analysis/moe_performance_attribution.json)

13.899→12.591秒/步减少1.308秒，时间下降9.411%；速度相对提高10.388%。两个百分比的分母不同。MFU从28.24%到31.18%是增加2.94个百分点，也不能直接叫“提升2.94%”。这些都是作者单rack表格的算术，不是本地GPU测量、统计置信区间或独立组件归因。

|比较|结果或缺口|允许的判断|
|---|---|---|
|逐项加入共享专家SwiGLU|12.638→12.652秒/步，多0.014秒|它在该前序代码上的条件效果为负；不能推出在所有组合中应删除|
|从最终tip移除同类改动|作者两组记录慢0.35%与0.22%|效果依赖最终组合；原始成对step trace未归档，不代填本地独立确认|
|共享计算节省约0.18秒、通信暴露约0.24秒|作者的scope解释，与总步时属于不同层次|并发scope的变化不能直接加减成完整步时，不强行让0.24−0.18等于0.014|
|表格最后一行Gain|作者另用相邻的新跑12.610秒作基线|上一表格行是12.611；来源已解释这一差异，不把小差异指认为数据错误|
|移除/保留小于噪声尺度的改动|作者按最终tip做移除对照与重复pairs|保留判据是该实验政策；本报告没有替它给出通用0.2%阈值|

源码变化为什么会产生这种依赖？#9833保存routed output后，重计算不再包含原先那批down GEMM、return all-to-all与combine操作，通信的先后关系随之改变；#9708的prefetch又与carry-offload共享memcpy stream。一个kernel时间缩短，既改变它本身的工作量，也改变旁边通信和拷贝可以被覆盖多久。profile中某scope总量减少，不保证完整训练步的关键路径同比缩短。

作者还报告shared GEMM的加速让通信暴露，并对小收益在最终tip做移除检查。这里可以观察到“加入时为负、最终移除却变慢”的条件反转；不能只靠它求出每个组件的独立效应或完整交互项，因为缺少共同base上的所有组合及原始trace。顺序表每一行的程序已经不同，把Gain相加会把这些条件关系隐藏起来。

如果两项改动能独立编译且数值合同兼容，可以固定checkpoint/批次/环境，分别跑base、仅A、仅B、A+B。用秒/步定义交互量`I=T_AB−T_A−T_B+T_base`；它只是这个共同实验下的额外时间变化，不是无条件可迁移的组件特征。若某些组合在源码合同上不可运行，就明确留空，改做最终图上的移除对照与profile；不能为了凑四臂而绕过有效性检查。

[性能交互记录模板](templates/performance_interaction_review.json)要求保存硬件/mesh、执行与lowering身份、wheel/flags、checkpoint和输入、预热及profile排除窗口、配对顺序、原始step trace、时间与HBM/通信指标。模板没有执行任何实验，所有测量为空。作者声称最终重构前后StableHLO一致，但本报告未重编译验证，不能把源码摘要或同一PR名称当作lowering一致。

对于配比干预，还需冻结这些系统条件：专家负载和drop变动可能改变计算/通信覆盖，使wall time收益与学习收益同时变化。应同时报告固定token预算下的任务结果、固定wall time下的有效学习进展、drop/路由与可恢复进度；不能把MFU更高自动翻译成每个数据桶更有学习价值。

因此当前能得出的结论是：这组候选在作者单rack测量中整包更快，部分优化存在明显条件依赖；独立组件归因、长窗稳定性、跨rack扩展与实际部署仍未验证。复算与绘图入口为`make moe-performance`，只读冻结来源，不访问GPU或更新PR状态。


## V58：局部梯度误差会不会真正传到参数

#9833作者以正的归一化sigmoid权重和bf16较宽的指数范围解释Hero选择EXPERT_SIDE的依据。本轮检查其数值边界。来源仍是[冻结diff](sources/engineering_current_2026_10_07/pull_9833_files.json)及[作者说明](https://github.com/marin-community/marin/pull/9833)，不是新训练事故。

本轮执行48个人工标量控制，跨float16、bf16、float32，各取4个权重指数与4个cotangent指数。专家输出人为设为y=3，row-dot生产者是替身；只有division/mask两条语句来自原patch。先区分输入cast成0与两个输入均非零、乘积却成0。float16的16格中12格已有输入cast成0，不能拿它们证明乘法额外丢信息。每种dtype各有1格在两个正输入可表示的情况下丢失乘积。这是选定CPU网格的存在性反例，不能解释为训练发生率，也不能外推GPU的subnormal处理。

用float32 sigmoid对人工logits[-50,0]归一化，小权重约3.8575e-22，在bf16中仍非零；cotangent取2^-70，约8.4703e-22，也非零。CPU乘积变0，EXPERT_SIDE重建dS为0，标量float32参考为2.5411e-21。bf16没有无限指数范围，“权重正”只排除了代数零除，不能保证乘积保留信息。

**这个局部反例不能直接证明router更新错了。** 把参考与候选dS送进同一个float32归一化sigmoid VJP，极端bf16例的两条logit梯度都为0：进一步乘上小权重后的参考梯度也在本CPU上丢失。局部dS差异未传到该人工router的最终梯度。另一组logits[-7,0]、cotangent=2^-20、float16乘法控制中，参考logit梯度非零而候选为0，差异确实穿过人工router。两例一起看，才知道误差在何处被衰减或抹掉。未执行真实top-k、完整路由实现、ragged专家或优化器更新。

### 数值验收怎么做

沿实际反向链分别查accepted分配上的乘积、dS、router logits/参数梯度、相同参数与优化器状态下的一步更新。随后才讨论固定评估loss与长窗口稳定性。局部差异是定位线索，更新差异更接近训练行为，但都不能单独量化最终能力损失。

按dtype、权重/cotangent大小及accepted/drop/padding分桶，保留绝对误差和有明确分母下限的相对误差。整体中位数可能掩盖尾部；跨dtype也不能只比ulp。输入cast为0、正输入乘积为0、非有限输入应分别统计。不要用同一种低精度乘法当未舍入参考，可离线用更高精度或指数记录查乘积范围。监控与真实GPU验收在本轮均未实现。

[10项CPU检查与全部数值](analysis/routing_gradient_envelope_cpu.json)、[未执行验收模板](templates/routing_gradient_acceptance.json)保留原语句、输入、来源SHA与未知字段。运行`make routing-envelope CPU_PYTHON=/tmp/marin-jax-cpu-072/bin/python`可复算。图不是生产失败概率；缺Hero输入分布、完整GPU反向、同状态更新和loss反事实，不能据此要求回滚。


## V59：把简化梯度实验接回原router的权重路径

V58的sigmoid VJP是人工参考，不能当作Marin整个router执行。本轮从固定84869ae8版本的[QBRoutedMoE](sources/routing_2026_10_05/grug_moe.py)用AST提取原moe_route block，执行原dot、top-k、gather、sigmoid、rounding barrier、归一化和cast。reshard替换为identity，partition spec为占位。K=2、目标2.5，一行一维activation与三专家router，输入全是人工构造。没有统计、QB更新、dispatch、ragged专家、真实mesh、checkpoint或优化器。[9项CPU检查及源码SHA](analysis/router_weight_path_cpu.json)

### 选择分数和权重分数不同

原代码在router_logits加stop_gradient(router_bias)后的biased_logits上取K+1项；最后一项形成alpha，其余K个ID用于选择专家。combine权重却从未加bias的router_logits按ID gather，再做sigmoid。bias改变选择，不直接进入权重的sigmoid。

人工logits[-50,0,-1]、bias[100,0,-100]时，选中ID[0,1]、alpha=-101。第0专家因bias被选中，但bf16权重仍约9.6615e-22；第1为2.5。“进top-k”不保证权重远离零。K+1阈值在biased域，也不能与unbiased权重大小混作一量。

人工专家输出[3,1]只用于加权和梯度检查：选中router参数梯度非零，未选中参数为0，bias梯度为0。这个结论限于加权和路径；QB bias另有状态更新，统计也可能另有用途，不能写成完整训练不更新bias或未选中专家永远无任何梯度。

### 2.5是目标，不是所有输入的严格守恒量

原实现使用float32 sigmoid，加optimization_barrier保留舍入边界，再乘2.5/(sum(sigmoid)+1e-9)，最后cast到x.dtype。省略epsilon、barrier、2.5或最终cast的参考，不能自动证明整个实现的合同。barrier存在已核对，具体GPU编译与性能作用未测。

设选中sigmoid总和为S，cast前总权重为2.5*S/(S+1e-9)。S远大于epsilon才接近2.5；S小时，总权重随S缩小。人工logits[-50,-51,-52]取前两项，最终bf16权重约[4.8317e-13,1.7764e-13]，总和6.6080e-13。每项sigmoid虽为正，却没有恢复为2.5。epsilon避免全零时除零，不能同时保证任意输入保持目标总权重。

更极端的有限logits[-100,-101,-102]在本CPU原sigmoid路径得到全零权重。数学上的严格为正，不等于浮点实现输出必定非零。GPU/backend的subnormal和指数计算路径仍需独立验证。普通[2,1,0]例得到bf16权重[1.3671875,1.1328125]、总和恰为2.5；不能外推所有cast后的总和都严格相等。四例eager/JIT输出一致，仅覆盖这四例。

### 对V58反例的正确使用

V58回答某种正权重/cotangent是否丢失乘积，以及人工sigmoid VJP是否保留差异，没有建模这里确认的完整权重路径。要诊断#9833的训练影响，须在同执行版本下将原router输出交给实际EXACT/EXPERT_SIDE，再查router参数梯度与同状态更新。不能拼接不同时期源码后宣称复现Hero。

下一项生产证据应记录每层选中unbiased logits、S/epsilon、cast后权重零值/幅值分布、accepted/drop，以及同输入下两种反向的参数梯度。仅看entropy、计数或平均权重不能排除少数由bias选中的极小权重。当前没有Hero分布，不能认定实际失稳、无效训练或应修改epsilon。

运行make router-weight-path CPU_PYTHON=/tmp/marin-jax-cpu-072/bin/python可复算。这里把验收条件落实到选择、归一化和cast，并未新增未经证实的生产bug。
