# 新MoE性能提案：省掉了什么，数值合同又改变了什么

V77最新复核：[完整API与固定head源码审计](analysis/engineering_current_v77.json)确认#9708已经closed但未合并，#9832/#9833仍open未合并。#9833新head仅修订注释和docstring，补明portable backend仍保存专家输出；两份文件去除文档后的计算AST相同。下面的V56性能窗口和CPU控制保留为历史证据，不能当作新的生产部署或GPU验收。


V56（10月7日较早快照）读取完整公开API：#8435/#8506/#8870正文没有变化，评论仍为27/58/30条，没有新增、删除或正文修改。[当前核对](analysis/engineering_current_v56.json)。网页没有展开全部评论，因而本轮使用API完整页，并对issue.comments与实际条数逐项核对。旧事故翻译不需要补写一个不存在的新结论。

新的工程材料来自[#9708](https://github.com/marin-community/marin/pull/9708)拆出的[#9832](https://github.com/marin-community/marin/pull/9832)与[#9833](https://github.com/marin-community/marin/pull/9833)。V56 API快照中三个PR都open且merged=false。它们是作者已测量的候选实现，不能称为生产已合并修复；非空merge_commit_sha也不能代替merged状态。两份子PR的全部文件列表分别为3和10项，没有漏下一页；各patch新增/删除行数也与API元数据一致。[19项独立源码审计](analysis/moe_proposal_source_audit.json)重算了issue比较与patch覆盖。完整修改覆盖不等于完整head依赖已可运行；本轮未构建PR checkout。[11份新增API文件来源](analysis/engineering_v56_acquisition.json)

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

EXACT模式保留专家输出y，计算combine权重w的梯度`dS=<dout,y>`。#9833新增EXPERT_SIDE：专家反向已有`dy=w×dout`和`dh=dy×W2ᵀ`，于是用`rowsum(dh×h)/w`重建同一个标量。反向返回每行一个float32 row-dot，减少返回通信。QuACK路径还省去保存完整y；portable ragged_dot路径仍在残差中保留y来计算row-dot。V77复核修正了此前把“省去保存y”泛化到所有backend的表述；省存储和省通信需要分别核对。

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

在QuACK路径省掉完整y后，候选carry-offload策略改为保存up projection之前的routed output。作者报告跨48层约18 GiB额外保存，峰值HBM增加约20 GiB；编译调度预算太紧会重新计算本来想保存的output，使优化收益消失。提案因此调整内存fraction/slop，并把ragged collective overlap limit收紧到1。作者记录多collective并发曾造成不同梯度或挂起；这不等于本报告已复现该故障，也不代表此前#8870的所有hang同属这个机制。

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


## V60：一处舍入误差如何经过归一化与优化器

本轮把三段固定源码接成CPU诊断：V59的原router block、#9833 portable分支原row-dot与division/mask语句、原gate/router衰减包装和真实Optax。router activation为bf16，诊断参数为float32；expert输出人为固定为3，实际MLP没有执行。reshard/spec仍为替身，leaf_key_paths用人工扁平路径代替。不同冻结来源的组合不代表同一个真实Hero执行版本。[10项检查、输入与原值](analysis/router_coupling_update_cpu.json)

### 没有下溢，也会影响多个选中专家

人工router参数[-4,0,-10]选中专家顺序[1,0]，bf16权重[2.40625,0.0869140625]。两专家人工输出相同，cotangent为bf16的0.001，即0.00099945068359375。两个weighted cotangent都大于0，本例没有乘积下溢。

参考dS为[0.00299072265625,0.00299072265625]。原portable row-dot语句先在float32中对人工专家输出与已经舍入的weighted cotangent相乘并归约；再执行原division语句，送回bf16权重VJP。候选变为[0.0030059814453125,0.00299072265625]：只一个assignment相差一个bf16步距。

参考router参数梯度在此CPU控制中为[0,0,0]，候选为[-1.2556e-6,6.3935e-7,0]。没有局部dS误差的另一个选中专家，也出现了参数梯度变化。原因是两个权重共享归一化分母，Jacobian不是对角阵。令s_i=sigmoid(z_i)、D=sum(s)+epsilon，cast前w_i=C*s_i/D，则加权和对z_j的梯度为C*s'_j*(dS_j*D-sum(dS_i*s_i))/D²。改变一个dS会改变公共和项。epsilon不可无条件忽略，最终cast的数值也需沿原路径验证；这个公式用于解释耦合，不代替CPU/GPU实现。

因此按assignment检查局部误差，还不足以解释router参数误差。应保存同token的整组dS及归一化权重，观察误差的共同部分和差分部分。相近的局部误差可以因共同项而抵消，也可以因不均匀误差改变相对路由压力。不能把“中位数约1 ulp”直接翻译为更新差异同样很小。

### 同样的梯度差异，更新尺度取决于moments

原优化器源码将标准.router路径放入Adam组，不能因为router为矩阵而改用AdamH。这里执行原专用衰减包装，但人工path不证明实际Hero参数树分组。beta1=.9、beta2=.95、epsilon约6.0446e-15、衰减.02取自最新归档声明。把声明adam_lr约7.5934e-4作为诊断常数，total_steps人为设1000；未执行实际LR schedule，不能当作生产当前学习率。

共同新初始化状态下，参考仅产生共同衰减；候选另产生Adam方向。两份更新之差范数约1.0739e-3。因为首次偏差修正后m_hat=g、v_hat=g²，当abs(g)远大于epsilon时，Adam项近似sign(g)，很小的非零梯度也可能对应接近学习率的变化。这里衰减在两份对照相同，未选中参数的更新也相同，不能把梯度为0误写成参数不动。真实apply_updates后的参数距离另有记录。

再先向双方同一状态注入人工梯度[.1,-.2,.3]预热一次，保持下一步参数输入共同不变：相同局部梯度差异产生的更新差异约7.4824e-9。此例历史moments主导了下一次更新，显著衰减了差异；这是构造的共同状态对照，不是某个真实checkpoint。也不能外推所有warm state都会降低误差。

### 验收要求为何必须绑定状态

同输入、同参数、同优化器count/mu/nu、同LR/衰减、同dtype和同路由策略，才是单步更新对照的起点。应同时覆盖新初始化与有代表性的真实恢复状态，不能用fresh对照估计中途切换风险，也不能用一个warm例证明全程安全。再向后做固定输入重放和固定评估，才能讨论累积偏移与训练质量。

本轮证明的是人工集成中误差可跨专家传播并受状态调节。未执行实际EXACT backend、QuACK专家、collective、生产参数分组或Hero checkpoint，没有测真实loss影响，不能据此回滚#9833。运行make router-coupling-update CPU_PYTHON=/tmp/marin-jax-cpu-072/bin/python可复算原语句与诊断图。


## V77：合并状态、源码修订与新的运行故障分开看

本轮重新取得三个issue的正文/完整评论、三个PR元数据、#9833的完整10项文件列表、新旧head比较，以及两个文件在两版head上的完整源码。另归档Kubernetes官方NoExecute短摘录；16份新来源保留URL、获取时间和SHA，旧461份非bookkeeping来源保持原字节。[来源账本](analysis/engineering_v77_acquisition.json)与[12项独立审计](analysis/engineering_current_v77.json)可复查。

|对象|V56历史状态|V77新快照|能够改变的结论|
|---|---|---|---|
|[#9708](https://github.com/marin-community/marin/pull/9708)|open，未合并|closed，merged=false|整包PR已关闭，不能称已合并部署；关闭原因未另行核实|
|[#9832](https://github.com/marin-community/marin/pull/9832)|open，未合并|仍open，head不变，正文改写|更新了说明的呈现；作者单rack测量不是本轮重跑|
|[#9833](https://github.com/marin-community/marin/pull/9833)|open，未合并|仍open，head增加1提交，正文改写|澄清不同backend的残差保存范围，不能写成新数值修复|
|[#8506](https://github.com/marin-community/marin/issues/8506)|58评论|59评论|新增一次NoExecute驱逐/整gang重试报告|
|#8435与#8870|27/30评论|正文及评论正文均未变化|不制造新的主帖或hang根因结论|

### head变了，不意味着算法变了

#9833旧head与新head b65be4c9550c5097f0a3add08933531a1c24d534比较为ahead一个提交，修改恰为ep_ragged_all_to_all.py与grug_moe.py两份文件。完整原文件逐个解析，移除各模块/类/函数开头的docstring后，AST完全相同；注释本来不进入AST。这证明此次差异不在两份文件的计算语句中，不证明整套PR已通过运行、kernel数值等价或生产稳定性，也不把docstring的可观察变化称作整个程序字节完全不变。

新说明的重要价值是纠正内存收益的范围。原portable `_RaggedDotExpertMlp.forward`返回的残差包含out，backward读取它，执行 `sum(out.astype(float32) * cotangent.astype(float32), axis=-1)`。因此portable backend保留完整输出用于row-dot；QuACK backend才可从专家反向取得row-dot而不保存完整y。两者都可减少返回完整输出的反向通信，但保存量与重计算图不同。此前报告对此范围的泛化已在第二、三节直接修正。

这里把“文档改写”和“代码修复”分开，但文档修订不是无关紧要：若按旧泛化选择portable backend并预算HBM，可能期待一项它并未实现的省存储收益。验收应填写实际backend、残差叶子/shape、编译内存计划和返回通信，而不是只看共同的EXPERT_SIDE配置名。原GPU性能数据仍是作者在特定QuACK/SM100窗口上的报告，本轮未重测。

### 新故障报告：驱逐原因与节点根因不是一层

[10月6日21:00:53 UTC的新评论](https://github.com/marin-community/marin/issues/8506#issuecomment-6025364160)由loom-oa-dev[bot]发布。中文翻译与状态边界已加入[运行索引](OPERATIONS_ZH.md)：task16在s14fys64节点因NoExecute taint被Kubernetes删除；Iris重排整gang；作者报告至21:00 UTC全部176个task运行、训练到step215756。触发taint的节点条件仍未确认。

按[Kubernetes官方说明](https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration/)，NoExecute可驱逐已运行Pod，是否立即驱逐或延后取决于匹配toleration与tolerationSeconds。这个effect名称不能单独证明硬件损坏、节点重启或网络隔离。公开评论也没有给出taint key、节点condition、Pod UID、事件序列、恢复checkpoint和重放量，本报告没有访问内部incident链接或生产控制面。

“被驱逐→整gang重试”是该评论报告的可追踪事件链；“节点为什么被打taint”仍缺证据。它不同于#8870的collective hang，也不同于9月存储写入暂停；不能把所有重试合成一种训练代码bug。反过来，单个任务退出能影响整gang，使恢复成本进入预训练预算，这一层值得单独记录。

下一次取证应保留节点taint key/effect/timestamp、condition与事件、Pod UID/删除原因、首次rank退出和gang attempt，随后核对实际恢复state.step、首批数据身份及首次持续训练进度。上报到step215756不是本报告独立重建的实时W&B结果，也不证明该节点根因已经修复。训练曲线仍使用已有10月7日独立快照，没有据这一条评论改写token总量、loss或配比收益。

对研究和部署的规则是：PR head变化先查计算差异，PR关闭先查merged，性能收益先查backend，重试恢复先查根因与状态/数据时钟。四类证据分别保存，避免把说明修订、候选提速、任务重排和长程质量改善写成同一项成功。


## V78：原portable专家的前向、反向与row-dot一起验收

V77澄清portable保存输出。现在执行固定PR head的原 `_RaggedDotExpertMlp`，并补齐同head的原Haliax ragged_dot包装：CPU auto selector选择XLA，实际执行ragged_dot_general、512行补齐与原输出切片。没有用普通matmul替换该包装；独立按专家分组的dense matmul只作为对照。14项检查通过，[梯度、有限元素计数与源码SHA](analysis/portable_expert_mlp_cpu.json)保存完整范围。

原类采用Silu，本例capacity=10、hidden=4、intermediate=6、两专家；active sizes=[3,4]，physical sizes=[3,7]，最后三行是静态尾部padding。参数和cotangent为人工FP32。AST加载跳过不执行的GPU/TPU定义和包导入，CPU没有backend环境覆盖；实际JAX/jaxlib=0.7.2，未重建生产执行包。没有运行QuACK、all-to-all、真实routing planner或生产输入。

<div id="portable-expert-placeholder"></div>

[图原文件](assets/portable_expert_mlp.svg)：有效普通梯度均有限，原row-dot的三个闲置行允许NaN；原调用方选择语句在本地身份行映射中将它们排除。图中百分比按各自数组元素数计算，不是token dropping比例或真实故障率。

### NaN尾部没有进入普通梯度，原因不是乘零

干净控制将尾部输入和cotangent清零；poison控制将两者都设NaN。原_apply先以where选择有效输入，再执行两个ragged GEMM，最后以where选择有效输出。反向重新对_apply建立VJP，输出选择也限制传回的cotangent。两种控制的有效输出与全部dx/dW13/dW2逐元素相同且有限；独立dense分组参考的前向及三种普通梯度在atol=1e−6、rtol=1e−5内一致。

但原backward还直接计算 `sum(out.astype(float32)*cotangent.astype(float32), axis=-1)`。尾部out=0、cotangent=NaN，得到三个NaN row-dot。普通梯度安全与row-dot全数组有限，是不同命题。接口已声明active计数之后的行允许未指定；该NaN不自动构成bug。

|本地poison控制中的返回量|元素数|非有限元素|验收范围|
|---|---:|---:|---|
|dx|40|0|有效与尾部普通输入梯度均有限|
|dW13|96|0|所有专家参数梯度有限|
|dW2|48|0|所有专家参数梯度有限|
|原row-dot|10|3|七个active行有限；三个尾部行未指定|
|原caller选择后的weight梯度|10|0|七个active行匹配参考；三个inactive行归零|

### 为什么caller还必须有自己的选择

原调用方先令divisible=accepted且weight非零，再以安全分母除row-dot，并通过where选择最终weight梯度。测试中将receiver行到assignment行的映射设为身份，七个有效权重均为正，cotangent=weight×dout；执行两条原除法/选择语句后，结果全部有限，有效行匹配直接的dot(out,dout)，闲置行归零。这个局部对照验证了选择语句的作用，**没有证明真实transport、排序或accepted映射正确**。

若改造通信接口，把未定义行参加全局求和，或者先按错误索引读取再选择，合法局部NaN可能进入有效结果。每个consumer都要承接active范围与映射合同；不能仅要求生产者所有padding都填0，也不能把一项局部CPU通过代替#9832的GPU未写buffer poison/recompute验收。

另执行空首专家active=[0,7]/physical=[0,10]和全inactive=[0,0]/physical=[0,10]：普通输出与三种普通梯度保持有限，空专家参数梯度为零；全inactive时三种普通梯度范数均为0，row-dot的10个闲置行仍NaN。这是局部专家调用的输入边界，不涉及整步optimizer是否推进或全空训练loss的分母。

### 保存的out对应哪一条梯度支线

本例原forward残差有六项，末项为完整[10,4]输出；raw数组有40个FP32值。但不能据此把编译峰值HBM增加写成160字节，实际内存还取决于alias、生命周期、重计算和offload安排。

独立对每行输出scale求JAX梯度，结果与原row-dot的七个active行一致。再故意把保存的out第一行每个分量加3，保留其他残差、参数与cotangent：dx/dW13/dW2逐元素不变，首行row-dot却从−0.3376771变成−2.4502907。原因是普通梯度由重新计算的_apply VJP产生，row-dot直接读取保存的out。仅对照输入和专家参数梯度，会漏掉这条支线的一致性问题。

这是人为破坏残差的控制，没有真实缓存损坏或offload错误证据。它给出可验证要求：保存与重算视图应对应同一次前向；全部有效梯度验收应包括row-dot及最终routing-weight梯度，而不是只检查专家参数梯度。构建remat/offload优化时，应分别核对这两条读路径。

### 可复用的验收顺序

先固定backend、activation、dtype及物理/active布局，说明padding仅在静态尾部；核对原前向与独立分组参考；再用尾部NaN检验所有有效输出和普通参数梯度；单独核对row-dot的输出scale导数；最后检查caller的accepted/索引/非零权重合同，再扩大到真实collective、recompute与GPU。实际batch或配比改变active分布后，要在新负载下重新审查这些边界，而不是把局部通过当作所有配比下的性能和质量保证。

本轮新增一份固定head包装源码，旧477份非bookkeeping来源保持原字节，来源总数479；实际生产输入、通信poison、GPU与节省内存量仍未知。复现：`make portable-expert-cpu CPU_PYTHON=/tmp/marin-jax-cpu-072/bin/python`。


V81将同一head原router接到实际portable MLP，再比较EXACT与EXPERT_SIDE的权重和router梯度。FP16控制出现乘积向上舍入与归零两类差异，BF16控制有非下溢舍入差；详见[同版本精度链](ROUTING_DROPS_ZH.md)。17项CPU检查不证明GPU或优化器更新等价。


V82再次取得#9831/#9832/#9833，仍open且未合并；#9833 head不变。完整Hero model/train把通用EXACT默认与Hero ragged的EXPERT_SIDE静态选择对应起来；原runtime helper13项host字典检查显示memory/slop/latency继承值可保留，overlap等则强制覆盖。详见[运行开关的证据链](ROUTING_DROPS_ZH.md)。实际生产部署与GPU仍未知。
