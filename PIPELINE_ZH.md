# 把研究做成能反复运行的决策管线

这条管线的输出是带证据范围的决定：补材料、保留候选、做独立确认、放大或回退。**它不会自动训练模型，也不会因报告写得完整就允许声称生产收益。** 每个阶段都留下文件，后续能复查为什么推进或停止。

## 1. 七个阶段的输入和产物

| 阶段 | 输入与问题 | 留下什么 | 哪些缺口需要先处理 |
|---|---|---|---|
| P0 冻结来源与问题 | 一个明确的变化、固定材料、指标定义 | 来源manifest、版本、研究问题 | 来源漂移、指标分母不明 |
| P1 诊断与筛选候选 | 可比终点、旧/比例等参照、替代策略 | 子集贡献、退步表、候选与陈述边界 | 参照弱、预算不一致、只看平均 |
| P2 设计局部交换 | 自己的库存、历史、目标与实际恢复点 | 供体/接收域、容量账本、冻结实验清单 | 容量不可行、目标事后改变、状态未核对 |
| P3 独立确认能力 | 冻结候选、新重复与确认评估 | 波动、实际任务结果、能力保底决定 | 选择偏差、只有BPB、退步不受控 |
| P4 顺序与恢复 | 已确认累计配比、两种时序、加载器 | 逐桶总量、整数块/cursor重放、中间与末期结果 | 顺序与总量混杂、实际token流未匹配 |
| P5 放大与切换 | 强基线、候选、风险对照、执行状态 | 更大代理结果、回退点、切换状态记录 | 方向不迁移、状态或评估路径不同 |
| P6 复盘与下一轮 | 结果、失败假设、残余未知 | 版本差异、反例、下一问题与验收 | 只增加抓取量，旧证据被覆盖 |

P4只有在研究顺序时才需要执行；若采用已经确认的静态方案，可明确写出跳过理由。某项数据在P1作为线索可用，不意味着已经满足P3/P5。工作台展示的当前公开示例就是这种情况。

## 2. 用第四轮发现跑一遍

P0固定已有公开快照、HF revision、d512范围与BPB口径。P1补上库存比例基线，分解16项macro，保留数学风险与197c的反例。得到的决定是：选中方案可以作为代码/学术方向候选，比例基线也应继续保留。

要进入P2，需要自己的库存、历史曝光、供体、目标与实际恢复状态。现有Marin公开材料不能替代这些输入。P3还缺冻结后的独立确认和生成评估；P4缺完整token ID重放；P5缺更大尺度的相同强基线与关键风险验证。

第七轮把P2中可离线完成的设计部分做成[数学四臂草案](TRANSFER_GUIDE_ZH.md)：从候选差值选出“累计量×域内Q档”两个因素，声明供体，编译到整数块并核对原加载器计数。它不是P2全部通过；库存、历史、实际预算和恢复内容仍缺。原取整额外改了两个桶，因此在进入P3以前先回到P2修正支持范围。执行计数的检查应前置，不能等结果出来后才发现干预范围变了。

第八轮把P4的设计进一步做成[顺序整数账本](ORDER_GUIDE_ZH.md)：从恢复next index划分完整/部分块，整数补偿逐桶累计量，共同边缘块保留数组和游标，再核对实际key和底层映射。没有通过P3的V7候选只用于条件草案；P4计数检查不补齐其能力证据。

这个例子不是“全部完成”。它说明证据足够支持候选判断时，就把下一步需要的材料写清，而不把后续阶段的结果代填。[第四轮结论](CONCLUSIONS_ZH.md)

## 3. 对自己的训练，先提交一份实验记录

可复制[实验记录Markdown模板](templates/experiment_record.md)或[JSON模板](templates/experiment_plan.json)。空字段代表待填，不是默认通过。

最小记录应包含：研究问题和目标能力；共同checkpoint与实际恢复内容；模型和优化/路由设置；token与时间预算口径；旧方案、比例基线与候选；供体/接收域与阶段权重；已有曝光和有效库存；确认题目、重复与解码；能力退步限制；结果会怎样改变决定。

建议第一轮使用4套配比×3个续训重复：旧配比、比例基线、候选、候选加数学局部交换。这是12次的起步设计，不保证统计充分。供体和增量由库存、预算与目标决定，本报告没有给出通用最优百分比。顺序研究等累计配比确认以后再做。

## 4. 管线中的回退分支

P1可以用[候选决策文档](DECISION_GUIDE_ZH.md)复算不同目标和限制下的入围集合，保留每项排除原因。这样的事后探索不能使P3通过。准备确认时另填[选择契约](templates/selection_contract.json)，冻结未来重复、主目标、风险阈值和独立评估。以候选自己作参照得到的零变化，不是强基线验证；没有符合条件者时，不得静默放宽阈值。

| 发生什么 | 返回哪里 | 具体动作 |
|---|---|---|
| 平均改善但关键任务退步 | P1/P2 | 保留取舍候选；调整目标或做明确供体的局部交换 |
| 名义容量不足 | P2 | 减预算、补独立库存或修改经解释的曝光上限 |
| 只有搜索评估改善 | P3→P1 | 检查选择目标与评估污染，冻结新候选再确认 |
| 顺序两组总量不同 | P4 | 修补偿、边界与整数块；暂不归因到先后 |
| 放大收益方向不稳定 | P5→P1/P2 | 核对阶段和系统差异；不要原样上线小模型比例 |
| CE突然跳变 | P0/P1 | 分清数据分布、drop、恢复状态与数值问题，补固定对照 |
| 运行恢复但根因不明 | P6 | 保存缓解、排除与未知，不删除故障记录 |

## 5. 报告自身也走一条构建管线

离线命令`make report`依次执行：归档重算→附录→代理/删域/强基线分析→规则和理解数据生成→科学图→数值验证→HTML→源归档哈希→报告检查。它只消费已归档材料，不读取在线最新run。

浏览器复核是另一项人工可见的检查：选择参照后数值更新、五个曲线案例能切换、规则缺证据不会通过、导入失败不覆盖现有记录、JSON导出可还原、窄屏布局无页面横溢、单文件无远程资源请求。验证结果和截图保存在analysis目录。

第七轮构建链还包括：原权重→两口径200桶预算差→三供体四臂连续合同→整数支持修正→原方法计数核对→误差CSV→可视化。原取整与修正后的结果并存，正负结果都保留。24个整块探针是已执行计数证据，GPU训练与有限token流字段保持未执行；重复基线仅有一套配方，不因文件出现三次就增加样本数。

第八轮增加：恢复窗口→完整块/边缘账本→最小整数补偿→对称可行范围→共同边缘索引检查→错误设计诊断→计数JSON。脚本拒绝负配额、非整数步幅和未经说明的边界取整；不把不完整块按期望比例假装成实际计数。原方法运行范围与真实数据读取范围分别记录。

第九轮增加报告理解检查：冻结十题与来源摘要→保存原回答→记录本页参考展开→按三维锚点人工双评→保留分歧/仲裁→另存修订→用新情境检查迁移。题库与工具检查属于本地构建；真实读者观察另收集。R16的图表准备可以满足当前P1表达范围，理解结果缺失不能改写成训练失败，也不使P3/P5通过。[协议](ASSESSMENT_GUIDE_ZH.md)

采集更新必须用新快照目录，先重建来源范围与run生效区间，再修改数字和结论。旧曲线不应在没有说明的情况下刷新。完整token流与新GPU实验不属于这条本地构建管线。

## 6. 如何选择下一项改进

优先处理会改变结论的缺口，其次处理会造成误读的表达，再补操作便利。每条待办写明：当前判断、可能的反例、最小追加证据、所需资源、完成/推翻条件。

例如“找到完整selector目标”可能改变候选选择解释；“扩大一个图的字号”改善阅读，但不增加因果证据。两者都可以做，记录中不要把它们当成同一种研究进展。最新队列见[持续改进记录](IMPROVEMENT_LOG_ZH.md)。

## 工程事故的检查支线

数据候选管线保持七阶段。核对kernel、通信和checkpoint事故时，采用[工程文档的六个动作](ENGINEERING_GUIDE_ZH.md)：固定身份→首个异常→必要条件→诊断预测→分层验收→修正复盘。它主要应用R13，并结合R03/R12/R14；不把供体、BPB或顺序项机械用于批准工程部署。各项保存“观察/解释/缓解/未知”，空表见[模板](templates/engineering_incident.md)。页面导出是未执行草案，不能替代生产验收。

## 源码检查如何进入实际实验

V18的[训练变更评审](CHANGE_REVIEW_ZH.md)将缓存/评分/边界、完整训练状态、实际更新规则、评估视图与预算接到P0/P2/P3/P5。它保留配置、内容、执行等价的区别，提供[变更记录](templates/training_change_review.json)。六份ladder配置的完整字段差异只支持完整配方比较，不把它当作共同状态下的单一配比确认。

## loss变化的统一诊断入口

V22的[诊断流程](LOSS_TRIAGE_ZH.md)把指标/模型视图、样本内容与计数、执行/更新规则依次接到上述阶段。它整合容量丢弃、HIST/TOPK估计和optimizer时钟，不把配置字段差异当成实际行为差异。可复制[诊断记录](templates/loss_change_diagnosis.json)保存首个异常、共同变化、替代解释和下一项区分检查；这是未执行模板，不新增训练或能力证据。


## 自定义打包的入口核查

在已有证据和预算阶段，对明确逐token对应的输入、权重或标签，先比较逐文档length、ID与插入特殊token后的坐标；最终shape不能代替这一步。再记录切片/drop造成的内容与有效目标变化。详见[打包字段案例](PACKING_FIELDS_ZH.md)。若缺真实cache对齐证据，保留为待验收；人工存储探针不支持生产配比收益。阶段数量与判断锚点不变。


## 加量前的历史覆盖核查

预算阶段先填[重复曝光输入](templates/repeat_exposure_input.json)：有限child的可索引序列数、计划抽取次数与已有component序列索引。把窗口内覆盖与对历史新增覆盖分别保留；库存或游标未知时，不以首次曝光通过评审。再确定供体与独立确认实验。详见[重复曝光账本](REPEAT_EXPOSURE_ZH.md)。这项核查细化已有阶段，不自动证明训练收益。

## V48应用：训练归约先于配比归因

沿用既有规则与管线。改变microbatch前，先写清目标分母，确认实际调用入口，并记录每块有效T；不能仅凭loss有限与梯度已相加验收。对有效均值，核对整个optimizer step的归约，避免重复除micro步数。再核对RNG、路由、梯度裁剪与状态更新；分母正确不自动证明真实模型拆批等价。[源码与14项检查](GRADIENT_ACCUMULATION_ZH.md)只证明原函数体的串行适配行为和人工反例，没有证明Hero事故或配比收益。下一步需要真实入口的loss函数、逐块N/T与逐参数梯度记录。

## V49应用：先核对危险分支，再核对实际触发

沿用已有锚点。有限loss不能替代有限cotangent/梯度；零权重不能自动排除非有限operand。先核对global T与inactive数值，再区分除法guard和上游计算的处理，最后验证完整状态更新。[CPU原函数结果与默认配方条件](MASKED_NUMERICS_ZH.md)不能自动升级成Hero事故证据。原配方若正常执行应有正T，因此人工全零mask只作回归/迁移反例。

## V50：空目标处理不能只把梯度或update改成0

[原AdamH模块CPU对照](ZERO_GRADIENT_STATE_ZH.md)确认：有历史moment时，零梯度仍产生非零更新；只丢弃最终update，状态仍推进，下一有效步也会改变。空目标应按实际global T与接受策略判断，不能按grad==0跳步，也不能用某个未使用expert的叶梯度为0触发整步跳过。moment/count、训练step、调度、EMA与QB应分别记录接受条件。本轮只执行optimizer局部对照，完整Hero步与事故触发仍未验证；规则锚点不变。

## V51：裁剪验收要看归约范围和后续状态

[分组章节的CPU对照](OPTIMIZER_GROUPS_ZH.md)确认三组各裁剪到1时合并范数为√3，且原AdamH首步近乎相同的参数更新可留下不同moment；固定后续参数与梯度，差异仍进入下一次update。验收需记录clip范围、系数/触发率、moment/count和后续有效步，不能只看首次更新相似。最新归档max_grad_norm=None；这不是Hero已启用裁剪的事件证据，规则锚点不变。


## V52：加载器恢复控制

恢复检查顺序：先冻结已完成step和历史batch前缀，再记录next offset与连续输入身份；把prefetch水位单列。有限库存分别验收完整末批、部分末批、合法终点和超范围step，随后才比较完整状态与下一步更新。[原函数检查结果](analysis/loader_resume_probe.json)。


## V53：混合数据身份

在next offset核对后，再验收域ID→子索引→库存取模→实际token身份。先比较完整块多重集，再核对部分边缘槽位，记录重复曝光与独特库存；浮点权重与seed标签不代替映射身份。[14项CPU控制](analysis/mixture_identity_cpu.json)。


## V54：桶内shuffle与切分

数据接续链补齐：库存内容/长度→切分清单→训练shuffle类型/key/窗口→预算截断→混合映射→恢复offset→实际token/hash。跨快照重新切分时更换评估版本，或明确冻结留出身份；按当前配置确认该分支是否启用。[检查结果](analysis/inner_shuffle_cpu.json)。


## V55：小实验库存缩放

配比pilot增加库存缩放审计：先算实际切分后库存，先执行初始化互斥约束，再分别复算合法方案的shuffle前缀、floor或max_train_batches上限；核对每个活跃域可行性及重复率偏差。分别做共同库存改权重与共同权重改库存的对照，在固定评估和额外预算中确认候选。[数据缩放分析](MIXTURE_IDENTITY_ZH.md)。


## V56：新MoE提案

kernel提案顺序：绑定PR head与完整diff→确认输入/未写内存/梯度合同→执行对应数值与poison测试→最终组合移除对照→长窗稳定性与自身保存恢复→再判断部署。性能表的逐行gain不相加，数据配比变更单独记录。[当前分析](RECENT_MOE_CHANGES_ZH.md)。


## V57：性能归因

整包优化增加性能归因步骤：源码合同/兼容性→共同身份与输入→预热和profile排除→配对测量→原始trace、HBM及通信覆盖→最终图移除或合法四臂→长窗/保存恢复/跨rack。报告端点算术与独立组件效应分开。[来源复算](analysis/moe_performance_attribution.json)。


V58入口：`make routing-envelope CPU_PYTHON=/tmp/marin-jax-cpu-072/bin/python`执行48个选定CPU输入及两组人工router VJP。下一阶段按`templates/routing_gradient_acceptance.json`采集完整GPU反向、同状态更新和固定评估；模板未执行，CPU存在性结果不触发生产回滚判断。


V59入口：make router-weight-path CPU_PYTHON=/tmp/marin-jax-cpu-072/bin/python执行原moe_route block，9项CPU检查。下一步需同执行版本的router→EXACT/EXPERT_SIDE→参数梯度，保留选择、epsilon、barrier、cast与accepted mask。


V60入口：make router-coupling-update CPU_PYTHON=/tmp/marin-jax-cpu-072/bin/python。10项人工集成检查执行原router/portable row-dot/衰减包装及真实Optax；真实专家、生产checkpoint、GPU和loss仍为下一阶段证据。


V61入口：make mix-event-identifiability。9项归档/代数检查确认108000配比与108778执行切换在67个时刻的事件列相同。按templates/mixture_execution_comparison.json补同代码配比对照；四臂仅在代码/恢复合同兼容时执行，当前未执行。


V62入口：make swarm-seed-pairs，复算6份公开观察、25终点与75差值。下一步冻结候选/保底项，使用未参与选择的续训重复和同身份评估；不得把三次同方向或compute-equivalent倍数写成已确认535B收益。


V63入口：make pending-router-view CPU_PYTHON=/tmp/marin-jax-cpu-072/bin/python执行原偏置helper和router block，9项CPU/来源控制。tree_at/reshard/spec为替身；真实checkpoint IO、全模型输出和GPU仍待验证。


V64入口：make weights-consumer-faults CPU_PYTHON=/tmp/marin-jax-cpu-072/bin/python。10个人工输入、9项原消费者控制；metadata本地IO真实，manifest/array/digest/template/tree替身。后续须真实save→独立restore→下一步，不能由此认定生产可靠。


V65 管线补充：在 metadata/layout 守卫之后加入同 attempt 的内容完整性核对，再执行 next-step 与固定 eval。记录摘要覆盖是全量还是抽样、回执覆盖哪些 rank。全量 chunk 键计数不能代替内容对照。探针已完成小数组本地 IO，生产恢复尚未执行。


V66 保存管线补充：为 stage、每个 future 的成功/失败、local 状态、全局 callback 和消费异常记录同一 attempt。失败 future 也应释放预算，但已写成的部分数组不发布成成功版本。异常已消费与失败版本已恢复是两个状态。单进程真实 IO 已检查，多 rank 与上层发布仍待执行。


V67 恢复管线补充：目录候选先绑定成功发布 attempt；随后独立比较预期 schema、manifest schema、TensorStore 元数据与实际 JAX 数组。内容对照及 full-state/next-step 另列。发现函数返回的最高 step 不能自动视为已验证回退点。保存验收模板增加对应证据栏，仍待真实执行。


V68 管线补充：先导出真实训练状态叶类型/schema inventory，再区分 NamedArray、普通数组、optimizer buffer 与计数器，逐类核对恢复约束。记录 NamedArray shape-check 状态；显式 dtype 对照和内容校验另执行。原 load wrapper 的本地六输入已测试，真实状态树尚未验收。


V69 管线补充：记录 candidate→原布局→legacy→较早候选的实际调用轨迹、首错及错误类别。只在既定政策允许时退回已验证版本，不笼统吞 ValueError。master-bearing 到 device 模式的恢复核对原布局 hook、权威副本值与转换后保存路径，真实结果仍待执行。


V70 续训→数据管线补充：标明 full resume 或 weights-only，绑定 marker/state 时钟；冻结已消费 batch 历史与 mixture/子集映射，核对首批 IDs 和配比阶段后再解释 loss。源配置切换 step 相同不足以保证累计曝光相同；真实 token 与 loss 尚待生产重放。


V71 切换验收：比较批次区间与阶段边界，核对 callback step/next_step、实际 domain IDs 与各域 valid targets/NLL；保留正常无跨界的证据。若重新转换新 schedule 修好批次边界，还须单独审查历史消费重现。配比边界模板待真实执行。


V72 配比确认管线补充：对固定预测重放旧新 mask/weight，记录分母、域内 NLL 与输出 z-loss；再用固定评估和能力保底确认模型收益。组成和参数变化可能交互，写明分解基准。CPU 人工控制已执行，真实 Hero 分解未知。


V73 诊断管线：绑定共同输入和 checkpoint 身份 → 核对有效目标质量与分母 → 两套预测×两套权重重放纯 CE/含罚项目标 → 三项及两条顺序闭合 → 固定分布评估与能力保底。支持集或预测缺失时停止分解；重放解释不替代训练反事实。


V74 评估管线补充：冻结预期域清单 → 核对逐域N/T/B与覆盖 → 明确根级/父级与空域政策 → 检查小数权重、重分批和零权重非有限值 → 验收原累计状态与最终字段。缺域或被污染状态保留故障证据，不参与配比候选通过判定。


V75 评估入口管线：完全相同[B,T]评分shape与[B,K]标签 → 有限非负数值/合法ID → 显式覆盖与标签重叠契约 → 绑定有序域名和评分位置身份 → 原归约 → 候选收益。新增离线callback数组检查；未作为生产JIT hook，也没有线上修复。


V76 拆批验收：确认真实调用路径 → 记录各块/整步有效质量 → 同输入整批与拆批loss及逐参数梯度 → 空块但总步非空控制 → 检查原累积除K与分子归约 → 再做真实模型/路由/状态/多设备验收。有限前向或事后乘零不作为梯度安全证据。


V77 工程复核管线：完整评论与PR元数据/分页 → 新旧head固定源码差异 → backend残差和通信合同 → 节点/Pod事件与首次rank退出 → 恢复state/数据时钟 → 独立稳定性与质量确认。说明修订、合并状态和生产恢复分别验收。


V78 portable专家验收：固定backend/activation/dtype与尾部布局 → dense分组前向/普通梯度参考 → inactive NaN → 输出scale导数/row-dot → saved-out与重算一致性 → caller accepted/索引合同 → 真正collective/recompute/GPU。保留未定义行允许范围。


V79 接受路径管线：实际需求/容量 → 原prefix裁剪 → dispatch起点/size与receiver压实 → active/physical尾部合同 → return镜像/身份 → 来源分组接受率 → 固定评估。host元数据重放是前置检查，不替代真实collective或GPU验证。


V80执行补充：在同checkpoint/样本的路由重放中，联结领域身份→selected分数→裁剪前权重→接受mask→保留权重质量→专家输出→梯度。先固定离散选择测试连续权重导数，再核查真实容量策略和GPU路径；随后固定评估，才能讨论数据配比。原函数CPU探针不代替后四项。


V81执行补充：先用同head router与实际专家核对两个权重反向，再覆盖FP32/BF16/FP16、subnormal与零权重边界；随后扩大到真实GPU通信、模型cotangent和同状态优化器。九个CPU正权重控制与三个人工零权重控制不是实际训练收益。


V82执行补充：绑定配置请求/代码SHA→helper前环境→helper后环境/重复flag→进程启动参数→backend与初始化时刻→实际编译及测量。若复用同进程，比较上一模式留下的变量；不要由remat_mode推断有效内存预算。公开PR未合并与真实部署未知分开记录。


V83执行补充：环境变更验收增加dispatcher白名单、hardware merge、child注入与backend读取时刻。分别记录请求值、提交值、子进程值和加载对象；若缺后一层，不把性能/质量变化归给未确认生效的参数。


V84把近轮源码控制整理为七个症状入口，按口径/输入、路由/梯度、状态和执行身份选择下一项实验。见[首页操作路径](SYNTHESIS_ZH.md)；入口绑定实际结果摘要，不自动判根因，也没有新的训练收益。


V85执行补充：启用或迁移clipping前，绑定组范围、实际梯度dtype、范数累积dtype、clip系数和clip后cast；对照独立高精度范数与同状态更新。norm=0、梯度全零和更新停止是三个不同命题。


V86执行补充：续训验收绑定parameter/mu/nu、Adam count、外层schedule count、trainer.step、N与实际lr，再用共同批次核对下一更新。主动重置优化器或延长预算应作为独立干预字段。


V87执行补充：构建后检查实际阶段数/正时长/边界碰撞、已消耗历史映射及逐桶曝光预算；分别记录raw或模拟epoching分支。延长N时冻结或主动改写配比与优化计划，不能隐式混用。


V88执行补充：比较候选与基线每块整数计数，检查支持消失及集中余数；累计表单列最后不足一块的实际排列，绑定key、组件顺序和spec/解析配置身份。


V89执行补充：保存float.hex、键顺序、Python/NumPy版本、归一化政策与编译quota；恢复和迁移用整数向量验收共同前缀。日志Python声明不能代替实际执行二进制。


V90补充：缓存/子数据集构造后，再对每阶段正权重支持与实际子域求差集。配置components检查不能替代实际子域检查；异常时先定位来源，不自动删域或重归一化。详见[入口支持边界](MIXTURE_IDENTITY_ZH.md)。


V91补充：构造后支持验收取全部阶段正权重并集。普通活跃训练缓存缺失已有拒绝路径；空concat另验非空children。验证记录顶层域和concat子域的实际覆盖、有效分母、缺失政策，部分面板单独标记。[构造分支控制](MIXTURE_IDENTITY_ZH.md)


V92补充：缓存元数据分类可以并行，缺失构建保持源码的原组件顺序。多机进入构建前核对有序任务与缓存动作清单；故障分别记录发现、future读取、线程池清理和向外返回时刻。不能用本地派发有序替代实际collective验证。[真实线程控制](CACHE_PROVENANCE_ZH.md)


## V93：把数据执行记录先查成“矛盾、缺证据、自洽”

前几轮的源码结论已能形成一组启动前记录检查。本轮新增[数据记录检查器](scripts/check_data_execution_record.py)，用途是检查提供给它的记录是否内部矛盾。它是本报告提出并实现的工具，**不是Marin原训练代码，不是完整训练启动检查，也不是生产运行证明。** 适用范围为有限缓存序列数据集；无限stream、动态库存与在线数据需要另定义合同。

五个检查入口对应源码链中的不同位置：

|入口|读什么记录|发现矛盾时先做什么|目前公开示例的状态|
|---|---|---|---|
|声明支持|components、各阶段完整权重字典|定位未知桶、非有限或负权重；不自动重配比例|归档声明自洽|
|实际子域与库存|构造后有序子域、组件类型、序列数、内容摘要、concat子清单|查空concat、活跃域遗漏、空库存；保留构造日志|缺实际对象记录|
|整数配额与运行时|dataset_index、每阶段整数配额、block_size、Python/NumPy/JAX、归一化政策、代码版本|查配额守恒、非活跃域分配、域顺序；保存环境与最终配额|局部配额存在，但实际域顺序和对应运行证据缺失|
|构建派发|各参与机ordered(name, action, cache_identity)|先比实际有序build清单，结合同步协议核对差异|没有生产host清单|
|评估面板|期望/实际域与子域、内容摘要、有效目标分母|先核对覆盖变化，再比较loss|没有实际面板身份记录|

声明检查根据[配置与实际子域边界](MIXTURE_IDENTITY_ZH.md)整理；整数检查来自[运行时与配额](MIXTURE_RANGE_ZH.md)；构建检查来自[真实线程派发](CACHE_PROVENANCE_ZH.md)；评估身份来自[实际构造省略路径](MIXTURE_IDENTITY_ZH.md)。检查器要求字段，并检查基本范围、守恒和相等关系，未重新执行全部原方法。正权重被取整为0会留下一条diagnostic，不自动判定它错误：原代码也允许并发出警告，应由实验目标判断是否可接受。

输出有三种状态。`conflict`表示提供的记录相互矛盾，返回逐项冲突；`needs_evidence`表示尚缺必需记录；`record_consistent_only`表示当前检查范围内记录自洽。后者不提供训练启动授权或收益结论；所有输出固定 `production_execution_verified=false`、`training_benefit_verified=false`。填一个合法形状的内容SHA只通过格式与记录一致性检查，检查器不去读取缓存验证这些字节。

先复制[空模板](templates/data_execution_record.json)。全部空值会返回needs_evidence，不使用默认值补成通过。另提供[明确标注人工的例子](templates/data_execution_record_synthetic.json)，只用于理解字段；它的摘要来自人工标签，不是实际token。运行：

```sh
python scripts/check_data_execution_record.py templates/data_execution_record.json /tmp/data-record-check.json
```

脚本只读输入、写结果、打印状态，不调用训练或修改缓存；返回码不是自动训练门禁，调用方应显式读取JSON状态。它目前也没有集成进Marin启动脚本。

本轮对公开材料做了一个诚实的应用：把10月7日223个component声明和三阶段权重放进[归档输入](analysis/data_execution_record_archived.json)，同时引用V89本地Python3.12配额。实际子域、有序dataset_index、host构建计划与评估身份仍为null，没有把spec顺序当成生产对象顺序。本地归一化探针未采集JAX版本，该项也保留null。得到的[实际检查结果](analysis/data_execution_record_archived_check.json)为needs_evidence：声明支持自洽，其余关键执行证据未齐。它没有说Hero数据读取失败，更没有因为缺公开证据而判定训练不健康。

[18组篡改与缺证据控制](analysis/data_record_checker_probe.json)覆盖未知桶、缺失正权重子域、空concat、配额不守恒、域顺序变化、缺运行时、跨机build顺序/身份不一致、同顶层评估域下少子域、内容变化、有效分母为0和非有限权重。另核对正权重零配额保留诊断、空模板不能通过、人工自洽记录不会升级为生产证明，以及当前归档示例保持needs_evidence。这18组是新检查器的测试，不能计作18个新增Marin源码执行实验。[控制脚本](scripts/probe_data_record_checker.py)

这个工具还没有覆盖阶段起点、动态batch前缀、int32中间乘法、边缘块真实排列、inner shuffle、读取token/hash复核、下一state有限性、完整checkpoint恢复和loss因果确认。它仅把数据记录中的第一层矛盾前置。后续应将记录从真实构造与读取路径导出，再用原方法和独立实际数组复核；只有执行身份清楚，配比和顺序的训练实验才更容易解释。


## V94：检查器自己也有边界，先修正错误的“记录自洽”

V93把源码结论做成工具，但工具的覆盖范围也必须审查。本轮发现它检查了关键关系，却漏了部分字段形状：未知组件kind、双方相同的字符串children、非字符串运行时字段均可能返回record_consistent_only；只提供部分host清单时，又没有预期参与机集合可供对照。这些是**本报告工具的问题，不是Marin训练源码故障**。

保留原V93脚本、输入和结果，在[严格版本](scripts/check_data_execution_record_v94.py)增加形状与host覆盖检查，再调用旧关系检查。这使历史结果可重放，也避免悄悄改写原探针的来源SHA。新工具依赖原脚本，两者需一起保留。

|记录控制|保留V93的实际结果|V94实际结果|修正理由|
|---|---|---|---|
|actual child kind为unsupported|record_consistent_only|conflict|无法按受支持类型解释记录|
|期望/实际eval children均是字符串xy|record_consistent_only|conflict|相等不代表字段形状正确|
|runtime.python为非空列表|record_consistent_only|conflict|运行时身份必须明确为字符串|
|没有expected_hosts，已提交host计划相互一致|record_consistent_only|needs_evidence|局部一致不能证明参与机覆盖|
|expected_hosts含h0/h1，但只提交h0|record_consistent_only|needs_evidence|需要缺失机器的实际清单|
|weight整数为10的400次方|本地OverflowError|conflict|binary64范围转换前先拒绝，不能让检查本身崩溃|

新版本还拒绝非空白名字之外的组件/host标签、重复concat子域、重复派发名、未声明host计划和错误eval子域类型。缓存构建动作之间的次序与身份仍由旧层核对。不存在构建时，可给每个预期host显式空build清单；不能以“没提交机器”代替“该机器没有构建任务”。expected_hosts本身也是提供的记录，不是工具独立发现的集群状态。

**当前归档仍返回needs_evidence。** 新要求没有补造生产rank、对象清单或token内容。当前报告已能指出需要补什么，但没有实际训练环境导出记录，所以不能把形状修复升格成生产验收。

```sh
python scripts/check_data_execution_record_v94.py templates/data_execution_record_v94.json /tmp/data-record-check.json
```

使用[新空模板](templates/data_execution_record_v94.json)，新增expected_hosts；[新人工模板](templates/data_execution_record_synthetic_v94.json)演示完整字段。旧模板/脚本保留作为V93历史记录。形状冲突先报告并停止进入旧层，因此某些输出只含新形状入口；修正输入后再运行，才能看到其余关系问题。

[12组回归对照](analysis/strict_data_record_probe.json)同时保留旧状态与新完整输出，覆盖上述漏检、缺host、额外host、重复任务、重复子域、超大整数、人工自洽示例、归档输入和空输入。它们测试的是新工具，不是新增Marin源码实验。本轮没有重新运行模型、缓存读取或分布式同步。

这次修正带来的工程规则是：验证器也应接受反例审查。关系相等、字段合法、记录覆盖、实际执行分别是不同条件。前两项能在离线记录上检查，后两项需要真实导出和独立复核。检查器只负责它明确实现的范围，不能因为输出简洁就把其他条件隐去。
