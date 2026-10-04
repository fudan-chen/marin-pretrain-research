# 来源、计算方法与复核边界

本报告的公共训练快照截止北京时间2026年10月4日约05:38。GitHub评论、代码和W&B历史在数分钟内分别抓取，属于一组归档快照，并非同一毫秒的原子快照。后续线上曲线会继续变化。

## 1. 这次实际读取了哪些材料

| 材料 | 读取范围 | 用途 |
|---|---|---|
| [#8435](https://github.com/marin-community/marin/issues/8435) | 主帖与全部27条评论 | 初始配方、scaling、架构解释、后续router与长度实验 |
| [#8506](https://github.com/marin-community/marin/issues/8506) | 主帖与全部58条评论 | 生产故障、部署、回退、checkpoint和存储时间线 |
| [#8818](https://github.com/marin-community/marin/issues/8818) | 主帖与22条评论 | Gate/router范数、敲除、缩放和WD对照 |
| [#8870](https://github.com/marin-community/marin/issues/8870) | 主帖与30条评论 | Silent hang的假设、排除、底层定位和缓解 |
| [#9126](https://github.com/marin-community/marin/issues/9126) | 主帖、评论与配比结果CSV | 旧/新配比、混杂条件、等效算力计算 |
| [#9615](https://github.com/marin-community/marin/issues/9615) | 主帖与19条评论 | 长度/QK、drop口径、loader顺序与cursor |
| #8443、#8480、#9062、#9183、#9179、#9332、#9333等 | PR/issue说明、相关评论与固定版本代码 | 验证实际改动对象，区分提出、合并与部署 |
| [公开W&B报告](https://wandb.ai/marin-community/marin_moe/reports/535B-A23B-18T-Token-Hero-Run-Scaling-Ladder--VmlldzoxNzc2MDM5Ng) | 浏览器阶段说明；7条生产run公共GraphQL元数据与抽样历史 | 重建生产轨迹，计算配对窗口 |
| [Harrier组成页](https://storage.googleapis.com/marin-public/held/harrier-k40-cluster-overview/2026.08.18/index.html?revision=uniform-sampling) | 40域、5档、400个公开样本与来源legend | 核实库存、标签和样本；不把库存当配比 |
| 初始/当前Harrier spec及当前运行配置 | 固定SHA代码、JSON、W&B config | 还原200桶三阶段抽样权重 |
| [旧pretrain仓库](https://github.com/fudan-chen/pretrain) | README、两篇学习笔记、训练报告、最新日记、采集与分析脚本、保存的数据 | 判断解释层缺少什么，保留已有证据纪律 |

被引用的内部Echo/Iris事故页无法由公开链接验证。本报告没有读取它们，事故事实仅依据公开评论；必要处明确保留“原因未确认”。训练数据token store不公开，本地没有重训535B或GPU内核复现。

第二轮新增材料位于 `sources/deepening_2026_10_04/`：固定HF revision的完整934行swarm与桶/内容/预算元数据，生产SHA的MixtureDataset、文本加载器与Python版本，以及固定当前代码SHA。新增原始文件没有覆盖第一版快照。训练曲线截止时间仍为05:38；新增分析范围见[第二轮章节](DEEP_DIVE_ZH.md)。

第三轮新增材料位于`sources/practical_2026_10_04/`：125个公开W&B配置与终点、两组顺序实验的早期与边界历史窗口，以及数据配比相关的一手论文。GraphQL原请求与URL、时间、SHA均保留，控制字段、权重差与端点差见[配置审计CSV](analysis/experiment_config_audit.csv)。80个删域实验在移除配比和运行专属路径后，记录配置与各自参照相同；这不等于核对了原启动代码SHA、实际checkpoint内容或完整token流。不同日期和后缀本身也不证明执行条件不同。[第三轮详细分析](PRACTICAL_ZH.md)

第三轮预算工具使用用户输入的库存与历史量，只计算计划曝光和整数取整后的约束。默认100B是自拟教学示例；它不推断真实库存、不预测Loss、不自动决定最优比例。连续权重下的阶段补偿不替代完整混合块与checkpoint游标重放。

第四轮新增材料位于`sources/findings_2026_10_04/`：选中配比三seed和197c替代候选的4个W&B配置、选中与比例基线的6个step393—430窗口。三个强基线组取同一固定HF版本的终点，与原始W&B记录交叉核对。观察前沿只比较597个mixprior seed0，不混入多seed均值。16项macro分解排除整体token汇总指标；“92.05%”仅为观察差值比例。[第四轮结论与原值入口](CONCLUSIONS_ZH.md)

四组新增配置在声明移除权重和运行专属路径后相同，最终checkpoint fallback一致；这减少记录配置混杂，不能确认实际恢复文件及启动代码相同。3对continuation重复不是3个独立初始化模型。小样本t区间忽略候选选择，只用于敏感性分析；前沿未包含未知选择目标和约束，不解释作者真实决策。

## 2. 生产曲线为什么必须按run拼接

新W&B run继承旧历史，试部署也会重放一段步骤。如果把它们简单合并，某些step会出现多个值，某次失败试验还可能被画成生产继续训练。这里按公开handoff记录规定生产run的生效区间：

| Run | 计入曲线的step区间 | 发生的变化 |
|---|---|---|
| hero-12d8b6f0-dee637 | [0,58014) | 初始生产 |
| hero-wd-gate-router-p02-step58k | [58014,81716) | Gate/router WD |
| hero-ragged_a2a-nccl2307-ep-step81k | [81716,108000) | Ragged EP及轮包 |
| hero-mix-996f4891-step108k | [108000,108778) | 新配比 |
| hero-nopdl-step108k | [108778,121638) | PDL关闭及PJRT更新 |
| hero-main-step121638 | [121638,146139) | Main切换 |
| hero-fa4sm100-nomask-step146k | [146139,快照末尾] | FA4、tail masks、GC |

公共GraphQL使用 `sampledHistory`，每个metric请求1500个点；实际返回点数和稀疏评估点数不同。最终生产CE保留7361个抽样点，dropless Paloma macro loss保留67个评估点。同metric、同run、同step完全相同的重复值去重，冲突值为零。全程图用固定1000step桶的中位数展示趋势，桶不跨run；原始抽样点同时保留。它不能排除没被抽到的瞬时尖峰。

81k和146k另查询200步连续窗口，并逐step配对。含启动开销的均值、全部窗口的中位数、官方稳态窗口不是相同统计量。本文保留自己的原始窗口和公式，不把几种口径混成一个收益数字。108k的配比窗口更换了批次分布，无法按CE差判断能力。

W&B的run状态 `crashed` 也可能来自主动取消、部署或watchdog。诊断要回到对应事件和checkpoint，不从run颜色直接推断数值发散。

## 3. 配比表与曝光怎样计算

库存取固定版本 `harrier_mix_2026_08_18.json` 的200个 `available_tokens`，总数23,106,103,007,622。实际权重取最新生产run的 `data.train_weights`，筛选 `c\d\dq\d`，每阶段200桶之和均为1；23个验证来源权重为零。原配比与固定版本phase0权重最大差约3.47e−18，属于浮点精度差。

domain比例是同cluster五档求和；质量比例是同档40域求和；阶段预算=(结束step−开始step)×11264×4096。曝光是三阶段预算乘配置权重后求和、除以桶库存；它未计入每混合块整数计数的偏移。第二轮另存 `analysis/loader_quantization.csv`，把配置与完整块有效权重分列。所有表都保留计算值与CSV，不用图上读数反推权重。

如果后续16K切换改变batch tokens、阶段边界或跳过数据，实际曝光需重新核算。本报告计算的是**当前4K配置下的全程计划**。网页里的source enrichment、quality档、抽样weight、重复epochs分别使用自己的分母。

## 4. 怎样重建和检查

在仓库根目录执行README中的离线命令，可从已归档JSON生成CSV、图和HTML。`scripts/validate.py`检查评论覆盖、库存总量、200桶×3权重归一、预算、配对窗口步数与数值、内部链接和生成文件。它验证这些计算与归档的一致性，不验证作者的所有因果解释。

[原始抓取清单](sources/source_manifest.json)保存URL、采集时间与SHA256，也保留抓取失败。错误记录如错误路径404不被当作技术证据。W&B归档单独保存请求specs、实际返回与采集时间；[归档完整性清单](sources/archive_manifest.json)覆盖全部原始快照。不要原地刷新归档然后继续把旧报告当作同一快照。

## 5. 仍然需要额外实验才能回答的问题

1. 新配比里的哪个domain/quality变化分别贡献了多少535B收益；当前没有200桶同checkpoint的完整损失分解和独立扰动实验。
2. 数学份额下降对535B解题能力的影响；新增d512任务BPB显示三seed均退化，但不是准确率，也不能定位单桶因果。
3. Cooldown、16K及更长上下文的最终能力和长期稳定性；当前快照尚未完成。
4. PDL缓解是否覆盖全部hang，以及9月底、10月初storage/coordination故障的起始原因；不能把恢复成功当根因确证。
5. 当前计划epochs与实际文档曝光的差距；不公开的token store及cursor无法在本机完整重放。

本报告针对这些问题给出可检验的实验设计，没有补造实验结果。
