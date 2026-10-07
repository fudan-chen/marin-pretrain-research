# Marin 535B 预训练研究

当前版本V108：从原 seed 派生、子域 key 分配接到原 shuffle/mix；验证未来才启用的域前置时，也可能提前改变当前内容顺序。区分 None/显式0、始终零域过滤与末尾追加控制。[数据派生与前缀验收](MIXTURE_IDENTITY_ZH.md)。

[MoE性能归因与图解](RECENT_MOE_CHANGES_ZH.md)复算12条作者步时，区分时间下降、速度提高与MFU百分点；解释单项负收益与最终组合正作用的反转，并提供未执行的性能交互记录模板。

V26对V25的micro/macro命名错误已[明确更正](EVAL_METRICS_ZH.md)。

建议按自己要解决的问题阅读，不必顺着版本号读完全部增补：

|你现在的问题|阅读入口|
|---|---|
|这次535B训练到底遇到了什么|[主报告](REPORT_ZH.md) → [主帖解读](ISSUE_8435_ZH.md) → [运行索引与更新](OPERATIONS_ZH.md)|
|数据比例为何这样设，loss说明什么|[数据配比](DATA_GUIDE_ZH.md) → [16域真实轨迹](MIX_TRAJECTORY_ZH.md) → [评估分母](EVAL_METRICS_ZH.md)|
|如何为自己的训练选配比和顺序|[候选比较](DECISION_GUIDE_ZH.md) → [供体与预算](TRANSFER_GUIDE_ZH.md) → [顺序账本](ORDER_GUIDE_ZH.md) → [生效时钟](BATCH_CLOCK_ZH.md)|
|换数据之后loss突然变化|[诊断流程](LOSS_TRIAGE_ZH.md) → [缓存身份](CACHE_PROVENANCE_ZH.md) → [去重](DEDUP_FILTERS_ZH.md)与[质量桶](QUALITY_BUCKETS_ZH.md)|
|换kernel或优化器能否安全接续|[变更评审](CHANGE_REVIEW_ZH.md) → [状态时钟](TRAIN_STATE_ZH.md) → [优化器](OPTIMIZER_GROUPS_ZH.md)与[接口契约](IMPLEMENTATION_CONTRACTS_ZH.md)|
|MoE和文档边界有哪些具体陷阱|[路由丢弃](ROUTING_DROPS_ZH.md)、[分位数均衡](QB_ESTIMATION_ZH.md)、[router精度](ROUTER_PRECISION_ZH.md)、[文档边界](DOCUMENT_BOUNDARIES_ZH.md)、[ShortConv](SHORT_CONV_ZH.md)|
|训练在跑，但不知道checkpoint是否可靠|[保存提交与恢复](CHECKPOINT_COMMIT_ZH.md) → [保存内存与写入计划](CHECKPOINT_MEMORY_ZH.md) → [工程证据链](ENGINEERING_GUIDE_ZH.md)|

离线HTML包含科学图、性能图解及原值入口，另有机制与提交控制流示意图。各原函数探针明确列出替代依赖，没有运行真实GPU训练、模型checkpoint恢复或分布式提交。各版本来源、验证范围与历史更正分别保留。

这是一份独立的中文研究报告，重点是读懂故障机制、核实Loss变化、还原实际数据配比，以及设计自己的配比与顺序实验。最新独立观察为[10月7日归档](LIVE_2026_10_07_ZH.md)，约9.915T/4K、目标18T；旧段落中的10月6日54.18%属于历史观察。旧数值图与其分析保持10月4日快照，工程issue于10月6日复核正文和评论未变。

直接打开 [index.html](index.html) 阅读全部内容。需要单文件分享正文时使用 [report_standalone.html](report_standalone.html)：图、样本片段、配比和查询器已内嵌。来源JSON/CSV链接需要配套目录，外部原文需要网络。本地笔记仅保存在当前浏览器，可导出，不会上传。

## 内容

- [八案工程证据链](ENGINEERING_GUIDE_ZH.md)：原观察、机制、缓解与未知；24项陈述对照、预测与推翻条件，以及六动作检查支线。
- [交付审计](DELIVERY_AUDIT_ZH.md)：逐项核对交付要求、实际验证与资源相关后续研究。

- [顺序、边缘块与索引](ORDER_GUIDE_ZH.md)：八组历史顺序对照的整块残差；保持原预算的共同边缘块设计、整数可行范围与错误反例。网页可切换历史、步幅与两种边缘处理，导出计数评审JSON。
- [预算分解与供体实验](TRANSFER_GUIDE_ZH.md)：四候选的200桶差值、域内相抵与数学时序；三份四臂草案、整数余数的额外干预及修正。HTML提供候选/预算/域/供体切换和完整方案导出。
- [候选选择与规则回放](DECISION_GUIDE_ZH.md)：597个seed0候选、七项BPB、三个参照；先执行退步限制再排序，解释零个符合条件者和自比通过的边界。

- [理解文档与自测](LEARNING_GUIDE_ZH.md)：三条阅读路线、18个术语、五个反例练习；HTML前部有交互理解与评审工作台。
- [十题理解检查协议](ASSESSMENT_GUIDE_ZH.md)：保存原回答、逐题证据和人工评分，保留分歧与修订，来源版本变化需重评。
- [18条判断规则](RUBRICS_ZH.md)：每条保留证据齐全/部分/反证锚点与来源，判断证据支持的结论范围，不计算总分。
- [七阶段决策管线](PIPELINE_ZH.md)：冻结来源、筛候选、局部交换、独立确认、顺序、放大切换与复盘；附实验模板。
- [持续改进队列](IMPROVEMENT_LOG_ZH.md)：按会改变结论的缺口排列，保留规则回放中的实际修正。
- [第四轮：结论与判断](CONCLUSIONS_ZH.md)：补上库存比例强基线，拆解16项macro贡献，检查597个单seed候选的取舍，并给出数学修复、独立确认与跨规模实验决策。
- [研究主报告](REPORT_ZH.md)：三个Loss误读案例、模型与系统结构、15类工程问题的症状/原因/改动/验证、scaling与旧仓库分析。
- [#8435逐条中文解读](ISSUE_8435_ZH.md)：主帖及27条评论的中文整理，数值、配置、含义和证据边界；不是逐句机械翻译。
- [数据配比与顺序](DATA_GUIDE_ZH.md)：200桶、三阶段、样本审计、固定验证关联、曝光预算、可证伪的配比与顺序实验。
- [第二轮深挖](DEEP_DIVE_ZH.md)：934条swarm观测、三seed/54项任务BPB、模拟曝光、200桶内部迁移、实际计数与长上下文游标复算。
- [第三轮：自己的实验方案](PRACTICAL_ZH.md)：80个删域、10个质量实验、数学Q4响应、125个原始配置与8组顺序比较，解释怎样控制曝光与替代关系。
- [58条生产运行索引](OPERATIONS_ZH.md)：按日期追踪尝试、回退、修正与仍未确认的原因，每条链接原评论。
- [来源与复核方法](PROVENANCE_ZH.md)：7段run拼接、抽样限制、计算公式、来源范围及尚不能回答的问题。

`sources/`保存公开原始JSON、代码、页面和W&B抽样；`analysis/`保存CSV、指标与验证记录；`assets/`保存14张PNG/SVG、HTML样式及离线预算工具。原始材料的权利属于各自来源，保留来源与原有权利；不能把原始训练数据误认为本仓库公开。

## 离线重建

已生成的报告不需要安装依赖。需要从归档数据重算时：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
make report
```

当前已在macOS、Python3.9.6执行。第三轮重建还需Node.js运行预算工具的数值验证，可用`NODE`变量指定现有运行时。已生成的HTML不需要Node。图表默认使用系统Arial Unicode；其他系统需要可用的中文字体，可设置 `MARIN_REPORT_FONT` 为字体绝对路径。SVG导出为字形路径，阅读已导出的图不依赖本机字体。`make report`只使用归档材料，不访问网络。

浏览器若限制file页面的本地存储，可用 `make serve`，打开 `http://127.0.0.1:8765/`。完整HTML包括桌面/窄屏排版、目录筛选、200桶域/质量/关键词筛选、曝光排序、样本片段与笔记导出。

## 采集与更新

`scripts/acquire*.py`是本次GitHub、页面、固定代码及配比结果的采集过程；`fetch_wandb.py`与`fetch_windows.py`查询公开匿名GraphQL，不需账户密钥。W&B是抽样历史，不是全量每步流；指定的200步窗口另行取数。错误路径的404保留在抓取清单，未作为证据。

采集脚本会覆盖对应归档文件。如果要更新研究，应在新的快照目录执行并另行更新报告日期、评论翻译、run生效区间与因果判断；不要直接刷新旧归档后继续使用旧文字。GitHub未来评论可能超过一页，更新时需要重新检查分页。本次全部评论数已核对。

## 第二轮复核与版本

第一版保留为Git标签 `report-v1-2026-10-04`。第二轮只新增来源目录，不刷新原训练曲线；`make report`会重算swarm、配对seed、两种计数口径和完整块cursor。`analysis/loader_probe_python312.json`保存Python3.12.13实际执行归档加载器两段原函数的结果；`loader_probe_python39.json`保存旧环境对照。主分析用准确求和情景，并逐计数核对3.12结果。

需要重新执行原函数探针，可在装有numpy2.0.2的Python3.12环境运行 `scripts/reproduce_loader_probe.py`。本次复现命令为 `uv run --no-project --python 3.12 --with numpy==2.0.2 scripts/reproduce_loader_probe.py`；这条命令可能下载运行时/依赖，与离线 `make report` 不同。

## 验证范围

检查包括评论覆盖、每个引用的comment ID、200桶三阶段归一、预算及曝光、两组逐step对照、内部链接、源文件SHA256和单文件图嵌入。结果在 [analysis/validation.json](analysis/validation.json)。浏览器交互与窄屏检查另存为 [analysis/browser_validation.json](analysis/browser_validation.json)。

第二轮页面检查保存为 [analysis/browser_validation_v2.json](analysis/browser_validation_v2.json)，包括八张内嵌图、54行附录、窄屏表格滚动与单文件无远程HTTP请求。

## 第三轮复核与工具

第二版保留为本地标签`report-v2-2026-10-04`。第三轮新增材料位于`sources/practical_2026_10_04/`，没有刷新旧曲线；303个原始归档文件保留校验和。125个W&B配置核对参照与实际阶段边界，80个删域的记录控制条件一致，但原启动代码SHA、实际恢复内容及token ID未逐一复现。

HTML的“自己的配比预算工具”可以填后续预算、库存、历史抽样、每阶段保底、曝光上限和早期增幅，也可用JSON填写最多200桶。它计算后期曝光补偿、直接交换阶段的曝光差、整数取整后的容量与保底问题，并导出输入和结果。默认100B配比是自拟教学数据，不是Marin配比或训练推荐；工具不会预测Loss或执行训练。

`node scripts/test_planner.cjs`验证12组预算性质与100个随机补偿计划，结果在[planner_validation.json](analysis/planner_validation.json)。报告重建通过368项检查；新增浏览器记录见[browser_validation_v3.json](analysis/browser_validation_v3.json)。

这些检查确认报告与公开快照一致。GPU训练、kernel故障、逐桶因果贡献、未来cooldown/长上下文结果没有在本机复现。报告明确区分作者记录、本地重新计算、机制解释与建议实验。

## 第四轮：结论与复算

第三版保留为本地标签`report-v3-2026-10-04`。第四轮新增4个W&B配置与6个恢复初期窗口，来源位于`sources/findings_2026_10_04/`；源归档增加到313文件，生产曲线仍保留原快照。`scripts/analyze_findings.py`离线重算三组continuation seed的强基线比较、16项macro贡献和597个seed0候选的观察前沿。

“92.05%的观察macro差值”是终点均值的算术比较，不是因果贡献率；候选前沿也是单seed观察，不能作为部署推荐。新章节把代码/学术收益、数学风险、小幅macro的不确定性和跨规模迁移分别说明。附录保留54项任务逐seed结果，原值与假设见[findings_audit.json](analysis/findings_audit.json)。

第四轮离线重建通过479项检查，另检查12张内嵌图、54行新附录、390px窄屏表格和单文件无远程请求，记录在[browser_validation_v4.json](analysis/browser_validation_v4.json)。第四版标签为`report-v4-2026-10-04`。

## 第五轮：理解与评审工作台

HTML提供五个Loss/工程案例的解释链、四项BPB的三seed点图和旧/比例参照切换。12张静态图补了读法、边界和原值入口；五张长表默认收起，打印时展开。规则1.0和七阶段管线是本报告提炼的方法，未经外部评分效度验证。

可以给18条规则填写证据与状态、查看所选阶段缺口，并通过JSON复制/导入/导出。记录保存在当前浏览器本地，和旧笔记使用不同存储项；工具不验证用户证据真伪，也不执行训练。空证据、反证和不适用均不能让必备项通过；不同规则版本需要人工核对。

`make report`新增规则/理解数据生成与9组逻辑、算术检查。第五轮报告检查通过521项，浏览器检查见[browser_validation_v5.json](analysis/browser_validation_v5.json)。完整实验计划模板位于[templates](templates/experiment_record.md)，默认均为未执行计划。原313文件源归档和训练曲线保持原快照，第五版标签为`report-v5-2026-10-04`。


## 第六轮：候选决策实验室

HTML的“候选决策实验室”按当前参照、主目标和启用风险阈值筛选597个seed0终点，展示七指标变化与排除原因。可切换完整/局部散点，局部图明确声明范围外数量且不改变筛选；手机端图和长表在容器内滚动。完整CSV及条件JSON可导出，默认情景是本报告自拟教学条件。JSON始终标记探索性重算；确认选择契约位于[templates/selection_contract.json](templates/selection_contract.json)。

完整selector没有在已检查入口恢复，检索范围见[审计](analysis/selector_provenance_audit.json)。新增来源另存目录，318个文件有校验和；原312个内容文件未变，抓取清单新增记录。完整离线重建通过546项检查，九情景另从原parquet独立复算。浏览器记录见[browser_validation_v6.json](analysis/browser_validation_v6.json)，测试包括空结果、无效输入、CSV按钮数据、局部图排名不变与单文件零远程请求。未检验OS实际保存下载文件、独立GPU确认或外部规则效度。第六版标签`report-v6-2026-10-04`。

## 第七轮：从权重差到可核对的供体交换

四候选均与996f逐桶比较，两种预算分列；增加预算分解图、1600行桶差值、320行域差值。197c的195桶变化中，名义累计重分配1.473T，约0.759T在域内相抵；数学累计增量和约8.070B时序余量分开计算。这些是账本，未估计桶因果贡献。

三份数学总量×域内Q档四臂草案有12条记录、8套不同整数配方。原取整会额外改变c28q4/c30q4；修正把计数变化限制在声明域，并执行归档两段加载器方法核对24个整块。方案完整保存连续/整数目标与残差，执行、库存、实际恢复和结果仍为空，`launchable=false`。重跑探针需Python3.12/numpy2.0.2，例如 `uv run --offline --no-project --python 3.12 --with numpy==2.0.2 scripts/probe_transfer_loader.py`；离线标志要求本地已有运行时和依赖缓存。`make report`核对已归档探针与当前目标，不会自行安装或训练。

离线重建与独立核对通过622项检查，含13组设计边界检查；网页测试见[browser_validation_v7.json](analysis/browser_validation_v7.json)。原连续案另有24次原方法计数，代码和输入权重摘要防止探针过期。单文件保留13张图、四候选与全部四臂数据；六张长表打印时展开。原318文件和生产曲线未变，规则版本仍为1.0。第七版标签`report-v7-2026-10-04`。

## 第八轮：相同累计量的顺序账本

按公开窗口推导，恢复后包含56/16个整块以及两端不完整块；八组历史对照的整块L1折算为2.29M—4.72M tokens，完整窗口实际差仍未知。新草案保留原3537更新预算，用共同边缘块保护前缀和终点；中间按+2k/−7k做整数补偿。计算锚点是V7尚未确认的M1Q1配方，不能跳过独立能力确认。

执行归档六段方法，核对48组历史和480组草案阶段计数、20份条件索引证明；Python12组边界检查与JS7组检查覆盖全部40个模式/步幅组合。改变两端的错误设计在关闭shuffle诊断下留下140序列L1差，明确不作为历史实际读取差。网页计数导出不含加载器权重；默认k10完整权重草案单列下载。

重跑探针命令为 `uv run --offline --no-project --python 3.12 --with numpy==2.0.2 scripts/probe_order_loader.py`，需已缓存依赖。`make report`消费并核对固定探针，不安装依赖或训练，完整报告通过707项检查。浏览器证据见[browser_validation_v8.json](analysis/browser_validation_v8.json)；规则仍为1.0，14图可离线阅读，原318来源未刷新。第八版标签`report-v8-2026-10-04`。

## 第九轮：理解检查与规则1.1

新增十题理解记录，逐题有参考、来源摘要、三维锚点和关键越界。保存原回答后再看参考，修订不继承旧评分；人工A/B分歧与仲裁依据保留，评分修改也存事件历史。30份作者校准例只检查工具记录逻辑，没有作为实际读者结果。导出没有总分、理解率或训练阶段决定。

自审发现旧R16的“正确复述”问题与“图表可读”通过条件不一致，更新为1.1，区分图表准备与理解证据；其余17项锚点和七阶段不变。旧1.0缓存保留，迁移按钮只重置R16并保存旧依据。13组理解记录边界检查、11组实验评审检查通过；完整报告通过755项检查，浏览器验收范围见analysis记录。仍为14张图，原318来源与生产曲线未刷新；真实读者观察、评分效度、实际token流和GPU训练没有新增结果。第九版标签`report-v9-2026-10-05`。

## 第十轮：工程证据链与可推翻的检查

新增[工程排障文档](ENGINEERING_GUIDE_ZH.md)和网页八案时间线：引用/pool、INLINE排除、headers复发、PDL机制、两次存储停滞、整包提速、tail消费契约、teardown时序。24项陈述逐项保留支持/反证/未知，并给出诊断预测与推翻条件；导出仅是未执行草案。

本轮实际修正主报告的训练CE质量表述、PDL确认范围、整包GC以及58事件第54条遗漏的清理后256进程停滞。工程GitHub另抓12份公开文件，原评论正文未变；训练数值仍冻结在10月4日。330项归档带摘要，规则1.1不变。原十题的源文摘要更新，旧题库与缓存留存，不把旧回答静默应用到新材料。第十版标签`report-v10-2026-10-05`；当前验收数与浏览器范围见[验证](analysis/validation.json)和[browser_validation_v10.json](analysis/browser_validation_v10.json)。

当前交付范围和资源相关后续研究见[交付审计](DELIVERY_AUDIT_ZH.md)。本次完整重建通过803项检查，浏览器新增11项工程/版本行为；未执行GPU或真实读者实验。

## 第十一轮：先统一测量，再判断配比是否跨规模有效

完整复算48个BPB端点，四个子集在不同ladder改变方向；配比目录新增14个固定公共来源，保留全部329个旧内容文件。初始配比差、系统差和eval batch差单列。公共代码机制用合成记录做算术回放，未声称确认历史根因。联合floor/斜率拟合、留出ladder诊断和48槽位确认草案均有脚本与原值；约8.482T是未执行的续训预算，不含共同checkpoint与生成评估成本。题库仍十题，旧V9/V10摘要保留，新版来源变化后不会自动继承旧评审。

执行来源追查又冻结四份公共文件。六run公共commit为空，两份保存入口仅加载另一个callable；这项结果收窄了`save_code=true`的复现含义，没有把公开机制升级成历史根因。

V11复核：完整`make report`通过1019项检查；17图、三个当前浏览器截图、单文件零HTTP请求与旧题库拒绝导入均已核对。实际GPU确认、历史评估执行版本和真实读者理解结果仍为空。[版本摘要与文件SHA](analysis/release_v11.json)。
