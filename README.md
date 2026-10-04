# Marin 535B 预训练研究

这是一份独立的中文研究报告，重点是读懂故障机制、核实Loss变化、还原实际数据配比，以及设计自己的配比与顺序实验。公共快照截止北京时间2026年10月4日约05:38；18T是目标，当前约51%，仍为4K。

直接打开 [index.html](index.html) 阅读全部内容。需要单文件分享正文时使用 [report_standalone.html](report_standalone.html)：图、样本片段、配比和查询器已内嵌。来源JSON/CSV链接需要配套目录，外部原文需要网络。本地笔记仅保存在当前浏览器，可导出，不会上传。

## 内容

- [候选选择与规则回放](DECISION_GUIDE_ZH.md)：597个seed0候选、七项BPB、三个参照；先执行退步限制再排序，解释零个符合条件者和自比通过的边界。

- [理解文档与自测](LEARNING_GUIDE_ZH.md)：三条阅读路线、18个术语、五个反例练习；HTML前部有交互理解与评审工作台。
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

`sources/`保存公开原始JSON、代码、页面和W&B抽样；`analysis/`保存CSV、指标与验证记录；`assets/`保存12张PNG/SVG、HTML样式及离线预算工具。原始材料的权利属于各自来源，保留来源与原有权利；不能把原始训练数据误认为本仓库公开。

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
