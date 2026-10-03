# Marin 535B 预训练研究

这是一份独立的中文研究报告，重点是读懂故障机制、核实Loss变化、还原实际数据配比，以及设计自己的配比与顺序实验。公共快照截止北京时间2026年10月4日约05:38；18T是目标，当前约51%，仍为4K。

直接打开 [index.html](index.html) 阅读全部内容。需要单文件分享正文时使用 [report_standalone.html](report_standalone.html)：图、样本片段、配比和查询器已内嵌。来源JSON/CSV链接需要配套目录，外部原文需要网络。本地笔记仅保存在当前浏览器，可导出，不会上传。

## 内容

- [研究主报告](REPORT_ZH.md)：三个Loss误读案例、模型与系统结构、15类工程问题的症状/原因/改动/验证、scaling与旧仓库分析。
- [#8435逐条中文解读](ISSUE_8435_ZH.md)：主帖及27条评论的中文整理，数值、配置、含义和证据边界；不是逐句机械翻译。
- [数据配比与顺序](DATA_GUIDE_ZH.md)：200桶、三阶段、样本审计、固定验证关联、曝光预算、可证伪的配比与顺序实验。
- [58条生产运行索引](OPERATIONS_ZH.md)：按日期追踪尝试、回退、修正与仍未确认的原因，每条链接原评论。
- [来源与复核方法](PROVENANCE_ZH.md)：7段run拼接、抽样限制、计算公式、来源范围及尚不能回答的问题。

`sources/`保存公开原始JSON、代码、页面和W&B抽样；`analysis/`保存CSV、指标与验证记录；`assets/`保存5张PNG/SVG及HTML样式。原始材料的权利属于各自来源，保留来源与原有权利；不能把原始训练数据误认为本仓库公开。

## 离线重建

已生成的报告不需要安装依赖。需要从归档数据重算时：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
make report
```

当前已在macOS、Python3.9.6执行。图表默认使用系统Arial Unicode；其他系统需要可用的中文字体，可设置 `MARIN_REPORT_FONT` 为字体绝对路径。SVG导出为字形路径，阅读已导出的图不依赖本机字体。`make report`只使用归档材料，不访问网络。

浏览器若限制file页面的本地存储，可用 `make serve`，打开 `http://127.0.0.1:8765/`。完整HTML包括桌面/窄屏排版、目录筛选、200桶域/质量/关键词筛选、曝光排序、样本片段与笔记导出。

## 采集与更新

`scripts/acquire*.py`是本次GitHub、页面、固定代码及配比结果的采集过程；`fetch_wandb.py`与`fetch_windows.py`查询公开匿名GraphQL，不需账户密钥。W&B是抽样历史，不是全量每步流；指定的200步窗口另行取数。错误路径的404保留在抓取清单，未作为证据。

采集脚本会覆盖对应归档文件。如果要更新研究，应在新的快照目录执行并另行更新报告日期、评论翻译、run生效区间与因果判断；不要直接刷新旧归档后继续使用旧文字。GitHub未来评论可能超过一页，更新时需要重新检查分页。本次全部评论数已核对。

## 验证范围

检查包括评论覆盖、每个引用的comment ID、200桶三阶段归一、预算及曝光、两组逐step对照、内部链接、源文件SHA256和单文件图嵌入。结果在 [analysis/validation.json](analysis/validation.json)。浏览器交互与窄屏检查另存为 [analysis/browser_validation.json](analysis/browser_validation.json)。

这些检查确认报告与公开快照一致。GPU训练、kernel故障、逐桶因果贡献、未来cooldown/长上下文结果没有在本机复现。报告明确区分作者记录、本地重新计算、机制解释与建议实验。
