# 一次工程事故的证据与验收

此表为空模板，未执行。不要仅由“重启后继续”填写根因已确认。

| 身份 | 值 / 原始证据 |
|---|---|
| run / attempt / step区间（UTC） | |
| commit / wheel / headers / runtime / 拓扑 | |
| handoff、临时/永久checkpoint与恢复内容 | |
| 此次整包全部变化 | |

| 时点（UTC） | 独立观测 / 日志位置 | 发生于stall之前还是退出之后 |
|---|---|---|
| last progress | | |
| first abnormal event | | |
| watchdog / kill / restart | | |
| save stage / upload / commit / restore | | |

| 解释 | 必要条件 | 当前支持 | 已排除什么 | 尚缺证据 |
|---|---|---|---|---|
| | | | | |

下一项诊断：

预期：

出现什么结果必须撤回当前判断：

| 验收范围 | 预先条件 | 实际观察窗/结果 | 未覆盖部分 |
|---|---|---|---|
| 同批次数值（指标与分母） | | | |
| 性能（device step / host iteration / tail latency） | | | |
| 同型故障的稳定观察窗 | | | |
| 临时 / 永久 save | | | |
| 自己保存后 restore / 回退点 | | | |

当前可公开陈述：

仍然未知：

原主张 / 后续反证 / 修正文：

规则：R01/R02/R03/R12/R13/R14。执行状态：not_executed。
