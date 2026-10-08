# 最后一条报错、最后一个事件与真正的训练失败

V131，2026-10-08。结论来自同一固定提交的训练循环、进度作用域和事件分发方法，配合人工故障；不是535B生产事故复现。可执行结果见 [11项控制](analysis/exception_provenance_cpu.json)，复跑入口见 [脚本](scripts/probe_exception_provenance_cpu.py)。

训练先出现非有限loss，随后退出时进度回调又失败，最外层报错可能来自回调。只读最后一行容易把排查方向带偏。与此同时，日志里的 `CHECKPOINT_FINISHED` 也不能证明checkpoint写完：这段代码用它表示作用域结束，保存调用抛错时仍会发出。要判断可恢复进度，必须继续核对异步commit、metadata发布与实际恢复，不能停在事件名称。

## 具体执行了什么

本轮使用提交 `84869ae8c91ffe64e9f761c5bd714542eb1876e0` 的三个对象：

- [训练循环](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/experiments/grug/moe_hero_ep/train.py)：原 `try/except/else/finally` 与紧随其后的 `tracker.finish()` 调用。
- [进度作用域](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/lib/levanter/src/levanter/callbacks/_core.py)：原 `progress_event_scope`，开始事件在 `try` 之前，结束事件在 `finally` 中。
- [事件分发](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/lib/levanter/src/levanter/callbacks/state_adapter.py)：原 `StateCallbackRunner.emit_event`，顺序调用每个hook，没有逐hook异常隔离。

执行原AST，训练、普通回调、保存和tracker换成人工记录器；事件分发接两个人工hook，验证前一个失败时后一个是否还能运行。旧V34探针的定义被复用，旧案例没有重跑、旧结果没有覆盖。未执行完整函数外围的资源上下文、真实JAX训练或真实保存。历史上传的训练文件也有相似结构，但本轮没有将另一版本的回调拼进它并宣称历史绑定。

## 逐项对应：触发、实际结果与原因

|人工触发|最外层异常或结果|跳过或仍执行的动作|为什么|
|---|---|---|---|
|全流程成功|无异常|周期保存、最终保存、wait、最终事件、tracker结束均到达|对照组证明收尾路径可达|
|训练loss非有限，其他服务正常|非有限loss错误|跳过强制最终保存，仍发最终事件；未到达后面的tracker结束|训练体进入except，记录后重新抛出；finally仍执行|
|loss失败，再让最终事件hook失败|最终事件错误；loss错误留在context|后一个事件hook未运行；没有最终保存|finally的新异常成为最外层，分发在第一个失败hook处停止|
|loss失败，再让logger失败|logger错误；loss错误留在context|仍执行最终事件；未到达tracker结束|except中的日志调用没有独立保护，裸raise之前已出现新异常|
|周期保存交接抛错，事件正常|交接错误|仍发CHECKPOINT_FINISHED；训练体记录错误并跳过强制最终保存|保存作用域的finally不区分成功与失败|
|保存交接和结束事件都抛错|结束事件错误；交接错误留在context|训练体except记录的是新的最外层错误|内层作用域退出时再次抛错，向外传播的异常已改变|
|保存开始事件hook抛错|开始事件错误|保存体未进入，配对结束事件也未发出|开始事件在作用域try之前，尚未进入yield|
|训练成功，强制最终回调抛错|强制回调错误|未执行最终保存、wait；仍发最终事件，没有训练体fatal日志|else中的异常不会重新进入前面的except|
|训练成功，最终wait抛错|wait错误|仍发CHECKPOINT_FINISHED与最终事件，没有训练体fatal日志|wait在else和保存作用域中；分别受两层finally影响|

“原错误留在context”不等于日志系统一定保留了完整异常链。探针检查Python异常对象的 `__context__`，未检查云端日志裁剪、rank聚合或监控解析。同样，logger故障是人工可达性控制，没有证据证明Marin的实际日志handler发生过这种错误。

## 看懂三个容易混淆的状态

<svg viewBox="0 0 900 160" role="img" aria-labelledby="exception-event-title" xmlns="http://www.w3.org/2000/svg"><title id="exception-event-title">结束事件不证明保存成功：保存调用、作用域结束与可恢复性需要不同证据</title><rect x="10" y="20" width="270" height="115" rx="10" fill="#eef2ff"/><rect x="310" y="20" width="270" height="115" rx="10" fill="#fff7ed"/><rect x="610" y="20" width="280" height="115" rx="10" fill="#ecfdf5"/><g font-size="17" fill="#172033" text-anchor="middle"><text x="145" y="53">保存调用</text><text x="145" y="84">可能返回，也可能抛错</text><text x="445" y="53">CHECKPOINT_FINISHED</text><text x="445" y="84">作用域退出时发出</text><text x="750" y="53">真正可恢复</text><text x="750" y="84">commit＋发布＋恢复验收</text><text x="295" y="84">→</text><text x="595" y="84">≠</text></g></svg>

图中的“≠”表示证据不等价。即使没有事件错误，周期性的 `on_step` 也可能只是没有满足调度条件，或者把任务交给后台；它外侧的结束事件没有携带这些区分。正常最终路径额外调用wait，但若wait抛错，结束事件仍可能出现。上一轮 [保存失败策略](CHECKPOINT_FAILURE_POLICY_ZH.md) 已将交接水位、发布水位和retention错误分开，这轮补足事件和异常水位。

`TRAINING_FINISHED` 也只证明执行曾到达这项通知；训练体失败时finally同样发出它。hook如果在通知过程中失败，后面的hook甚至可能收不到通知。事件数量、正常退出码和成功提交数应该分别记录。

## 排错管线：先还原顺序，再选择重启点

1. 保留最早的异常及完整链：至少记录rank、阶段、step、时间、异常类型、cause/context。最外层错误用于解释退出，链中更早的错误用于追查首个触发；两者都不能单独自动判根因。
2. 把阶段拆开：训练计算、普通回调、周期保存交接、最终回调、最终保存、wait、进度通知、tracker结束。缺少训练体fatal日志不等于训练正常结束，else路径失败就不会经过那条日志。
3. 核对checkpoint：从可验证的发布记录和恢复读取选起点。不要凭最后一个训练step、交接step或FINISHED事件选最新checkpoint；也不要把清理旧checkpoint失败自动解释为新checkpoint未发布。
4. 核对恢复输入：记录恢复state.step、recipe版本、各阶段边界、数据key、加载器构建与样本身份。先证明恢复后的样本和状态符合实验设计，再比较loss。
5. 对照后再行动：若首个失败来自数值计算，检查更新前后状态和有限性；若来自保存/事件，先查对应服务与提交轨迹。事件报错本身不足以支持修改学习率或数据配比。

对数据配比研究的具体影响是：重启前后loss变化，可能同时跨过checkpoint回退、样本重放、阶段边界和评估状态变化。如果只保存“最后训练到step X”和一张连续曲线，很难知道配比干预实际在哪个状态、哪批输入上生效。先补恢复账本，再讨论配比收益；本轮没有估计这种偏差在Hero上的大小。

## 评审规则与候选修正

**事件语义规则：** 含FINISHED的名字必须说明是“作用域退出”“成功交接”还是“已提交”。检查报错路径与未触发调度路径，不能只验成功路径。对于未更名的旧事件，监控应避免赋予它保存成功含义。

**异常归因规则：** 故障测试要同时注入主操作与收尾错误，并检查最外层异常、完整链、后续hook和最终提交记录。只断言“抛出了异常”无法发现归因信息发生了变化。

**收尾规则：** 日志/通知失败是否允许终止训练、是否必须继续通知其他hook，应明确决策。候选实现可保留主异常并单独记录次要故障，或聚合错误；需要验收成功路径、双故障、无主异常的收尾故障以及KeyboardInterrupt/SystemExit语义。不能简单吞掉所有BaseException。这里给出合同要求，没有向上游应用补丁，也没有宣称候选实现已验收。

**恢复与配比规则：** 事件不能替代恢复验收，恢复step不能替代样本身份验收。若这两项缺失，重启前后的loss关联应标记为观察，不能升级为配比因果证据。

下一份能改变结论的材料是：真实运行的事件hook注册清单、完整跨rank异常链、实际commit/发布记录以及同checkpoint恢复结果。当前11项控制证明固定源码的条件行为，不能证明真实535B遇到了同样故障、存在GPU hang或因此改变了loss。
