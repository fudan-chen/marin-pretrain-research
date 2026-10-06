# loss有限，为什么更新仍可能损坏

上轮确认日志存在阶段与视图差异。这轮继续检查训练循环的失败边界：明确的数值检查只看`train/loss`；它不能替代对更新、状态或几何约束的验收。先发生更新异常，后发生loss变化的案例尤其容易被这种检查漏掉。

核对固定来源[train_hero_ep.py](sources/scale_2026_10_05/train_hero_ep.py)中的主循环try/except/else/finally。新增[17项控制流探针](scripts/probe_failure_loop.py)，执行原AST片段，替换训练步、callbacks、checkpointer、JAX同步与事件范围。结果在[原值与事件记录](analysis/failure_loop_probe.json)。没有真实GPU、autodiff、checkpoint I/O，也没有穷举外部callbacks/kernel中可能另有的检查。

## 检查发生在更新之后，保存之前

本循环依次执行：取batch与可选diagnostic → 调train_step并接收next_state → current_step读取新state.step → 发TRAIN_STEP_FINISHED → 检查本步loss有限 → 普通callbacks → checkpoint.on_step → 下一轮。

正常退出时，else强制执行末步callbacks和checkpoint，并等待保存结束；发生异常则except记录并重新抛出，跳过该强制末尾路径；finally仍发TRAINING_FINISHED。

三个事件不要混同：TRAIN_STEP_FINISHED早于loss检查，表示这一调用已返回并等待了loss；TRAINING_FINISHED也会在失败时发出；checkpoint.on_step只证明到达交接调用，不能证明内容有限、写入提交或恢复通过。

## 人工故障逐项对应源码行为

|人工输入或故障|实际原循环片段行为|证明范围|
|---|---|---|
|一正常步后结束|普通callbacks/save交接后，再强制末步callbacks/save并wait|这是调用顺序，fake recorder不实现真实频率/去重/写盘|
|本步loss为NaN|state.step已到1，发finished后抛错；未调用本步callbacks/save或强制末尾save|不是更新前阻断，也没有回滚next_state|
|loss=2，但next_state参数为NaN|没有抛错；callbacks与save recorder看到了NaN状态|有限loss这一检查不检查next_state参数；不能断言真实backend已提交坏文件|
|第一步更新为NaN，第二步loss才NaN|第一步先交接state.step=1；第二步在state.step=2抛错|晚发现不能证明之前交接的state健康|
|参数每步缩小7倍，下一loss=9.14但仍有限|两步均不触发finite guard，普通/强制交接照常|有限值检查不检测有限的范数塌缩或有限loss尖峰|
|callback或save交接抛错|日志记录异常，不再走强制末尾重试|不证明资源已清理或异步写入状态已恢复|
|diagnostic先抛错|没有调用synthetic train_step，state.step仍为0|异常可发生在优化器之前；真实设备状态未测|

主循环只有一处显式isfinite调用，对象是train/loss。这是本次审查片段的事实，不能扩大成“整个仓库没有其他数值防护”。实际训练可能配置自定义callback，也可能由kernel/runtime报错；本地没有验证它们的覆盖率。

## 为什么三个检查互不替代

|检查|能抓住的例子|抓不住的例子|
|---|---|---|
|loss是否有限|已传播到当前前向的NaN/Inf loss|当前更新才损坏；有限loss尖峰|
|更新和状态是否有限|NaN/Inf update、参数或moment|#8073报告的有限范数缩小、错误方向、错视图/错计数|
|更新几何与完整状态一致性|范数不变量违例、同输入更新不一致|未经确认的数据/评估口径变化；长窗能力退步|

历史[#8073](MUON_GEOMETRY_ZH.md)曾报告参数范数150.73降到21.41、下一步loss到9.14；这些值本身都有限。因此不能从“finite检查存在”推出它能抓住那种事故。不过该案例在另一条训练路径，不能据此断言535B生产循环已发生或漏检同一故障。

[离线事故检查器](scripts/analyze_optimizer_bundle.py)可检查明确全局叶的非有限计数、几何差异与分母一致性；它不在训练循环中自动拦截，也不决定接受、跳步或回滚。

## 数值验收需要自己的水位

此前[checkpoint提交分析](CHECKPOINT_COMMIT_ZH.md)区分训练步、交接、metadata可见和真实恢复。现在需补一条独立的**拟议数值验收水位**：哪个状态已通过明确列出的数值与更新检查。它不能由state.step、finished事件、checkpoint metadata或读回成功自动推定。

|拟议记录|应保留什么|没有证据时|
|---|---|---|
|trained step|输入进度、输出state.step与日志step|不称为健康状态|
|numerically checked step|参数视图、检查对象、组轴、容差、原值和通过项|保持unknown，不用有限loss代填|
|save handed-off step|候选state摘要与交接时刻|不称为已提交|
|metadata-visible step|提交标记与该次对象绑定|不称为数值健康或可恢复|
|restored step|读回内容身份与同输入一步更新对照|不由“发现目录”代填|

[数值状态记录](templates/numerical_acceptance_record.json)保留全部为空；这是建议检查方案，不是Marin当前实现的新增机制。保存前的检查也可能增加同步与归约成本，需按[监控成本](OBSERVABILITY_ZH.md)测量。

不要直接把“发现异常”改为“跳过这一步”。moment、scheduler、loader/cursor、pending QB、EMA和state.step的推进必须有明确语义，跳更新与重放数据也会改变有效配比/曝光。没有同起点状态轨迹验证时，跳步只能叫新的干预策略，不能叫已证明无损的修复。

原train_step的JIT声明donate_argnums=(0)。[JAX官方buffer donation说明](https://docs.jax.dev/en/latest/buffer_donation.html)解释了输入buffer可能被复用、之后不应继续使用。因而保存一个Python旧state引用不代表其设备buffer仍可用于回滚；是否成功复用取决于实际编译和形状条件。本轮没有执行donation或设备恢复；建议回退以内容绑定、已核对的checkpoint为依据，再验证数据与下一步状态接续。

这轮允许的结论是：在所审查循环中，“loss有限”和“步已完成”不是完整状态健康证明；失败路径抑制强制末尾save有明确控制流依据；早期检查应分清非有限、几何和状态一致性。是否出现过污染提交、哪个真实checkpoint可恢复，仍需原训练事故包与实际恢复记录。
