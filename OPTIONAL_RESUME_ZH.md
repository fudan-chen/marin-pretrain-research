# 可选恢复：同一份坏checkpoint，换个路径为何得到不同结论

V132，2026-10-08。执行固定源码的发现、布局选择、严格数组读取和恢复策略，使用真实本地TensorStore/OCDBT小状态，12项控制通过。结果说明：**“没有找到可恢复状态”与“从未写过checkpoint”在这份实现中并不总能可靠地区分。** 这是人工存储夹具揭示的条件行为，没有证明535B实际因此重置。

[全部原值与调用轨迹](analysis/optional_resume_evidence_cpu.json) · [复现脚本](scripts/probe_optional_resume_evidence_cpu.py) · 前置：[异常链与恢复归因](EXCEPTION_PROVENANCE_ZH.md)。

## 先看一个能改变排错方向的反例

人工保存step20，保留可读metadata与manifest，但不把Adam二阶矩 `opt_state/nu/w` 列入manifest。其余数组真实写入并等待commit。恢复使用 `allow_partial=False`。

传入父目录，发现step20，数组读取失败；原legacy包装重试也失败；最终抛 `FileNotFoundError`。传入同一个step20具体目录，仍有真实读取与legacy重试失败，但 `load_checkpoint_setting=None` 最后返回初始state，step为0。把setting改成True，同一具体路径就明确失败。

这不是数组读取器放过了缺叶。它确实拒绝了输入。差别发生在更外层：恢复策略怎样判断“这里以前存在checkpoint”。

## 原码的判断条件

[固定checkpointing.py](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/experiments/grug/checkpointing.py) 的逻辑是：

```python
candidates = _checkpoint_candidates(checkpoint_search_paths)
written = [candidate for candidate in candidates
           if candidate not in checkpoint_search_paths]
```

数组读取都失败之后，required=True明确抛错；optional=None只有 `written` 非空才按“旧checkpoint存在但无法恢复”报错，否则返回传入state。注释想保护的场景是：不能在一次失败恢复后静默从step0重训。

问题在于，`candidate != search_root` 是路径关系，不是存储历史证据。父目录下的step20满足这个条件，直接传入的step20不满足；它拥有metadata也没有改变这一分类。因此，“written_count=0”在这个具体路径反例里只表示候选等于配置路径。

扫描还有另一层影响：它只把有metadata且能解析JSON的目录加入集合；JSON解析失败会warning后跳过。若父目录下唯一checkpoint的metadata损坏，`written`也可能为空。至于JSON文本为 `null`，解析本身成功，随后在try外调用 `metadata.get` 才触发AttributeError，所以它并不走相同的“跳过”分支。

## 真实本地IO对照

|夹具与调用方式|optional=None的结果|证据与解释|
|---|---|---|
|父目录；step20有正常marker，manifest缺Adam nu叶|明确失败|发现了不同于根路径的历史候选；数组严格读取拒绝|
|直接指定同一step20；同样缺nu叶|返回初始state，step0|真实读取和legacy重试均失败；written却为0|
|同一具体坏目录，required=True|明确失败|强制恢复分支不会转为首次初始化|
|具体坏目录，传入ShapeDtypeStruct模板|返回模板后，原入口guard调用初始化器|执行原guard，初始化器为记录器；得到人工初始step0|
|父目录；完整数组，但子目录marker是非法JSON|返回初始state，step0|损坏marker的子目录被跳过；根目录无manifest，布局hook拒绝|
|直接指定上一行的子目录|真实恢复step20及Adam count1|具体路径被补为候选；读取路径不要求metadata解析成功|
|完整数组但无marker；传父目录／传具体目录|前者step0，后者真实恢复step20|发现发布候选与显式数组读取的合同不同|
|父目录；子目录marker为JSON null|AttributeError，扫描中止|解析成功不等于metadata是字典；异常出现在JSON try之外|
|真正空目录／正常父目录完整checkpoint|前者初始step0，后者恢复step20|两项正常控制，排除“探针总是失败”|

“无marker的具体目录也能读”不能自动判为bug：直接指定目录可被用作显式导入路径。它证明的是消费者没有把发布标记设为所有读取方式的必要条件。若业务要求只从成功发布的checkpoint续训，就需要在更外层明确这项约束。

非法JSON、JSON null和数组缺叶虽然都可称为“损坏”，实际走不同分支，不能用一个“遇到损坏就回退”的概括代替源码分析。缺叶FileNotFoundError还会触发原legacy重试；上一章V69的数组存储NOT_FOUND/ValueError则直接中止，不会套用本章的可选恢复兜底。

## 为什么Shape模板这条分支也重要

固定训练入口先构建初始state。在非partial且没有明确禁用恢复时，为节省内存，它将具体数组替换成携带shape/dtype/sharding的模板并释放初始化数组。恢复之后有如下guard：

```python
if released_initial_state and any(
    isinstance(leaf, jax.ShapeDtypeStruct)
    for leaf in jax.tree.leaves(state)
):
    state = _init_state(model_key)
```

因此，可选恢复返回模板并不会自动在下一步报“没有数组”：入口准备了重新初始化路径。本轮对具体坏checkpoint执行原恢复策略，确实拿回Shape模板，再执行这段原guard，记录到一次初始化调用。初始化器返回人工初始state；没有执行完整Transformer构建、原释放数组helper或整个训练入口，不能将此控制写成“已经让生产训练静默重启”。

这也解释了为什么监控不能只检查训练是不是还能跑、loss是不是有限、MFU是不是正常。它们没有回答“这次启动是否继承了预期祖先”。

## 与配比和训练预算的直接关系

如果原本要从step20切换配比，实际却从step0初始化，那么权重、moment、优化器计数和加载器起点都可能变化。根据原入口，加载器从实际 `state.step` 构建读取位置，当前配置还会重新构建阶段表。此时“新配比run优于旧配比run”的对照已不再共享恢复祖先。这个结论是条件推理，本轮没有执行数据加载或测量语言loss。

预算也需区分：一次意外初始化后的新step不能接在旧累计token曲线上当作不中断训练。应保留attempt身份、预期祖先、实际恢复路径、恢复step、首次更新的optimizer count和数据位置。对外展示可以有墙钟连续线，但训练祖先和有效累计预算必须可追溯。

## 可执行的恢复合同

**首次启动、完整续训、weights-only导入分别声明意图。** optional适合允许首次初始化的入口；业务已经指定了续训祖先时，缺少完整状态应让启动失败，不能仅靠optional兜底。required=True在本轮具体坏目录上有效，但它不自动验证发布身份、叶级dtype、metadata/state.step一致性或下一步等价。

**“历史存在”至少保留独立证据。** 可读发布标记、损坏标记、manifest/数组残留和真正空目录需要分别记录。残留并不证明成功发布，也不该被悄悄等同于从未写入。遇到不确定历史，可进入隔离与人工确认分支；不能直接清理残留或默认新建同名训练。

**状态机应有明确返回结果。** 建议返回 `initialized_fresh / resumed / failed_resume / imported_weights` 以及候选、跳过原因、实际step和祖先身份。不要只返回state，让调用者靠state有没有Shape模板猜发生了什么。这是候选合同，未修改上游。

**验收至少覆盖本章的路径对照。** 同一夹具在父目录、具体目录、多个搜索根下测试；分别注入缺叶、metadata非法JSON、合法JSON但错误类型和未发布残留。正常空目录仍允许明确首次启动，正常已发布状态仍应恢复。还需真实多rank与业务配置回放，才能评估修正是否适用于生产。

下一步最有价值的材料是实际启动配置中的 `load_checkpoint`、完整搜索路径与所选候选日志，以及首批state/optimizer/data身份。当前控制证明固定策略的分类边界，不证明Hero发生过数据重放、预算丢失、意外初始化或loss异常。

## 复现范围

运行 `make optional-resume-evidence-cpu CPU_PYTHON=/tmp/marin-loss-mass-v109/bin/python`。本轮JAX0.7.2、Optax0.2.5，状态只有一个两元素参数、真实Adam矩与pending。原host writer、manifest、tree/leaf reader、布局hook和恢复策略参与；metadata由探针手工写入，collective仅记录，sharding/StoragePath继承单设备本地适配。临时目录退出后删除；没有操作任何生产checkpoint，没有运行旧探针main或覆盖旧结果。
