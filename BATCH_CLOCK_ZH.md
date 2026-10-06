# 配比何时生效：batch、样本索引和日志的三种时钟

调整顺序前，先把切换点换算成累计序列数。生产代码中的配比阶段从训练step转换为序列索引；batch发生变化时，不能用“当前batch × step”代替历史累加。恢复时若把过去的batch声明一起改了，相同checkpoint step可能对应不同的下一个逻辑样本。下列例子为人工构造，用来解释接口；没有发现Hero实际发生了这种恢复事故。

[可执行检查器](scripts/audit_batch_clock.py)、[27项核对与反例原值](analysis/batch_clock_probe.json)可复查本章。源码均来自既有归档；本轮重新联网核对的schedule字节与归档一致，没有新增独立实验。这里只检查调度、构造前条件和原回调函数，未执行真实DataLoader、JAX、多卡或token流恢复。

## 1. 切换step必须穿过历史batch账本

定义第s次更新使用的batch为b(s)，其开始序列索引为q(s)。代码实现的是 `q(s)=sum(b(i), i=0..s-1)`；当前批次的索引范围是 `[q(s),q(s+1))`。归档[BatchSchedule](sources/deepening_2026_10_04/schedule_production.py)预计算分段offset，再加当前段的增量；逆函数返回一个序列索引所属的更新step。

人工例子：step 0、1、2的batch为4，从step 3起为8。这里的step是待执行更新的零基索引。

|step|该次batch|开始序列索引|该次索引范围|错误的“当前batch × step”|
|---|---:|---:|---|---:|
|0|4|0|[0,4)|0|
|2|4|8|[8,12)|8|
|3|8|12|[12,20)|24|
|4|8|20|[20,28)|32|

配比配置若在step 4切换，转换结果就是20。[MixtureDataset构造器](sources/deepening_2026_10_04/mixture_production.py)要求每个后续阶段开始索引能整除block_size，并且递增。在block_size=8时，20余4，正常启用assert的构造路径会拒绝；它不会自动把切换点向下取整。step 3和4都不对齐，step 5为28仍不对齐：这个例子改用后续step不能解决余数。需要重新设计batch/切换前缀或block大小，并记录改变的曝光预算。

原转换函数允许最后一个负step转为−1，但构造器的对齐/递增要求并不接受该例中的−1。不能把“转换函数没有报错”作为可训练的验收条件。检查器直接拒绝负阶段；这属于更严格的评审约束，不是声称源码转换函数本来就拒绝。

固定长度4096时，20序列对应81920个**名义位置**。这不是loss分母：padding、loss_weight、文档边界和评分口径还需按已有[文档边界](DOCUMENT_BOUNDARIES_ZH.md)与[评估分母](EVAL_METRICS_ZH.md)核对。上下文长度也变化时，应分段累加b(s)×L(s)，本检查器只接受固定长度或null。

## 2. 恢复时不能只冻结checkpoint step

Hero入口以 `train_loader.iter_from_step(int(state.step))`恢复批次，并把声明的batch schedule传给DataLoader。本章没有重放loader内部，因此以下结论限定为调度接口：改变已消耗前缀的声明，会改变其计算的逻辑索引；实际会不会读到同一token，还取决于缓存、key、混合阶段、loader实现和真实cursor。

|人工恢复对照：均在完成4次更新后|前四次batch|计算的下一个序列索引|可以得出的结论|
|---|---|---:|---|
|原声明|4、4、4、8|20|基准逻辑位置|
|把整个历史改为batch 8|8、8、8、8|32|接口位置前移12；不能沿用“相同step就是相同数据”的判断|
|另一个前缀，累计量相同|4、4、6、6|20|下一索引相同，但更新分组不同；梯度、优化器历史和训练轨迹不能据此当作相同|
|只改变恢复之后的batch|前四次仍为4、4、4、8|20|已消耗batch前缀声明一致；仍未证明真实数据与状态一致|

检查器分别输出 `sequence_offset_delta` 和 `consumed_batch_prefix_equal`，不把二者合并成一个通过灯。对于真实恢复，保存旧schedule并保持已完成更新前缀；把新段从恢复点开始追加。若确实要重写过去，必须绑定旧cursor和新索引映射，而不是依靠当前batch反推历史曝光。

## 3. 曲线中的mixture/weight究竟是什么

原[阶段回调](sources/scale_2026_10_05/train_hero_ep.py)用 `StepInfo.step` 算该次更新的开始索引，定位block和stage，仅在stage变化时记录该stage的归一化权重。既有[状态时钟](TRAIN_STATE_ZH.md)已核对 `StepInfo.step=state.step−1`：这是刚完成更新的零基日志step，不应直接判作错一位。本轮执行原回调函数并记录输出，依赖只用人工dataset和日志收集器替代。

|看到的字段或数量|真正代表什么|不足以证明什么|
|---|---|---|
|`mixture/weight/A`|阶段声明经归一化后的比例|该批次中A占多少，或A贡献了多少有效loss token|
|每个完整block的A计数|源码先截断weight×block，再把余数全部给计数最大的域|边缘块、随机置换后的单个batch或实际内容比例|
|`mixture/stage`首次出现的日志step|回调按该更新开始位置选择的阶段|真实数据文件身份、缓存内容或独立配比收益|
|某域在新stage没有字段|归一化字典省略零权重；回调没有主动写入该域的0|不能仅凭缺失点宣称记录平台已将旧值置零|

人工配置A=.34、B=.33、C=.33，block=8时，整块计数为4、2、2，实际完整块比例是.5、.25、.25。回调仍记录.34、.33、.33。这里没有采样噪声：偏差来自整数分配规则，有限块大小会把小幅配比调整吞掉，或把余数集中给最大域。生产block更大时需用真实200桶逐项复算，不能把这个小块例子的偏差幅度外推到Hero。

另外，人工阶段从A/B各.5变为只有A时，新日志只写A=1及stage，不写B=0。我们没有验证W&B对缺失键的绘图、插值或summary策略，所以不宣称图必然错误；读取数据时应以完整stage快照或实际计数解释，不能无条件向前填充旧域权重。

## 4. 把这项核对接进配比实验管线

本章不是新加一个总分，而是把现有曝光与同预算要求细化为可检查的输入条件：

1. 冻结旧、新batch schedule、阶段step、block_size、序列长度、缓存身份和恢复step；先输出每阶段累计序列索引。
2. 要求阶段索引整除block_size；不合格时返回设计阶段。不要静默取整，取整会改变阶段预算。
3. 单独比较已消耗batch前缀和恢复点累计offset；仅累计量一致仍需解释更新分组差异。
4. 将声明权重、整块整数计数、边缘块和实际loss分母分别存档；对移除的域建立完整快照或明确零值。
5. 只有绑定真实cursor、样本/token流与固定评估后，才进入配比收益比较。调度审查通过只说明声明内部一致。

可复制[人工输入](templates/batch_clock_example.json)后修改。输出保存输入SHA256，并拒绝覆盖已有文件：

```bash
.venv/bin/python scripts/audit_batch_clock.py templates/batch_clock_example.json /tmp/my-batch-clock-review.json
```

示例故意设置不对齐阶段和被改写的前缀，输出应为 `construction_alignment_ok=false` 和 `requires_cursor_and_prefix_review`。这是练习材料，不是Hero生产配置。检查器验证输入并执行原BatchSchedule与原整数分配方法；它没有修改训练配置、证明历史恢复或替代真实token重放。

对自己的配比实验，最值得先做的不是寻找一个漂亮的阶段比例，而是确认“阶段边界、完整块计数、真实loss分母”能互相对上。同step、相同声明权重和有限loss都不足以完成这项确认。确认后再用供体四臂、共同边缘块与固定评估检验收益，才能把顺序影响与实现造成的曝光差异分开。
