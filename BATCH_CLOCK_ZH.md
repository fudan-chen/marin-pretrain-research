# 配比何时生效：batch、样本索引和日志的三种时钟

调整顺序前，先把切换点换算成累计序列数。生产代码中的配比阶段从训练step转换为序列索引；batch发生变化时，不能用“当前batch × step”代替历史累加。恢复时若把过去的batch声明一起改了，相同checkpoint step可能对应不同的下一个逻辑样本。下列例子为人工构造，用来解释接口；没有发现Hero实际发生了这种恢复事故。

[可执行检查器](scripts/audit_batch_clock.py)、[27项核对与反例原值](analysis/batch_clock_probe.json)可复查本章。源码均来自既有归档；本轮重新联网核对的schedule字节与归档一致，没有新增独立实验。早期27项只检查调度、构造前条件和原回调函数；下方V52另执行三个原loader异步host方法。两者均未执行完整DataLoader、JAX、多卡或真实token流恢复。

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

Hero入口以 `train_loader.iter_from_step(int(state.step))`恢复批次，并把声明的batch schedule传给DataLoader。早期表格仅验证调度接口；下方V52进一步执行原异步索引与取数函数，但未重放完整loader：改变已消耗前缀的声明，会改变其计算的逻辑索引；实际会不会读到同一token，还取决于缓存、key、混合阶段、loader实现和真实cursor。

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


## V52：预取位置与恢复位置，走进原异步加载器

前三节的公式能指出声明差异，但没有证明加载器究竟请求哪些索引。此次执行归档`DataLoaderIterator`的三个原函数：`_produce_batches`、`_dataset_get_available_batch_number`、`_do_retrieve_batch_of_batches`。函数体由AST原样提取，BatchSchedule也执行归档源码。异步store返回索引本身作为身份；CPU mesh替换为空上下文，设备布局替换成人工本地范围，JAX batchification替换为host结果摘要，慢请求watchdog替换为直接await。因此，这是原host索引控制流的执行，不是完整DataLoader或实际样本恢复。[原加载器](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/lib/levanter/src/levanter/data/loader.py)

沿用人工schedule：前3步batch=4，此后batch=8。`fetch_batch_size=4`代表一次检索四个训练批次，不是每批四条样本。

|控制条件|原函数请求或返回|含义与边界|
|---|---|---|
|从step 0启动，一次预取4批|首个store请求覆盖索引0–19，首次返回只含0–3|host取数已经超前；这些提前读取不等于模型已经训练|
|模拟完成1步，再从step 1重建|下一批身份为4–7|原函数按state对应的batch号重建；不从此前取数末尾20接续|
|step 4，保留历史batch|返回20–27|历史累计量20决定真实identity store请求|
|step 4，把过去都改成batch=8|返回32–39|同step已经不是同输入；这里从公式推断提升为原取数函数控制|
|只在step 4以后改成batch=6|返回20–25|起点保持，后续更新分组已变；不是同一训练轨迹|
|把fetch深度由4改成1|首个返回批仍为20–27|此人工store/layout中返回内容不变；取数时序与资源成本没有验收|
|两个人工设备范围完全重叠|每个batch的store请求仍只有8个独立索引|原函数先去重设备范围；不是跨进程通信或真实mesh测试|
|有限库存22条，允许尾批padding|最后返回step 4，只有身份20、21，global_size=2|原规划保留两条真实数据；没有执行JAX padding或证明其loss权重|
|同库存，不允许尾批padding|仅返回4个完整批次，最后为12–19|20、21未交付；这是有限数据控制，不代表Hero无限混合数据丢样本|
|有限库存20条，恰好完整结束|返回4批，没有额外空批|正向边界控制|
|库存22条，从step 5/6启动|step 5为空迭代；step 6触发原assert|超出可达终点需要区分合法结束与非法恢复；不认定实际Hero遇到此情况|

预取控制在第一次yield后主动关闭async generator。它确实执行了原取数函数并记录请求，未构建生产background queue或测量进程崩溃。这足以说明**取数位置、交付位置、完成更新位置必须分开记录**；不足以证明生产prefetch永远可重放。缓存版本、随机key、混合映射或底层store改变，即使相同索引也可能返回不同token。

原host重建路径并没有独立持久化本探针store的cursor：batch号经当前schedule重新计算offset，再取得样本。因此恢复评审至少绑定完成step、历史batch前缀、当前next offset、数据身份与连续若干批次的token/hash；若数据源有在线增长、非确定性读取或外部游标，还需额外恢复合同。确认下一批后，继续核对optimizer、router、EMA/RNG等完整状态；数据一致不能替代更新一致。[状态审计](TRAIN_STATE_ZH.md) · [保存提交与独立恢复](CHECKPOINT_COMMIT_ZH.md)

[11项原函数控制结果](analysis/loader_resume_probe.json)保留每批身份、原函数行号和源码SHA。[可执行脚本](scripts/probe_loader_resume.py)要求Python 3.10以上，因为原函数使用`zip(strict=False)`；本次运行是Python 3.12.13，没有修改原函数来兼容3.9。入口为`make loader-resume CPU_PYTHON=/你的Python3.10以上/bin/python`。实际Hero next tokens、完整checkpoint恢复、JAX batchification和background线程仍为空。
