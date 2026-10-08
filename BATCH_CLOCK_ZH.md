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


## V70：把真实恢复步号接到下一批样本与配比阶段

这次把原 checkpoint 发现/Grug 恢复、原 BatchSchedule、原三个 loader 异步方法与原 MixtureDataset 串起来。checkpoint 数组真实读写，数据子集是 A/B 身份字符串；没有真实 token、训练更新或完整 DataLoader。7 项检查通过，[原始样本顺序与接口范围](analysis/restore_data_clock.json)可复查。

人工 root 下有两份完整小状态：old 的 marker/state 都为 90；candidate 的 marker 为 100、state 为 20。原策略按 marker 选择 candidate，返回 state.step=20，没有在这条受控路径拒绝两种时钟的不一致。归档训练入口第 1101 行使用 `train_loader.iter_from_step(int(state.step))`；本轮依照这一源码调用关系，把读回步号传给原 loader 方法，没有执行整个训练 main。

人工 batch 声明为前 3 step 每批 4，之后每批 8；配比在 step21 从 A 全量切到 B 全量，原转换函数给出序列边界 156。结果如下：

|控制|采用的 step|累计序列 offset|返回样本集合，完整顺序见图|
|---|---:|---:|---|
|恢复状态时钟|20|148|A:148…155|
|仅用于比较：误用 marker 时钟|100|788|B:632…639|
|step 不变，历史 batch 改为全程 8|20|160|A:160…167|
|下一配比边界|21|156|B:0…7|

<div id="restore-data-clock-placeholder"></div>

图中的身份是实际原混合代码输出的顺序，不是按数字排序后的展示。即使一个块全来自 A，原 block permutation 也会改变该块样本顺序；不能拿序号不单调直接判 loader 错。用相同声明和 key 再取一次，顺序一致。

### 对配比与 loss 分析的影响

第一，候选的 metadata step 用于选择路径，恢复的 state.step 用于消费数据；二者不能在研究账本里不加核对地互换。人工不一致例子会让“已进入 B 阶段”的推算与实际下一批 A 样本冲突。此时把恢复附近的 loss 变化归因于 B 配比，就连处理发生的时刻都没有核对。

第二，batch 的历史累积影响下一序列索引。改为全程 batch8 后，step20 的下一批移了 12 个逻辑序列；对应的 step21 配比边界也从 156 改为 168。配置中同一个切换 step 和同一组权重，不代表此前每域已消费数量相同。要比较配比收益，应冻结历史 batch 声明与 mixture key/子集映射，并记录恢复后实际首批身份。

第三，这些结果只说明恢复与配比对齐的必要检查，没有测 loss，也没有确认 Hero 实际发生 marker/state 不一致。人工 marker 是故意注入的；正常保存调用路径把 state 与 step 传给 checkpointer，是否有路径复用、旧 marker 或历史工件问题需要独立材料。

### 完整续训与 weights-only 必须分别记账

对自己 run 的完整续训，验收应比较同 attempt 的 marker step、恢复 state.step、optimizer 时钟及声明的累积样本 offset；不一致时先阻断配比归因和未经审核的训练继续。对于有意的 weights-only 初始化，源 checkpoint 的 step 可以是 100，而新 run 的 state.step 是 0；这是另一种恢复意图，不能用完整续训的等号直接拒绝。要明确源权重 step、新时钟起点、optimizer/pending 哪些继承或重置，再制定数据消费账本。

下一步验收字段已补进保存模板：resume mode、marker/state 对照、历史 schedule digest、恢复首批身份与实际配比阶段；真实结果仍为空。完整生产验证还缺真实下一批 token IDs、同一存储/内部 shuffle 的映射以及 next-step 指标。复现：`make restore-data-clock CPU_PYTHON=/tmp/marin-jax-cpu-072/bin/python`。


## V71：配比日志标签是否代表刚训练完的整批样本

本轮执行原 StepInfo、LambdaCallback、StateCallbackRunner、原 stage hook，以及原 loader 方法和实际 CPU 混合类。用 A/B 身份数据检查五种输入，6 项检查通过，[每批样本、域计数和回调日志](analysis/mixture_boundary_logging.json)均可复查。传给 callback 的 loss=0 是占位输入，没有模型 loss 测量。

### 正常构建路径有批次边界保护

归档 build_train_dataset 把配置中的切换 step 用同一 BatchSchedule 转成累计序列 offset。人工旧 schedule 为前三批4、之后8；step21 的边界为156，batch20 消费 [148,156)，batch21 消费 [156,164)。两批分别全 A 和全 B，没有半批跨阶段。

MixtureDataset 的构造检查是边界对 block_size 对齐。这个检查本身不要求边界对任意 loader 的批次对齐，但正常构建通过 q(step) 转换提供了后一项条件。因此不能把任意“block 对齐但 batch 不对齐”的输入当成正常训练路径的 bug。

### 原 callback 没有在本例错一拍

StateCallbackRunner 接收训练完成后的 state.step；原 StepInfo.step = int(state.step)-1，next_step = int(state.step)。stage hook 用 StepInfo.step 的 batch 起点 offset 判断阶段。五个控制中，回调 step 都等于刚返回批次的编号。本例旧 schedule 的 batch21 训练后 state.step22，stage 日志记在21并显示阶段1，与 CE 日志采用的 completed-batch 编号一致。

这里证明的是原接口在受控调用中的对齐。hook 的周期条件用 next_step；运行环境是否采用同一执行包，以及实际日志是否继承/覆盖，仍要另外绑定。本轮没有运行训练主循环。

### 故意拼错两个 schedule 后，日志只反映批次起点

冻结旧转换后的序列边界156，却把 loader/callback 改为全程 batch8，则 batch19 为 [152,160)：4条来自A、4条来自B。原 hook 按152所在阶段记录 stage0、A配置权重1，并省略B键。它记录的是批次起点的配置阶段，未测量整批实际域占比。下表与图都使用原混合方法的实际返回值。

|人工控制|批次编号|域样本计数|日志 stage|
|---|---:|---|---:|
|正常旧 schedule，切换前|20|A8|0|
|正常旧 schedule，切换后|21|B8|1|
|冻结旧边界，使用新 loader|19|A4、B4|0|
|用新 schedule 重新转换，切换前|20|A8|0|
|用新 schedule 重新转换，切换后|21|B8|1|

<div id="mixture-boundary-log-placeholder"></div>

新 schedule 将 step21 边界重新换算为168，正常两批再次全 A/全 B。不过 V70 已证明历史 schedule 改写会改变此前消费身份；修好边界对齐不代表重现原续训轨迹。这个跨批控制只是检查不匹配工件的后果，没有证据表明 Hero 实际把旧边界和新 loader 拼在一起。

### 从标签继续走到 loss 的有效证据

先核对三种量：配置权重、实际域样本数、实际有效 loss target 数。即使本例 A4/B4，若真实序列有效长度或 loss mask 不同，也不能推成两个域各贡献一半目标。模型预测误差、MoE drop 和权重系数会再改变损失贡献；本轮身份数据没有这些信息，不计算损失比例。

阶段附近的验收应保存 `[q(step), q(step+1))` 与边界相交结果，抽查或记录实际 domain IDs，再记录各域 valid targets、NLL sum 及权重。对配置日志中的阶段变更，用此记录确认哪批真正包含新域、哪批全部进入新阶段；不能把 stage 标签本身当作已测域组成。若实际没有跨界，则保留正常对齐证据，而不是制造额外的切换延迟解释。

[配比边界验收模板](templates/mixture_boundary_review.json)把 schedule、边界、日志时钟、样本域与有效目标放在同一记录中，生产结果为空。执行范围：原 callback 类通过 AST 提取，Generic 的 TrainerState bound 未用于运行；synthetic state 只提供完成 step；记录 tracker、身份子集及 host loader 适配器替代真实训练与 batchify/watchdog。来源沿用462份归档，没有新增生产观测。复现：`make mixture-boundary-logging CPU_PYTHON=/tmp/marin-jax-cpu-072/bin/python`。


## V87：训练预算会重新定义配比时间表

V71检查loader与日志的边界，本轮向上追到实际Harrier配置构建函数。固定head b65be4c9550c5097f0a3add08933531a1c24d534的原函数、原spec校验及三个预算/阶段辅助函数均执行；LmDataConfig、DatasetComponent和ctx替换成记录字段的适配器。200桶、三阶段权重与库存来自同head完整JSON。没有执行真实配置post-init、数据缓存、MixtureDataset、训练、checkpoint或GPU。[原构建代码](https://github.com/marin-community/marin/blob/b65be4c9550c5097f0a3add08933531a1c24d534/experiments/grug/moe_hero_ep/harrier_mix_2026_08_18.py) · [15项host检查](analysis/phase_budget_probe.json)

### 对齐不只是近似误差，还可能覆盖一个阶段

令N为本次total_steps，B为固定batch，数据block为49,152条序列。可对齐的step间隔m=49,152/gcd(49,152,B)。initial→main使用ceil(N×108000/390251÷m)×m，向上对齐；cooldown辅助函数先取max(1,int(0.8N))，再向下对齐到m，且最少为m。两者方向不同。

阶段随后用dict把(0,switch,cooldown)绑定三份权重。若switch与cooldown相同，后写入的cooldown覆盖main。这是源码明确注释的短诊断策略，不能直接当作生产bug；不过实验名写着“三阶段”，不证明实际读到了三阶段。若阶段起点超过终止步，它会存在于配置里，却没有任何训练批次进入它。

|原构建函数的独立输入对照|m|返回阶段起点 / 阶段|解释范围|
|---|---:|---|---|
|N=390251，B=11264|48|0 initial；108000 main；312192 cooldown|指定Hero尺度参数的构建输出，不证明历史实际执行|
|N=390252，B=11264|48|0；108048；312192|只多计划一步，main晚48步；块对齐造成离散跳变|
|N=780502，B=11264|48|0；216000；624384|人工延长预算，两次边界都随N重算|
|N=100，B=1024|48|0 initial；48 cooldown|两边界碰撞，main被覆盖；实际48/52步分配|
|N=1，B=1024|48|0 initial；48 cooldown|cooldown只被声明，本次运行不会读到|
|N=100，B=49152|1|0 initial；28 main；80 cooldown|不同batch对齐关系让三阶段都出现|
|N=390251，B=2048|24|0；108000；312192|本例边界相同，名义总token仍下降到约3.274T|

同一人工恢复step=121638，在原N=390251时位于main，在N=780502时位于initial。参数和optimizer state是否完整恢复，不决定这张重新构建的阶段表；它是另一个输入。同一step也不能证明同一域身份/库存offset，因为完整映射还依赖旧阶段已消耗的域配额和key。这里只执行原声明构建，没有将该反例写成Hero的真实阶段回退。

<svg id="phasebudget-figure_1" viewBox="0 0 1000 250" role="img" aria-labelledby="phasebudget-title" style="width:100%;height:auto"><title id="phasebudget-title">同一恢复步，在不同训练预算下对应不同配比阶段。源码声明对照，不是真实训练轨迹。</title><rect width="1000" height="250" fill="#fafaf7"/><text x="12" y="77" font-size="15">N=390,251</text><rect x="150.000" y="55" width="110.698" height="35" fill="#d5e6f2"/><text x="205.349" y="78" text-anchor="middle" font-size="13">initial</text><rect x="260.698" y="55" width="209.293" height="35" fill="#61a8ad"/><text x="365.344" y="78" text-anchor="middle" font-size="13">main</text><rect x="469.991" y="55" width="80.009" height="35" fill="#dcb269"/><text x="509.995" y="78" text-anchor="middle" font-size="13">cooldown</text><text x="260.698" y="47" text-anchor="middle" font-size="12">108,000</text><text x="469.991" y="47" text-anchor="middle" font-size="12">312,192</text><text x="12" y="149" font-size="15">N=780,502</text><rect x="150.000" y="127" width="221.396" height="35" fill="#d5e6f2"/><text x="260.698" y="150" text-anchor="middle" font-size="13">initial</text><rect x="371.396" y="127" width="418.586" height="35" fill="#61a8ad"/><text x="580.689" y="150" text-anchor="middle" font-size="13">main</text><rect x="789.982" y="127" width="160.018" height="35" fill="#dcb269"/><text x="869.991" y="150" text-anchor="middle" font-size="13">cooldown</text><text x="371.396" y="119" text-anchor="middle" font-size="12">216,000</text><text x="789.982" y="119" text-anchor="middle" font-size="12">624,384</text><line x1="274.677" x2="274.677" y1="40" y2="177" stroke="#a32a36" stroke-width="2" stroke-dasharray="5 4"/><text x="274.677" y="198" text-anchor="middle" fill="#a32a36" font-size="13">同一人工恢复步 121,638</text><text x="150" y="232" font-size="13">横轴为绝对 step；上行恢复在 main，下行恢复在 initial。未重放真实 checkpoint 或 token。</text></svg>

图中按共同绝对step轴显示两份构建输出。相对进度都保持相同目标，固定恢复step所处阶段却不同；这说明“按比例扩展实验”与“忠实续训原计划”需要不同配置。

### 八轮上限检查保证了什么

原_validate_spec使用固定18.75T参考预算：initial预算是18.75T×108000/390251，main使用15T减去initial，cooldown为3.75T；逐桶累计除以available_tokens，检查≤8。该函数不接收实际N、B或序列长度。校验的是固定spec在这份参考预算下的曝光约束，不能自动保证任意训练预算仍≤8。

本轮另按原构建返回阶段、连续声明权重与每步B×4096计算**名义完整序列token曝光**。N=390251时总量约18.005T，最大桶为c27q0，约6.7408轮；人工N加倍后总量约36.010T，该桶约13.4817轮。固定spec校验仍通过，原构建函数在raw分支仍能返回字段。因此扩大raw预算需要重新核算曝光，不能把原静态校验当作任意预算的保证。这不是测得真实重复率：真实每块整数配额、被读取边缘块、文档身份、padding/mask和有效loss token均未执行。

还需区别模拟epoching分支。原函数以传入analytic experiment_flops≤1e23决定是否设置target_budget与experiment_budget，等号仍开启；只提高到下一可表示浮点数便切换为两个None。开启时experiment_budget=N×B×L，超过固定18.75T会被预算辅助函数拒绝；关闭时这一检查不在该辅助分支内。这里观察的是原函数控制流和记录配置，未证明完整运行入口接受任意人工参数。analytic FLOP不是实际测量消耗，也不能从None推断训练收益。

短实验开启模拟预算时，真实数据加载器还可能按预算比例截断每桶库存；上面的名义曝光使用完整库存作为分母，不能拿它代表截断后的真实epoching。具体截断与身份变化见[混合身份](MIXTURE_IDENTITY_ZH.md)，本轮不重复执行缓存和取数。

### 配比与顺序实验的验收要求

先导出实际返回的阶段表，再计算每阶段是否有正训练时长、是否发生同step覆盖、每阶段累计序列/token预算及各桶名义曝光。小实验与目标大训练应同时比较阶段时长比例、每桶独特库存、重复预算与有效目标质量，不能只比较同名的三份weight字典。

忠实续训时保存绝对阶段边界和已消耗前缀，延长停止预算应明确是否保持原阶段计划；若按新的N整体重算，则把它当成新的顺序干预，并审核旧数据映射。不要只改num_train_steps后沿用原loss解释。V86显示同一个N还可能改变优化器衰减，因此配比时间表、外层lr计划和状态时钟应共同导出后评审；本轮没有执行外层学习率，也没有证明三者实际错配。

新增3份同head源码/spec，492份旧非bookkeeping来源字节保持，完整来源496份。[获取台账](analysis/phase_budget_acquisition.json)记录SHA。复现入口`make phase-budget-probe HOST_PYTHON=/tmp/marin-jax-cpu-072/bin/python`需要Python≥3.10的严格zip，未运行JAX数值kernel。


## V99：训练等数据时，哪些报警真的在等待期间运行

继续沿loader调用链，发现报警覆盖范围必须分开理解。“看到了watchdog”并不意味着读取有deadline；“没有stalled日志”也不能排除loader正在等待。这不是新增Hero卡顿事故，而是固定源码的行为边界。本轮重新下载b65be4c9550c5097f0a3add08933531a1c24d534版本loader，与既有归档逐字节一致，未增加源文件。[同版本字节绑定](analysis/loader_current_source_binding.json)

<div id="loader-stall-flow-placeholder"></div>

本轮在既有原host三个异步函数之外，执行原`run_and_report_slowness`和`__next__`，未改函数体或10秒阈值。异步读取/长度使用事件适配器，等待实际经过原阈值；同步next使用可释放的人工阻塞迭代器。共13项检查，原始日志与请求保留。[结果](analysis/loader_stall_cpu.json) · [脚本](scripts/probe_loader_stall_cpu.py)

|等待位置或故障输入|原函数对照结果|排查含义|
|---|---|---|
|get_batch事件未释放|10秒warning出现，读数task仍pending；没有自动重试|这是慢请求观察器，不是超时恢复器|
|同一请求随后释放|返回原请求0–3的身份|报警不丢弃请求，也不自动改变cursor|
|finite async_len事件未释放|观察10.3秒仍pending，无读取watchdog报警；尚未请求get_batch|长度查询发生在watchdog包裹的读取之前|
|调用方主动取消原读取链|人工read收到CancelledError|取消由调用方发起；没有验证真实线程/远程IO可中断|
|同步next阻塞后成功返回|阻塞期间没有该stalled日志，成功返回后记录耗时|该日志是事后观察，不能作为运行中的心跳|
|同步next阻塞后抛错|原错误传播，没有该事后stalled日志|日志缺失不能排除一次慢失败|
|fetch=4，人工身份4不可读|一次请求0–15，首批尚未yield就抛OSError|预取成功与已交付训练批次需区分|
|fetch=1，同一坏身份|先交付0–3，下一请求4–7再失败|失败覆盖范围改变；本轮没有吞吐或内存测量|

原`_produce_batches`先确认可用长度，再请求batch-of-batches，最后逐批yield。`_do_retrieve_batch_of_batches`把多个批次的本地主机索引合为一次get_batch，整体返回后才拆回。慢请求wrapper对读取建立task，每10秒仅记warning；finally取消自己的watchdog。同步`__next__`的耗时判断则在next成功返回后。日志覆盖的是这些明确位置，不是整条loader的每一个等待。[原加载器](https://github.com/marin-community/marin/blob/b65be4c9550c5097f0a3add08933531a1c24d534/lib/levanter/src/levanter/data/loader.py)

人工故障store声明无限，在遇到身份4时主动抛OSError；不是V96的有限库存越界，也不是源码自己生成的故障。预取四批时，早期有效身份虽在请求范围内，尚未形成可交付批次；这个控制没有证明真实后端会部分读取、重试或缓存它们。这里只证明原host管线的yield在整次get_batch成功之后。fetch=1的正向例子也不代表生产应一律关闭预取。

### 卡顿与恢复管线的具体规则

先将等待分成长度/就绪查询、get_batch读取、后台队列消费、batchify/设备传输、训练执行，分别保留开始和完成时刻。读取warning存在时，检查该请求仍pending、后端错误、队列和各host进度；不要把warning直接写成“自动重试中”。读取warning不存在时，也要检查有限长度查询和初始化。没有执行本轮未覆盖的真实后台队列或初始化路径，不能用这里的安静对照概括它们全部日志。

显式超时若要加入，应另写合同：作用在哪个等待、能否取消底层IO、各host如何共同退出、是否重试同一身份、什么状态允许提交下一训练step。只在单host外面加wait_for，而其他host继续进入collective，可能需要进一步协调；这是设计审查事项，本轮未实施或复现跨host死锁。wall-clock报警阈值也不等于可读性deadline，不能直接拿原10秒阈值作为停止标准。

恢复仍依据完成的state.step及历史数据声明，不依据“后台已经请求到哪里”。在重新取数前记录已请求、已交付、已完成更新三种水位；重读未完成的预取身份不等于模型训练重复。对于外部有副作用或不可重放的数据源，另建游标和提交合同，不能套用稳定随机访问identity store的结论。

数据配比实验也受此影响：若调整后某域读取更慢、损坏请求更频繁或host掉队，按墙钟比较到达的loss会混入有效训练量差异。先核对成功完成的序列和有效目标预算，再比较共同进度的域loss；吞吐退化与统计收益分别报告。本轮没有生产步时、真实故障率或loss，不能量化Hero受影响的程度。


## V100：后台队列的停止、错误和耗尽，原模块怎样执行

V99只执行host管线，没有执行后台队列。这轮新增固定head的完整`background_iterable.py`与`thread_utils.py`，执行原模块、真实标准库queue/thread/asyncio及真实tblib traceback依赖，再把原loader host producer接入原CPU background subclass。CPU mesh仍是空上下文，子域仍是人工身份数据；没有完整DataLoader、设备batchify或真实IO。[两份源码与旧文件完整性](analysis/loader_background_acquisition.json)

**最重要的区别：设置stop flag、唤醒消费者、取消生产请求和等待线程退出，是四件不同的事。** 对照发现正常耗尽后再次next会阻塞，且已进入空队列get的消费者不会被stop唤醒。另一个现象是stop(wait=True)等待生产线程；它不主动取消本轮事件控制的异步请求。这些不能直接归因到Hero，也不能据此断言真实远程请求一定不可取消。

|原模块控制|实际结果|机制或合同|
|---|---|---|
|生产两条有效数据后抛OSError|先取good0/good1，后收到原OSError及生产frame|错误排在既有有效项后面，不是无条件抢先清空队列|
|producer factory构造时抛ValueError|消费端收到ValueError|构造失败也通过原异常wrapper传递|
|异步有限生产0、1后结束|先0、1，再StopIteration|首次耗尽正常|
|同一buffered iterator耗尽后再次next|0.15秒观察仍等待，生产线程已退出、队列空|结束sentinel被取走，原消费端未记录持久exhausted状态|
|异步请求未释放时stop(wait=True)|观察窗口内stop和生产线程仍等待；请求未收到取消|join等待线程结束，stop flag不取消正在await的请求|
|消费者已经进入空队列get，再stop(wait=False)|消费者仍等待；释放请求后生产线程退出，消费者仍等待|stop没有向已阻塞的get发唤醒信号；停止后的_enqueue也不放入终止项|
|生产者因容量1的满队列等待enqueue|stop后按原queue.put超时轮询退出|满队列backpressure与生产请求等待不是同一种阻塞|
|未缓冲路径的原AsyncIteratorWrapper|耗尽后重复next仍返回StopIteration|该wrapper有自己的_exhausted标志，正向控制正常|
|原未缓冲wrapper遇人工RuntimeError / OSError|RuntimeError变成StopIteration；OSError正常传播；buffered路径保留RuntimeError|异常类型与缓冲模式改变了失败表现|
|原loader接真实background queue，fetch1/4遇同一人工坏身份|fetch1先交付有效首批再报错；fetch4首批前报错|V99的host故障边界在原队列传递中仍成立|

原队列消费端先检查stop flag，再阻塞于`q.get()`；进入get之后不会重新检查，stop也没有唤醒队列。正常耗尽只入队一个sentinel，消费者取走它后抛StopIteration，但没有记住已结束。Python要求迭代器一旦抛StopIteration，后续调用继续抛出；因此buffered重复耗尽控制属于明确协议缺口。[原后台模块](https://github.com/marin-community/marin/blob/b65be4c9550c5097f0a3add08933531a1c24d534/lib/levanter/src/levanter/utils/background_iterable.py) · [Python迭代器合同](https://docs.python.org/3/library/stdtypes.html#iterator-types)

这些不是测得“永久挂死”的生产记录。0.15秒用于观察特定等待状态，实际源码分支也说明当前没有可消费项或唤醒路径；探针在记录之后主动注入sentinel清理阻塞消费者。**这个注入只用于结束实验，不是原实现的恢复动作。** 所有人工等待均释放并回收相应线程，不能把清理后的退出说成stop自己成功唤醒。

正常for循环通常只遇到一次StopIteration，DataLoader的新一轮迭代也构造新iterator；不能因此宣称每次评估都会卡住。问题条件是重复访问同一个已耗尽buffered iterator，或者另一线程/关闭协调器在消费者已进入空队列等待时发stop。生产线程若正在等真实存储或collective，还需另核取消与退出合同。本轮没有测这些后端。

未缓冲路径另有异常合同缺口：原AsyncIteratorWrapper在_run_async_task中把RuntimeError解释为迭代结束，但try范围包括future.result，因此人工数据producer主动抛出的RuntimeError也被转成StopIteration。OSError的对照保留原异常，同一RuntimeError经过buffered队列则仍作为错误到达消费者。不能把这个差异叫作真实库存耗尽。原DataLoaderIterator在max_buffered_batches=0时使用该wrapper；同head训练入口声明buffer capacity=512、fetch=4，正常构建会使用buffered路径；因此不能用未缓冲错误解释这份入口声明的正常训练。实际Hero执行版本和运行参数仍需另绑定。[训练入口声明](https://github.com/marin-community/marin/blob/b65be4c9550c5097f0a3add08933531a1c24d534/experiments/grug/moe_hero_ep/train.py#L421)。[原线程工具](https://github.com/marin-community/marin/blob/b65be4c9550c5097f0a3add08933531a1c24d534/lib/levanter/src/levanter/utils/thread_utils.py)

这条路径需要把“事件循环已经关闭或任务调度失败”和“producer读取抛错”分开处理；捕获范围不能包住所有future.result的RuntimeError。下面的消费端候选没有修改AsyncIteratorWrapper，因此没有解决未缓冲错误转结束的现象。训练或评估提前结束时，首先核对错误来源与真实库存，不以StopIteration单独证明完成了计划预算。

### 一个局部候选修正，以及它没有解决的部分

新增[消费者候选](scripts/background_queue_consumer_candidate.py)，只作为本地实验类，不改上游：消费端记terminal状态；用0.05秒queue.get轮询，让空队列等待可以重新观察stop flag；保留有效项与原异常的FIFO顺序，异常交付后也进入terminal。该方案选择stop后不再交付缓冲项，必须在实际系统审查丢弃与恢复水位政策。

5项候选控制通过：正常流与重复耗尽、有效项后原错误、原生产frame、已阻塞消费者无sentinel注入退出，以及释放生产请求后线程清理。取数请求仍未被取消，stop(wait=True)的join语义也未改变。0.05秒是候选轮询参数，不是硬实时退出保证；本轮没有CPU开销/吞吐测量，也没有多消费者、多host或设备缓冲验证。

候选依赖固定模块的私有sentinel/exception协议，不能直接当作跨版本通用补丁。更完整修复需要同时定义异常后的终态、停止是否丢缓冲、生产请求的取消/关闭、唤醒策略和join deadline；队列容量满时必须验证终止信号能否送达。这里没有发布PR、替换Marin组件或宣布生产问题已修复。

### 带到自己的预训练管线

检查成功路径之外的退出路径：空队列+等待消费者、满队列+等待生产者、底层请求pending、有效项后错误、首次与重复耗尽。分别验收消费者结束、生产task取消/自然完成、线程回收及其他host退出；其中任何一个成功都不能替代其余项。

关闭或故障时仍区分已请求、已交付和已完成更新；排在队列里的有效批次是否训练，取决于消费与状态提交。配比调整改变访问成本后，不应将更大的prefetch水位或更多排队项当作更多训练曝光。先对齐完成step/有效目标量，再比较域loss与墙钟效率。

[19项原模块＋5项局部候选控制](analysis/background_queue_cpu.json)保留原错误traceback、线程/消费者观察和清理记录。[可复跑脚本](scripts/probe_background_queue_cpu.py)使用Python3.12.13、tblib3.2.2；本轮仅在临时CPU环境补装tblib。真实Hero关闭事故、远程IO取消、多host退出和完整DataLoader均未知。


## V121：从step393恢复，为什么step384的配比边界仍然重要

V120验证了完整配比历史保留训练支持集合，不能将这组删域结果归因于child shuffle key改变。这轮继续检查另一条独立路径：相同key不保证相同的每桶逻辑消费历史。阶段权重参与原`MixtureDataset`的累计计数和块内置换，旧阶段被怎样解释仍然重要。

### 1. 新取得的共同生产run声明

匿名公开GraphQL取得共同checkpoint生产run `h100-ladder-d512-ep8-bs1024-791tpp-10pct-20260826-rno2a` 的配置与摘要，[原始响应](sources/resume_boundary_2026_10_08/producer_meta.json)和[请求及抓取记录](analysis/resume_boundary_acquisition_v121.json)均归档。它的阶段是0/3120；80个删域续训配置为0/384/3120。两边初始权重、223个组件、shuffle与模拟曝光预算一致。生产run最后日志`_step=392`，名义累计位置1,648,361,472，等于393×1024×4096；记录状态finished。这支持公开记录对应393次更新，不能替代实际checkpoint数组和元数据的读取。

因此存在一个明确的声明差异：生产run在384–392仍处于初始权重，而续训配置将这段过去归入新配比。共同恢复路径虽名为step-393，重建加载器时仍可能按续训配置重新计算已经过去的逻辑前缀。当前没有核验每个run最终选中了哪份checkpoint，也没有恢复历史启动源码SHA，下面严格称为原函数条件复算。

<div id="ablation-resume-boundary-placeholder"></div>

读图：前两行来自公开配置，第三行是未执行的桥接方案。393在384–432这一混合块内部；不是合法的块边界。图不表示实际token已经重复，也不表示桥接方案已有训练收益。

### 2. 原加载器如何解释这9步

记录batch=1024，混合块大小49,152序列，即48次更新；上下文长4096。原`BatchSchedule`与阶段重标度函数给出：step384对应393,216个序列槽，step393对应402,432个槽。恢复点落在第8号块的第9216个槽，即9次更新，名义位置37,748,736。

原混合器先按阶段计算每桶整数quota，再累计此前完整块的计数。每个槽保存编码的dataset ID及桶内局部index，整块随后被置换。恢复只给全局槽位置时，必须用当前声明的历史重建这些映射。本轮运行原混合类、原块置换、原schedule和原逻辑index decoder；没有实际token、child shuffle、模拟库存切片或checkpoint恢复。[脚本](scripts/probe_ablation_resume_boundary.py)、[完整控制与边界](analysis/ablation_resume_boundary.json)。

在给定混合key=7下，对80份记录逐一比较：生产声明的块8前9216槽，与续训声明的同一前缀。前8个完整块的命名整数base offset一致；差异来自当前这个部分块的不同配方及置换后逻辑身份。

|条件复算的量|80份配方的范围|含义|
|---|---:|---|
|前缀每桶计数差的半L1|2027–2940条序列|总槽数不变，但预算在桶间重新分配|
|两种前缀各自未匹配的逻辑身份数|7788–8066|不只改了桶计数，同桶内具体index也可能不同|
|拼接旧前缀与新后缀后，块8内相交的逻辑身份数|5129–5977|人工逻辑拼接中，新后缀会重访旧前缀中的这些身份|
|新后缀第一批1024槽与旧前缀相交数|116–167|上述交集在恢复后的第一批即可出现|

半L1与身份交集回答不同问题，不能把2027–2940解释成总重复量，也不能把5129–5977直接乘4096称为真实重复token。逻辑身份是`(cell, unwrapped_index)`；实际token还取决于缓存、child shuffle、模拟切片、有限库存取模、真实key及运行源码。尤其这里提供了人工key7，不是恢复历史运行key。

四个代表配方还调用原`get_batch`读取有限身份store，逐条与原decoder核对下一批1024个逻辑身份。另用key0和11重复一个配方，几何边界不变、交集数值变化；这说明数值依赖置换，不能作为生产固定常数。未改变权重的分段控制则没有前缀或块内身份差异。

### 3. 为什么只存“每桶已经读了多少”仍可能不够

块内顺序经过置换，一个部分块中的某桶身份不一定是从0开始的连续前缀。给定key7，生产声明的块8前9216槽里，c30q4出现480次，但局部index从5到2456，低于最大index的位置中有1977个未在这段前缀出现；该桶本块quota为2458。[逐桶前缀计数](analysis/ablation_resume_boundary_cells.csv)与结果中的`partial_block_population`保留原值。

所以“读了480条”不等于“下次从局部index480继续”。若换了这个块的配方或排列，只保存计数无法描述哪些局部身份已经被消费。恢复合同应保留足以重建部分块的身份：块号、槽位置、阶段及整数quota、置换key/版本、子数据映射；否则需要显式的部分块交接方案。这里讨论的是原块采样机制，不是要求所有预训练系统都保存一个庞大已读token集合。

### 4. 不能把边界简单改成393

原混合类在普通解释器下assert阶段起点必须是整块倍数。将第二阶段直接移到393，402,432不能被49,152整除，本轮原构造函数拒绝。这个检查是源码assert，本轮没有禁用assert，也没有据此推断生产解释器选项。

一个合法的未执行对照是先保持原配比，完成块8，再于step432开始新配比。原函数复算中，它保留整个旧块8的映射；但恢复后还要走39次旧配比更新，改变了干预时机和累计曝光。它不是等价替换，更不是本轮已经验证的训练修复。另一方向是显式交接部分块及每桶映射；同样需要真实读取与续训验证，不能只加一个cursor字段就宣称解决。

实验设计应把两种变化分开：配比本身的收益，与数据前缀交接策略的影响。对照必须使用同一实际恢复状态，并记录部分块身份、供体预算、后续曝光补偿和固定评估；否则一次“从同checkpoint出发”的比较仍可能包含不同数据历史。当前80次删域结果不能因本轮条件反例而全部判无效，也不能把它们的BPB差异归因给这段边界差异——尚无这种训练因果证据。

12项源码/条件控制通过。1份新公共来源使归档达到574份；既有非账本来源保持。实际checkpoint内容、历史key、真实重复token数量、loss影响和桥接续训结果仍未知。


## V122：历史源码与恢复证据

V122已取得生产与一个续训run上传的历史混合器快照，并用该源码重跑80组条件复算，结果与V121一致。该续训日志明确报告读取step-393并进入后续训练；实际状态内容、导入路径和token仍未确认，不能推广为80个run均已核查。[源码、日志与证据层次](RUN_CODE_PROVENANCE_ZH.md)。
