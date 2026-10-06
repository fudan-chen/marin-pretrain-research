# PTB为何变化：先确认评估是否在重抽文本

10月6日的PTB评估先升后降，对宏平均端点改善贡献很大。[上一轮观察](LIVE_2026_10_06_ZH.md)保留了样本身份未知。这轮追查固定源码后，可以收紧其中一种解释：**归档入口没有显示每轮随机重抽一批评估文本。** 它默认不shuffle，最新配置没有评估数量截断，每次loader默认从索引0重新遍历。

这不能直接证明历史运行使用完全相同的token：运行SHA、缓存内容哈希、分词/打包产物和真实分布式读流仍未绑定。也不能因为排除了一种常见解释，就把PTB变化归因到配比、kernel或能力。

依据包括[评估入口](sources/scale_2026_10_05/train_hero_ep.py)、[TaggedEvaluator](sources/scale_2026_10_05/eval.py)、新归档的[固定loader](sources/eval_identity_2026_10_06/loader.py)、[最新配置](sources/live_2026_10_06/meta.json)和[21项CPU核对](analysis/eval_identity_probe.json)。函数片段使用人工有限dataset、NumPy标签和批次规划记录器；不执行后台读取、JAX排列、多卡数组或模型评分。静态检查与原方法执行分别记录，没有把固定版本冒充历史执行版本。

## 1. 数据从哪里来、为什么预期会重复

|路径|源码/配置事实|能够排除什么，仍缺什么|
|---|---|---|
|选择validation|最新声明`num_validation_sequences=None`；PTB和twitterAAE组件为validation缓存|这份声明没有每轮从train随机划分validation；真实缓存内容未取回|
|标记评估域|`tagged_eval_sets`把组件名加入tags；DomainTaggedDataset按组件长度建立拼接offset|域名对应这份声明的组件；实际组件顺序与内容仍需绑定|
|限制评估样本|`max_eval_batches=None`；非null时乘eval_batch_size得到**每个dataset**的样本上限|当前没有该数量截断；这个字段不是全局评估batch上限|
|shuffle|builder未传shuffle，TaggedEvaluator默认False；启用时使用固定seed 0|默认路径没有随机重抽；固定seed仍依赖数据长度、顺序和排列实现|
|每次开始位置|`evaluate`新建[计时包装器](sources/eval_identity_2026_10_06/logging.py)，其构造函数立即调用iter(loader)；loader的`__iter__`调用`iter_from_step(None)`，规划从0开始|默认不会接续上一轮的末尾cursor；真实运行是否改过入口仍未知|
|评估结束条件|遍历loader直至数据耗尽，没有用`len(loader)`截断循环|进度条total不是评分次数控制器；真实分布式读取和评分未重放|

[validation构造源码](sources/deepening_2026_10_04/datasets_production.py)先建立validation token datasets；配置存在`num_validation_sequences`时，才会从train拆出指定数量替换相应validation。本次配置为None，不应把训练集的shuffle配置自动解释成评估集每轮重采样。

PTB声明路径为`s3://marin-us-east-02a/marin/paloma/ptb-llama3/2026.06.28`，twitterAAE也是同日的相应组件路径。这只是地址身份，不是不可变内容承诺。配置的cache_catalog也是None；当前没有可绑定的内容哈希，不把日期目录当作哈希。缓存与分词的其他边界见[缓存身份](CACHE_PROVENANCE_ZH.md)。

## 2. 原方法在人工数据上的行为

用A=[0,1,2]、B=[0,1,2,3]两份人工dataset调用原DomainTaggedDataset方法，拼接offset为[0,3,7]。请求[6,0,3,3,2]时，内部排序、分组读取，再恢复原请求顺序；两个3保留为重复读取B的第0项，tag与对应样本保持配对。这个测试核对批量映射机制，不证明真实文本或I/O一致。

|人工输入与原方法执行|实际得到什么|意义|
|---|---|---|
|两域长度3和4，不设cap|总长度7、边界3进入B的第0项|拼接索引与域标签对上|
|每域cap=2|A前两项+B前两项，总长4|数量限制作用于每个域的前缀，不是随机子集|
|长度10、batch4、默认末批策略|[0..3]、[4..7]、[8,9]|所有逻辑项恰好一次；两次默认规划相同|
|同样输入，fetch grouping从3改为1|逻辑流仍为[0..9]|这份人工规划中后台抓取分组不改变索引顺序|
|显式从batch1开始|[4..7]、[8,9]|显式恢复会改变起点；不能与默认评估混同|
|关闭末批padding|只保留两个完整逻辑batch|末批策略影响样本覆盖，必须进入评估身份记录|

生产默认会pad末批。原padding函数调用`tree_zeros_like`，覆盖example树，意图让额外的tokens、loss_weight与tags为零；本轮只静态核对该调用，没有运行Optax/JAX零化或验证多卡padding。探针输出的[8,9]是**逻辑样本范围**，不是已经构造的4行设备tensor。

## 3. 确认一个进度条问题，也说明它为何不解释PTB

固定loader的`__len__`使用：

```python
find_step_containing_offset(total_length) + 1
```

当长度N正好整除固定batch B时，索引N已经属于下一批，所以返回N/B+1。原规划函数在到达数据末尾时不会生成这批：本轮长度8、batch4的人工例子，`len(loader)=3`，实际只产生2批。

|长度/batch|`len(loader)`|原规划实际批数|可以报告的影响|
|---|---:|---:|---|
|8/4|3|2|进度总数多一批|
|10/4，保留末批|3|3|这个例子总数正确|
|10/4，不保留末批|3|2|总数也不反映丢弃末批策略|

TaggedEvaluator把这个长度交给tqdm显示total，实际`for batch in iterator`一直遍历到耗尽。**本轮没有发现因此漏评或多评一个真实batch的证据。** 把“进度条未到100%”当作评估遗漏，或者拿这个问题解释PTB大幅变化，都超出了源码和人工执行能支持的范围。

对固定batch且保留末批的长度估计，常见的正确计数是ceil(N/B)；关闭末批时为floor(N/B)。调度batch会变化时，应另用累积offset设计计数，不能直接套这个固定batch公式。本章没有修改上游loader或在生产应用修复。

## 4. 现在怎样调查PTB，才会改变结论

本轮把下一步从“猜测是否随机重抽”推进到绑定以下对象：

1. **文本与token流**：固定validation缓存ledger/内容哈希、tokenizer身份、打包长度与边界策略、域组件顺序，记录每域实际评分样本数与loss分母。
2. **模型视图与执行**：checkpoint、参数/EMA、pending router状态、compute dtype、dropless backend与执行SHA。最新配置EMA声明为None；不能只凭EMA关闭就证明两次前向相同。
3. **重复评分**：同checkpoint、同输入流连续评估两次，比较每域CE/BPB和逐批N/T/B。逻辑流相同但分数不同，应优先调查前向状态、数值与分布式行为；输入流不同，先调查数据身份。
4. **后续轨迹**：绑定成功后再比较后续同step窗口。重复评估核对可重复性，多个checkpoint检验训练轨迹；二者不能互相替代，也不能当成独立seed。

这些材料当前没有结果。本轮能支持的是“归档默认路径设计为重复遍历，存在一个进度长度估计问题”；不能支持“PTB已经证明可重复改善”或“只有优化器/配比导致变化”。

这项检查可以加入既有配比管线的评估冻结阶段：冻结的不只是指标名称，还包括域级实际输入和分母；先检验同checkpoint重评，再让域端点进入候选排序或保底判定。原方法可用[探针脚本](scripts/probe_eval_identity.py)离线重跑，源码与配置的哈希保留在原值账本中。
