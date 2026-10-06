# 打包字段对齐与权重坐标

最终数组shape相同，不代表输入token与权重仍属于同一篇文档。同名文档索引也不够：逐token字段必须在每篇文档中保留相同长度和坐标。固定打包器在人工存储接口上接受了一个“总长度相同、逐文档长度不同”的输入；它符合通用多字段打包的约定，却违反了token与权重逐位置对应的要求。

这不是当前Hero故障结论。10月6日公开配置快照的223个component均声明文本格式、`pack=None`，固定构造实现选择连续流。以下结果用于准备自定义打包、权重与旁路评估时的验收。[配置快照](sources/live_2026_10_06/meta.json)

## 通用接口与调用方不变量

`pack_documents`按同一组文档范围组织各字段，并检查字段的文档数量一致；每个字段用自己的长度与offset读取、生成segment。通用多模态或多字段数据不必具有相同序列长度，因此不能无条件要求所有字段相等。若某字段是另一个字段的逐token权重或标签，则调用方须另外保证其逐文档长度及顺序一致。[固定源码](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/lib/levanter/src/levanter/data/packing.py)

|接口层|实际核查或使用的量|不足以证明的性质|
|---|---|---|
|通用pack规划|各字段文档数量、各自容量、segment数量上限|逐文档token/权重长度相等|
|各字段数据重建|各自offset、切片、padding及segment|不同字段的同一数组位置属于同一文档|
|`PackedTokenDataset`调用方|输入字段segment，另一个字段的权重值|权重字段原有segment是否与输入一致|
|LM例子因果mask|窗口末位及padding后继等规则|权重在有效位置是否已经错位|

`PackedTokenDataset`将`seg_ids['input_ids']`与原权重数组交给LM构造器，未在此处比较权重字段的segment。[调用实现](sources/deepening_2026_10_04/datasets_production.py)

## 相同shape如何遮住错位

人工输入有两篇文档：输入长度为`[3,1]`，正确权重长度也是`[3,1]`；错误权重长度为`[2,2]`。打包容量设为4，两组字段都能装下。输入内容是`[10,11,12]`和`[20]`，数值仅用于标识人工位置。

|输出|正确对应|错误对应|
|---|---|---|
|输入token|`[10,11,12,20]`|`[10,11,12,20]`|
|输入segment|`[0,0,0,1]`|`[0,0,0,1]`|
|权重segment|`[0,0,0,1]`|`[0,0,1,1]`|
|原权重|`[1,0.5,0,1]`|`[1,0.5,1,0]`|
|应用原末位mask后|`[1,0.5,0,0]`|`[1,0.5,1,0]`|
|有效权重和T|1.5|2.5|
|人工loss固定为`[1,2,9,8]`时的N|2|11|

原方法在两种情况下都产生同shape输出。错误输入将第二篇文档的权重放到了第一篇文档末位；默认末位mask只屏蔽整个窗口末位，不能重建权重原来的文档坐标。N的差别来自人工权重和人工loss，不是模型前向结果，也不是生产精度误差。这里的N/T按评估加权定义计算；训练目标对分数权重的处理还须另查[损失接口契约](IMPLEMENTATION_CONTRACTS_ZH.md)，不能直接混用。

已有[下一token契约](EVAL_TARGET_ALIGNMENT_ZH.md)处理的是评分ID与输入后继的关系。即使目标ID都正确，权重仍可能选错目标。导出器会把权重纳入摘要，两次权重改变能被发现；但两次都使用同样错误的权重，仍需要调用方的坐标证据才能诊断。

## 截断策略必须进入曝光账目

人工过长文档`[10,11,12,13,14]`在容量4时，left保留前四个位置，right保留后四个位置；已正确对齐的权重沿同方向切片。raise在pack规划时拒绝过长文档，drop跳过整篇，并保留后续文档原来的索引。这些原方法行为已在人工存储上核查，但未读取实际TensorStore或运行GPU。

|验收对象|建议记录|影响|
|---|---|---|
|token对齐字段|逐文档length、document ID与特殊token插入位置|总length和最终shape相同仍可能错位|
|切片策略|原length、保留区间、丢弃区间、权重坐标约定|左截与右截可能保留不同内容或只留下prompt|
|drop策略|被丢文档的来源与length分布|配置比例不等于留下来的实际曝光比例|
|segment上限|每pack文档数、真实输入数、有效目标数|达到文档数上限时仍可能有较多padding|
|权重变换|缓存权重表示输入位置还是预测位置，是否已shift|重复shift或漏shift都可能选择错误目标|

固定chat路径会把assistant token选择mask向左roll一次，再交给因果构造器；prebuilt路径则通过可选`loss_weight_transform`处理权重，不能假定所有缓存采用同一种坐标。[数据构造实现](sources/deepening_2026_10_04/datasets_production.py)

建议在进入通用packer前，对明确声明为token对齐的字段比较逐文档length和ID；加入BOS/EOS后再次核对；抽样导出输入、后继目标、原权重、变换后权重及各字段segment。若长度或ID不一致，先修复缓存，不能以补齐到同shape代替坐标恢复。不同语义字段可具有不同长度，这项约束须限定到逐token对应字段。

## 验证范围

[探针](scripts/probe_parallel_packing.py)执行固定`pack_documents`、`GreedyPrepackedDataset`方法和原因果mask，共14项检查，含匹配/错位字段、padding、左右截断、raise、drop、segment上限及223个声明配置。PyTree限于单层字典，TensorStore读写由人工NumPy存储、offset结果与异步副本替代；严格zip由兼容适配器处理。没有实际TensorStore读、JAX设备计算或Hero字段错位证据。[原值结果](analysis/parallel_packing_probe.json)

这轮得到的规则是：**逐token字段按文档检查坐标，不以最终shape或相同loss摘要作为对齐证明。** 它为自定义配比与评分管线增加验收入口，没有给出新的训练配方收益。
