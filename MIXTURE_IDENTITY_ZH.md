# 混合数据的映射身份：同key、同配比与实际样本

## 1. 同key、同配比为何仍不一定是同一条数据流

前文的逻辑索引合同依赖“共同数据映射”。这轮把这项条件拆开验证：执行归档`MixtureDataset`的原类体，以及原`_compute_block_assignment`，使用真实JAX 0.7.2 CPU的`fold_in`与`permutation`。AsyncDataset基类、StopStrategy容器、local_cpu_mesh和立即完成future替换成最小依赖，子dataset是有限identity store。没有导入完整Levanter模块，也没有执行实际inner shuffle、packing、token store、GPU或checkpoint恢复。[原混合源码](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/lib/levanter/src/levanter/data/mixture.py)

人工block_size=8，A/B各占一半；每域身份为`域名:库存索引`。首先核对同一窗口拆成不同get_batch请求、非排序请求与重复索引：返回身份都按请求对应。说明下面差异来自映射条件变化，而不是仅改变取数请求分组。

|改变的条件|保持的条件|原类执行结果|可以解释什么|
|---|---|---|---|
|key由7改为8|A/B整数配额、库存、24个混合索引|完整3块的身份多重集相同，顺序不同；前3槽的多重集不同|完整块和边缘窗口需要分别核对|
|dataset插入顺序A/B改为B/A|同key=7、各域每块4条、相同库存与24索引|完整窗口身份多重集相同，排列不同|只有名称权重和key，还不足以重建相同序列|
|等权A/B/C的插入顺序改为C/B/A|同key、block=8、三个浮点权重相同|ABC计数4/2/2，CBA计数4/2/2；余数接收者由A变C|整数余数与并列argmax还依赖有序dataset_index|
|每域库存长度3改为4|同key、配比、24混合索引|取模后的身份不同|库存长度是映射身份的一部分，不能只保存dataset名称|
|同长度库存的内部映射反转|同混合索引、key、配额|所有子索引读取另一组人工身份|相同cursor不能代替内容版本/hash|
|从混合索引16起只读A|前16槽A/B各一半、key与库存|前16槽逐项相同，后8槽读取A:8至A:15的一个排列|阶段变化仍承接已消耗的域内累计offset，不从A:0重新开始|
|有限库存为空|restart策略|原路径抛出空有限dataset的ValueError|空库存应在训练前验收，不用有限loss替代数据条件|

插入顺序进入两处计算。构造器从过滤后的datasets字典建立`dataset_index`，打包ID的高16位是该有序列表的位置；先按此顺序生成未排列数组，再执行真实JAX排列。因此，同一个随机key只是对当前数组排列的控制，不能保持更换数组后的域身份。计数规则则先截断每域的weight×block_size，再把余数加给计数最大项；出现并列时`np.argmax`取最先项，插入顺序会决定余数归属。

不能在恢复时直接“统一排序字典”作为修复：排序本身可能改变既有映射。应保存实际有序dataset_index和打包数组摘要；新规则在新配方中明确版本化，并用同checkpoint输入重放核对。这里的小块等权反例没有外推成生产200桶的偏差幅度，也没有认定Hero字典顺序实际发生过变动。

有限restart子域的逻辑索引经`index % async_len()`落到库存。这次24槽、A/B各12次，库存各3条时只有6个不同身份，每个身份出现4次。总曝光24与独特库存6是两个量；相同总曝光可能包含更多重复，不能仅由loss下降判断增加了多少新知识。人工store的身份不是文档、token或独立语义样本；真实去重、序列切分与inner shuffle还需绑定。[去重与库存](DEDUP_FILTERS_ZH.md)

配比/顺序评审因此新增五项：实际有序域列表；过滤后支持范围与每块整数计数；未排列打包ID和key摘要；子域有限性、库存长度与内容版本；窗口内实际身份/token hash及重复分布。完整块可比较多重集，边缘块还要比较实际读取槽位。报告14项检查只证明人工共同映射下的这些关系；真实Hero映射仍为空。

[14项真实CPU与源码控制](analysis/mixture_identity_cpu.json)保留两组完整排列、部分窗口、并列计数与取模流。[脚本](scripts/probe_mixture_identity_cpu.py)使用已有[CPU依赖](requirements-cpu-numerics.txt)，执行`make mixture-identity CPU_PYTHON=/你的环境/bin/python`。它承接[加载器恢复](BATCH_CLOCK_ZH.md)，把“从哪里恢复”继续追到“同一个位置究竟读到什么”。


## V54：桶内排列和训练/验证切分还依赖什么

上节的identity store固定了子域内部映射；这次继续执行原`PermutationDataset`、`SlicedAsyncDataset`、`BlockShufflingDataset`类体与原`_split_into_trainval_sets`函数，并补取同一固定代码版本的[_prp.py](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/lib/levanter/src/levanter/data/_prp.py)。PRP函数与类体原样执行，真实JAX CPU负责生成key和种子，原NumPy实现负责Feistel/linear索引映射。替换AsyncDataset基类及slice构造接口、CPU mesh上下文和有限identity store；不是完整模块导入或实际缓存读取。[新增源码来源](analysis/prp_acquisition.json)

|控制|原代码执行结果|对实验的含义|
|---|---|---|
|Feistel与linear，长度1至65|130个小域控制全部是双射|核对这些域内无重复/漏映射；不是对任意库存长度的证明|
|长度22，IO block=4，window_blocks=3|映射完整覆盖0–21；最后2槽只含20、21|尾块保留在最后，只在尾部内部排列；不能把这种层次shuffle当作全库存任意排列|
|同长度、key、窗口设置，独立构造|映射逐项相同；重复/乱序请求按映射返回|在此固定人工快照下可重建，不代替真实缓存身份|
|window_blocks从3改为2|同key和库存下，整体序列改变|窗口配置属于数据流身份，不只是IO性能参数|
|库存长度22变23|尾部映射改变|数据增长会改变排列范围；相同key不能固定所有已有槽位|
|同一22条快照，从训练库存拆4条validation|独立构造得到相同切分；train/val互斥并覆盖0–21|固定key在相同快照上确实支持这条接口合同|
|用23条快照重新切validation|新validation与旧train交集为人工身份1|同seed跨快照不保证留出身份不变；这是可复现控制，不是实际Hero泄漏|
|关闭split前shuffle|最后4条18–21进入validation|顺序结构会决定留出内容；随机与位置切分需要分别记录|
|负索引和长度端点22|原block shuffle分别抛ValueError/IndexError|索引边界有明确拒绝，不能静默当作restart取模|

**最新10月7日归档声明`num_validation_sequences=None`。** 归档LM数据构建代码只有设置该字段，才从train库存拆出validation；这次人工跨快照交集不能用来认定Hero训练污染了验证集。Paloma的外部语料重复与去重问题是另一条证据链，也不能由索引互斥证明不存在。[评估身份](EVAL_IDENTITY_ZH.md) · [去重范围](DEDUP_FILTERS_ZH.md)

切分函数使用固定key=0，先对当前库存长度建立Feistel映射，再按当前length−num_validation_sequences切片。它保证独立构造train和val时采用相同排列，前提是两次看到同一库存长度、顺序和映射实现。数据快照改变后，排列域与切分边界都可能改变；“仍使用seed 0”不足以保持验证身份。人工交集只说明旧train/new-val的身份重叠，不是同一次切分内部重叠。

层次block shuffle先打乱完整IO块，再在若干块组成的窗口内排列样本；最后不完整块保持在末尾。这是IO局部性与排列方式的选择。如果训练预算只覆盖一段前缀，就应检查这个前缀实际覆盖了哪些文档、质量桶与尾部；不能只比较完整库存计数。这次仅验证索引身份，没有测磁盘吞吐、缓存质量分布或模型loss。

配置顺序也需要保留：归档train_sets先拆train/val，再做训练shuffle，然后按experiment_budget/target_budget截断，最后按max_train_batches截断。预算截断作用在shuffle后的逻辑前缀；改预算、窗口或缓存长度可能同时改曝光身份。最新声明experiment_budget和target_budget均为None，因此这里只给出配置迁移的核查位置，没有把预算截断归因于Hero曲线。

对自己的实验，可以把“数据身份”记录成一条可重放链：缓存内容与长度 → 切分身份 → 训练shuffle算法/key/窗口 → 截断范围 → 混合域ID/配额 → 恢复next offset → 实际token/hash。验证集固定后，数据增长应明确采用冻结留出清单或重新定义评估版本；重新切分的曲线不能自动与旧曲线作同样本比较。若样本内容可重复，仍需文档/token层去重检查，序列索引不相交只是一层条件。

[14组CPU与源码控制](analysis/inner_shuffle_cpu.json)包含130个PRP小域检查，以及完整人工排列、两份切分身份、交集、配置声明和四份源码SHA。[脚本](scripts/probe_inner_shuffle_cpu.py)使用已有[CPU依赖](requirements-cpu-numerics.txt)；执行`make inner-shuffle CPU_PYTHON=/你的环境/bin/python`。实际Hero inner shuffle、split leakage、token store和GPU/TPU结果仍为空。
