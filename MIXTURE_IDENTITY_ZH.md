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
