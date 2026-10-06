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

配置顺序也需要保留：归档train_sets先拆train/val，再做训练shuffle，然后按experiment_budget/target_budget截断，最后按max_train_batches截断。这些是方法体内的分支顺序，不代表各字段能同时配置：原__post_init__禁止模拟预算与num_validation_sequences/max_train_batches并用。预算截断作用在shuffle后的逻辑前缀；改预算、窗口或缓存长度可能同时改曝光身份。最新声明experiment_budget和target_budget均为None，因此这里只给出配置迁移的核查位置，没有把预算截断归因于Hero曲线。

对自己的实验，可以把“数据身份”记录成一条可重放链：缓存内容与长度 → 切分身份 → 训练shuffle算法/key/窗口 → 截断范围 → 混合域ID/配额 → 恢复next offset → 实际token/hash。验证集固定后，数据增长应明确采用冻结留出清单或重新定义评估版本；重新切分的曲线不能自动与旧曲线作同样本比较。若样本内容可重复，仍需文档/token层去重检查，序列索引不相交只是一层条件。

[14组CPU与源码控制](analysis/inner_shuffle_cpu.json)包含130个PRP小域检查，以及完整人工排列、两份切分身份、交集、配置声明和四份源码SHA。[脚本](scripts/probe_inner_shuffle_cpu.py)使用已有[CPU依赖](requirements-cpu-numerics.txt)；执行`make inner-shuffle CPU_PYTHON=/你的环境/bin/python`。实际Hero inner shuffle、split leakage、token store和GPU/TPU结果仍为空。


## V55：小规模实验的库存缩放，是否真的保持重复曝光率

配比实验常希望用较小训练预算，模拟生产训练的每域曝光轮数。归档`LmDataConfig.train_sets`有对应接口：当experiment_budget和target_budget都非None时，令r为二者比值，在train/val切分和训练shuffle之后，对每域保留`int(length×r)`条逻辑序列，再应用可选max_train_batches截断。[原方法](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/lib/levanter/src/levanter/data/text/datasets.py)

此次执行原train_sets函数体，复用前两节的原PRP、slice与mixture类体。同时执行原__post_init__约束。缓存构造返回人工identity store，AsyncDataset提供人工同步长度桥；生产key_iterator替换为明确记录的fold_in key序列。没有执行完整LmDataConfig初始化、真实tokenization或模型训练，不能据此确认生产key分配或收益。库存单位均为人工序列，不是loss有效目标或独立文档。

人工参考预算96个混合槽位，A/B各占一半；小实验预算24，r=1/4。原mixture返回两域参考各48次、小实验各12次，名义曝光比例确实缩小了四倍。但库存截断改变了重复率：

|原函数控制|结果|含义|
|---|---|---|
|A库存7，B库存23，r=1/4|保留A:1条、B:5条|逐桶取整，不是整套库存统一抽足25%|
|参考A曝光48/库存7，小实验A曝光12/库存1|6.857轮变12轮，相对高75%|名义预算比正确，A重复率仍不匹配|
|参考B曝光48/库存23，小实验B曝光12/库存5|2.087轮变2.4轮，相对高15%|同一个r对不同域产生不同取整误差|
|A库存3，B库存23，r=1/10，A/B仍正权重|A保留0条，原restart混合拒绝空有限dataset|预算比大于0不保证每个活跃域可训练|
|先shuffle23条，再r=1/4|保留真实原排列的前5条，不是原库存0–4|预算截断选择的是当前shuffle后的逻辑前缀|
|绕过初始化：23条先拆4条validation，再r=1/4|方法体内训练库存19条再截成4条|正常__post_init__拒绝这组字段；只验证函数分支，不是可用配方|
|绕过初始化：23条r=1/2，再max_train_batches=2、initial_batch=4|方法体内先截11条，再截8条|正常__post_init__拒绝此组合；未声称生产可以两种cap并用|
|experiment_budget大于target_budget|原方法ValueError|已有顺序约束|
|experiment_budget=0且target_budget=0|原__post_init__未拒绝；随后方法出现ZeroDivisionError|原初始化与方法边界控制；未核对外层配置解析/launch检查|
|max_train_batches存在，但无initial_batch或要求超过库存|原方法assert拒绝|上限按initial_batch换算序列数；不等于动态batch历史累计|

**先验收完整入口，后解释函数内部。** 原__post_init__明确要求：如果max_train_batches或num_validation_sequences非None，则experiment_budget和target_budget必须都为None。检查器正常调用先执行该原约束；上表两个组合另行显式绕过初始化，用来核对方法分支，并保存初始化拒绝结果。不能把这些分支控制推广成合法配置。

连续理想条件是：曝光E变为rE，库存A变为rA，因此E/A不变。代码库存为n=floor(rA)，当n大于0且名义曝光恰好按r缩放时，重复率的相对倍数成为`rA/n`。差异为`(rA−n)/n`，小库存更敏感；实际混合整数配额和部分边缘块还可能造成曝光端的额外偏差。若n=0，不能再用这条除法估计“轮数”，需要先解决可行性。

不要把所有0库存自动抬到1当作修复：这会另改小实验的库存占比、重复率和内容选择。评审应先拒绝或显式重新设计此条件，保留实际保留数与误差；可选择增加pilot预算、调整缩放方案或单独分析稀有域，但每种方法都改变实验合同。即使取整误差很小，缩小库存也减少内容多样性，配比排名仍可能随模型规模、训练阶段、重复次数和跨域迁移变化，不能把“曝光轮数接近”当作生产收益已证明。

据此做配比研究，至少拆开两类干预：在共同库存上改权重，测新增曝光和供体退步；在共同权重上改库存/去重，测内容多样性与重复曝光。然后固定同一评估身份，在额外预算上确认候选，并用独立随机性观察排序是否稳定。若同时改权重、库存截断、shuffle窗口或恢复状态，loss差异只能归于这个完整组合，不能单独归于新配比。[供体与预算](TRANSFER_GUIDE_ZH.md) · [选择确认](CHANGE_REVIEW_ZH.md)

最新10月7日归档的experiment_budget与target_budget均为None。这节提供小实验设计和配置迁移的检查，**不是Hero实际发生预算截断或重复率事故的记录**。也未测模型loss、真实每桶库存或生产重复率。

[16项原CPU与源码控制](analysis/budget_inventory_cpu.json)保留人工参考/小实验输入身份、两域曝光、截断顺序、错误类型与依赖SHA。[脚本](scripts/probe_budget_inventory_cpu.py)要求已有[CPU依赖](requirements-cpu-numerics.txt)，执行`make budget-inventory CPU_PYTHON=/你的环境/bin/python`。它使用原方法与显式依赖适配；实际pilot模型结果、生产重复率和token store仍为空。
