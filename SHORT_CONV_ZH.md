# ShortConv：文档边界、分片halo与数值验收

attention不跨文档，不能直接推出整个模型不跨文档。Hero还在K投影、attention输出和MLP输出上执行ShortConv。六份ladder配置均声明开启这三个位置，kernel width为4。模型把query侧segment IDs传给卷积，卷积再独立处理文档边界。[模型调用](sources/contracts_2026_10_05/model.py)

本章归档六份固定实现与原测试，执行16项检查，包括原FP32 reference、原halo函数体与API选择逻辑。NumPy替代JAX，shard_map/ppermute由单进程模拟，sharding guard不执行；没有运行Pallas、BF16、梯度或真实分布式通信。[结果](analysis/short_conv_probe.json)

## 1. 它怎么隔离文档

每个channel独立做因果卷积：当前位置乘weight[0]，此前W−1个位置分别乘后续tap权重并相加。对于lag>0，只有当前与被读位置的segment ID相同，tap才保留；序列之外补零。lag=0始终保留。[reference](sources/short_conv_2026_10_05/reference.py)

人工单channel输入`[1,10,100,1000,10000,100000]`，三tap权重全为1；这是width=3的可读算例，不冒充width=4历史输入：

| segment元数据 | 原reference输出 | 含义 |
|---|---|---|
| 无segment IDs | `[1,11,111,1110,11100,111000]` | 因果位置可以混合；不能靠attention的独立mask代替 |
| `[0,0,1,1,2,2]` | `[1,11,100,1100,10000,110000]` | 每个新文档的首位置不读旧文档 |
| `[0,1,0,2,2,2]` | 第三个输出101 | t2重新读到同号的t0，即使中间出现另一个文档 |

第三行是故意违反连续文档编号语义的反例。函数只看tap两端ID相等，不检查中间是否跨过边界；所以一个序列内重新使用旧ID会产生错误连接。它不是已观察到的Hero数据故障，实际loader编号是否满足约束需要独立核对。[文档构造与attention边界](DOCUMENT_BOUNDARIES_ZH.md)

编号可以在不同batch样本之间重复，因为卷积不沿batch读数据；同一全局序列跨context shard时，则必须保留一致的segment语义。不能在每个shard里从零重编号，并假定halo交换仍能判断文档身份。

ShortConv初始化为weight[0]=1、其余tap=0。这个状态下，正确mask与缺失mask都输出原输入，因此只测初始化会漏掉边界问题。探针使用非零历史tap来暴露差异。源码中lag=0不依赖segment值，即使位置ID是−1也可能保留该位置激活；padding是否退出语言目标还要查query/loss mask，不能指望卷积清零。

## 2. context shard怎样保留左侧上下文

[halo路径](sources/short_conv_2026_10_05/api.py)在每个序列shard前拼接上一shard末尾W−1项：激活和segment IDs一起传。rank 0没有前驱，激活补零，segment补OOB_SEGMENT=−1。GPU tile需要时在右端补齐；卷积后丢弃左halo与右padding输出，只留下本地原位置。

右padding不会改变此前保留输出，是因为算子因果；不能把这个结论推广到非因果卷积。halo过大则仅取相邻shard的尾部还不够，所以API要求每shard序列长度至少W−1，并检查全局长度能被序列轴分片数整除。

探针执行原halo函数体，模拟两个序列shard：

| 输入情形 | 检查结果 | 证据范围 |
|---|---|---|
| 未打包连续序列 | 与池化reference相同 | FP32/单进程模拟 |
| 同一文档穿过shard边界 | 与池化reference相同 | 同上；不是实际ppermute验证 |
| 文档边界在shard内部 | 与池化reference相同 | 同上 |
| 文档边界正好在shard起点 | 与池化reference相同 | 同上 |

故意去掉halo的错误对照，在连续文档的第二shard首位置只得到1000，正确三tap结果为1110。漏传激活会丢上下文；漏传或重编号segment则可能错误丢弃/接受tap。必须把二者作为同一元数据协议检查。

原API还拒绝channel分片和未声明的batch分片，以避免隐藏all-gather；本轮以no-op替代这些guard，没有验证真实sharding布局和通信代价。通过halo算术检查不等于通过分布式执行验收。

## 3. fused kernel为何不能随意改计算顺序

reference按lag升序逐次乘、逐次加。BF16下每次乘加的舍入位置都会影响结果。Pallas的`exact_reference_rounding=True`保留参考算子的逐操作舍入，而不是简单写一个FP32累加器再在最后转回BF16。[GPU实现说明](sources/short_conv_2026_10_05/pallas_gpu.py)

源码为不同输出设置了不同验收范围：

| 对象 | 固定源码给出的数值范围 | 本轮实际验证 |
|---|---|---|
| BF16前向，exact开关开启 | 与reference逐位相同 | 未运行BF16/Pallas |
| 序列不分片的BF16 dx | 与reference逐位相同 | 未运行自动微分或GPU反向 |
| 序列分片边缘的dx | 本地与邻居贡献分开舍入后相加，可不同 | 未运行真实边缘梯度 |
| dw | 跨token求和的结合顺序不同，按误差验证 | 未执行原GPU测试 |
| FP32前向/梯度 | 编译融合/重结合可造成误差 | 仅原reference与模拟halo的FP32算术 |

`exact_reference_rounding=False`保留FP32跨tap累加，源码称其数值更准确，但不再逐位匹配reference。更接近高精度参照和保持旧训练轨迹是两种验收要求；改开关前应先决定要维护哪种契约。

原测试把dw与FP64定义式参照比较，而非要求其与旧dw逐位相同；它检查kernel误差不明显劣于reference。本轮归档该测试但没有执行，不把“仓库有测试”写成“真实环境测试通过”。[原测试](sources/short_conv_2026_10_05/test_short_conv.py)

## 4. 后端回退与dtype必须显式记录

API对weight与x不同dtype直接报错，避免reference的类型提升与Pallas的输出dtype产生不同结果。模型调用方是否把它们放到相同dtype，仍须结合实际mixed precision路径检查。

明确指定字符串`pallas_gpu`时，不可用会失败；传`(pallas_gpu, reference)`时可以警告后回退。默认有GPU支持时先试Pallas，否则reference。探针以“GPU不可用”替代依赖，分别验证失败与回退路径；没有确认当前生产后端。

因此，run名字或调用处未改不代表实际执行了相同后端。比较吞吐必须保留所选实现、警告、形状与tile设置。reference回退可以恢复可运行性，但不能自动当作已复现原吞吐。

## 5. 配比与packing实验怎样使用这些检查

配比改变文档长度和packing方式，可能改变文档起点密度、有效tap数量、halo边界与实际计分位置。固定模型宽度并不意味着相同卷积上下文；这些变化可能属于完整数据干预，也可能是无意改变的样本构造。

| 核查规则 | 保存什么 | 失败后先做什么 |
|---|---|---|
| 文档编号在全局序列内一致 | 实际segment映射、文档起点与重用检查 | 回到样本构造，不能只修attention mask |
| 激活与segment halo对齐 | shard尾部/首部样本身份、halo长度 | 用共同输入检查两类halo，不立即调配比 |
| 测试不能只用identity init | 非零历史tap、短文档与边界输入 | 补能触发跨边界读取的算例 |
| 前向、dx、dw分开验收 | dtype、exact开关、误差口径与边界分层结果 | 根据目标冻结容差，避免逐位结果互相代填 |
| 后端与回退有记录 | 实际backend、warning、tile、资源 | 回到实现可用性与成本检查 |

这些规则接到[loss诊断流程](LOSS_TRIAGE_ZH.md)的样本与执行两站。本轮确认的是固定源码的边界/halo机制和人工反例，没有证明真实训练跨文档泄漏、context通信错误或能力退步。
