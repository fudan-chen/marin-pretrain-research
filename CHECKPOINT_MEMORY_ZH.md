# Checkpoint内存：预算、写入分摊与训练停顿

上一章分清了保存交接与真正完成。这一章追问：为什么保存时host内存会升高，为什么同一份模型可能有少数rank成为写入瓶颈，以及减小staging预算会付出什么代价。结论来自固定源码和小型CPU检查，尚没有真实Hero写入计划、RSS或GPU切片测量。

## 1. Host byte budget控制的是在途快照，不是整个RSS

[HostByteBudget](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/lib/levanter/src/levanter/utils/byte_budget.py)在一个event loop上累计in-flight bytes。非空时，新的请求若使总量超过limit，就等待释放；当前没有在途快照时，超出limit的单个请求允许独占进入，避免一个大请求永远无法开始。因此，这个limit是调度目标，不能当作绝对最大快照大小，更不能当作RSS硬上限。

原类直接在CPU asyncio中运行：limit=10时，6+6的第二个请求等待；释放后开始，peak为6。单个14字节请求在空预算下开始，peak为14，并阻塞后面的1字节请求。completion thread的release能唤醒等待任务；reserve中的异常也释放容量。这里的字节数是人工调度输入，没有分配14字节的模型快照，更没有测量RSS。

|量|源码中是什么|需要额外测量什么|
|---|---|---|
|`max_staged_host_bytes`|每个process的在途staging目标|实际配置、local process数量、超大请求|
|`gate.peak_bytes`|当前save在budget中累计的最大在途估算字节|各次stage是否按真实复制量计费，device slice结果|
|`_STAGED_BYTE_OVERHEAD=4`|日志估算host占用的乘数，源码说明仅用于reporting|实际copy/encode/cache/allocator占用；不能把4当已测出的恒定放大率|
|RSS|进程驻留内存，含训练基线、offloaded状态、快照、编码、缓存、allocator|当前RSS曲线、可达引用、allocator retained bytes和写入完成水位|

固定源码的默认target是32×512 MiB=16 GiB/process，cache pool soft limit为1 GiB，CPU copy concurrency为16。这些是代码声明，不代表已确认的Hero运行参数。四个local process若都用默认值，名义target合计64 GiB；源码的4倍日志估算对应256 GiB。这只是配置算术，不是节点实际峰值，也没有计入训练/offload基线；超大单请求和预算之外的内存还可能改变它。

## 2. 为什么预算小了，训练可能更慢

`issue_write`先acquire，再生成host snapshot。staging失败会release；正常路径将release挂到commit future的完成回调，无论成功或失败都释放。它不会仅因device-to-host复制结束就释放，因为TensorStore写入仍可能引用这份snapshot。

`asyncio.run(write_all())`需要把所有写入都安排完成才能返回。若一个process需要写的总量大于budget，后面的stage就要等前面的commit释放容量。所以“异步保存”仍可能在交接阶段等待一部分存储写入。减小budget可减少在途内存，但也可能延长训练暂停；增大budget则可能让保存更早交接，却提高与训练同时存活的host内存。[写入路径](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/lib/levanter/src/levanter/tensorstore_serialization.py#L704)

此外，上轮manager任务若未完成，本轮数组serializer会先等待它；这个等待与本轮budget等待是两个阶段。诊断时分别记录previous-commit wait、stage wait、copy、local commit和global callback，不能把总save耗时全叫“拷贝慢”。

快照必须与训练之后可修改或donate的buffer分离。固定代码对pinned-host数组使用`np.array(..., copy=True)`；GPU路径转到pageable CPU array，再做私有NumPy快照，并删除临时JAX staging array。直接复用训练buffer看似节省一次copy，却需要证明写入完成前训练不会修改它。本轮只读这段代码，没有执行GPU DMA、真实buffer donation或切片。

## 3. 副本多，不代表所有副本都写一份

`plan_array_write`只依赖路径、全局shape、sharding与配置，让各process得到一致的计划。它统计同一shard index的副本数量，寻找可整除的轴，将每个shard拆给若干副本写。如果最大片段太小、没有合适整除轴、副本数不均匀或pinned-host切片不满足安全条件，则回到单writer。

|计划条件|原函数行为|工程含义|
|---|---|---|
|可安全整分，且每个writer片段达到minimum|按选中的副本数切分第一个可整除轴|分摊同一个逻辑shard，不复制成多份存储|
|小数组，或不能整分/安全切片|`crc32(path) % replica_count`选一个副本写整shard|小参数按路径分散，但不保证每个rank总字节绝对均衡|
|非JAX host array|process 0写，其他process不写|host叶子可能集中在process 0，需单独计数|
|writer数小于现有副本数|只让被选writer写对应区间|不能用总副本数平均估计每个rank的工作量|

原planning函数用人工sharding检查：shape=(8,4)，4副本，minimum人为设为1字节，得到四段[0,2)、[2,4)、[4,6)、[6,8)，完整且不重叠。这个fixture把四个副本都设为同一process可寻址，因此该process合计写128字节；没有把它冒充四台真实设备各写多少。使用默认minimum=16 MiB时，同样的小数组不切分，仅一个副本写。pinned-host一维向量也按源码安全条件回到单writer。

所以需要从实际state树生成按rank的write-plan直方图：总写入字节、最大的整shard、最大的snapshot、单writer/副本切分比例。仅看模型参数量或集群rank数，无法判断最忙rank是否超过预算。当前没有生成Hero真实计划，不能从这个fixture认定其writer不均衡已是生产瓶颈。

## 4. Chunk目标也有一个整分边界

`_capped_chunk_shape`先把轴长度至少设为1，再反复把最长的偶数轴减半，直到chunk达到目标或没有可减半轴。保持整分可让writer边界落在完整chunk上；代价是“max_chunk_bytes”的名字没有在所有shape下兑现为严格上界。

|人工输入，float32|配置目标|函数返回|实际chunk字节|
|---|---:|---|---:|
|(64,64)|1024|（16,16）|1024|
|(33,33)|1024|（33,33）|4356|

还有一个容易混淆的大小：chunk是存储对象的分块，而stage请求按整个writer region的字节量计费并生成快照；一个writer region可以含多个chunk。因此，减小`max_chunk_bytes`不会自动把host快照切成同样的小块，仍需查看实际writer region和stage请求大小。

第二项不是Hero实测，也没有使用默认512 MiB目标。它证明的是通用接口边界：全为奇数且无法继续整分时，函数会返回仍超目标的chunk。HostByteBudget又允许超大单请求独占进入，因此不能把这两个配置字段合起来当作绝对内存/单写入安全保证。

迁移到自己的模型时，应先报告实际plan里超过chunk目标或staging目标的条目。如果运行要求严格上界，可以选择发起写入前明确拒绝不满足约束的shape；若要支持部分边缘chunk，则需要重新证明副本writer不会共享同一chunk并验证最终覆盖。直接把奇数轴随意截小，可能破坏此前的writer边界契约。本轮没有给作者代码打补丁，也没有验证一种新的分块算法。

## 5. RSS没下降，要从引用与allocator两层查

生产早期保存后的writer RSS上涨，作者追到manager已等待future却仍保留引用；随后还讨论pinned BFC pool。它们分别涉及对象是否可回收和内存是否退还OS。[对应事故与纠正](ENGINEERING_GUIDE_ZH.md#1-保存后内存变高先区分内存还活着还是留在分配器)

固定序列化代码在所有local commit future结束后才调度heap trim；环境检测到jemalloc时跳过这条glibc trim路径。这个安排说明“commit未结束”与“已结束但allocator保留”要分开观察。trim不是清除活跃buffer引用的替代品；RSS曲线本身也不能证明future引用已经释放。

建议一次保存测量同时记录：每rank实际plan、budget peak、当前RSS、offloaded基线、commit future的数量/完成数、仍可达快照引用，以及保存交接/commit耗时。先用同一个state和固定写入后端比较budget档位，再选择满足内存余量的最快档位；不要在同一轮同时改副本切分、chunk、concurrency和allocator而把结果归给一个参数。

本轮17项检查包含原asyncio budget、配置默认值、原shape/replica规划与区域覆盖、整分失败边界；规划使用synthetic JAX类型与sharding，真实GPU/存储测量仍为空。[脚本](scripts/probe_checkpoint_memory.py) · [原值与作用范围](analysis/checkpoint_memory_probe.json)。它接续[保存提交管线](CHECKPOINT_COMMIT_ZH.md)，优先补实际writer布局和每rank内存证据，再决定是否调保存频率与budget。
