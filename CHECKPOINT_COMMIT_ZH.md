# Checkpoint：从“已经保存”拆出真正可恢复的进度

535B训练里，checkpoint不是一个后台文件操作，而是决定多少训练量能在故障后保住的系统。生产记录中的配额拒写、永久metadata删除被拒、异步写入停滞和下一次save的barrier超时，发生在不同层次。把它们都压成“checkpoint有问题，重启解决”，无法指导自己的训练。

本章读固定代码`84869ae8c91ffe64e9f761c5bd714542eb1876e0`的五份实现。源码用于解释接口和失效边界；没有把它绑定为每次历史事故的实际执行版本。[checkpoint.py](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/lib/levanter/src/levanter/checkpoint.py)、[TensorStore序列化](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/lib/levanter/src/levanter/tensorstore_serialization.py)、[来源与SHA](analysis/checkpoint_commit_acquisition.json)。

## 1. 保存返回、标记可见、恢复成功，是三件事

<div id="checkpoint-flow-placeholder"></div>

[下载控制流SVG](assets/checkpoint_commit_flow.svg)。实线框对应源码阶段，虚线框为建议额外执行的恢复验收，不表示作者代码已经执行该验收。

|阶段|这份源码做了什么|这个阶段不能证明什么|
|---|---|---|
|各rank决定是否save|由process 0广播保存决定与temporary/permanent；避免各rank时钟差导致不同step写不同目录|广播决定不代表写入已完成|
|布局描述|process 0先写`manifest.json`，包括数组路径、shape、dtype、chunk布局与格式版本|manifest不是数组完成标记|
|staging与异步写入|`_serialize_arrays`等待上一轮manager任务，再安排写入与commit；`on_staged`记录host复制阶段|数据离开device，不代表远端完整提交|
|完成标记|manager的commit callback调用`_save_metadata`，发布`metadata.json`里的step、timestamp、is_temporary|标记可见，不代表已独立恢复并验收状态|
|保留/清理|metadata发布之后才调用用户callback；temporary保留配置数量，permanent分支清理已记录temporary|清理被调度不代表删除成功，也不代表新的保存经过恢复测试|
|发现与恢复|发现函数按可读metadata筛选，再按数字step、时间和路径排序；真正读取数组在load路径|“latest可发现”没有验收所有数组、optimizer与下一批训练|

`tree_serialize_leaves_tensorstore`在调用数组serializer之前就写manifest。因此，运维不能数目录或数manifest来宣布最新可恢复step。另一方面，`save_checkpoint`给已提供manager的调用方返回目录时，callback可能还没发生。类包装器此时更新`_last_save_step`与`_last_save_time`；这些变量记录该次保存已发起/交接，并不是独立的远端完成水位。

下一次save的数组路径开头会等待上一轮manager完成。于是“上一轮异步任务停住、训练先继续、下一次save等不到”符合这条控制流。它解释症状如何传递，不能确定前一轮最早卡在worker、网络、存储还是GPU。metadata的timestamp来自没有时区偏移的`datetime.now().isoformat()`；对照UTC事故日志时需记录host时区，不能直接把它解释成UTC。manager配置了30分钟timeout，也不能只凭这个配置推断每一次底层网络请求都会在30分钟内退出；底层factory、客户端和回调阻塞还需按实际版本核对。

## 2. 对应三类生产事故，不共用一个根因

|原记录|做了什么/结果|源码怎样解释|还缺什么|
|---|---|---|---|
|114000永久保存：metadata copy成功，删除临时源被`DenyHeroCheckpointDeletion`拒绝|识别为永久前缀权限与copy-delete动作冲突；没有在调查中修改policy|对象存储rename可由copy再delete实现；目的对象存在与调用成功可能分离|该事故执行版本与完整请求轨迹；不能把普通lock-release的AccessDenied混入|
|146585 staging：104.56 TiB超过100 TiB，405拒写；清理后写入恢复|清理约49TB，保留若干durable handoff|配额控制作用于写请求，manifest/数组/metadata阶段都应单独记录|清理之后146582的256/704进程停写仍原因未知；不能继续归给配额|
|184731临时保存未完成，下次checkpoint barrier timeout|重启恢复训练，公开记录未找到先行原因|下一次save等待上一轮异步任务，末端超时可能是传播结果|前一轮首个异常、各rank local commit状态与存储请求；barrier不是充分的起因证据|

原文位置：[永久metadata事故](https://github.com/marin-community/marin/issues/8506#issuecomment-5708008076)、[配额与清理后第二次停滞](https://github.com/marin-community/marin/issues/8506#issuecomment-5817400840)、[184731未完成保存](https://github.com/marin-community/marin/issues/8506#issuecomment-5922288472)。更完整时序在[58条运行索引](OPERATIONS_ZH.md)及[八案工程证据链](ENGINEERING_GUIDE_ZH.md)。

这份固定源码已在`_save_metadata`里区分后端：远端对象直接`write_text`，本地使用临时文件加rename。源码注释说明，受保护前缀不允许删除temporary source，故不对远端metadata做sibling rename。这里可确认的是当前归档实现的动作，不凭此断言此前每个生产故障都已完全修复。

## 3. 原函数故障注入揭示的四个边界

本轮执行原始atomic、metadata、候选发现和save包装函数，配合本地临时目录、最小StoragePath、process-index桩以及捕获callback的假serializer。17项检查全部通过。真实磁盘fixture只用于metadata读写；没有写模型数组、TensorStore、S3、分布式manager或真实checkpoint。[脚本](scripts/probe_checkpoint_commit.py)与[检查结果](analysis/checkpoint_commit_probe.json)保存原函数行号、替代依赖和五份source SHA。

|故障注入/fixture|实际看到什么|应怎样理解|
|---|---|---|
|模拟对象存储先copy，再拒绝delete|目的metadata可见，临时源仍在，调用仍抛PermissionError；清理也被拒|不能用“目的对象存在”覆盖调用错误；本例模拟策略，不是执行真实S3 policy|
|假异步serializer只发布布局并捕获callback|save返回时manifest存在、metadata不存在，发现函数排除它；触发callback后才可发现|manifest与完成marker功能不同；可发现fixture仍完全没有数组|
|metadata发布失败 / 用户callback失败|前者阻止后续保留callback；后者发生在metadata成功之后，marker继续存在|报告必须指出失败在marker之前还是之后；“save失败”不能一概解释成没有marker|
|复用已有目录，注入callback之前失败|旧metadata没有被撤回，目录仍可被发现|重新写同一路径时，marker的存在不能证明这次attempt成功；没有据此证明数组已损坏|

数值step排序、max_step、排除前缀不误伤同名兄弟路径、process 0独占发布、保留metadata字段不可被应用字段覆盖，也纳入检查。回退到旧可加载候选的异常类型契约见[接口章节](IMPLEMENTATION_CONTRACTS_ZH.md)；本轮没有重复用假loader冒充真实恢复。

“metadata-only可发现”不是说生产代码会故意提前发完成marker。fixture故意没有数组，目的是确认**发现函数本身的验收范围**。正常writer的完成顺序仍依赖真实manager、各rank提交与回调；这些是下一层检查，不能由本地辅助函数替代。

## 4. 防止白训练，要追踪四个进度水位

一个step字段不够。运维至少同时展示：训练已完成step、保存已发起step、metadata已发布step、独立恢复验证step。它们可能长时间不同步。速度报表也应同时看device吞吐和已验证可恢复进度在wall time里的增长；后者才包含重训、保存停滞与恢复成本。若用名义token位置换算，注明它不同于有效loss目标。

容量规划不要只算一份权重。按实际state叶子求`Σ numel×dtype_bytes`，核对master、compute参数、optimizer、EMA以及布局里是否都被序列化；再加上已保留版本、新save正在写的对象、OCDBT元数据/版本残留与容量余量。FP32 master模型不能按BF16推算全部容量，也不能按rank数量把逻辑数组重复倍乘；具体存储大小由真实shard写入布局和后端对象决定。当前没有这些生产叶子与对象的实测计数，报告不给一个假精确的TB预算。

这份源码在metadata发布后即可清理temporary，并没有在清理callback里先执行restore smoke test。因而“保留几个temporary”与“保留几个独立验证过的回退点”是不同政策。大版本变更前应另留已验证的durable handoff，直到候选通过自己save后restore、下一批训练、固定eval和新permanent保存。具体保留数量、容量上限和停止条件必须按预算事先定，不把默认keep=1当作足够可靠的结论。

## 5. 一条可以交接的保存验收管线

1. **发起前**：冻结run、代码/wheel、step时钟、状态树与loader cursor；选新目的路径，记录temporary/permanent前缀、权限与剩余容量。复用路径时单独标记attempt，不能沿用旧marker判断成功。
2. **上传中**：按rank记录stage、local commit、全局commit、manifest与metadata请求起止。旧任务未完成时，优先收集其首个异常；新save末端barrier只作为后续症状。
3. **提交后**：读回metadata与manifest，验证step及布局身份，记录全局callback结果与清理结果；不因marker可见掩盖callback错误。
4. **恢复验收**：独立新进程读取完整状态，核对master/optimizer/pending bias及下一批输入；运行下一步和固定eval。验证应包含新temporary、新permanent及候选自己保存的checkpoint。
5. **清理与复盘**：只清理授权范围内的版本，保留预定且经过恢复验证的回退点；实际容量与删除成功读回确认。将根因已证实、缓解、恢复进度和未知分别填写。

[保存验收记录模板](templates/checkpoint_commit_review.json)把四个水位、各阶段证据和回退点放在同一交接里。它是待执行模板，实测状态为空；本轮没有进行任何生产删除、权限修改或真实恢复。最优先的后续检查是：取得一次真实save的各rank提交轨迹与自己save后restore结果，再判断哪种保存阶段最值得优化。


## V64：推理恢复消费者如何拒绝错误输入

本轮执行固定weights.py的完整restore_weights函数及checkpoint_stores_master函数。临时目录中的metadata.json读写是真实本地IO；manifest读取、数组加载、Transformer模板、digest算法与tree_at是明确替身。原偏置计算使用真实JAX。10个人工输入、9项控制检查不构成实际TensorStore/OCDBT恢复测试。[原值和调用轨迹](analysis/weights_consumer_faults.json)

### 先判定永久元数据，再选择权重视图

原入口要求调用方提供metadata_digest，并判断metadata.get('is_temporary') is False。true、缺失字段或整数0均被拒绝，且尚未进入layout/array读取。JSON的false与0不是相同的接口声明；不能用一般的真假值判断替代此合同。digest不一致也在layout读取前拒绝。本轮digest使用人工SHA256回调，仅验证原条件分支，没有执行或证明实际digest算法。

通过后，manifest中出现master_params及其子路径时选择master；没有master时选择params。控制中即便本地有一个陈旧master_params目录，只要manifest说params，就不会因为目录存在而改选master。有manifest时权威布局应来自manifest，不能把多个历史文件的存在性混作当前状态。

只有manifest缺失所引发的FileNotFoundError才进入旧布局探测。原消费者另有OCDBT KV marker探测；本轮只执行本地目录marker分支，没有执行OCDBT。人工malformed manifest的ValueError直接传播，不会被当作“没有manifest”而继续。缺失与损坏应该区分，但实际manifest parser对各种损坏抛什么异常仍需其真实路径验证。

### 权重加载成功，还必须有pending状态

该推理入口的请求模板包含权威权重键和pending_qb_betas，且传入allow_partial=False，candidate保持调用方指定路径。返回前block_until_ready，再应用pending。四个正常控制返回相同的人工centered bias[1,-2,1]。这些调用证明此入口的请求合同，不能证明替身返回的数组来自真实checkpoint、shape完整或payload无损。

人工loader将缺失数组错误传播到消费者时，外层消费者不会改选另一个checkpoint。必须同时限定：真实load_grug_checkpoint内部包含root到legacy wrapped布局的FileNotFoundError重试，两次仍使用同一candidate；本轮数组loader是替身，没有执行这个内层重试，不能宣称整条恢复链只有一次底层读取。

推理restore_weights不恢复optimizer，因此不能拿它的成功作为续训恢复合格。训练恢复仍需完整params/master、opt_state、pending、step和下一输入身份。metadata一致也不能自动证明数组提交完整；实际save/commit、manifest解析、strict array load与后续执行分别需要对应证据。

### 可转用的验收规则

记录失败发生于元数据、布局选择、数组读取、pending应用还是首次执行，并保留异常类型；不要把损坏输入随意当成旧布局。对照需明确消费者用途：推理权重恢复和完整训练状态恢复是不同合同。保持指定candidate和权威视图，避免回退后只记录“恢复成功”。

运行make weights-consumer-faults CPU_PYTHON=/tmp/marin-jax-cpu-072/bin/python可复算。下一项强证据仍是候选代码自己保存、独立新进程实际恢复并运行下一步，而非增加替身控制后宣布生产恢复完成。


## V65：真实存储读回，为什么“有限值”仍不足以验收

这次沿用归档源码的 `build_kvstore_spec` 与 `_create_ocdbt_spec`，实际执行 TensorStore 0.1.69 的本地 Zarr3/OCDBT 读写。输入是 6×4 的 float32 人工数组，三块各占两行，值为 1 至 24；等待 copy 与 commit 完成后，另起 Python 进程读回。8 项检查通过，[原值与错误记录](analysis/tensorstore_roundtrip.json)可复查。

|受控操作|新进程读到什么|能确认什么|
|---|---|---|
|全数组写入并等待提交|24 个值全部一致，shape/dtype 一致|这一个本地样例能够跨进程读回|
|请求不存在的 pending 数组|打开时报 NOT_FOUND|缺数组元数据与缺数据块不是同一分支|
|用 int32 约束打开 float32 数组|FAILED_PRECONDITION|本例会拒绝 dtype 不一致|
|创建数组，只写前两行|前两行一致，其余四行全零|元数据可见不代表所有预期值写入|
|完整写入后删除逻辑键 c/0/0|前两行全零，其余行一致，全部有限|读成功和 finite 检查会漏掉本例的内容变化|

<div id="tensorstore-io-placeholder"></div>

为什么会这样？在本例的默认 Zarr3 配置中，没有存储的 chunk 读作填充值零。数组元数据描述形状、类型与块布局，不能单独证明原定的 24 个值都已写入。这里删除的是 OCDBT 内的逻辑 Zarr chunk 键，使用 [KvStore.write(key, None)](https://google.github.io/tensorstore/python/api/tensorstore.KvStore.write.html)；没有删除或破坏 OCDBT 物理 blob，也没有测试断电、损坏 manifest 或多 rank 故障。

**验收规则应增加内容完整性证据，但不能简单要求每个 chunk 键都存在。** 合法的全填充值块可以不占独立键；真实参数也可能含零。应将保存时预期状态与恢复内容对应：记录 step、树/叶身份、shape/dtype、各 rank 写入及全局提交回执，并在预算允许时核对保存端与恢复端的内容摘要。抽样校验只提供抽样范围内的证据，下一步训练与固定 eval 是行为补充，也不能替代字节完整性判断。摘要生成与数据写入必须属于同一 attempt，避免用旧摘要验收新路径。

这个实验没有执行 Marin 的完整 save/load 函数、CheckpointArray 类、分布式 coordinator 或生产 checkpoint。元数据 entry 用 SimpleNamespace 提供 shape/dtype/chunk_shape。因此不能声称 Marin 保存链缺少保护，不能把这里的零值解释成 Hero 的历史事故。它只把“缺块可能怎样被底层读出”的一个分支从推测推进到了真实本地 IO。

复现：在 Python 3.12 环境安装 requirements-tensorstore-io.txt；本轮依赖隔离在 /tmp/marin-tensorstore-lib，命令为 `make tensorstore-io CPU_PYTHON=/tmp/marin-jax-cpu-072/bin/python`。临时数组由探针创建并自动清理。下一步需要完整序列化链的同 attempt 保存回执、内容摘要和独立恢复，才能评判系统层面的保护覆盖。


## V66：原数组保存链的真实 IO 与错误传播

V65 只跑了原 spec 构造函数。这次把执行范围推进到原 `_serialize_arrays`：原 HostByteBudget、原写入 context、原 host-array snapshot、真实 TensorStore 0.1.69 和 JAX 0.7.2 的 GlobalAsyncCheckpointManager 均参与执行。两个 6×4 float32 数组各占 96 字节，预算设为 96 字节，使第二个数组必须等待前一个释放预算。9 项检查通过，[逐项结果](analysis/serialize_arrays_real_io.json)保留回调、错误和读回值。

|检查对象|实际操作与结果|工程含义|
|---|---|---|
|保存快照的所有权|原函数返回后将两个调用方数组改成 -999；等待完成后，存储仍为保存时的 1…24 与 101…124|host 分支的 `np.array(copy=True)` 保住本例保存值；没有测试 GPU donation|
|峰值预算|两份 96 字节数据，峰值记录为 96|本例预算阻止两份 snapshot 同时占用；不是整个进程 RSS 上界|
|失败时释放预算|第一个数组的已有 float32 元数据与请求 int32 冲突；第二个数组仍完成，内容与预期一致|失败 future 的回调也会释放预算，不把后续写入永久堵住|
|保存成功的边界|上述混合成功/失败案例出现 `local_failed`，manager.wait 抛 ValueError，成功 callback 未执行|部分数组已写完不能代表整次保存成功；本例异步 open 错误没有被吞掉|
|错误归属的时刻|不先取出上次失败，直接发起下一次原函数；入口 wait 抛上次错误，新路径尚未创建、未 staging|新 save 处看到报错，可能来自上一轮在途任务；应按 attempt 追踪首个异常|
|已报告异常的保留|安装版 manager 的 check_for_errors 清除已取出的异常；再次 wait 返回正常|异常只报一次，不代表失败 checkpoint 变成完整；不能靠后一次 wait 正常抹去失败账本|

这里有两种不同的“完成”。释放 host 预算意味着相关写入 future 已结束，成功或失败都可以释放；它不表示数据保存成功。成功 callback 是另一条水位，要求 manager 汇合写入 future 后进入成功分支。本例使用真实单进程 manager，但没有多 rank barrier，不能把它提升成全局提交验证。

源码也解释了为何 manifest 可见不是提交证明：`tree_serialize_leaves_tensorstore` 在调用 `_serialize_arrays` **之前**由 process 0 写出 manifest；manifest 的数组字段列出 path、shape、dtype、chunk_shape，格式字段列出版本与 driver，没有保存值的内容摘要。它承担布局清单的职责，最终 metadata/成功标志要看上层 callback 链。这个顺序由归档源码核对，本次没有执行完整 tree serializer 或生产 metadata 发布。

把 V65 与 V66 放在一起，较准确的结论是：底层缺逻辑 chunk 的填充值语义，不能单独用来判断保存系统是否漏检；本例原保存链确实传播了一个真实异步打开错误。但还有未覆盖的分支：提交成功后内容变化、分片覆盖不足、多 rank 首错、metadata 发布失败及 consumer 是否只选择成功版本。应逐项取证，不能将一个失败案例的绿灯当作全保存链安全证明。

验收管线应保存每个 attempt 的失败状态，即使 manager 已消费异常也不清除失败记录。下次 save 的入口 wait、local 回调、成功回调分别关联到它们所属 attempt。评估或恢复候选同时核对布局身份、成功版本选择及内容证据；“当前 wait 无异常”仅说明当前没有待报告异常。

范围：原源码用 AST 提取执行，不导入完整 Marin 包。heap trim 显式替为 no-op；host 分支不使用 write plan，传入 None；entry 仍为 SimpleNamespace。JAX manager 来自本轮安装环境，源码摘要与 check_for_errors 源文保留在结果中，没有证明 Hero 当时使用同一版本。没有执行设备分片、tree flatten、真正 CheckpointArray、完整 restore、生产 metadata 或分布式提交。复现：`make serialize-real-io CPU_PYTHON=/tmp/marin-jax-cpu-072/bin/python`。


## V67：发现候选、严格恢复与内容完整性的接口边界

这次执行原 checkpoint discovery、原 Pydantic manifest 类和原 `_restore_ocdbt` → `_deserialize_leaves` → read-plan → shard-read → JAX materialization 链。完整样例由 V66 的原 host writer 写入，两份数组读回值一致；单 CPU 设备、EVERY_REPLICA 模式、96 字节读取预算下，9 项检查通过。没有把数组加载器换成替身。[原值、错误与源码摘要](analysis/restore_candidate_real_io.json)保留复现证据。

|受控目录或调用|原发现/恢复行为|实际边界|
|---|---|---|
|step10：完整写入 a、b，标志有效|发现并恢复两份 JAX 数组，内容一致|原 writer 与原 reader 的本地 host 样例贯通|
|step20：a 完整，b 仅写首块；人工放 metadata|发现为候选；allow_missing=False 仍读出 b 后四行零|严格参数限制缺叶，不保证每个预期数据块都已写入|
|step30：manifest 列有 b，但 b 元数据不存在；人工放 metadata|发现最高 step30；实际恢复报 NOT_FOUND|发现阶段未读取所有数组，因此候选资格不是恢复验收|
|step40：有 manifest 与部分数组，无 metadata|不进入候选集合|manifest 自身不充当完成标志|
|删除人工 step30 标志，再次发现|选择 step20|重新选择路径仍不能证明内容完整；本例没有自动故障回退|
|请求 manifest 中不存在的 required_missing 叶|allow_missing=False 抛 FileNotFoundError|这一层严格检查叶存在性确实生效|
|manifest 声明 b 为 6×4，实际 b 元数据为 2×4|原 leaf reader 恢复 2×4 数组，没有在该层比较 manifest shape|布局字段存在，并不证明这一层把它当约束使用；完整状态重建的上层检查未执行|

这里的 step20/30 标志由探针人工写入，用于测试消费者边界；并不是原生产 callback 发布。V66 的真实异步错误案例没有触发成功 callback，所以不能把这里的候选选择推成“生产系统会将失败保存标记成功”。真正需要追查的是：谁有权发布 metadata、复用路径是否残留旧标志，以及发布后发生内容变化时如何验收。没有这类生产证据，保持事故归因未知。

### shape 字段为什么没有在这一层拦住不一致

原 `_restore_ocdbt` 从 manifest 取得的是 array_paths，用来筛选路径。构造读取 spec 时只传 checkpoint_root 与 path；原 `_leaf_read_plan` 随后从 TensorStore 元数据取得 shape/dtype，再按目标 sharding 读入。这条路径没有使用对应 manifest entry 的 shape 作为 TensorStore 打开约束。上层 NamedArray 重建、模型 schema 或状态合并是否拒绝不一致，是另一项必须单独验证的合同。本例没有执行它们，不能断言完整模型会接受错误形状。

### 把验收拆成四项可核对的合同

1. **候选资格**：metadata 是否由当前 attempt 的成功发布链产生；step、时间与路径是否对应。目录扫描结果只回答“哪些路径符合发现条件”。
2. **状态布局**：预期叶与实际叶、预期 shape/dtype 与实际存储/恢复 shape/dtype 都核对；不能用“清单里有字段”代替“消费者执行了约束”。
3. **内容完整性**：保存预期值与恢复值的摘要或明确覆盖的抽样证据。strict、finite、目录键数、形状正确各自都不能独立完成这项验收。
4. **恢复后的行为**：完整 optimizer/master/pending、下一批输入、下一步更新和固定 eval。行为验收补充前面三项，不把一次 loss 正常当作字节内容证明。

[保存验收模板](templates/checkpoint_commit_review.json)已增加这四项的证据栏，并保留失败 attempt 是否已报告的状态；所有真实验收结果仍为空。这不是已经通过的生产检查清单。

范围：LocalStoragePath 为 pathlib 适配器，替代 rigging 存储接口，仅用于本地 manifest 读写；候选扫描使用真实 fsspec。原 schema 使用实际 Pydantic 2.11.7，AST 环境显式 rebuild forward annotations；reader 使用真实 JAX 0.7.2/TensorStore 0.1.69。heap trim 和 host write-plan 范围继承 V66。没有远端一致性、ONE_REPLICA collective、完整 tree restore、生产 metadata 发布或真实旧标志事故。复现：安装 requirements-cpu-numerics.txt 与 requirements-tensorstore-io.txt 后，`make restore-real-io CPU_PYTHON=/tmp/marin-jax-cpu-072/bin/python`。
