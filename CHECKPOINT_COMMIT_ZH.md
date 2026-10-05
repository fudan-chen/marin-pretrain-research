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
