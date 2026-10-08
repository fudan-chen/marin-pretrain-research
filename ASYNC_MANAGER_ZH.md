# 保存之后谁还持有数据：执行历史异步manager

保存内存问题需要沿引用链追到底。训练仓库把commit futures交给JAX异步manager，manager等待它们，再执行发布回调。调用方看到线程结束时，还需要分别确认错误是否已被消费、对象引用是否释放、保存是否真的可恢复。本轮将此前只有接口说明的依赖纳入源码与执行证据。

## 一、来源绑定与执行范围

从已归档的d512 ladder producer和c00删域continuation代码artifact清单取得同一个依赖路径：`.venv/lib/python3.12/site-packages/jax/experimental/array_serialization/serialization.py`。下载13,477字节，[两份清单的size与MD5均匹配](analysis/async_manager_acquisition_v129.json)，[原文件](sources/async_manager_2026_10_08/serialization.py)保留SHA。它绑定两个小规模run的上传依赖，不证明535B现场实际导入了同一文件。

执行其中完整`AsyncManager`类体，使用真实Python线程和真实JAX单进程查询/monitoring。分布式client为None，异常类有最小适配，未执行多rank barrier。运行时为Python 3.12.13、JAX 0.7.2、NumPy 2.5.3、TensorStore 0.1.69；执行的是新取得的上传类体，不能称作完整历史环境复现。[探针](scripts/probe_async_manager_cpu.py)、[10项控制与版本](analysis/async_manager_cpu.json)

控制包含标准库Future、受控闭包与一次真实TensorStore内存KV/Zarr提交。后者仅32字节，不写生产存储，不保存模型或恢复checkpoint。没有测RSS、GPU或真实事故发生率。

## 二、原方法的三个不同状态

```text
_add_futures：manager保存整个future列表
    ↓
_start_async_commit：启动真实后台线程
    ↓
_thread_func：逐个future.result → 单进程发布回调
    ↓                  ↘ 异常存入manager._exception
wait_until_finished：join线程 → 清除_thread → check_for_errors
    ↓
线程清理完成；_commit_futures与_on_commit_callback仍可被manager引用
```

源码的`wait_until_finished`会将`_thread`置None，但不清空future列表和发布回调。`_add_futures`下一次直接替换列表，而不是把新future追加到所有历史任务之后。因此，这个字段本身的保留不能解释成“每次保存永久累积一份历史快照”。若RSS持续逐次抬高，仍需寻找其他根引用、回调链、任务状态或allocator行为。

## 三、引用存在，究竟保留了什么

|执行控制|等待之后观察到什么|支持什么判断|
|---|---|---|
|标准库Future以NumPy数组作为result|线程已清理，列表仍含1个future；数组weakref仍活着|这个受控Future的result引用足以保留数组|
|用下一份future列表替换上一份|旧数组weakref失效，新列表长度仍为1|该字段会释放旧列表；未证明无限历史累积|
|发布回调闭包单独捕获数组|手动清future列表后数组仍活着；再清回调后失效|回调是另一条独立引用路径，需分别检查|
|真实TensorStore内存写入，允许后端长期引用源数据|等待后实际future列表仍长1，但源数组weakref已失效；读回32字节正确|这次完成future未继续保留可观察的源数组；不能用受控Future替代真实后端结论|

TensorStore控制中，copy完成后删除本地snapshot、write返回对象和commit变量；原manager完成等待后做GC，再检查weakref及读回值。清空manager列表后，源数组仍然不存在。没有把“源数组已释放”解释成“全部编码缓冲、KV数据或allocator内存都已释放”。

标准库Future持有数组结果，与实际commit future返回值和内部对象图不同。报告若只展示前者，就宣称真实TensorStore长期保留整个模型，会超出证据。反过来，32字节内存后端的回收也不能证明大规模分片、不同driver或生产版本不会保留数据。

手动清字段是用于定位所有权的诊断动作，没有接入训练代码。清理必须等使用者结束；在commit前删引用可能破坏写入所需数据，不能作为通用省内存技巧。

## 四、异常只抛一次，第二次等待不能覆盖失败

原`_thread_func`捕获异常并存到`_exception`；`check_for_errors`先将该字段清空，再向调用方抛出异常。单进程控制得到：

|失败位置|第一次wait|发布回调|第二次wait|正确的训练记录|
|---|---|---|---|---|
|local commit Future抛错|RuntimeError: local commit failed|未执行|正常返回|本次保存失败，保留旧可恢复水位|
|commit成功，发布回调抛错|RuntimeError: publish failed|尝试1次并失败|正常返回|数据提交与发布失败分开记录；不能晋升完成水位|

第二次返回说明没有尚未消费的本地异常，不是原保存任务被自动修复。多rank路径还会等待协调KV中的成功值，本轮没有执行，不能照搬单进程的第二次返回行为到集群。

这也影响日志与重试：错误消费位置可能是下一次save之前的等待、显式wait或结束阶段。应将错误绑定原保存任务身份，持续保留失败状态与首次异常；不要按“谁最后调用wait”把事故归到下一次保存。原方法没有在这两项控制中重试失败commit或失败回调。

另外，受控未完成Future使后台线程停在`result()`，回调尚未执行；放行Future后，原线程结束并只调用一次回调。该控制证明这里的等待顺序，没有证明真实存储延迟、分布式同步或对象存储完成语义。

## 五、如何更新保存验收与内存排障

**保存水位分别记录。** 训练完成、快照交接、local commit、发布完成和独立restore分开。第一次wait失败应阻止本任务晋升；后续wait正常不能抹掉失败。与既有[提交协议](CHECKPOINT_COMMIT_ZH.md)和[观测时刻](OBSERVABILITY_ZH.md)共同使用。

**引用按所有者记录。** 观察manager列表、callback、Future结果/回调、writer任务与外部缓存；对数组用可达性或weakref，对allocator另测retained bytes。future数量、array可达性和RSS分别回答不同问题。本轮没有直接从weakref推算释放字节或RSS下降。

**清理作为独立变更验证。** 已结束任务的列表与回调可以评估显式释放，但要先保存错误诊断、确认线程/写入使用结束，再比较提交、发布和恢复行为。该依赖的私有字段接口还涉及版本兼容，不能把手动清字段当成已验证生产补丁。

**复现真正相关的后端。** 内存driver先检验调用合同；真实checkpoint还需对应TensorStore driver、数组分片、写并发与manager生命周期。再分别观察writer/non-writer ranks和反复保存后的稳态、峰值及可恢复水位。当前这些仍未完成。

这轮将“manager可能保留引用”的说法收紧为实际执行结果：两个字段在所审查原类体中保留；受控对象图能保留数组，真实小TensorStore控制未保留该源数组；错误一次性消费需要调用方保存任务级失败事实。它补全排障合同，没有证明或推翻作者当时的生产RSS事故根因。
