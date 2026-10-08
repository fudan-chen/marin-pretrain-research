# 保存失败以后，调度器实际会做什么

上一章确认异步manager只向调用方抛出一次错误。本轮继续向上：谁消费保存请求、谁更新调度步数、谁决定再次保存、谁清理旧版本。**交接步数记录的是任务已提交，不是新checkpoint已发布；错误发生在哪一层，决定应重试写入、重试发布还是只处理清理。**

## 一、把调用链接起来，而不是只测底层Future

执行固定Marin快照`84869ae8…`中Checkpointer的五个原方法：`on_step`、`save_checkpoint`、`request_checkpoint`、`_consume_checkpoint_request`、`_get_current_step_save_interval`；同时执行全局保存函数的原嵌套`my_callback`，接到上一章归档的JAX原AsyncManager类体。[固定训练仓库源码](sources/checkpoint_commit_2026_10_05/checkpoint.py)、[依赖源码](sources/async_manager_2026_10_08/serialization.py)

这两份来源没有被证明是某次历史运行实际部署的一对版本。本轮是接口组合控制，不能称作完整历史事故重放。

Shell手工设置最小字段，省略真实构造器、清理线程和文件发现；单进程广播使用identity适配。writer、metadata与retention后端只记录内存事件，后台提交用可控标准库Future，线程与错误等待用原manager。没有实际写盘、删除、模型、分布式同步或restore。[探针](scripts/probe_checkpoint_schedule_cpu.py)、[11项控制及全部事件](analysis/checkpoint_schedule_cpu.json)

运行时为Python 3.12.13、JAX 0.7.2、NumPy 2.5.3 CPU。事件`writer_submit`表示虚拟writer接受任务，不表示所有数组已复制；`_last_save_step/time`在原类方法的writer调用返回后更新。发布事件可能在后台更早发生，本轮没有测量这些时刻的生产延迟。

## 二、源码中的请求与水位顺序

`on_step`先检查step0与同step去重，再消费显式请求，再计算step/time策略。决定保存后，它构造一个保留策略回调，调用类`save_checkpoint`。

类方法调用全局writer返回后，更新`_last_save_step`和`_last_save_time`。原全局保存的发布回调先写metadata，再运行保留策略回调；永久保存分支请求清理已有临时版本，临时保存分支先登记新临时版本，再按保留数清理。

```text
同step去重 → 消费请求 → 决定本次保留策略
    ↓
writer接收/交接 → 更新_last_save_step与_last_save_time
    ↓ 后台执行顺序独立于调用方返回时刻
commit完成 → metadata发布 → retention回调
```

因此，去重水位、时间节流水位、metadata发布水位需要分别解释。把一个`last_save_step`当作所有完成事实，会同时误读重试与回退。

## 三、四个失败位置，对应四种处理边界

控制起点有人工旧发布水位5，在step10请求永久保存：

|注入位置|原on_step/等待结果|交接水位|人工发布水位|保留策略是否执行|
|---|---|---:|---:|---|
|writer交接前抛错|on_step报staging failed|仍为0|仍为5|未执行|
|后台commit抛错|on_step返回，wait报commit failed|10|仍为5|未执行|
|metadata发布抛错|on_step返回，wait报metadata failed|10|仍为5|未执行|
|retention回调抛错|on_step返回，wait报retention failed|10|已到10|已尝试并失败|
|成功对照|on_step和wait正常|10|已到10|发布后请求prune(0)|

这里的发布水位是内存事件记录，不是实际对象存储marker。保留策略用记录器替代，既未执行原删除队列，也未确认真实删除安全。

前两种后台失败没有进入retention；metadata失败也没有进入retention。这支持所选原回调的顺序保护，不能用来证明所有存储后端或所有调用者都不会提前删除。

源码在metadata写入之后记录Saved日志，再进入retention回调；该日志也不是全部后续动作已完成的信号。

retention失败则已经越过metadata发布。此时“保存任务报错”需要拆开看：数组提交、marker发布、保留清理分别到哪一步。不能仅凭错误词就重写整份数据，也不能凭marker存在就认定checkpoint内容与restore已验证。

## 四、请求会不会自动重试

### 交接前失败：请求已被消费

原`_consume_checkpoint_request`在writer调用之前将请求置None。控制中同步staging失败保留旧交接水位，但请求没有自动放回。随后改为成功后端，不增加请求、step策略或time触发，在step11调用不会提交。

这说明显式请求是一项触发，而非带失败恢复的任务队列。需要可靠重试的调用方必须另存请求身份与结果，不能只观察水位尚未前进就假定请求还在。

### 交接后失败：同一步普通调用被去重

后台commit或metadata失败后，交接水位已经是10。消费该错误，再以`force=False`调用step10，原方法提前返回，提交次数仍为1。不能通过“再调用一次同step”获得隐含重试。

`force=True`控制能绕过这一去重，第二次虚拟提交成功并发布step10。它仅证明开关行为；没有验证对同一真实路径重复写入的原子性、覆盖安全或多rank一致性，不构成生产重试建议。

### time策略：节流时间也已经前进

另一控制只启用100秒time策略。第一次保存交接后commit失败，1秒后进入下一update：新请求为空，step策略关闭，距上次交接仅1秒，因此没有立即重试。它遵循当前调度实现，没有以“最近一次成功发布时间”计时。

本轮没有执行训练主循环的异常处置。上层若选择捕获错误并继续，就需要明确失败后的请求、时间策略和去重策略；不能把Checkpointer本身当作自动重试系统。

## 五、应如何设计自己的故障策略

**用attempt记录替代单个成功标签。** 记录来源state、提交step、请求身份、保留类型、writer返回、local commit、metadata发布、retention与restore状态。保存首次错误，并绑定原attempt；manager后一次wait正常不能覆盖它。

**将调度触发与失败处置分开。** 时间间隔和step去重可以控制正常保存频率，失败重试则应另有明确触发与预算。交接前错误是否重新排队、交接后是否允许同step重试，都要由策略决定；当前报告没有实现或部署这个策略。

**先判断应重做哪一段。** commit失败要调查写入；metadata失败要调查发布；retention失败先核对已发布数据和保留策略。保留经过验证的旧回退点，避免清理失败引发不必要的全量重写。实际成功与否仍由存储证据及独立restore确认。

**重试保留独立身份。** 若实现新的重试流程，可评估独立attempt路径与明确发布选择；使用同一路径则必须验收覆盖/部分写入语义。force只是绕过去重，不提供这些保证。分布式重试还需所有rank对step、路径和保留策略达成一致，本轮没有验证。

这套建议来自原调用合同与局部故障注入。没有新增Hero事故判断、真实删除结果或训练收益；它帮助解释为何“已经发起保存”“已经看到metadata”“错误已经报过”不能合并成同一种可恢复状态。
