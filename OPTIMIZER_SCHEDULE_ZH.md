# 恢复相同状态后，新配置仍可能改写下一步学习率

V133，2026-10-08。14项CPU控制补上V86未执行的外层 `inject_hyperparams`：使用固定源码的schedule构造、Hero build、真实Optax注入与Adam，检查同一内存state在不同训练预算下的下一步。没有磁盘恢复、完整Hero优化器或语言模型训练结果。

核心结论有两条：**state里缓存的学习率不是下一步的承诺；一个optimizer里也可能有多份不同用途的count。** 若切换配比时同时调整总步数或重置部分计数器，更新规则也会改变，不能把loss变化全部归给数据。

[14项控制、所有状态与更新](analysis/optimizer_schedule_resume_cpu.json) · [复现脚本](scripts/probe_optimizer_schedule_resume_cpu.py) · 前置：[V86衰减状态控制](OPTIMIZER_GROUPS_ZH.md)。

## 真实执行了哪条源码链

固定提交 `84869ae8c91ffe64e9f761c5bd714542eb1876e0` 的[基础scheduler](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/lib/levanter/src/levanter/optim/config.py)本轮新增归档，23388字节。执行原 `lr_scheduler`、`_get_cycle_minima`、`_convert_frac_or_steps`、Linear builder及context；选择线性schedule时绕过插件注册与配置解析，未执行完整原模块。

[Hero optimizer build](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/experiments/grug/moe_hero_ep/optimizer.py)分别构造Muon/AdamH用的learning_rate与Adam用的adam_lr，然后通过真实 `optax.inject_hyperparams` 包装原multi_transform。实际更新的是单个两元素人工router叶，原create_mask将其分到Adam组。原专用衰减包装参与；两个未选中组的MuonH/AdamH变换替换为identity，叶路径由单叶适配器提供。没有证明真实模型分组、MuonH、AdamH或通信数值正确。

训练入口在恢复之前用当前 `trainer.num_train_steps` 构建optimizer。恢复读回的是状态，而新构建的闭包仍由当前配置决定。因此需要实测：新闭包如何解释旧state，而不能只检查state是否能被接受。

## 同一个state，新预算会改变什么

人工参数固定为[2,4]、梯度固定为[0.25,0.25]，预热50次只生成Adam状态，参数不随预热更新。这不是训练轨迹。两组peak LR分别设为0.02、0.01，warmup为总步数的10%，线性衰减至peak的10%；专用衰减初值0.02。所有N和预热长度都是人工值。

预热后外层count、两份WrappedScheduleState.count和Adam count均为50。缓存adam_lr为0.0061，对应前一次update使用的count49；下一步按原N=100重新求值得0.0060。不能拿这两者差异当作off-by-one bug。

|独立对照|下一步实际adam_lr|专用衰减系数|两个参数的最终update（约值）|
|---|---:|---:|---|
|原N=100与完整state|0.0060|0.010|−0.006120，−0.006240|
|完整state，重建N=200|0.0085|0.015|−0.008755，−0.009010|
|只改变衰减N=200，schedule仍按100|0.0060|0.015|−0.006180，−0.006360|
|只改变schedule N=200，衰减仍按100|0.0085|0.010|−0.008670，−0.008840|
|重建N=40，count仍为50|0.0010|0|−0.001000，−0.001000|
|只将缓存hyperparams改为123|0.0060|0.010|与原N=100对照相同|
|只将外层count改为0|0.0060|0.010|与原N=100对照相同|
|只将两份schedule count改为0|0|0.010|0，0；Adam状态仍推进|

第三、四行是人为解耦的机制对照，不是原配置提供的两个独立预算字段。它们让我们分别核对：lr变化按比例缩放相同方向，而衰减N变化会在相同lr下改变方向。第二行同时包含两项变化。

所有分支输入参数、梯度和Adam moments相同；下一步内层Adam状态逐叶相同，更新却不同。把缓存hyperparams写成123后，原注入器在update时重新求值，得到与基线一致的结果。它说明这些被声明为schedule的字段在此版本中由schedule状态和新闭包共同决定，并不表示所有手动hyperparams或所有Optax变换都遵循这一行为。

缩短N=40也不会让参数自动冻结：线性学习率停在设定的最小值0.001，专用衰减已为0，但Adam方向继续产生更新。若要用提前停止实现预算限制，需要另一个stop条件，不能指望衰减到头等于停止训练。

## 几份count分别控制什么

本轮真实依赖为JAX0.7.2、Optax0.2.5。实际state中同时有外层注入count、learning_rate与adam_lr的WrappedScheduleState.count，以及Adam自身的count。

将外层count单独改为0后，下一次update外层count变为1，两个schedule count却从50变为51；学习率和参数更新与基线相同。因而“我把optimizer.count归零，所以学习率重新warmup”对本控制不成立。

只将schedule count改为0，外层count仍从50到51，Adam的下一状态仍与基线逐叶相同，但学习率回到warmup起点0，参数update为零。**没有参数变化，不代表优化器没有消耗这次梯度。** moment和偏差校正计数已经推进，下一步不能等价地当作这一批从未执行。

这两项都是刻意破坏时钟一致性的控制，目的在于识别字段的消费者；不要据此建议只改某一个count。正常恢复应验证完整状态与下一步，主动重启优化则应定义一致的新状态及学习率计划。

## 为什么改N尤其容易混进配比干预

原scheduler把warmup、decay等fraction换算成当前周期的步数，并不是从恢复点开始再走一遍同样长度。例如同一个绝对count50，在N100和N200下处于不同的衰减进度；N再增大到使warmup超过50时，甚至可能重新处于warmup区间。这是全程计划被重建的结果，不能只记录“延长了剩余训练”。

专用衰减又读取Adam count并使用闭包内的总N。训练step、调度count、Adam count、累计token及数据阶段边界，需要各自保存映射。更改batch schedule后，count相同也不自动意味着训练过相同token量。前述数据阶段与加载器控制见 [batch时钟](BATCH_CLOCK_ZH.md)。

配比实验至少分为三种：保持共同checkpoint与原优化计划，只改变采样输入；改变配比同时改变预算/优化计划的组合干预；独立研究重置或重规划。三者都能研究，但必须用相应对照解释结果。这里没有新的语言loss曲线或最优配方结论。

## 接续与重规划的验收管线

1. **保存计划定义。** 原总步数、warmup/decay的单位与fraction、周期端点、两组peak/min LR、专用衰减N，以及实际构建源码与依赖版本。只存一个当前lr无法重建未来计划。
2. **清点时钟。** 恢复前后state.step、外层count、各schedule state、各Adam count和数据offset分别记录。不能按字段名字把所有count自动改成训练step。
3. **从同一状态走一步。** 同一参数、完整state和输入梯度，比较实际用到的两组LR、decay系数、update、moments和下一状态。完整训练再增加同一batch的forward/gradient对照。本轮只完成选中Adam组的更新层。
4. **明确允许什么变化。** 忠实续训要求原计划接续；延长预算应规定恢复点LR是否连续、之后斜率如何变化，而不是默认重新拉伸整个schedule。本轮没有实现连续拼接候选或证明其收益。
5. **再评估配比。** 如果计划被同时改变，增加计划对照或将其明确作为组合干预。最后用固定评估和独立重复判断收益，局部update差异不是长期loss改善证据。

本轮原eager/JIT下一步在选中组的控制内一致。尚缺磁盘中完整注入状态的读回、真实Hero分组与runtime绑定、完整forward/gradient、GPU分片和长期训练结果。下一步应先取得一次真实恢复的这些时钟与下一步LR，而不是继续从缓存字段猜测实际计划。
