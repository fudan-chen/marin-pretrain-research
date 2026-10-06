# 零梯度、空目标与训练时钟

把无效梯度改成0，不代表这一训练步什么都没做。原AdamH模块在人工矩阵上的真实CPU结果是：有历史moment时，零梯度产生范数0.188205的非零参数更新，count从1推进到2，而参数范数仍保持。只丢弃最终update、保留推进后的状态，也会改变下一个有效步的方向。

这不是训练事故复现。实验使用未修改的[完整AdamH模块](sources/adamh_2026_10_05/adamh.py)，真实JAX 0.7.2、Optax 0.2.5、Chex 0.1.90及人工2×2矩阵；没有真实Hero loss、checkpoint、分组multi_transform或完整train_step。参数与lr=0.1是机制对照，不能把更新大小外推到实际训练。

## 三种处理到底改变了什么

|处理|当前参数动作|optimizer状态|CPU结果|
|---|---|---|---|
|新初始化状态，输入零梯度|近零update|count仍从0到1；moment为0|update范数0|
|已有一次历史梯度，继续提交零梯度|仍应用历史方向|mu乘0.9，nu乘0.95，count从1到2|update范数0.188205，参数范数保持|
|计算零梯度步，但丢弃最终update|参数保持在p1|moment与count已推进|下一次同一非零梯度的update与完整冻结相差0.005558|
|这个空步完整保留参数与optimizer状态|当前参数不变|当前count/moment不变|后续有效步从原状态计算；只是本实验对照，不是已有训练策略|
|新初始化的独立Optax Adam|zero update|count推进|普通Adam也有状态时钟|
|新初始化的独立Optax AdamW，toy decay=0.1|仍有衰减update|count推进|update范数0.022361|

“梯度为0”与“没有有效目标”必须分开。合法目标可能恰好给出零梯度，moment继续推动参数是优化器设计的一部分。本章不建议按grad==0跳步；空目标应由实际有效分母T及预先确定的处理策略识别。

## 为什么moment衰减，参数仍然移动

AdamH先更新mu、nu并递增count、做偏差修正，然后形成逐元素Adam方向U。下一步不是直接使用U：它以U/‖U‖决定方向，以lr×‖P‖决定候选步幅，再投影回原参数范数。若零梯度让mu/nu整体衰减、且epsilon影响很小，U的一个公共幅度因子会被方向归一化抵消。不能用moment幅度变小推断参数更新同比变小；参数与方向的夹角也会影响投影后的实际步幅。

从同一个p1连续提交20次零梯度，count到21，mu/nu按原系数衰减，参数范数为2.236068，相对p1的距离达到3.330322。这些都是人工固定lr下的局部结果。图中moment和update曲线各自按初值归一化，不能把两条纵轴份额当成同一物理量。

公开10月7日元数据声明gate_router_weight_decay=0.02。[分组优化器源码](sources/optimizer_2026_10_05/optimizer.py)在该字段大于0时，给选中的attn_gate/router权重加衰减，并按Adam count退火。这里的AdamW是独立Optax控制，不是Hero那条自定义gate/router路径的运行结果；声明与历史执行仍需绑定。

## 复现环境也会改变冒烟测试能否运行

|执行方式|实际结果|允许的解释|
|---|---|---|
|原AdamH模块，CPU eager、SingleDevice输入|运行成功|验证人工矩阵的局部状态与更新|
|同一模块，无mesh的CPU JIT|ValueError：reshard需要非空mesh|JAX 0.7.2下typeof得到空AbstractMesh NamedSharding，原_pin_sharding尝试reshard；这是本环境复现边界|
|同一模块，单设备named mesh CPU JIT|运行成功，与eager在1e−6容差内一致|未修改原模块；不能因此证明多卡SPMD正确|

不能为让CPU测试通过就静默删除_pin_sharding。真实问题#8073的修复正依赖布局固定；本轮保留原文件并提供非空named mesh。无mesh异常不是Hero发生数值塌缩的证据。记录包含异常原文、原模块SHA和独立Optax函数来源SHA。

## 完整训练步应逐个决定哪些时钟推进

|状态/动作|固定Hero源码里的动作|空目标验收要补的证据|
|---|---|---|
|optimizer参数与moment|调用optimizer.update，再apply_updates|实际T、接受/跳过策略、更新前后参数与moment/count|
|训练step|next_state.step=state.step+1|是否以消费输入、完成更新或有效目标作为时钟|
|EMA|以新params更新EMA（开启时）|若仅丢弃最终update，EMA是否仍追向参数；本轮未执行EMA|
|QB|训练前应用pending betas，next_state接新metrics中的QB betas|跳过范围是否包括这些状态；本轮只静态读此路径|
|学习率与回调/保存|外层调度与日志/保存有各自入口|不能由组内count冻结推断整个循环冻结，需要实际完整步记录|

[源码](sources/scale_2026_10_05/train_hero_ep.py)这些动作并不因为某个输入梯度为零就自动全部停止。若策略要求跳过无效步，应明确跨rank一致的触发、数据游标如何处理，以及训练step、optimizer count、scheduler、EMA、QB、回调/保存各自的接受条件。这里只给验收要求，未植入跳步补丁。

[14项原模块/CPU与静态核对](analysis/zero_gradient_state_cpu.json)覆盖fresh/warm零梯度、moment/count递推、named-mesh JIT、丢update与保状态的后续差异、20步轨迹及独立Adam/AdamW对照。脚本导入完整原AdamH文件，未替换其内部函数，关闭源码目录pyc写入。完整Hero训练步和真实分组绑定仍未验证。

固定环境见[requirements](requirements-optimizer-cpu.txt)，执行`make optimizer-cpu CPU_PYTHON=/你的环境/bin/python`。正常report构建使用冻结CPU记录。配方的正分母条件参见[上一章](MASKED_NUMERICS_ZH.md)；目前没有实际Hero空目标步的证据。因此这些结果用于恢复、异常处理和管线迁移验收，不能用来解释某次生产loss变化。
