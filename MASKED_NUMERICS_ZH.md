# 零权重、非有限值与反向梯度

零权重表示某个位置不应贡献目标，但浮点运算不会自动把这个位置从计算图删除。本轮在独立JAX CPU环境运行固定源码中的归约、参考CE与fast backward scan函数体，确认了两个不同风险：全零权重可以让前向loss返回0而反向出现NaN；inactive位置的NaN/Inf也不能靠最后乘0隔离。

这比上一轮NumPy适配检查多了一项真实证据：已经执行JAX自动微分和CPU编译。但没有执行Hero checkpoint、完整模型、GPU/TPU内核、分布式collective或完整optimizer step。也没有证明真实Hero batch出现了触发条件。

## 从源码追到哪一步

固定源码版本为84869ae8c91ffe64e9f761c5bd714542eb1876e0。

|位置|实际运算|需要检查的触发条件|
|---|---|---|
|[API归约](sources/contracts_2026_10_05/api.py)，_apply_reduction|先loss乘weight，mean时用where选择总分子/总weight或0|总有效weight为0；或零权重位置的loss已经非有限|
|[Grug归约](sources/scale_2026_10_05/grug_loss.py)|分片分子/分母求和后，同样用where保护除法|应检查聚合后的全局分母；单rank为空不等于全局为空|
|[原参考CE](sources/contracts_2026_10_05/reference.py)|dot得到logits，再求logsumexp与label logits差|本轮运行原CPU分支；不是训练选择的完整xla_fast_bwd入口|
|[原fast backward scan](sources/contracts_2026_10_05/xla.py)|概率乘上游cotangent，再与x、w做GEMM|NaN cotangent会传播；非有限概率/operand乘0仍可能非有限|

Grug传给fused kernel的是`reduction=None`：内部只做加权，外部才汇总分子/分母并求mean。因此下面API mean和Grug mean是两项不同入口的检查，不是对Hero loss连续执行两次mean。一个rank或microbatch为空也不能证明归约后的全局T为0。

JAX的[where说明](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.where.html)与[FAQ](https://docs.jax.dev/en/latest/faq.html#gradients-contain-nan-where-using-where)解释了反向中的这一限制：选择有限分支，不保证另一个分支的未定义运算不会污染导数。这里的CPU检查针对具体原函数，而不是仅凭文档猜测。

## 实际CPU结果

|检查|前向结果|反向或后续结果|
|---|---|---|
|原归约，有限loss、全零weight、mean，eager与jit|0|对loss的cotangent全NaN|
|原参考CE接原mean归约，全零weight，eager与jit|0|hidden与head梯度全NaN|
|同一原参考CE，改用sum归约、全零weight|0|hidden与head梯度全0且有限|
|提出的安全分母归约，有限loss、全零weight|0|cotangent全0；这是独立对照实现，未修改上游|
|正分母，但零权重位置loss为NaN或Inf|NaN|最后乘0不能排除非有限值|
|log计算后才where选择inactive结果，再安全归约|0|inactive输入仍有NaN梯度|
|在log之前选择有效operand，再安全归约|0|inactive梯度为0；这个例子只验证log的处理|
|原fast backward scan接入原mean产生的NaN cotangent|不重算前向|hidden与head梯度全NaN|
|同一scan接入安全归约产生的零cotangent|不重算前向|hidden与head梯度全0且有限|
|同一scan，inactive row的hidden含NaN、该row cotangent为0|不重算前向|该row hidden梯度与head梯度仍出现NaN|
|原Grug单设备归约接原CPU参考CE delegate，全零weight|0|hidden与head梯度全NaN|

原mean检查与reference CE分别执行了eager和jit。另有一个有限float32正例：3行输入、词表3项，以2行/2词分块，包含尾行和词表padding；原scan与reference的hidden/head梯度在1e−6绝对/相对容差内一致。容差只用于这个小例子，未标定Hero数值误差。

原scan直接接受提供的cotangent，在CPU运行其scan与GEMM；没有运行完整custom_vjp入口、自动backend选择、tensor core或BF16训练路径。Grug检查将mesh明确设为None，backend dispatcher替换为原CPU reference delegate；它验证归约组合，不验证多卡训练。

## 怎样解释，怎样修

第一类是除法图的问题：T=0时，即使where前向选择0，另一分支仍含N/T。先把除法分母变成一个在T=0时也有限、非零的值，再按预先定义的空步策略选择结果，才能排除这段除零反向风险。CPU对照中的安全归约保留了正分母值，并在有限输入全零weight时得到零cotangent。这个局部改写不负责修复上游NaN。

第二类是上游运算的问题：NaN×0与Inf×0仍会得到NaN。只在loss算出后做where，可以改善前向值，却仍可能留下未定义导数；log例子中，需要在log之前选择安全operand。实际CE不能机械套用log例子的修法。应针对被排除的位置检查logits、概率、激活、梯度与custom backward，并确认inactive operand如何处理；对活跃位置的非有限值应定位和拒绝，而不是靠nan_to_num制造有限曲线。

即使数值上得到零梯度，也不等于完整训练步不发生变化。已有moment、weight decay、EMA、QB pending状态和scheduler可能继续改变状态。应先决定是接受空步、跳过哪些状态，还是报错；再核对实际执行。参见[训练状态](TRAIN_STATE_ZH.md)、[失败边界](FAILURE_BOUNDARIES_ZH.md)和[梯度累积分母](GRADIENT_ACCUMULATION_ZH.md)。

|验收层次|最少材料|不能用什么代替|
|---|---|---|
|触发是否实际存在|实际global T、各rank/各块T、weight来源、输入身份|不能用人工全零mask或单rank空块证明Hero触发|
|活跃与inactive数值|各类位置的非有限计数、CE前后、cotangent、grad_x/grad_w|不能用loss有限代替梯度有限；MoE assignment drop也不自动等于CE mask|
|单设备原函数|相同参数/输入的eager、jit、reference与custom backward对照|本轮CPU结果不能代替GPU/TPU、分布式或历史环境|
|完整训练步|参数、moment、计数、EMA/QB/调度状态更新前后|零梯度或loss=0不能代替状态不变|
|归因与修复|实际事件触发记录、修复前后同输入/状态重放|本轮未修改Marin代码，不能宣布修复了其训练事故|

## 复现与证据范围

[CPU脚本](scripts/probe_masked_numerics_cpu.py)使用AST选取原函数体，并通过future annotations载入。参考CE显式选择原_default_logsumexp CPU分支，CPU exp为jnp.exp；不运行TPU accuracy分支。没有用NumPy模拟自动微分。所有数组转回host后记录，非有限值以字符串NaN/Inf写入合法JSON。

实际环境：Python 3.12.13，JAX 0.7.2，jaxlib 0.7.2，NumPy 2.5.3，backend=cpu，x64=False。这是独立研究环境，未证明与Hero执行环境一致。固定依赖见[CPU requirements](requirements-cpu-numerics.txt)，重跑入口为`make cpu-numerics CPU_PYTHON=/你的环境/bin/python`；常规report构建读取冻结结果，不隐式安装JAX。

本轮21项检查的原值、函数行号、适配说明与源码SHA见[结果记录](analysis/masked_numerics_cpu.json)。当前缺口仍是真实Hero全零分母或inactive非有限operand的事件证据、历史运行版本绑定、真实内核及完整训练步重放。下一步优先获取这些材料，决定是否达到修复条件。

## 这个触发条件是否符合公开Hero配方

[默认权重审计](analysis/default_target_weights.json)读取10月7日归档配置：223个组件均声明普通text格式、pack=None、长度4096、batch 11264。固定版_effective_pack对text返回False；dataset_for_component选择TokenSeqDataset，不指定loss_weights_key，再由CausalLmDataset构造默认causal mask，未传ignore_id或外部segment_ids。原causal_loss_mask只有最后位置为0，每条有4095个有效位置。因此正常整批条件下，T应为11264×4095=46,126,080，严格为正。

这不是实际global T测量：代码版本与声明必须绑定实际执行、cache必须正常提供整批序列、权重必须没有外部变更。该审计的9项检查使用原函数体、NumPy和dataset构造描述器，没有运行真实loader或读取实际缓存。长度1或空completion的另一个配方可给出空mask；不能把这些人工迁移反例写成目前Hero发生了全零分母。

正分母排除的是全零归约条件，不保证每条序列被屏蔽末位的hidden、logits或上游导数有限。inactive非有限风险仍需另查；不能因为T为正就宣布数值验收通过。

故当前结论是：数值危险分支已在CPU复现，但公开正常配方的条件推导不支持用它解释Hero历史loss。调查应先记录实际T与inactive非有限值，确认触发，再进入修复。
