# 优化器分组：配置、状态时钟与实际更新

看到`weight_decay=0.1`，不能直接把Hero的所有权重更新理解成AdamW。这个训练入口使用自定义MuonH、AdamH、Adam三组变换，字段是否生效要沿`build`追查。本章新增一份固定源码，执行16项原掩码和衰减包装辅助函数检查。[源码](sources/optimizer_2026_10_05/optimizer.py) · [探针原值](analysis/optimizer_probe.json)

早期探针用NumPy、平面参数树和恒等Adam，只检查分组、衰减附加项与状态转发。V51增加真实Optax CPU分组裁剪对照和原AdamH模块检查；人工标签不证明真实模型分组，也没有执行完整Hero optimizer build、训练或checkpoint恢复。

## 1. 参数名决定进入哪一组

`create_mask`先检查名称，再检查维数。不是“所有二维矩阵都用MuonH”。

| 叶子路径/条件 | 组 | 为什么需要记录 |
|---|---|---|
| token_embed、router_bias、准确以`.router`或`.attn_gate`结尾 | Adam | embedding/router/gate虽可为矩阵，名称分支先命中 |
| output_proj或lm_head | AdamH | 输出投影采用独立变换 |
| gated_norm | MuonH | 比普通`.weight`规则优先 |
| 其余以`.weight`结尾 | Adam | 归一化增益和小SConv kernel靠这条规则分流 |
| 其余2/3/4维叶子 | MuonH | 扫描层轴与专家轴使矩阵堆叠成3/4维 |
| 其余 | Adam | 兜底，不应由名称猜测具体算法 |

因此，模块改名、包装层改变路径、`weight`与矩阵属性命名变化，可能改变优化器组，即使张量数值和形状相同。应导出实际参数树的“路径、shape、组、衰减mask”，而不仅保存optimizer名称。探针中的`router_extra`是假想路径，展示准确后缀的边界，不表示真实模型存在这个参数。

旧版`code_optimizer.py`把包含`.router`的路径分到Adam，新版改用准确后缀；标准`.router`与`router_bias`在两版仍进入Adam。不能从规则变窄推断真实模型已有叶子被误分，须核对实际参数树。

## 2. 缺失、零与正值不能混为一谈

新版`gate_router_weight_decay`默认`0.0`，仅在`>0.0`时启用专用包装；其余走普通`scale_by_adam`。旧版12d8b6f0的归档文件没有该专用衰减分支。

六份ladder optimizer记录为：

| 配置组 | 专用衰减字段 | 通用weight_decay | max_grad_norm |
|---|---|---:|---|
| 三份旧rav-ladder | 缺失 | 0.1 | None |
| 三份新h100-mix25 | 显式0 | 0.1 | None |

缺失字段是“尚未声明”，不是显式传入Python的`None`。在固定新版类按默认构造时，缺失会落到0；如果直接传入`None`，`None > 0.0`并不成立为合法比较。报告中的旧值null表示归档字段缺失，不能拿它推断运行时采用了None语义。

这个自定义`build`没有读取通用`self.weight_decay`来添加全参数衰减；专用分支读取的是`gate_router_weight_decay`。所以归档通用0.1并不能证明该分支为router施加0.1衰减。MuonH/AdamH内部又有自己的几何更新，不能把缺少AdamW附加项说成“整个系统没有正则化”。

在这两个源码版本下，缺失→显式0不构成专用衰减开关变化的证据。跨组的学习率、beta2、epsilon等差异仍在；真实执行SHA与解析后的配置尚未全部绑定。[配置差异核查](CHANGE_REVIEW_ZH.md)

## 3. 衰减作用于权重，排除router_bias

专用mask只匹配准确以`.attn_gate`、`.router`结尾的叶子。`router_bias`虽然在Adam组，却被专用衰减排除；token embedding、普通norm也不匹配这个mask。

正值分支的更新顺序是：

```text
按当前组做可选gradient clipping
→ scale_by_adam，得到自适应方向a_t
→ 对gate/router权重加 λ_t × p_t
→ 整个Adam组乘 −adam_lr_t
```

因此目标叶子的更新是`−adam_lr_t × (a_t + λ_t p_t)`。专用衰减项也乘Adam组学习率，不能只看lambda判断每步收缩。它是自适应方向之后的附加项，不参与该次Adam矩估计；不同于把L2项加到原梯度再送入Adam。

权重衰减可能影响router logits的幅度，进一步影响专家选择、sigmoid combine和beta估计，但实际幅度与训练效果未测。router_bias还走独立的pending beta设置路径，不能把衰减mask里没有bias解释成它被冻结。[路由状态](QB_ESTIMATION_ZH.md)

## 4. 衰减时钟来自Adam count

包装在更新前读取`state.count`，然后让底层Adam更新自己的状态，再使用：

`λ_t = λ_0 × max(1 − input_count / total_steps, 0)`

它直接返回底层next_state，没有另设衰减状态结构。源码设计意图是保留原Adam状态布局，方便从无衰减分支继续使用已保存的count与moments；结构相同不代替checkpoint兼容性实测。

人工设置λ0=0.2、total_steps=100；底层Adam用恒等方向替代，Adam组学习率人为设为0.01：

| 输入count | λ_t | 人工router参数 `[3,6]` 更新后 | 人工router_bias `[5,10]` 更新后 |
|---:|---:|---|---|
| 0 | 0.2 | `[2.989,5.983]` | `[4.995,9.995]` |
| 50 | 0.1 | `[2.992,5.989]` | `[4.995,9.995]` |
| 100 | 0 | `[2.995,5.995]` | `[4.995,9.995]` |
| 150 | 0 | `[2.995,5.995]` | `[4.995,9.995]` |

方向各元素人为设为0.5。这里验证的是原衰减包装的算术，不是Adam算法数值结果。

只恢复权重而重建optimizer时，count与moments可能重置；完整恢复则应保留。若重建的total_steps变了，即使count相同，衰减系数也可能不同。训练state.step、Adam count、外层学习率schedule count必须分别记录，不能假定三者永远相等。探针未执行真实恢复；真实差异应通过同checkpoint的一步输出与完整状态摘要确认。

## 5. clipping的位置决定它约束什么

`clip_by_global_norm`在三条组变换内部各出现一次，然后通过multi_transform按参数标签组合。结合[Optax参数分组接口](https://optax.readthedocs.io/en/latest/api/combining_optimizers.html#optax.partition)，应将这里理解为组内梯度范数裁剪，不能把它自动翻译为全模型只做一次global norm裁剪。早期这里只核对三个源码调用位置。V51用真实Optax 0.2.5 multi_transform执行人工标签对照，确认该组内归约语义；未执行完整Hero分组构建。

六份历史配置及10月7日归档声明的max_grad_norm均为None，因此没有证据表明这些运行启用了该裁剪分支。即便开启，裁剪发生在Adam自适应方向与附加衰减之前，不等于把最终参数更新范数限制到同一个阈值。MuonH与AdamH也各有后续变换，需分别核查最终update/parameter比。

## 6. 配比实验要冻结的不只是optimizer名称

配比改变梯度分布。如果同时重置moments、改变组标签、衰减时钟或学习率，loss改善可能来自不同更新规则；即使全部保持，optimizer对新分布的适应也是完整配比干预的一部分。

| 核查动作 | 保存的证据 | 允许的结论 |
|---|---|---|
| 确认真实分组 | 完整参数路径、shape、组与mask摘要 | 哪些叶子采用哪条更新路径 |
| 确认有效配置 | 解析后的默认值、专用/通用字段、执行SHA | 配置差异是否真的改变分支 |
| 确认状态时钟 | state.step、Adam count、schedule count、total_steps | 恢复后学习率与衰减是否接续 |
| 确认一步更新 | 固定权重/输入/state，保存梯度与最终update | 修复是否改变更新语义，不能代填长期能力 |
| 确认完整干预 | 固定预算和评估，记录稳定性与任务退步 | 新配比及其状态适应的整体效果 |

这些动作接入[训练变更评审](CHANGE_REVIEW_ZH.md)。专用字段不生效、mask误匹配、恢复时钟重置是三种不同的问题，需要不同证据；本轮没有认定Hero实际存在其中任何一种故障。

## V51：真实CPU裁剪对照与状态验收

控制实验用三个人工标签与梯度范数3、4、12，对应三条clip-only变换。标签按源码分组结构命名，不是实际Hero参数路径；真实Optax只在每个标签的active子树里计算其global norm。

|归约控制|三个组的输出范数|合并输出范数|结论范围|
|---|---|---:|---|
|不裁剪，identity分组|3、4、12|13|对应无clipping的API控制，不是完整训练更新|
|每组阈值1|1、1、1|1.732051|每组受约束，合并范数可以超过1|
|全模型先裁剪到1|3/13、4/13、12/13|1|保留组间相对比例；不同于逐组裁剪|
|原AdamH，首步gradient阈值0.01|输入gradient范数0.010000|参数update范数0.177336|梯度范数界不自动成为参数update范数界；两者单位不同|
|原AdamH，首步clipped/unclipped|首次参数update差3.79e-07|moment差0.546723|参数首次几乎相同，状态仍不同|
|同一后续非零gradient|后续update差0.149793|—|短期更新相似不能验收策略等价|

AdamH对Adam方向做归一化。在首步、epsilon影响很小的人工例子中，整个梯度乘正比例后，mu也同比缩放、nu按平方缩放，逐元素方向比例可能近乎抵消；但保存的mu/nu幅度已经改变。下一步固定同一参数输入、加入相同新梯度后，旧moment与新梯度的相对份额不同，差异重新进入方向。这个对照只在首步使用两种不同输入，后续参数与梯度输入共同；不是两条完整“持续开启/关闭裁剪”的训练曲线。lr=0.1、epsilon=1e−8和2×2矩阵均为人工控制，不外推到535B。

验收应保存原梯度的全模型/各组范数、每组clip系数与触发率、clipping后的梯度、moment/count、最终update/parameter比，并观察后续有效步。若microbatch采用不同有效分母，其梯度尺度也可能改变裁剪触发；应先固定归约口径，再比较裁剪策略。[梯度累积审计](GRADIENT_ACCUMULATION_ZH.md)的反例不能直接代填本次训练的触发频率。

当前公开归档max_grad_norm=None，危险/差异分支是迁移与配置变更的验收对象，不能作为已发生的Hero裁剪事故。没有执行真实create_mask、MuonH Newton–Schulz、完整multi_transform组链、调度、GPU/TPU或历史状态重放。

[10项CPU与源码检查](analysis/group_clipping_cpu.json)记录原值、人工标签、模块SHA和Optax函数来源SHA。[脚本](scripts/probe_group_clipping_cpu.py)使用真实Optax分组控制与未修改的AdamH模块；完整Hero build仍未知。使用V50的[CPU依赖](requirements-optimizer-cpu.txt)，入口为`make clipping-cpu CPU_PYTHON=/你的环境/bin/python`；通常report构建只读取冻结结果。


## V85：梯度有限，范数仍可能坏掉；裁剪全零也不等于停止更新

本节针对**可选裁剪分支**。公开10月7日配置归档的max_grad_norm=None；没有证据表明Hero实际启用了它。因此以下是迁移和配置变更的数值合同检查，不是Hero裁剪事故。本轮取得与近期train.py同一head b65be4c9550c5097f0a3add08933531a1c24d534的完整optimizer.py和adamh.py；执行真实Optax 0.2.5 clip与原AdamH模块，没有完整Hero build、模型loss或GPU。

### 溢出发生在梯度平方，不必发生在梯度本身

安装的Optax 0.2.5源码中，global_norm先对每个leaf做abs_sq和sum，再将各leaf求和并sqrt；clip比较g_norm<max_norm，否则每个leaf执行(t/g_norm.astype(t.dtype))*max_norm。它没有在这里显式把每个梯度leaf转成FP32。对有限FP16矩阵，平方已经可能超出表示范围；先出错再求和，后面的有限性检查无法恢复原范数。

用2×2同值人工梯度，执行原库函数、独立float64范数参考和“先整体转FP32裁剪，再cast回原dtype”的比较控制。[19项CPU检查](analysis/clipping_precision_cpu.json)包括eager/JIT一致性；比较控制没有修改上游，也不是经过训练验证的修复。

|梯度dtype / 每项数值 / 阈值|原计算范数|原裁剪后的独立float64范数|FP32裁剪再cast回后的独立范数|
|---|---:|---:|---:|
|FP16 / 300 / 1|Infinity|0|1|
|FP16 / 0.0001 / 0.00001|0|0.0002000332|0.00001001358|
|BF16 / 300 / 1|600|1|1|
|BF16 / 0.0001 / 0.00001|0.0002002716|0.00001001358|0.00001001358|
|FP32 / 300 / 1|600|1|1|
|FP32 / 0.0001 / 0.00001|0.0002000000|0.0000100000|0.0000100000|

第一行的四项梯度均有限，真实范数600；原FP16平方/累积得到无穷，有限梯度除以它成为全零，输出也全有限。第二行的实际范数约0.0002000332，约为阈值20倍；平方下溢得到范数0，trigger选择原梯度，裁剪根本没有触发。两种情况都不能靠“裁剪结果没有NaN”排除。

FP32比较控制避免了这两例的范数失败，但cast回低精度仍有舍入。小值控制的最终范数0.00001001358略高于阈值；不能把该控制写成“任何dtype下严格保证≤阈值”。本轮的BF16/FP32六个对应值没有出现FP16这两类失败，不能外推任意幅值、leaf数量或分布式归约都安全。

### 范数失败会怎样进入已有动量

人工FP32参数从[[2,0],[0,1]]开始，原AdamH用梯度[[1,4],[2,3]]预热一次。然后从同一预热参数和moment/count分成两路：FP16范数溢出得到的全零梯度，以及FP32范数控制得到的范数1梯度；两路均转FP32后送入同一原AdamH，排除其输入dtype差异。

全零一路的count仍从1到2，mu和nu分别按0.9、0.95衰减，参数update范数为0.1882047；正确保留梯度的一路update范数为0.1802129，两路update差的L2为0.0146983。这个合成控制把“范数溢出→有限全零梯度→已有状态继续更新”接到实际原模块。它没有语言loss，没有完整分组、学习率调度或生产checkpoint；数值不表示Hero的实际误差量级。

### 应该检查哪个dtype

不能从activation dtype直接推断进入裁剪的gradient dtype。近期原train_step在FP32 pinned-host master模式下先_FP32_POLICY.cast_to_param(grads)，再调用optimizer.update；另一分支直接传grads。该顺序是固定源码静态证据，本轮没有执行完整master路径，也不知道某次Hero实际gradient dtype。

V51已验证裁剪位于组内变换，而不是全模型只做一次；本节则检查同一leaf的范数数值。组范围、实际梯度dtype、范数平方/累积dtype和裁剪后cast必须一起记录。给一个max_grad_norm数字不足以定义完整行为，训练CE有限或activation为BF16也不能代填这些字段。

迁移验收可以按四层保存：原梯度有限性及独立高精度范数；原clip使用的norm/dtype与系数；clip后的梯度；同状态optimizer更新与下一状态。发现有限全零时，先辨别是真正无梯度还是范数溢出；发现norm=0时，检查是否为平方下溢。提升范数精度是本控制中的有效区分手段，但是否要改实现、如何保持分片和性能，仍需实际执行包、GPU与训练评估。

新增2份同head源码，旧489份非bookkeeping来源字节保持，来源现492份；[获取台账](analysis/clipping_precision_acquisition.json)可复核。归档声明裁剪关闭、真实gradient dtype、GPU与训练收益分别记录，不能把可选分支反例改写成生产根因。复现：`make clipping-precision-cpu CPU_PYTHON=/tmp/marin-jax-cpu-072/bin/python`。
