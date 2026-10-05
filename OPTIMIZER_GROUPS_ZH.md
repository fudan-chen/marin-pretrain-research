# 优化器分组：配置、状态时钟与实际更新

看到`weight_decay=0.1`，不能直接把Hero的所有权重更新理解成AdamW。这个训练入口使用自定义MuonH、AdamH、Adam三组变换，字段是否生效要沿`build`追查。本章新增一份固定源码，执行16项原掩码和衰减包装辅助函数检查。[源码](sources/optimizer_2026_10_05/optimizer.py) · [探针原值](analysis/optimizer_probe.json)

探针用NumPy、平面参数树和恒等Adam替代依赖，只检查分组、专用衰减的附加项与状态转发。它没有运行真实Adam矩估计、Optax multi_transform、JAX分片或训练，不证明真实checkpoint可以无误恢复。

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

`clip_by_global_norm`在三条组变换内部各出现一次，然后通过multi_transform按参数标签组合。结合[Optax参数分组接口](https://optax.readthedocs.io/en/latest/api/combining_optimizers.html#optax.partition)，应将这里理解为组内梯度范数裁剪，不能把它自动翻译为全模型只做一次global norm裁剪。这是源码结构与接口语义推断，本轮只检查三个调用位置，没有执行真实multi_transform验证。

六份配置的max_grad_norm均为None，因此没有证据表明这些运行启用了该裁剪分支。即便开启，裁剪发生在Adam自适应方向与附加衰减之前，不等于把最终参数更新范数限制到同一个阈值。MuonH与AdamH也各有后续变换，需分别核查最终update/parameter比。

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
