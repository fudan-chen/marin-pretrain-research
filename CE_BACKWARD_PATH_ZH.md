# 同一前向，两条反向：参考实现的异常能否代表训练后端

V139，2026-10-08。上一章检查非法标签，本章使用合法标签，比较完整 logits 参考、直接求导的 streaming 参考，以及原 XLA streaming custom VJP 的两条反向。新增 14 项 CPU 控制，执行原装饰器、原 `defvjp` 注册、原前向与原 slow/scan 反向，不再只是手工向 scan 传 cotangent。

结论需要同时保留两半：选定低精度控制中，参考函数的直接自动微分会产生 NaN；原 custom VJP 的两条反向在同一输入上保持有限并与完整参考一致。不能只报道前一半，再把它当作生产内核已坏的证据。

## 先看实际执行了哪条求导路径

[原源码](sources/contracts_2026_10_05/xla.py)通过 `jax.custom_vjp` 装饰 streaming 入口，再用 `defvjp` 注册前向残余与反向规则。前向保存 x、labels、w 和最终 logsumexp；反向物化 loss/lse 的 cotangent，然后按 fast_backward 选择普通循环或 scan。根据 [JAX 官方说明](https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.html)，注册后，反向模式使用指定的导数规则，而非直接对原函数体求导。

因此，对 streaming 参考函数直接调用 `jax.grad`，与对注册 custom VJP 的入口调用 `jax.grad`，并不是同一项测试。只运行参考求导，无法证明显式反向的行为；只直接调用反向子函数，也无法证明注册、残余与 cotangent 接续正确。本轮把这些连接一起执行。

## 合法输入如何得到全为负无穷的局部块

控制输入为一个隐藏特征 1，分类头 `[-100000,-100000,0]`，目标为合法 ID 2。输入数组均为有限 FP32，但要求 CE logits dtype 为 FP16 后，前两项变成 `-inf`。这不是 FP32 logits 的普通运行范围，而是刻意触发缩窄转换的诊断输入。

当词表分块大小为 2，第一块全为 `-inf`，第二块包含目标类别与末尾 padding。第一块的 logsumexp 也为 `-inf`；最终加上第二块后，前向仍得到有限 logsumexp 和零 CE。可是直接对分块归约过程求导，会经过全为 `-inf` 的局部 logsumexp，其反向在本次运行中出现 NaN。最终前向有限，并未消除这个中间导数问题。

<figure aria-labelledby="ce-backward-route-title">
<svg viewBox="0 0 900 300" role="img" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto"><title id="ce-backward-route-title">Selected CPU control: identical finite forward, different backward routes</title><defs><marker id="ce-backward-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#52657a"/></marker></defs><rect width="900" height="300" rx="12" fill="#f4f7fb"/><g font-family="system-ui,sans-serif" font-size="17" fill="#18324d"><rect x="20" y="110" width="240" height="80" rx="9" fill="#fff" stroke="#71869a"/><text x="38" y="141">FP16 streaming forward</text><text x="38" y="169">selected CE = 0 (finite)</text><path d="M260,150 L320,150 L320,70 L370,70" fill="none" stroke="#52657a" stroke-width="2" marker-end="url(#ce-backward-arrow)"/><path d="M320,150 L320,230 L370,230" fill="none" stroke="#52657a" stroke-width="2" marker-end="url(#ce-backward-arrow)"/><rect x="380" y="28" width="490" height="85" rx="9" fill="#fff1ee" stroke="#a84b3a"/><text x="399" y="61">Direct differentiation of reference blocks</text><text x="399" y="90">head gradient = [NaN, NaN, 0]</text><rect x="380" y="188" width="490" height="85" rx="9" fill="#edf8f2" stroke="#327454"/><text x="399" y="221">Original registered slow / scan custom VJP</text><text x="399" y="250">head gradient = [0, 0, 0] (finite)</text></g></svg>
<figcaption>仅表示本次合法标签、FP16 极端转换控制。不是对任意输入的稳定性承诺，也不是 GPU 性能图。</figcaption>
</figure>

## 原值对照：分块参数与反向规则都要记录

|路径与精度|词表块大小|前向 CE|hidden/head 梯度|说明|
|---|---:|---:|---|---|
|完整 logits 参考，FP16|无分块|0|均有限，零|合法目标位于有限 logit 类别|
|streaming 参考直接自动微分，FP16|2|0|head=[NaN,NaN,0]|局部全负无穷块的求导边界|
|streaming 参考直接自动微分，FP16|3|0|均有限，零|整个词表放同一块，所选异常消失|
|原注册 custom VJP，slow 反向，FP16|2|0|均有限，零|与完整参考一致|
|原注册 custom VJP，scan 反向，FP16|2|0|均有限，零|与完整参考一致；本例 JIT 也一致|
|streaming 参考直接自动微分，BF16 或 FP32|2|0|均有限，零|本例负大数在这些 logits dtype 中没有变成负无穷|

显式反向为什么避开了所选异常？它使用最终全词表 logsumexp，直接计算概率与目标减项；在本例中，负无穷类的概率为零，目标类概率为一。不需要对那个全为负无穷的局部 logsumexp 逐层反向。原 w 和 x 仍是有限 FP32，后续梯度矩阵乘法在所测输入上也保持有限。

这是一条有条件的实测解释。若实际参数已经含 Inf、最终 lse 非有限、目标非法或 cotangent 本身异常，不能沿用本例的保护结论。把 block_size 改成全词表大小只验证了一个诊断对照，也会改变内存；本轮没有把它作为可部署修复推荐。

## 不只测零梯度：正常目标与罚项也经过原注册链

为了避免只用饱和零梯度给出过弱的通过证据，另设分类头 `[-1,0,1]`、同一合法目标 2。完整参考、直接 streaming、slow custom VJP、scan custom VJP 的 loss、hidden 梯度与 head 梯度在 FP32 小例子的既定容差内一致。

还比较诊断目标 `CE + 0.1 × lse²`。这会同时向两个输出传入 cotangent，检验原残余和两路反向的接续。四条路径的值与两种参数梯度再次一致。0.1 只是人工测试系数，不是 Hero 配方或建议的 z-loss 权重；这项控制也不验证整个公共 API 对罚项的组装。

## 给 kernel 切换评审增加什么规则

1. 记录实际求导入口：完整参考、参考自动微分、注册 custom VJP、公共 dispatcher 或真实设备 kernel。不能只写“测过 backward”。
2. 固定 x、w、目标与上游 cotangent；同时保存输入 dtype、logits dtype、反向 GEMM dtype、词表块和 batch 块。BF16 激活不等于 FP16 logits。
3. 同时检查 loss、lse、输入梯度与权重梯度。覆盖正常非零梯度、罚项 cotangent、末尾 padding，再研究极端转换与全空目标。
4. 参考异常和候选异常分别定位。参考路径可能并不适合作为某个数值边界的唯一裁判；候选在一个反例上通过也不证明所有输入正确。
5. 回到真实训练前，绑定实际 backend 选择、编译参数及状态，再做同起点更新和长窗验收。本地源码链通过不能替代设备内核、性能与训练稳定性证据。

这对配比实验的意义是：若两次运行同时换了数据与后端，不能默认“相同前向 loss”就代表相同梯度路径。先完成固定输入的后端对照，再解释各域 loss 与能力差异。本轮没有为任何 Marin 桶推导新增边际收益。

## 复现与未覆盖项

[14 项控制及 16 组原值](analysis/ce_custom_vjp_cpu.json) · [脚本](scripts/probe_ce_custom_vjp_cpu.py)。运行 `CPU_PYTHON=/tmp/marin-loss-mass-v109/bin/python make ce-custom-vjp`。源码版本固定为 `84869ae8c91ffe64e9f761c5bd714542eb1876e0`，原 [reference.py](sources/contracts_2026_10_05/reference.py) 与 [xla.py](sources/contracts_2026_10_05/xla.py) 的字节 SHA 和脚本 SHA 均保留。

通过 AST 加载原函数和注册表达式，CPU logsumexp/exp/logaddexp 分支被显式选择；没有导入全部模块或调用公共 backend dispatcher。未执行 GPU/tensor core、自动调优、多设备 custom VJP、真实 Hero FP16 logits 事件、完整优化器更新或长期 loss 曲线。本轮没有上游补丁。历史[零权重数值章节](MASKED_NUMERICS_ZH.md)中“未执行完整 custom VJP”的表述保留原版本范围，不能用于否定本轮新增的受控注册链执行。
