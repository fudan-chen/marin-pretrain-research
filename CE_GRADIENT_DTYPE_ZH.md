# FP32 loss 不保证 FP32 梯度计算

V140，2026-10-08。继续执行原注册 custom VJP，核对两个不同位置的精度：batch 块之间的权重梯度累积，以及进入反向 GEMM 之前的 dlogits 转换。14 项 CPU 控制、14 组原值说明，前向 loss 相同且梯度有限，也不保证更新量相同；同时，新 scan 反向确实避开了所选旧累积路径的误差。

## 先把四种 dtype 分开

[原 scan 反向](sources/contracts_2026_10_05/xla.py)的流程是：投影累加 → 按请求转换 logits → 提升到 FP32 计算概率和 dlogits → 将 dlogits 转为 `result_type(x.dtype, w.dtype)` → 反向 GEMM 请求 FP32 累加 → 输出转换回输入/权重的 dtype。

这意味着四项声明不能互相替代：logits 的 dtype、softmax 的 dtype、反向 GEMM 操作数的 dtype，以及梯度累积/输出的 dtype。[JAX 官方说明](https://docs.jax.dev/en/latest/_autosummary/jax.lax.dot.html)将 `preferred_element_type` 定义为输出 dtype，并在未指定 DotAlgorithm 时作为累加 dtype 的编译器提示；它不是硬件执行精度证明，也不会找回进入 GEMM 前已经舍掉的数。

普通 slow 反向则在每个 batch 块产生权重梯度后，转为 w 的 dtype，再加进同 dtype 的累计 buffer。scan 反向在跨 batch 块累计时保留 FP32，最后才转成 w.dtype。精度变化发生在加法之前还是之后，会改变结果。

## 普通 BF16 输入：512 次小增量为何只积到一半

控制使用 512 行输入，每行隐藏特征都是 1，分类头为两个零权重，标签都为合法 ID 0。输入与头为 BF16，logits 为 FP32，目标是平均 CE。每行贡献的 head 梯度为 `[-1/1024,+1/1024]`，精确总量是 `[-0.5,+0.5]`。不存在非法标签、非有限 logits 或全空目标。

|原注册反向路径|batch 块大小|平均 CE|最终 head 梯度|观察|
|---|---:|---:|---|---|
|slow，BF16 参数|1|约 ln(2)|[-0.25,+0.25]|每块回到 BF16 再累积，损失一半总量|
|slow，BF16 参数|32|约 ln(2)|[-0.5,+0.5]|所选较大块对照正确|
|slow，BF16 参数|512|约 ln(2)|[-0.5,+0.5]|一个 GEMM 先完成求和，再转换|
|scan，BF16 参数|1|约 ln(2)|[-0.5,+0.5]|跨块 FP32 累积保留总量|
|scan，BF16 参数|512|约 ln(2)|[-0.5,+0.5]|与所选小块 scan 一致|
|slow，FP32 参数|1|约 ln(2)|[-0.5,+0.5]|排除块调度本身必然丢量的解释|

为何卡在 0.25？在本例 BF16 累积中，到达这个量级后，相邻可表示值间隔已大于下一次增量；加法结果舍入回原值，后续同样的增量无法继续推进。这是逐次舍入的效果，不是所有梯度都整体乘了某个常数，也不能靠事后统一乘 2 修复一般输入。

源码中的[模型注释](sources/contracts_2026_10_05/model.py)已经明确说明，`xla_fast_bwd` 改变了反向数值，并记录作者在 GB200、指定 CE shape 上相对 FP32 参考的 rel-RMS 改善：grad_x 从 2.170e-3 到 1.445e-3，grad_w 从 1.946e-3 到 1.433e-3。这是作者记录，**本轮没有复现那组 GPU 数字**。我们的 50% 差值属于刻意选定的 CPU 微型累积例子，也不是 Hero 的梯度误差估计。

同一固定模型还声明 `b_block_size=65536`；不能把本次 block=1 的反例套成该声明已经丢失一半梯度。当前目的在于解释为什么分块与累积 dtype 属于更新语义，以及新实现具体保护了什么。

## FP16 输入：最终梯度可表示，中间 dlogits 仍可能先归零

第二组仅用一行输入，隐藏特征刻意设为 4096，分类头仍为零，目标 ID 0。x/w 为 FP16，logits 与概率计算为 FP32。设目标为 `CE/T`，模拟某个全局平均目标向这一行传入 `1/T` 的上游 cotangent；**没有分配或执行完整全局 batch**。4096 用于放大 GEMM 后的梯度，从而区分“最终输出不可表示”与“更早已经丢量”，不是实际 Hero hidden 的观测。

当 T=2^26，进入 GEMM 前的 dlogit 大小为 2^-27，小于本次 FP16 转换可保留的范围；转换后变成零。但乘上隐藏特征 4096 后，期望的 head 梯度大小是 2^-15，可以由 FP16 表示。

|输入/权重 dtype 与路径|T 或控制|反缩放后的 head 梯度|结论范围|
|---|---|---|---|
|FP16，slow|2^26|约 [-3.051758e-5,+3.051758e-5]|先 GEMM 后转换保留所选最终梯度|
|FP16，scan|2^26|[0,0]|GEMM 前 dlogits 转换丢量|
|FP16，scan，loss 乘 8|2^26|与 slow 相同|本次二次幂控制可精确恢复|
|FP16，scan|46,126,080|[0,0]|另一种归一化尺度也触发所选路径|
|FP16，scan，loss 乘 128|46,126,080|约 [-4.386902e-5,+4.386902e-5]|非零，但不等于 slow 的约 4.440546e-5|
|BF16，scan|46,126,080|约 [-4.434586e-5,+4.434586e-5]|本例没有整体归零，仍有舍入|
|FP16 x、FP32 w，scan|46,126,080|约 [-4.440004e-5,+4.440004e-5]|共同 GEMM dtype 为 FP32，所选早期归零消失|

大尺度 T 来自归档声明的 batch=11264、max_seq_len=4096，在“每条完整序列只屏蔽末目标、无其他 mask”的假设下计算 `11264×4095`。它不是实际权重和的测量，也不能替代真实 shard_map 反向的 cotangent 记录。

缩放对照说明转换位置重要，不是建议直接给 Hero 加一个 loss scale。实际缩放必须在优化器更新前正确反缩放，处理溢出、累积与状态；非二次幂控制仍有量化误差，GPU 对次正规数的行为也未测试。本轮没有实现动态缩放器或训练补丁。

## Hero 声明给出了哪些反证与未知

[step146k 归档元数据](sources/wandb/hero-fa4sm100-nomask-step146k_meta.json)声明 compute/output 为 BF16、param 为 FP32。因此，本次“x/w 都为 FP16”的失败条件不能直接用于描述这份 Hero 声明。计算 dtype、主参数 dtype 和实际 CE 操作数 dtype 仍需要分别确认，不能只凭其中一项推断另一项。

同样，fast 路径的改写有明确受控收益：本次 BF16 跨块累计反例被避免。不能从 FP16 的另一条反例推出“scan 总比 slow 差”，也不能从作者平均误差更小推出每个输入都更准。

## 对配比和后端切换的验收规则

1. 把 batch/词表分块参数与 dtype 作为执行身份保存；同一 loss 曲线不能证明更新量相同。
2. 梯度比较至少覆盖一个普通非零累积例子；同时查看零值比例、逐叶误差和累计总量，不能只查 finite。
3. 记录归一化位置与实际 cotangent 的范围。全局平均、分片局部平均、microbatch 累积和 loss scaling 会让转换前的小量处于不同尺度，不能无条件互换。
4. 保持相同全局目标，分别改变操作数 dtype、累计 dtype、块大小，再定位差异。增大块可能增加内存，需另外验收性能，不能默认可部署。
5. 配比变化若同时改变有效目标数或 dtype/后端，先核对更新语义；本轮反例没有提供任何域的质量排名或最优比例。

## 复现与边界

[14 项断言和逐项原值](analysis/ce_gradient_dtype_cpu.json) · [脚本](scripts/probe_ce_gradient_dtype_cpu.py)。运行 `CPU_PYTHON=/tmp/marin-loss-mass-v109/bin/python make ce-gradient-dtype`。脚本仅加载[上一轮探针](scripts/probe_ce_custom_vjp_cpu.py)的定义前缀，原 custom VJP 装饰器与注册表达式保留；不会重写上一轮结果。四份来源、基础脚本与当前脚本 SHA 均绑定。

固定源码 `84869ae8c91ffe64e9f761c5bd714542eb1876e0`，JAX 0.7.2 CPU，显式选择 CPU exp/logsumexp/logaddexp。没有执行公共 dispatcher、GPU/tensor core、真实全局 batch、实际 Hero 梯度、优化器一步更新或长期训练效果。本轮没有上游补丁。
