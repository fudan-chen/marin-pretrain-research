# 路由均衡：分位数估计、shard分组与数据顺序

路由均衡有自己的状态更新，不能简单当作“加一个均衡loss”。本章对照固定源码，执行16项原辅助函数检查，分析两条估计路径怎样受批次结构影响。[原值与检查](analysis/qb_partition_probe.json)

首先限定适用范围：六份归档ladder配置均声明`qb_estimator=HIST`、`qb_hist_bins=10000`。模型源码默认TOPK，不代表这些配置采用默认值；真实部署还需要执行SHA绑定。本章TOPK算例说明另一条可选路径的性质，不能据此认定这些HIST运行有同样的分组问题。HIST算例也没有复现真实模型、collective或历史训练效果。

## 1. 当前批次决定下一步bias

[MoEMLP](sources/contracts_2026_10_05/model.py)先在`router_logits + stop_gradient(router_bias)`上取K+1项。第K+1项作为每token的alpha，前K项作为选择的专家。用于估计beta的margin却是`router_logits - alpha`：前者未加bias，后者已受当前bias影响。

[训练入口](sources/scale_2026_10_05/train_hero_ep.py)在本步计算前应用上一批次的pending beta；本批次估出的beta保存为下一步pending。`apply_qb_betas`将beta取负，并减去每层专家均值，直接设置bias，不是把一个sign步长累加到原bias。bias的stop_gradient也不等于数据没有影响bias：数据通过这条显式状态更新路径影响下一步路由。

```text
本步开始：读取 pending beta → 设置中心化 bias
本步前向：biased logits 选专家与 alpha
本步统计：unbiased logits − alpha → 估计新的 beta
本步结束：新 beta 存为 pending，下步才用于训练
```

因此，配比切换第一步仍带着上一批次形成的路由状态。不能把第一步变化全解释为新配比的稳态表现；也不能事后擅自清空pending来“消除干扰”，那会额外改变干预。应先决定研究的是保留生产状态的完整切换，还是明确控制路由状态的机制对照。[训练状态与恢复](TRAIN_STATE_ZH.md)

## 2. TOPK：局部分位数的平均不等于全局分位数

[原辅助函数](sources/routing_2026_10_05/grug_moe.py)对每个shard计算：

- physical count = max(1, floor(local positions × K/E))，用于静态top_k形状；
- logical count = clip(floor(valid tokens × K/E), 1, physical count)，读取对应的局部次序统计量；
- beta按各shard有效token数加权，零有效token的shard返回零且权重为零。

有效数加权解决的是不同shard计数不等；它不能把局部分位数变成全局分位数。分位数不是可通过这样平均合并的统计量。

使用4个token、2个专家、K=1、当前bias为零的人工logits：

| token | expert 0 | expert 1 |
|---|---:|---:|
| t0 | 100 | 0 |
| t1 | 99 | 0 |
| t2 | 0 | 1 |
| t3 | 0 | 2 |

第K+1项均为0，因此这些数也就是margin。全局精确TOPK次序阈值为`[99, 1]`，以下两种分组使用同一组token、相同有效数：

| 分组 | 局部beta | 原reduce函数的beta | 中心化bias | 下一人工query `[60,0]`选择 |
|---|---|---|---|---|
| shard A=t0,t1；B=t2,t3 | `[100,0]`、`[0,2]` | `[50,1]` | `[-24.5,24.5]` | expert 0 |
| shard A=t0,t2；B=t1,t3 | `[100,1]`、`[99,2]` | `[99.5,1.5]` | `[-49,49]` | expert 1 |

局部阈值和归并执行的是原函数；top_k采用NumPy替代，bias与下一query是显式人工算术。没有执行真实模型或跨设备通信。这个反例证明该估计器一般不具备分组不变性，不证明真实训练产生上述数值或能力退步。shard内部单纯换行序不会改变次序统计量；问题是改变哪些token被放到同一个shard。

当局部有效数很小，目标数会夹到1，估计退化到局部最大margin。例如E=384、K=8，少于48个有效token时也是取第一项。应记录局部有效数和实际logical count，不能只看全局batch。有效数是否真的很小，需要真实输入证明。

## 3. HIST：全局计数减少分组依赖，但网格也会变化

[HIST实现](sources/contracts_2026_10_05/model.py)先在全部有效margin上求全局lo、hi，所有专家共用这一个范围。它对每个专家做直方图，再跨设备累加整数计数，在上尾累计数跨过目标rank的区间内插值。padding不参与范围和目标rank；全空批次由外层返回零，退化相等范围用`lo+1e-6`保护。

在共同网格、相同margin与valid mask下，整数直方图可合并。人工算例中换token行序不改变结果，这不同于前一节局部分位数加权。它仍是有网格近似的估计，不是直接读取精确次序统计量。

需要注意所有专家共用范围：

| 人工输入 | 全局范围 | bins | 区间宽度 | expert 0的估计beta |
|---|---|---:|---:|---:|
| 上述原margin | 0～100 | 10000 | 0.01 | 99 |
| 仅把expert 1最后一项由2改成1000000 | 0～1000000 | 10000 | 100 | 约66.6667 |

expert 0的四个margin始终是`[100,99,0,0]`，精确第二大值始终99；它的近似阈值却发生变化。原因是另一专家扩大全局范围，99与零落入同一粗区间，插值无法恢复区间内部的实际分布。

这执行了原`_bincount_upper_quantile`，以单个池化数组模拟全局统计，psum替换为恒等操作，没有验证真实通信。1000000是刻意放大的机制反例，不是Hero观察。不能据此声称10000 bins不足、发生数值爆炸，或者HIST比TOPK差。需要归档真实margin范围、区间宽度、目标rank附近的拥挤度和精确参照误差，才能讨论实际影响。

更多bins可以减小同一范围下的区间宽度，但增加计数存储/归并开销，也不能让估计自动等同于精确分位数。裁剪范围、分expert网格或更换估计器都改变算法，须独立验证；不能直接作为无害bug修复上线。

## 4. 怎样接到配比与顺序研究

如果研究完整配比干预，数据改变margin分布、专家选择与beta，路由状态的变化是干预效果的一部分。如果希望隔离“数据本身更有价值”，则应另外设置固定路由状态的短诊断；这个诊断不能替代完整训练结果。

数据顺序也要分清两个层次：跨步先后决定哪个批次生成下一步bias；同一步的shard分组决定哪些样本共同进入局部估计。全局领域比例相同，不保证这两个层次相同。不要为了得到漂亮的均衡曲线而按来源重排shard，并把所得收益归到混合比例。

| 规则 | 必须保存 | 出现问题后怎样返回 |
|---|---|---|
| 配置和执行路径对上 | estimator、bins、执行SHA、mesh token轴 | 配置与代码默认冲突时，先核实运行路径 |
| 状态时序对上 | 本步实际bias、输入pending、输出beta | 切换首步单列；恢复丢失pending回到状态核查 |
| TOPK分组核查 | 每shard有效数、logical count、局部beta、样本映射 | 对共同冻结margin重新分组，检查估计差异 |
| HIST网格核查 | 全局lo/hi、区间宽度、目标区间计数 | 对共同冻结margin算精确参照，检查误差与异常专家 |
| 输出接受路径核查 | selected experts、combine权重、实际接受索引 | beta更均衡但drop变坏时，回到容量与权重检查 |
| 能力结论核查 | 相同评估口径、固定预算、稳定性与任务退步 | 只有计数改善时保留工程诊断结论，不宣称能力提升 |

这些规则接入[训练变更评审](CHANGE_REVIEW_ZH.md)，与[MoE丢弃](ROUTING_DROPS_ZH.md)一起检查。默认保留原状态作为完整变更对照，额外控制实验显式记录；本轮没有实测配比收益或给出新的生产配方。

## V114：原全局histogram的真实CPU边界

V19使用NumPy依赖替代检查了局部阈值与直方图；V113注入异常beta，验证后续训练与恢复检查的边界。这轮直接执行冻结`eee467…`的原`_qb_beta_hist`、`_bincount_upper_quantile`、原token轴选择和原setter中的中心化语句，使用真实JAX 0.7.2单设备`shard_map`及`psum/pmin/pmax`，没有把collective替换为identity。[9项控制](analysis/qb_hist_real_cpu.json) · [完整输出](analysis/qb_hist_real_cpu_output.txt) · [脚本](scripts/probe_qb_hist_real_cpu.py)。

输入为人工4×3 margin，K=2、10000 bins；单CPU mesh的replica_dcn/data/expert轴均为1。这是真实单设备collective接口执行，不是跨设备/跨host验证；没有执行完整模型的logit与margin构造，也不是线上运行版本绑定。

|人工margin与有效掩码|原估计器beta有限|原中心化bias有限|实际机制|
|---|---|---|---|
|普通有限范围|是|是|常态基线|
|全部无有效token|是，全部为0|是|valid_tokens=0时lo/hi/beta归零|
|全部为1|是|是|普通退化范围可返回有限值|
|无效位置含NaN/+Inf/−Inf|是|是|无效位置排除出范围与计数|
|有效位置含NaN|否|否|异常进入有效范围归约，未被清洗|
|有效位置含+Inf|否|否|范围无上界，后续插值非有限|
|全部有限3e38|是|否|beta保持有限，中心化归约溢出|
|同时含有限−3e38和+3e38|否|否|`hi-lo`在float32下溢出到非有限范围|

最后两项把V113的故障注入向上推进到了原估计器接口：在人工极端margin输入下，原wrapper确实可能返回非有限beta，或者有限beta进入原中心化后产生非有限bias。它们没有证明这些margin可以由真实Hero前向产生，也没有测到生产异常。值3e38特意接近float32边界，不能把它当作真实路由量级。

`hi_grid=max(hi,lo+1e-6)`的保护受浮点可表示性限制：大数附近加1e-6可能仍是原值。因此不能仅凭源码有epsilon就推断所有量级下bin_width严格正且有限；也不能据本控制成功返回有限beta，反推其所有中间运算都健康。当前没有更改原归约或插值算式。

实际得到的工程规则是：padding过滤与有效输入健康检查分别验收；空有效集合返回零是该估计器的合同，不表示训练loss的分母也已被处理；同时记录margin范围、range差值、beta及应用后bias。有限性检查还需接到真实模型、完整训练视图与分布式失败协议，不能只用单CPU结果决定生产跳步或清零。

V113仍正确保留“原QB估计器是否曾在Hero产生异常”为空。V114只新增了原接口在人工输入下的可执行边界，未测GPU、实际margin构造、跨host归约、阈值误差对任务能力的影响或上游修复。
