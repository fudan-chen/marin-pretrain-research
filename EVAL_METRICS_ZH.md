# 评估指标：从日志名字追到真正的分母

V25里，我把`eval_dropless/paloma/bpb`当成macro，因而把它与16域等权均值方向相反写成了聚合疑问。这个判断的命名前提错了。本轮沿源码查到：该字段是父级micro；真正macro的字段为`eval_dropless/paloma/macro_bpb`。补齐归档后，等权重建与真正macro一致。需要修正的是报告，不是据此认定线上评估有bug。

## 1. 字段先对照，再解释曲线

以下来自固定源码`84869ae8c91ffe64e9f761c5bd714542eb1876e0`，不能直接证明每次历史run都执行此版本。[构造日志函数](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/lib/levanter/src/levanter/eval.py#L450)读取`EvalResult`，按叶子tag与父级tag使用不同命名。

|Paloma日志字段|来源|在该源码中的含义|
|---|---|---|
|`paloma/<subset>/loss`|`tag_micro_losses[leaf]`|该叶子域有效token平均CE|
|`paloma/<subset>/bpb`|`tag_micro_bpb[leaf]`|该域BPB累计结果；仍需看batch聚合|
|`paloma/micro_loss`|`tag_micro_losses[parent]`|叶子CE按累计有效token量加权|
|`paloma/macro_loss`|`tag_macro_losses[parent]`|非空叶子域CE等权平均|
|`paloma/bpb`|`tag_micro_bpb[parent]`|叶子BPB按累计token权重加权；不是macro|
|`paloma/macro_bpb`|`tag_macro_bpb[parent]`|非空叶子域BPB等权平均|

不应从字段是否包含`micro`推断它是否是micro。BPB父级字段没有这个后缀；CE父级却有。根级`eval_dropless/bpb`又来自另一个`micro_bpb`结果，不能与父级`eval_dropless/paloma/bpb`自动视为同一字段。

本轮提取并执行原始`construct_log_dict`、`_join_prefix`以及`_construct_tag_hierarchy`函数，用刻意不同的人工数值检查字段映射。十项日志与静态数据流检查涵盖父级/叶子、根级汇总、层级索引和无tokenizer时的输出。另执行原始`RunningMean`类的四项检查，使用NumPy `where`及dataclass初始化替代依赖，核对累计单位、比值权重和零total更新。合计十四项；这里只执行日志、层级与running mean辅助函数，没有执行`evaluate`、JAX、真实数据或分布式归约。[探针脚本](scripts/audit_eval_metrics.py)与[逐项结果](analysis/eval_metric_audit.json)保留源文件SHA和函数行号。

## 2. 67个真实日志step把命名错误分离出来

重新读取六份归档的全部36个Paloma指标，以生产接续边界选点：16项CE、16项BPB、四项父级指标。67个完整step共2412个值；没有插值。重建macro的最大绝对残差分别为CE 2.161×10⁻⁷、BPB 1.490×10⁻⁷。数值接近支持“这16域构成此处的等权macro”，不能单独确认历史代码或每次验证样本身份。

|窗口|记录macro ΔCE|记录micro ΔCE|记录macro ΔBPB|记录micro ΔBPB|
|---|---:|---:|---:|---:|
|104999→107999|+0.009042|−0.000176|+0.002571|−0.000120|
|107999→110999|−0.001871|+0.000275|−0.000937|+0.000064|
|110999→113999|−0.001838|+0.000944|−0.000572|+0.000338|

macro与micro不必同向：前者让每个验证域拥有同等发言权，后者让token较多的域拥有更大影响。这个差异本身不构成bug。在配比评审里，两者的用途要事先指定：保留多域表现可以看macro与逐域限制；验证总体token分布上的预测效果可以看micro。它们都不能直接替代任务能力评估，也不能在看到结果后挑一个有利指标。

原始CSV和67点残差在[轨迹摘要](analysis/mix_trajectory.json)，曲线与逐域变化见[修订后的配比章节](MIX_TRAJECTORY_ZH.md)。V25的release记录与浏览器回执保留为历史材料；当前分析使用`__parent__`区分父级，避免把一个父级指标当成第17个语料域。

## 3. 一个变量叫bytes，累计的却是token

源码的`bpb_per_tag_mean`调用为：

```python
state.bpb_per_tag.add(bpb_per_tag, this_weights_per_tag)
```

更新running mean的权重是有效token权重，不是byte量。后续取出`state.bpb_per_tag.total`，赋给`total_bytes_per_tag_cpu`。这个变量名容易误导读者：它沿这条数据流累计的是token权重。[原始RunningMean.add](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/lib/levanter/src/levanter/utils/stat_utils.py#L24)直接执行`new_total = self.total + total`，因而确认单位由传入的total决定。父级micro BPB再用这个total加权叶子BPB。[累计器与父级聚合](https://github.com/marin-community/marin/blob/84869ae8c91ffe64e9f761c5bd714542eb1876e0/lib/levanter/src/levanter/eval.py#L587)

|层次|源码计算|何时与“全局NLL/byte”相同|
|---|---|---|
|单batch、单域|`N_j / max(B_j,1) × log2(e)`|分母非退化时就是该批比值|
|一个域跨batch|以`T_j`加权各批比值|例如各批`B_j/T_j`相同；一般不能保证|
|父级macro|非空叶子BPB等权平均|这是域等权目标，通常不是全局比值|
|父级micro|以各域累计`T`加权叶子BPB|即使先修域内公式，也一般不等于按byte加权|

这里`N`指进入累计器的加权loss分子，`T`为权重和，`B`为token byte表按相同权重求和。loss回调是否已经加权是另一项接口约定，见[小数权重检查](IMPLEMENTATION_CONTRACTS_ZH.md)；本章没有重复把它当成已确认的生产错误。

原类的人工输入进一步核对这个区别：两个batch的N分别2、6，T分别2、6，B分别20、6。以token权重累计N/B得到0.775，total为8；以byte权重累计则为8/26≈0.307692，total为26。这些比值尚未乘log2(e)，不是Hero实际BPB。它验证了原running mean公式和单位传递，没有执行真实评估或证明线上排名变化。

有两层不能混淆。第一层是本报告将micro误写成macro；修正名字并核对真实macro即可解决。第二层是固定公共代码的BPB平均顺序与权重；它是否影响真实候选排名，需要冻结checkpoint、输入和分母后重新分批验证。不能用第一层的更正否定第二层的数学差异，也不能用第二层的反例替第一层错误辩护。

## 4. 修复指标之前，先明确测量目标

如果产品目标是按域等权的BPB，应该先为每个域累计NLL与byte，然后对域的比值做等权平均。如果目标是所有验证byte上的总体BPB，则累计所有域NLL和byte，最后只取一次比值。两者都合理，但回答的问题不同，字段、报表和验收阈值必须区分。

不能直接把当前`add(..., token_weights)`全部改成byte权重：父级mask目前从token累计量判断非空，后续变量也依赖状态；至少需要明确保存各域`N,T,B`以及评估序列ID，再设计接口迁移。本轮未修改作者训练代码，也未重测候选排序。

分布式场景还要确认归约后的分子与分母涵盖同一批位置。源码声明per-tag输出为复制sharding，可帮助理解预期布局，不能据此证明真实设备上的覆盖、数值或通信执行正确。本轮没有执行跨设备归约，不能把静态sharding声明写成通过了分布式验收。

验收管线补上一条具体规则：**任何参与候选排序的日志字段，先记录字段→源码返回项→累计状态单位→聚合公式四个对应关系。** 对同一组预测做重新分批、域顺序交换和最后不满batch检查；若目标是全局比值，结果应只受数值舍入影响。逐域NLL、token、byte及mask身份是验收产物，不能只保存最终BPB。真实mask全零、byte分母小于1或小数权重时，单独报告覆盖与退化处理，不把默认安全分母当作正常文本统计。

[评估口径交接模板](templates/eval_metric_contract.json)将上述对应关系、输入身份、N/T/B产物与重新分批检查放在同一记录中。当前模板为计划，实测字段均未填写；它不是已经验证的线上契约。
