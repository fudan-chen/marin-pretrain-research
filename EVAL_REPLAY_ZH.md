# 把重复评估调查变成可执行记录

[评估身份核查](EVAL_IDENTITY_ZH.md)追到默认重复遍历，但尚未取得真实token流和同checkpoint重评结果。现在新增[对照检查器](scripts/compare_eval_replays.py)，将这项调查的输入和输出固定下来。它比较两份逐批记录，分别检查模型视图、输入与分批声明、分母以及数值差异。检查器已经可运行；历史PTB记录仍未取得，没有把合成样例写成真实重评。

关键限制先说清楚：输入哈希由记录导出者声明，检查器不读取原始token数组或checkpoint来独立验证哈希。两个哈希相同，只能在当前工具范围内说明“声明一致”。运行SHA、参数视图或backend变化时，不能把它写成同一执行的重复评分。

## 1. 每次评估需要保留什么

一份JSON包含全局身份字段和逐批、逐叶子域的records。身份包括checkpoint SHA256、执行SHA、参数视图、pending beta是否应用、计算dtype、backend、tokenizer与byte表SHA256。输入摘要声明覆盖有序tokens、weights、mask与tags，字段名为`input_sha256`；V39的`ordered_tokens_weights_masks_tags_v1`保留为声明标记。V41新增[NPZ离线导出与typed v2规范](EVAL_ARRAY_EXPORT_ZH.md)，能绑定提供的数组；真实GPU采集与生产身份仍未验证。

|逐批字段|含义|采集要求|
|---|---|---|
|batch、domain|评估批次号与叶子域|保留真实顺序；同一批同一域唯一，不额外重复父标签|
|input_sha256|该记录输入的摘要声明|由导出者绑定tokens/权重/mask/tags；仅有文本哈希不能代替它|
|weighted_loss_sum，N|该批该域加权NLL之和|全局归约后的值；不能填batch均值|
|loss_weight_sum，T|该批该域loss权重之和|不是简单的序列数或名义token数；允许零权重记录|
|weighted_byte_sum，B|byte查表后的加权和|与原评分token移位、byte表和weights一致|

当前工具只接受`array_scope=global`及互斥叶子域的声明。它不合并rank，不验证是否重复归约，不接受局部shard直接冒充全局。若使用重叠标签，不能重复累加后再计算父级micro；需要另行建立完整层级及归约记录，本工具会要求使用互斥叶子域契约。

同一个batch中可能有多个域，故batch号可以重复，(batch,domain)不能重复。检查器保留记录顺序，不为使结果相同而自行排序或删掉重复请求。导出时应固定域记录顺序；仅导出顺序改变也会返回需要核对，而不是自动宣称数据变化。

## 2. 输出先回答哪个对象不同

|输出状态|检查器确认的范围|后续动作|
|---|---|---|
|identity_or_input_review_required|身份字段、记录顺序、batch/domain、输入摘要或T/B声明有差异|先定位第一处不同记录和身份字段；如果是有意的backend实验，冻结差异表再分析|
|numeric_difference_under_matching_declarations|上述声明一致，某些逐批N超过数值容差|核对前向状态、kernel、归约和导出精度；不是自动根因判定|
|within_tolerance_under_matching_declarations|声明一致，逐批N差异未超过容差|仅对这两份记录成立；不能证明统计显著性、跨seed稳定性或原输入哈希真实性|

数值阈值作用于逐批N：`abs(N2-N1) <= atol + rtol*max(abs(N1),abs(N2))`，默认atol=1e−6、rtol=1e−5。它是可审查的诊断默认值，没有在Hero实际GPU误差上校准；函数调用者可显式传入其他容差。不要把“通过默认容差”当成生产验收标准已确定。

工具始终输出`root_cause=null`和`actual_GPU_replay=null`。身份差异被标记后，仍会保留能够对齐记录的N差值，供有意的执行对照阅读；状态不会把不同backend自动改称同执行重复。

## 3. 为什么总N/T/B相同还不够

工具重建的是已核对[TaggedEvaluator](sources/scale_2026_10_05/eval.py)的**域级日志口径**，不是另一种理想BPB：

- 域CE：ΣN / ΣT。
- 域日志BPB：Σ[T × N/max(B,1) × log₂(e)] / ΣT。
- 对照用byte加权比值：ΣN / ΣB × log₂(e)。最后一项只供比较，不冒充原日志值。

这里的micro按互斥域的T加权，重建非空域父级口径；macro对有正T的叶子域等权。根级macro在原源码中未排除零权重域，根级CE对逐批T还有1的分母下限；本检查器不重建这两种根级边界行为，见[EVAL_METRICS的V74控制](EVAL_METRICS_ZH.md)。T=0的域单独列出，不能无提示变成一个正常评分。零T但非零N/B的记录被拒绝；负值、NaN、重复记录、缺失身份和溢出的聚合也被拒绝。

|人工记录|总N/T/B|重建日志BPB|
|---|---|---|
|两批：各N=2、T=1，B分别1与3|4 / 2 / 4|4/3 × log₂(e)|
|合为一批：N=4、T=2、B=4|4 / 2 / 4|1 × log₂(e)|

两份记录总CE都为2，但日志BPB不同。差异来自原实现对逐批N/B采用T加权：分批改变时，最终比值一般不能只由总N/T/B决定。因此输入内容与顺序之外，batch分组和每批B也必须进入重复评分身份。

本工具用Python float64公式重建，不执行JAX float32 RunningMean；数学口径相同不意味着每一位数相同。真实记录若要对齐已有日志，还应保留归约与日志dtype，并验证重建残差。当前没有这种历史记录结果。

## 4. 从练习走到真实记录

[合成输入](templates/eval_replay_synthetic.json)使用人工哈希和两批小数，只用于演示，backend明确为synthetic。用同一文件比较会得到“声明相同且在容差内”，不能把这个结果写成模型重评成功：

```bash
.venv/bin/python scripts/compare_eval_replays.py \
  templates/eval_replay_synthetic.json \
  templates/eval_replay_synthetic.json \
  /tmp/eval-repeat-review.json
```

CLI保留两份输入文件的SHA256，并拒绝覆盖输出。真实管线的接入顺序是：

1. 从实际评估accumulator导出逐批全局N/T/B及域记录；同时绑定真实输入摘要和模型视图。[离线数组导出器](EVAL_ARRAY_EXPORT_ZH.md)已实现；实际前向采集与多rank范围仍需接入，不能从最终summary反推这些字段。
2. 同checkpoint重新评分并导出第二份记录；先运行声明对照，记录第一处输入/分母差异。
3. 声明对齐后，检查逐批N差异、域CE/BPB和与原日志的重建残差；异常回到数据身份或执行调查。
4. 重评可重复性通过后，再用多个checkpoint检验PTB/twitterAAE轨迹；这仍不替代独立seed或配比反事实。

[27项合成检查](analysis/eval_replay_validation.json)覆盖身份字段、输入与分母、记录顺序、零权重、非法/溢出值、BPB分批反例以及真实CLI的输出哈希和拒绝覆盖。它们验证检查器行为，不证明Hero实际输入、数值稳定性或评估改善。

这把上一轮“需要同checkpoint重评”变成了明确的记录格式、比较顺序和返回分支。当前继续缺少的是实际运行导出记录；我们不会用合成输入的绿灯填补这个缺口。
