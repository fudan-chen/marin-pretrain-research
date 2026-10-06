# 从评估数组导出可比较的记录

[重复评分检查器](EVAL_REPLAY_ZH.md)已经能对照逐批记录，但V39尚未提供数组导出路径。现在新增[离线导出器](scripts/export_eval_arrays.py)：读取已收集的全局NPZ数组，计算域级N/T/B、实际输入摘要和byte表摘要，输出检查器能直接使用的JSON。

它补齐了“数组→记录”的离线环节，**没有运行模型、收集GPU数组、汇总多rank或验证checkpoint**。输入数组仍需从真实评估过程中采集；当前验证全部为人工数组，未取得Hero记录。

## 1. 输入数组与manifest

可从[空白manifest](templates/eval_array_manifest.json)填写实际运行信息；未填字段不能运行。manifest给出模型视图声明、ordered domains、mask_spec、global范围，以及按实际评估顺序列出的batch/NPZ路径。每个NPZ只能包含以下数组和命名为mask_*的显式mask叶子；禁止pickle/object数组。

|数组|形状与要求|用途|
|---|---|---|
|input_tokens|整数[B,L]，合法vocab范围|绑定模型实际输入；不是仅保留评分target|
|scoring_token_ids|整数[B,T]，合法byte表索引|绑定loss位置使用的token；导出器不自行推测移位|
|losses、weights|非负有限浮点[B,T]|逐位置NLL及实际loss权重；不是训练总loss，也不是名义token计数|
|tags|0/1的[B,D]，每行至多一个叶子域|只支持互斥叶子域；正权重行必须属于一个域|
|bytes_per_token|非负有限一维表|评分token查表后计算B，表内容也计算摘要|
|mask_*|有限实数、整数或布尔数组|保存实际mask表示的各叶子；另在mask_spec写清语义与布局|

模型视图identity字段包括checkpoint_sha256、execution_sha、parameter_view、pending_beta_policy、compute_dtype、backend、tokenizer_sha256。byte_table_sha256可以不填，由导出器计算；如果填了却与提供的表不一致，则拒绝。不同batch的byte表也必须相同。

mask可能是文档ID、布尔条件或其他显式叶子，导出器绑定它们的字节，不解释AttentionMask语义。若模型使用隐式causal条件，采集者须显式记录相应标志及mask_spec；不能为了通过校验随意造一个mask叶子。NPZ本身无法证明采集的就是实际前向输入。

## 2. N/T/B如何产生

对batch中属于域d的行，用float64独立计算：N=Σ(loss×weight)，T=Σweight，B=Σ(bytes_per_token[scoring_token_ids]×weight)。零权重位置不会贡献这些总量，但仍留在输入摘要中。

评分token的对齐由采集者负责。归档入口使用移位token并在末尾补一个位置；如果实际运行使用另一种对齐，必须按实际loss位置输出，不能从input_tokens长度直接推断。losses的来源也必须是评估NLL；训练loss可能包含额外项，不能混用。

|对象|现在实际验证了什么|仍未验证什么|
|---|---|---|
|N/T/B|对提供的人工数组计算与参考值一致|Hero前向、全局归约和真实mask/权重|
|输入摘要|从提供的数组按规范重新计算|数组是否等于真实运行输入|
|byte表摘要|从提供的完整typed表计算；拒绝虚假预填摘要|表是否等于运行时tokenizer字节定义|
|checkpoint/tokenizer身份|要求格式合法并进入对照字段|文件内容、模型状态和实际执行的对应关系|

这比只比较导出者手填的input_sha256更强，但仍不能把“给定文件内部可复查”推广成“真实训练身份已证明”。

## 3. v2摘要的具体定义

新格式标记为typed_global_eval_arrays_v2。每个域记录的摘要绑定**整个batch输入**及当前域名，不只截取该域的行；这样分批与其他行变化也会进入检查。

摘要帧按以下规则连接后计算SHA256：

1. context按键排序、UTF-8、无额外空格的JSON，包含scheme、ordered domains和mask_spec。
2. 数组按名称排序，包含input_tokens、scoring_token_ids、weights、tags与所有mask_*；不包含losses或byte表。byte表单独进入身份字段。
3. 每数组先写名称/dtype/shape的规范JSON头，再写little-endian、C-order字节。每个帧先写8字节little-endian长度。
4. 以上得到batch_input_sha256；域摘要再对含domain与batch_input_sha256的排序键、无额外空格UTF-8 JSON计算SHA256。整批数组只需哈希一次，各域再绑定名称。

|变化|摘要处理|原因|
|---|---|---|
|C-order改为Fortran内存布局|相同typed值摘要相同|物理内存布局不是这里要比较的输入变化|
|只改变字节序|规范化后摘要相同|绑定typed值而非文件的主机字节序|
|改变dtype宽度或shape|摘要不同|实际typed输入与布局声明不同|
|改变loss，输入数组不变|输入摘要相同，N不同|可以进入声明一致下的数值调查|
|改变token、权重、tags或mask|摘要不同|先检查输入身份，不自动解释为模型数值变化|

原v1声明记录仍可比较；v1与v2不同标记会触发身份核对，不能自动继承“哈希一致”。检查器仍不自行读取NPZ，使用v2记录时应保留导出manifest和所有NPZ以供复核。

## 4. 运行与接入

```bash
.venv/bin/python scripts/export_eval_arrays.py manifest.json first.json
.venv/bin/python scripts/export_eval_arrays.py repeat-manifest.json second.json
.venv/bin/python scripts/compare_eval_replays.py first.json second.json comparison.json
```

输出文件拒绝覆盖，导出记录保留manifest文件与每个NPZ文件的SHA256。manifest的batch项形如`{"batch":0,"path":"batch-000.npz"}`；相对路径按manifest所在目录解析。记录顺序按manifest保持，不自动排序。

采集接入时，应在实际eval_loss_fn得到loss、weight和评分token后保存对应模型输入、mask和域标签；先确认这些是全局数组，再做导出。不能把local shard改名为global，也不能将各rank已重复的全局总量再累加。这个多卡采集步骤当前尚未实现或验证。离线工具一次加载一个NPZ并规范化数组字节，不是受控的流式采集方案，真实批次的内存与导出开销尚未测量。

[23项检查](analysis/eval_array_export_validation.json)使用人工两行、两域输入，覆盖N/T/B参考值、typed摘要、loss/input差异、非法标签/值/范围、byte表声明和真实CLI拒绝覆盖；它们不是生产评估成绩。旧[27项记录比较检查](analysis/eval_replay_validation.json)也重新通过。

已有能力是离线、可复核的“全局数组→逐批记录→重复评分对照”；尚缺实际前向采集、rank范围证明及Hero记录。拿到这些材料后，才能把PTB调查从声明比较推进到真实输入与数值重放。


## 下一token坐标验收

新增可选`target_contract="causal_next_token_v1"`，空白manifest默认启用。它要求输入与评分ID同形、末位权重为零、所有正权重位置的评分ID等于右侧输入token；零权重ID仍受范围与摘要约束。缺省通用接口会明确标记未验证此性质。该检查仅验证提供的数组坐标，不验证loss来源、真实前向或缓存。详见[目标对齐与覆盖](EVAL_TARGET_ALIGNMENT_ZH.md)。
