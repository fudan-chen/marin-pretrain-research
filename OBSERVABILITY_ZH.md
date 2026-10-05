# 监控不是旁观者：时刻、成本与事故重放

监控可以帮助定位优化器故障，也可能改变计算负担和时序。先证明记录的是哪个阶段、哪个参数视图，再用它解释更新。本章把最近三章连接到一份可执行的事故包检查，而非继续增加loss截图。

新归档[watch.py](sources/watch_2026_10_06/watch.py)和[tree_stats.py](sources/watch_2026_10_06/tree_stats.py)均固定到`84869ae8c91ffe64e9f761c5bd714542eb1876e0`；调用端为已归档[train_hero_ep.py](sources/scale_2026_10_05/train_hero_ep.py)。[22项原辅助函数/静态检查](analysis/watch_probe.json)执行dispatcher、配置、diagnostic构造及平坦数组统计路径，用记录器、NumPy norm、平坦树、strict zip兼容器、假梯度/恒等JIT等替代依赖。没有执行真实JAX归约、scan层拆分、直方图或GPU profile。

## interval控制了什么，没控制什么

|模式或选项|源码实际行为|不能直接推出的结论|
|---|---|---|
|inline，interval=10|训练JIT每步调用统计；循环仅在步数满足interval时输出日志|把interval调大就减少了每步统计计算|
|diagnostic，interval=10|到期时额外做一次loss_and_grads，block_until_ready，然后正常训练一步|这是不改变负担的纯日志抽样|
|diagnostic targets|只接受params/grads；updates与opt_state被构造器拒绝|WatchConfig支持四种target，故所有模式都支持四种|
|include_norms=False|原tree_stats末尾仍无条件计算prefix/norm/total|关闭此flag就完全没有norm归约|
|include_per_parameter_norms=False|原平坦路径先构造逐叶范数，之后过滤输出；JIT可能消除未使用项|仅凭Python调用记录断言设备执行或省掉这些归约|

默认WatchConfig只选grads/params，interval=10；因此默认记录本身没有update和optimizer状态。配置的is_enabled要求targets非空、interval正、norm或histogram打开；per-parameter flag单独打开不能启用监控。即使只开histogram，启用后tree_stats的total norm路径仍存在。本地探针用假norm调用记录核对行为，**没有测量它占多少毫秒或显存**。

关闭逐参数输出后的叶norm没有进入返回字典，JIT可能dead-code eliminate；本地NumPy记录器只能证明Python调用路径，不能证明这部分GPU表达式存活。应看优化后HLO和profile。total norm则无条件进入返回字典，两者不能混同。

这些不是已验证的Hero性能故障。它们是源码事实与需要profile的优化机会：应分别测试“减少输出”“减少统计项”“改变监控模式”，不能把它们视为同一种操作。

训练循环在diagnostic完成以后才设置step_start；duration在训练loss ready后记录，并传给callbacks。它不包含之前的diagnostic计算、batch加载，也不包含之后的callback/checkpoint时间。因而比较监控成本需要整轮墙钟和分段profile，不能只检查这个duration。inline归约的实际同步与日志成本也须按执行轨迹测量。

## 一个日志step中混合了哪些阶段

|对象|inline调用实际传入|用于重放时必须注意|
|---|---|---|
|params|应用pending QB beta后的qb_params，本步更新前|不是更新后的params；不能把它配成下一步前向的起点|
|grads|本步前向/反向产生的grads|计算参数视图、loss mask和数据身份必须一致|
|updates|optimizer返回的有符号增量|调用端用p_after=p_before+updates，不是再减一次|
|opt_state|opt_state_in，更新前输入状态|mu/nu是旧状态；要先用本步梯度更新，再计算本步方向|
|count|若按model_tree_type筛state，只处理匹配模型树的子树|标量count可能没有被输出；没有日志值不等于count为零|

在FP32_PINNED_HOST master路径，optimizer更新的是master_params_in/master_grads；watch却仍传qb_params/grads。两者可能具有不同精度视图。使用watch参数范数和master update做精确不变量验收前，需要拿到optimizer真正的输入参数。这个调用差异不能独立证明生产恢复错误，但足以要求记录parameter_view。

历史[#8073事故记录](MUON_GEOMETRY_ZH.md)来自另一路径；不能把当前EP源码的字段时刻直接套给旧事故包。没有时刻说明时，旧mu/nu还是新mu/nu应保持未知；‖mu‖/√‖nu‖更不能替代逐元素真实方向范数。

## 最小事故包应能区分什么

|需要保存的证据|具体目的|缺失时的限制|
|---|---|---|
|实际optimizer输入p_before和有符号update|独立算p_after及组范数保持比|compute/EMA/local shard的范数不能替代master全局范数|
|本步梯度、更新前完整状态、有效lr/beta/eps与各count|重建moment与方向；检查状态接续|只有moment范数不能还原逐元素方向|
|moment/NS之后、投影之前的真实U|把方向异常与投影输出不一致分开|只有原梯度不够，MuonH方向还经过moment和NS|
|投影中间V、计算得到的new_param_norm|比较实际分母与独立重算分母|没有分母原值，不能由缩小倍数直接确定归约错误|
|全局shape、组归约轴、dtype、mesh/spec、执行SHA|绑定数学组、精度与分片路径|局部shard重算正常不能证明全局归约正常|
|数据/cursor/pending beta、实际p_after|对齐本步起点和更新落地|不能排除重放错步、错数据或错视图|

全张量抓取可能引入聚合、host占用和I/O；建议优先明确故障叶与触发窗口，实际代价与可恢复性需在目标环境测试。这里没有把事故包写入成功当成GPU重放成功，也没有给所有训练通用的触发阈值。

## 已可执行的离线检查器

新增[analyze_optimizer_bundle.py](scripts/analyze_optimizer_bundle.py)。它读取**已规范化的全局参数叶NPZ**与声明合同，用float64重算：

- p_before+update是否与记录p_after一致；本组范数是否在声明容差内保持。
- 若有真实U：按给定lr/epsilon重建投影，检查输出是否一致。
- 若有V和记录分母：比较记录norm与独立重算norm。
- 缺少方向/分母/p_after时明确列出缺失，不自动补值或判根因。

它不计算Optax moment，不复现NS、BF16或SPMD，不验证来源SHA的真实性。输出root_cause始终为null；“某分母不一致”仍需正确时刻、全局输入与原执行轨迹才能归因。

[空白合同](templates/optimizer_bundle_contract.json)不能直接执行，需填入实际对象；支持键为p_before/update，以及可选direction/projection_intermediate/computed_new_param_norm/p_after。方向时刻必须声明after_moment_or_NS_before_projection，参数视图必须声明optimizer_parameter_before_update，全局array_scope为global。拒绝局部shard、错轴、错形状和object数组；NPZ读取禁用pickle，结果文件存在则拒绝覆盖。

填好自己的文件后运行：

```bash
.venv/bin/python scripts/analyze_optimizer_bundle.py \
  --arrays /path/to/normalized-global-leaf.npz \
  --contract /path/to/filled-contract.json \
  --output /path/to/new-observation.json
```

/path/to是待替换路径。工具输出含输入文件SHA256，记录声明元数据与计算范围，但不能将人工填写的来源标记自动变成已验证事实。

[23项合成检查](analysis/optimizer_bundle_validation.json)覆盖正常投影、七倍分母反例、范数正常但方向不同、错时刻/轴/局部shard、非有限输入、零参数组、四维联合约束，以及真实NPZ CLI读写/拒绝覆盖。它们是人工分类和输入边界验证，**没有取得或重放历史事故npz**。

## 怎样从观察走向根因

先检查同一时刻/视图/符号/组轴；再核对方向与投影；最后才对照原分片计算。若范数异常但U和V的独立重算正常，保存并比较实际分母；若范数正常但方向不同，追踪moment、NS和状态恢复；若p_after与有符号增量对不上，先核查数据包的步与视图。

这条支线连接R12/R14/R15及[loss诊断](LOSS_TRIAGE_ZH.md)，不是自动部署审批。它将“还需要什么证据”变为具体缺项；实际root cause、修复收益和训练稳定性仍需原环境验证。
