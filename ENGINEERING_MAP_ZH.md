# 预训练排障总图：从症状选择证据，而不是从版本号读报告

本章把主报告15类工程问题接到后续源码深读，并给六种常见起点安排检查顺序。它是本报告作者的交接方法，不是Marin官方验收标准；对应的18条rubrics仍为1.1，没有另造一套总分。

第一张表回答“现在先查什么”，随后逐案分开作者已经做过的动作、本地到底核对了什么、以及下一项仍未执行的检查。链接与覆盖检查可以发现漏项和错指，但不能自动证明根因、GPU正确性或人的理解。

## 1. 六个入口，先留下一个能改变判断的产物

|你看到的症状|先读哪里|先留下什么|本轮能怎样决定|
|---|---|---|---|
|训练loss在换配比处跳变，但训练正常|[LOSS_TRIAGE](LOSS_TRIAGE_ZH.md) → [EVAL_METRICS](EVAL_METRICS_ZH.md) → [MIX_TRAJECTORY](MIX_TRAJECTORY_ZH.md)|原metric key、配比生效step、固定验证叶子/micro/macro与输入身份|先判断尺子/分布/状态是否改变，再讨论数据质量；不由训练CE跳变直接改比例。|
|部分rank不推进，最后停在AllReduce/barrier|[ENGINEERING_GUIDE](ENGINEERING_GUIDE_ZH.md) → [OPERATIONS](OPERATIONS_ZH.md)|各rank last progress与首个异常，触发机制的必要条件和实际执行包|先查最早未进入后续collective的rank；末端等待不能独立证明通信库首因。|
|训练继续，但保存水位落后或下一次save阻塞|[CHECKPOINT_COMMIT](CHECKPOINT_COMMIT_ZH.md) → [CHECKPOINT_MEMORY](CHECKPOINT_MEMORY_ZH.md)|四个水位、旧manager任务状态、manifest/metadata请求和每rank local commit|布局存在、路径返回、marker可见和restore通过逐项判断；保留已验证回退点。|
|恢复后OOM或保存后RSS不断抬高|[CHECKPOINT_MEMORY](CHECKPOINT_MEMORY_ZH.md) → [CHECKPOINT_COMMIT](CHECKPOINT_COMMIT_ZH.md) → [TRAIN_STATE](TRAIN_STATE_ZH.md)|分阶段buffer共存、每rank计划、当前RSS、可达引用和allocator retained bytes|先定位活跃对象与调度等待，再改变一个内存控制量；不连续重复确定性OOM。|
|候选kernel更快，准备接入生产|[CHANGE_REVIEW](CHANGE_REVIEW_ZH.md) → [ENGINEERING_GUIDE](ENGINEERING_GUIDE_ZH.md)|五个冻结对象、同起点输入/状态、数值窗口与回退和保存恢复结果|整包接受与组件收益分开；短窗、固定eval、生产稳定与恢复分别验收。|
|准备修改数据比例或顺序|[DATA_GUIDE](DATA_GUIDE_ZH.md) → [DECISION_GUIDE](DECISION_GUIDE_ZH.md) → [TRANSFER_GUIDE](TRANSFER_GUIDE_ZH.md) → [ORDER_GUIDE](ORDER_GUIDE_ZH.md)|旧/简单基线、供体、逐桶预算与累计量、独立重复、预定保底项|先检查竞争参照与预算/顺序混杂，保留逐任务限制；选择集收益不冒充独立确认。|

这些是建议执行顺序，尚未在新的集群事故中验证。若首个产物缺少版本、时间或输入身份，先补身份；如果已有反证否定触发条件，撤回对应假说；不要用继续跑更多步替代能区分解释的测量。

## 2. 十五类问题的动作、源码与证据边界

阅读本地探针时，先看输出中的scope、substitutes与actual_*字段。CPU原辅助函数、模拟sharding、固定源码推断、公开指标复算和作者生产观察是不同证据，不能互相补齐。当前所有案例的“本轮新增集群执行”均为空。

### 3.1 · Checkpoint 写得出来，却读不回来

**症状与机制。** 恢复报memory kind不匹配。作者报告deserialize device buffer与pinned-host模板不匹配；当前reader源码保留目标sharding并逐leaf完成放置。

**作者做过什么，结果到哪一步。** 作者修复逐leaf的device materialization与目标memory kind放置，并保留具体化sharding里的memory_kind。

**本地核对的范围。** 读取路径已归档；未运行真实恢复或验证HBM/host落点。

**下一项检查（未执行）。** 小state先保存再独立恢复，逐leaf核对dtype、shape、sharding与memory kind，再扩大到真实state。

**接受或继续调查的边界。** 落点与完整状态均匹配才接受；仅打印Loaded不通过。

源码/patch入口：[tensorstore_serialization](sources/checkpoint_commit_2026_10_05/tensorstore_serialization.py)
阅读：[TRAIN_STATE](TRAIN_STATE_ZH.md) · [CHECKPOINT_COMMIT](CHECKPOINT_COMMIT_ZH.md)
原始证据：[来源1](https://github.com/marin-community/marin/pull/8443)
使用既有规则：R13, R14，见[18条判断规则](RUBRICS_ZH.md)。

### 3.2 · 一味加 host RAM，仍然在恢复后被杀

**症状与机制。** 恢复显示Loaded后仍OOM。作者以恢复后驻留量、4 ranks/node和冷启动对照定位共存状态/读缓冲问题；多项内存措施一起生效。

**作者做过什么，结果到哪一步。** 作者组合使用读缓冲分块/donation、释放初始状态、延后data cache和rank轮流恢复，报告fleet峰值882→735GiB及step164续训。

**本地核对的范围。** 历史数字来自作者，当前未重测RSS；源码探针不能证明940GiB配置会通过。

**下一项检查（未执行）。** 分阶段记录init、reader staging、restored state、data cache与allocator；按node叠加local ranks。

**接受或继续调查的边界。** 峰值与稳态均有余量、下一步可运行才接受；确定性OOM先停止重复拉起。

源码/patch入口：[tensorstore_serialization](sources/checkpoint_commit_2026_10_05/tensorstore_serialization.py)
阅读：[CHECKPOINT_MEMORY](CHECKPOINT_MEMORY_ZH.md) · [CHECKPOINT_COMMIT](CHECKPOINT_COMMIT_ZH.md)
原始证据：[来源1](https://github.com/marin-community/marin/pull/8480)
使用既有规则：R12, R13, R14，见[18条判断规则](RUBRICS_ZH.md)。

### 3.3 · 每隔几次保存，host RSS 的地板抬高

**症状与机制。** 保存后RSS地板逐次抬高。作者纠正为completed futures仍持引用，allocator保留另算；两条链不能互相代替。

**作者做过什么，结果到哪一步。** 作者先后尝试缓存/并发限制、malloc_trim与jemalloc，并纠正completed futures引用；没有在报告里认定第一项措施根治。

**本地核对的范围。** 本地检查预算/规划，未执行真实manager的引用清理，也未测RSS。

**下一项检查（未执行）。** 同一state重复保存，分别测writer/non-writer RSS、future完成/可达引用、allocator retained bytes。

**接受或继续调查的边界。** 引用回落与allocator行为分别判断；RSS没回基线不能单独否定引用修复。

源码/patch入口：[tensorstore_serialization](sources/checkpoint_commit_2026_10_05/tensorstore_serialization.py) · [byte_budget](sources/checkpoint_memory_2026_10_05/byte_budget.py)
阅读：[CHECKPOINT_MEMORY](CHECKPOINT_MEMORY_ZH.md) · [ENGINEERING_GUIDE](ENGINEERING_GUIDE_ZH.md)
原始证据：[来源1](https://github.com/marin-community/marin/issues/8506#issuecomment-5373722833) · [来源2](https://github.com/marin-community/marin/issues/8506#issuecomment-5389725084)
使用既有规则：R13, R14，见[18条判断规则](RUBRICS_ZH.md)。

### 3.4 · Loader 缓存不是越大越好

**症状与机制。** 增大loader cache，训练速度基本不变。小规模loader benchmark已远高于训练所需速率；更大cache未证明端到端收益。

**作者做过什么，结果到哪一步。** 作者比较1GiB与125GiB cap，较小cache仍提供约18.6倍输入余量；没有报告11rack端到端提升。

**本地核对的范围。** 报告复述作者吞吐；没有11rack尾延迟与真实loader stall测量。

**下一项检查（未执行）。** 保持训练、I/O和输入相同，仅变cache cap，记录consumer等待、cache hit、尾延迟与RSS。

**接受或继续调查的边界。** 满足输入余量后优先保留较小内存配置；只有实际训练等待降低才称端到端加速。

源码/patch入口：该案没有定位为某段训练代码的根因，不强行补一个源码归因。
阅读：[DATA_GUIDE](DATA_GUIDE_ZH.md) · [ENGINEERING_GUIDE](ENGINEERING_GUIDE_ZH.md)
原始证据：[来源1](https://github.com/marin-community/marin/issues/8506#issuecomment-5401613236)
使用既有规则：R02, R12, R14，见[18条判断规则](RUBRICS_ZH.md)。

### 3.5 · NVLink、IB 与节点掉线，不能混为训练数值不稳定

**症状与机制。** NVLink/IB/节点掉线，或退出阶段CUDA报错。先行硬件证据与teardown报错有不同因果位置；retry恢复进度不等于修复硬件。

**作者做过什么，结果到哪一步。** 作者whole-gang retry从完整checkpoint恢复；硬件是否修好和teardown是否首因分别保留判断。

**本地核对的范围。** 已有作者时序证据；没有本地设备诊断，未定位为训练代码根因。

**下一项检查（未执行）。** 按UTC记录last progress、first abnormal event、watchdog、kill与restart，并保存各rank健康证据。

**接受或继续调查的边界。** 有先行独立证据才归硬件；只有退出错误则保持最初原因未知。

源码/patch入口：该案没有定位为某段训练代码的根因，不强行补一个源码归因。
阅读：[ENGINEERING_GUIDE](ENGINEERING_GUIDE_ZH.md) · [OPERATIONS](OPERATIONS_ZH.md)
原始证据：[来源1](https://github.com/marin-community/marin/issues/8506#issuecomment-5365222655) · [来源2](https://github.com/marin-community/marin/issues/8506#issuecomment-5442658185) · [来源3](https://github.com/marin-community/marin/issues/8506#issuecomment-5856665151)
使用既有规则：R01, R13, R18，见[18条判断规则](RUBRICS_ZH.md)。

### 3.6 · Ragged EP 第一次切换：短窗数值和速度通过，稳定性没通过

**症状与机制。** ragged候选短窗更快，随后silent hang。少丢弃改变前向，短窗数学/速度通过；生产同步仍失败，作者回退旧代码与完整状态。

**作者做过什么，结果到哪一步。** 作者在两次silent hang后回退旧代码和旧checkpoint，保留独立run/checkpoint树并核验同批次loss重放。

**本地核对的范围。** 独立复算公开同step指标；没有重放token ID或多rack执行。

**下一项检查（未执行）。** 同handoff和输入比较数值/路由，先验证独立checkpoint树与回退，再覆盖生产拓扑稳定窗。

**接受或继续调查的边界。** 分别批准数值、速度、稳定、保存恢复；出现同型hang即按预定条件回退。

源码/patch入口：[pooled](sources/routing_2026_10_05/pooled.py) · [ragged](sources/routing_2026_10_05/ragged.py)
阅读：[ROUTING_DROPS](ROUTING_DROPS_ZH.md) · [CHANGE_REVIEW](CHANGE_REVIEW_ZH.md)
原始证据：[来源1](https://github.com/marin-community/marin/issues/8506#issuecomment-5518830230)
使用既有规则：R03, R12, R13, R14，见[18条判断规则](RUBRICS_ZH.md)。

### 3.7 · “两个 executable 交替”是曾经的解释，后来被排除

**症状与机制。** hang总在日志/watch附近。曾提watch/plain executable交替，后查INLINE路径排除了必要条件；相邻时间不是机制。

**作者做过什么，结果到哪一步。** 作者核对INLINE执行路径后排除watch/plain交替解释；这不是把所有hang都归入另一确定根因。

**本地核对的范围。** 来源中有作者路径审计；未取得历史实际编译执行包。

**下一项检查（未执行）。** 核对执行版本、WatchMode、编译路径与触发条件；定位最先停住的rank。

**接受或继续调查的边界。** 必要条件不存在就撤回该假说；小环境未复现只保留阴性范围。

源码/patch入口：[code_train](sources/code_train.py)
阅读：[ENGINEERING_GUIDE](ENGINEERING_GUIDE_ZH.md) · [IMPLEMENTATION_CONTRACTS](IMPLEMENTATION_CONTRACTS_ZH.md)
原始证据：[来源1](https://github.com/marin-community/marin/issues/8870#issuecomment-5531534090)
使用既有规则：R01, R13, R14，见[18条判断规则](RUBRICS_ZH.md)。

### 3.8 · NCCL 编译 headers 与 runtime 不一致：实在的缺陷，但不是全部解释

**症状与机制。** NCCL编译headers/runtime不一致。依赖diff修正了明确版本缺陷；候选200步通过后1590步仍复发，不能认定全部hang解决。

**作者做过什么，结果到哪一步。** 作者更换ARM64 PJRT依赖wheel修正headers/runtime缺陷；200步通过后1590步复发，稳定性结论收窄。

**本地核对的范围。** PR patch与作者复发记录已归档；本地没有加载wheel或验证实际runtime。

**下一项检查（未执行）。** 记录headers、wheel摘要和实际加载库版本，随后按同故障特征观察完整拓扑。

**接受或继续调查的边界。** 版本一致性通过与长期稳定通过分别记录；保留已验收回退点。

源码/patch入口：[pull_9062_files](sources/engineering_2026_10_05/pull_9062_files.json)
阅读：[ENGINEERING_GUIDE](ENGINEERING_GUIDE_ZH.md) · [CHANGE_REVIEW](CHANGE_REVIEW_ZH.md)
原始证据：[来源1](https://github.com/marin-community/marin/pull/9062) · [来源2](https://github.com/marin-community/marin/issues/8506#issuecomment-5614639876)
使用既有规则：R01, R12, R13, R14，见[18条判断规则](RUBRICS_ZH.md)。

### 3.9 · GPU dump 收窄机制：PDL 下可能提前读取旧分组边界

**症状与机制。** grouped GEMM中部分warps退出，load warp等barrier。dump约束PDL读取旧边界的假说；PDL-off与wheel整包缓解，不提供单组件收益。

**作者做过什么，结果到哪一步。** 作者部署PDL-off与新wheel整包，记录194 paired steps及2052步无hang窗口；组件收益与最终机制没有独立确认。

**本地核对的范围。** launcher diff、dump叙述与公开窗口已核对；未执行GPU边界确认测量。

**下一项检查（未执行）。** 固定wheel/拓扑，记录各warp真实cu_seqlens及前驱完成事件；单独比较PDL开关。

**接受或继续调查的边界。** 机制需边界读值支持；同型warp状态再现要改写假说，不只延长观察窗。

源码/patch入口：[pull_9183_files](sources/engineering_2026_10_05/pull_9183_files.json)
阅读：[ENGINEERING_GUIDE](ENGINEERING_GUIDE_ZH.md) · [CHANGE_REVIEW](CHANGE_REVIEW_ZH.md)
原始证据：[来源1](https://github.com/marin-community/marin/issues/8870#issuecomment-5688372126) · [来源2](https://github.com/marin-community/marin/pull/9183) · [来源3](https://github.com/marin-community/marin/issues/8506#issuecomment-5690222518)
使用既有规则：R12, R13, R14, R18，见[18条判断规则](RUBRICS_ZH.md)。

### 3.10 · 存储策略不让删 checkpoint，却误伤了原子提交

**症状与机制。** 永久metadata copy完成却报AccessDenied。copy-delete提交动作与保护策略冲突；当前固定代码远端metadata direct write。

**作者做过什么，结果到哪一步。** 原调查识别copy/delete与保护策略冲突且未改policy；随后部署保存测试通过。当前固定源码远端metadata使用direct write。

**本地核对的范围。** 原atomic/metadata助手在模拟权限下核对；没有测试真实S3 policy。

**下一项检查（未执行）。** 区分目的对象可见、callback成功和恢复通过，核对temp/permanent前缀实际权限。

**接受或继续调查的边界。** 临时、永久及自己save后restore分别通过；不从已有目的对象掩盖调用失败。

源码/patch入口：[atomic](sources/checkpoint_commit_2026_10_05/atomic.py) · [checkpoint](sources/checkpoint_commit_2026_10_05/checkpoint.py)
阅读：[CHECKPOINT_COMMIT](CHECKPOINT_COMMIT_ZH.md)
原始证据：[来源1](https://github.com/marin-community/marin/issues/8506#issuecomment-5708008076) · [来源2](https://github.com/marin-community/marin/issues/8506#issuecomment-5734989379)
使用既有规则：R13, R14，见[18条判断规则](RUBRICS_ZH.md)。

### 3.11 · Checkpoint 留得多，反而把训练拖停

**症状与机制。** 配额拒写后清理，仍有部分rank停写。配额拒写有证据，清理后256/704停写起因未知；下次save等待旧commit是传播链。

**作者做过什么，结果到哪一步。** 作者清理约49TB并保留指定handoff，服务恢复写入；清理后另一次256/704停写起因仍未确认。

**本地核对的范围。** 本地验证manifest/metadata与假async边界；未定位真实worker或存储首因。

**下一项检查（未执行）。** 采集每rank local commit和各请求首个异常，记录四个水位与容量/保留版本。

**接受或继续调查的边界。** 恢复写入可记缓解；清理后的第二次停滞另立事故，不能沿用配额根因。

源码/patch入口：[checkpoint](sources/checkpoint_commit_2026_10_05/checkpoint.py) · [tensorstore_serialization](sources/checkpoint_commit_2026_10_05/tensorstore_serialization.py)
阅读：[CHECKPOINT_COMMIT](CHECKPOINT_COMMIT_ZH.md) · [CHECKPOINT_MEMORY](CHECKPOINT_MEMORY_ZH.md)
原始证据：[来源1](https://github.com/marin-community/marin/issues/8506#issuecomment-5817400840)
使用既有规则：R13, R14, R18，见[18条判断规则](RUBRICS_ZH.md)。

### 3.12 · 正确的提速改动也需要数值对照

**症状与机制。** FA4/mask/GC整包提速，CE略偏移。三项一起变更，训练CE窗口只支持该窗口数值差；active-only消费者允许特定路径去tail mask。

**作者做过什么，结果到哪一步。** 作者部署FA4、去tail mask与协调GC整包，200步有吞吐改善与小CE偏移；当时eval和自己save后restore尚未覆盖。

**本地核对的范围。** 本地复算200点；NaN-tail测试仅阅读PR，未在SM100执行。

**下一项检查（未执行）。** 毒化unused tail，核对active输出、输入梯度、所有专家参数梯度；另做固定eval和save/restore。

**接受或继续调查的边界。** 整包接受条件与组件消融分开；消费者变更后重新审计mask契约。

源码/patch入口：[pull_9333_files](sources/engineering_2026_10_05/pull_9333_files.json)
阅读：[ENGINEERING_GUIDE](ENGINEERING_GUIDE_ZH.md) · [CHANGE_REVIEW](CHANGE_REVIEW_ZH.md)
原始证据：[来源1](https://github.com/marin-community/marin/pull/9332) · [来源2](https://github.com/marin-community/marin/pull/9333) · [来源3](https://github.com/marin-community/marin/issues/8506#issuecomment-5804146010)
使用既有规则：R03, R12, R13, R14，见[18条判断规则](RUBRICS_ZH.md)。

### 3.13 · Router “提高精度”改变的是路由策略

**症状与机制。** 提高dot精度，score更准但路由改变。top-k离散边界使小误差改变专家集合与QB下一步状态；20步单rack不能保证生产切换。

**作者做过什么，结果到哪一步。** 作者比较旧/新router算术，精度指标改善但真实路由改变；短窗结果不支持中途切换，选择未部署。

**本地核对的范围。** fork三分支静态核对与人工BF16舍入；未运行真实GPU dot、fixture或长期训练。

**下一项检查（未执行）。** 分别测顺序、集合和combine，设计训练/评估policy交叉对照，固定schedule与checkpoint。

**接受或继续调查的边界。** 算术参考更接近不自动批准中途切换；历史作者选择未部署。

源码/patch入口：[model](sources/router_precision_2026_10_05/model.py)
阅读：[ROUTER_PRECISION](ROUTER_PRECISION_ZH.md) · [QB_ESTIMATION](QB_ESTIMATION_ZH.md)
原始证据：[来源1](https://github.com/marin-community/marin/issues/8435#issuecomment-5785310423)
使用既有规则：R03, R12, R14, R15，见[18条判断规则](RUBRICS_ZH.md)。

### 3.14 · 参数存了，不代表评估用了完整状态

**症状与机制。** checkpoint-only eval漏了pending QB更新。raw params与下一次train forward实际使用状态不同；实验结果重算不等于生产路径合入。

**作者做过什么，结果到哪一步。** 作者废弃漏pending state的a1结果并重做a2；实验fork的纠正没有被表述为生产主线已修好。

**本地核对的范围。** 原状态控制流与导出探针已核对，依赖替代；没有真实checkpoint恢复。

**下一项检查（未执行）。** 同state比较下一步train视图、eval视图和export视图，绑定pending的替换语义、重复应用的幂等性及清空后的bias；V111已执行人工模型的原训练步对照，真实模型验收仍待完成。

**接受或继续调查的边界。** 记录实验fork/主线状态；完整状态与metric路径一致才批准评估口径。

源码/patch入口：[state_adapter](sources/state_2026_10_05/state_adapter.py) · [export_vllm](sources/state_2026_10_05/export_vllm.py)
阅读：[TRAIN_STATE](TRAIN_STATE_ZH.md) · [EVAL_METRICS](EVAL_METRICS_ZH.md)
原始证据：[来源1](https://github.com/marin-community/marin/issues/9352)
使用既有规则：R03, R14，见[18条判断规则](RUBRICS_ZH.md)。

### 3.15 · 拉长上下文：drop 变多，先看分母与数据相关性

**症状与机制。** 固定token budget下拉长context，drop上升。独立文档减少可增加局部相关与专家集中；字段sender不必表示裁剪发生在sender。

**作者做过什么，结果到哪一步。** 作者用同180k checkpoint做单rack三seed短诊断，报告长context drop上升及小幅速度下降；生产长上下文质量仍未验证。

**本地核对的范围。** 原drop分母/裁剪辅助检查和作者三seed诊断；没有生产长context结果。

**下一项检查（未执行）。** 固定checkpoint、token budget和路由策略，分开测assignment丢弃、受损token与有效目标。

**接受或继续调查的边界。** 不只比较drop倍数；qk、LR、WD、拓扑变化保留为混杂，按真实能力约束验收。

源码/patch入口：[ragged](sources/routing_2026_10_05/ragged.py) · [ep_common](sources/routing_2026_10_05/ep_common.py)
阅读：[ROUTING_DROPS](ROUTING_DROPS_ZH.md) · [DOCUMENT_BOUNDARIES](DOCUMENT_BOUNDARIES_ZH.md) · [QB_ESTIMATION](QB_ESTIMATION_ZH.md)
原始证据：[来源1](https://github.com/marin-community/marin/issues/9615)
使用既有规则：R02, R03, R12, R14，见[18条判断规则](RUBRICS_ZH.md)。

## 3. 配比与工程为什么必须共用一次变更记录

数据比例改变训练分布，router或drop改变前向，optimizer或pending state改变更新/评估状态，kernel和checkpoint改变执行与可恢复进度。只记录“这次改了什么配置”无法分开这些效应。

因此沿用[五个冻结对象](CHANGE_REVIEW_ZH.md)：输入与数据身份、模型/路由、optimizer与状态、评估口径、执行资源。一次试验先声明哪些对象改变，再选择对照；声明单变量，但其他对象也变了，就收窄为整包观察。

例如108k换配比之后、第一次固定评估之前，还发生PDL/PJRT接续；该生产窗口不能单独估计配比收益。15域均值、16域macro、token加权micro还可能给出不同方向；字段对应和预定保底项应先冻结。见[真实轨迹](MIX_TRAJECTORY_ZH.md)与[指标分母](EVAL_METRICS_ZH.md)。

研究可以继续提出候选和检查，但当前没有独立GPU重复、真实token流重放、完整checkpoint恢复或外部读者评分。缺这些证据不证明候选失败，只限制我们能推荐到哪一步。

## 4. 怎样维护这张总图

用[人工映射配置](config/engineering_case_map.json)维护15案和六入口，运行[构建脚本](scripts/build_engineering_map.py)生成[可机读索引](analysis/engineering_case_map.json)。脚本核对主报告15个标题完整覆盖、原文URL、规则ID和文件SHA，不自动给主张评分。

新增材料时先找它改变哪个机制、结果或退出条件，再修改该案；不同事故或不同attempt分别留时序。新增源代码不能自动将“历史执行版本未知”升级为已确认。旧release和纠正记录继续保留，网页构建日期不改变历史训练快照。

