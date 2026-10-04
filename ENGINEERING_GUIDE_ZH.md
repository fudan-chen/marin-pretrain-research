# 工程排障：怎样把一次恢复变成可复用的判断

最有用的工程经验往往是：先前的解释为什么不成立，下一次该记录什么。535B生产记录里既有明确的配置缺陷，也有只做到恢复进度的事故。把这两种情况都写成“解决了”，下一轮就会沿着错误的机制调参。

本篇把八个案例做成[交互证据链](index.html#engineering-lab)。每案分开列观察、机制、动作、结果和未知，再检查三句具体陈述。选择器展示作者已写好的判断，不给自由文本自动打分。原评论、PR diff、SHA和日期在[原值表](analysis/engineering_data.json)中。

10月5日另抓取三个issue与三个PR的12份公开API文件；#8435/#8506/#8870评论仍为27/58/30条，正文未变。原训练曲线、200步CE复算和数据配比仍采用10月4日冻结数值，不把本次GitHub刷新冒充训练进度刷新。[刷新差异](analysis/engineering_refresh_audit.json)

## 1. 保存后内存变高，先区分内存还活着还是留在分配器

writer tasks 0–127两次保存后的RSS最低值分别增加约11.7和4.4 GiB，非writer的128–175没有同样变化。这支持优先查保存路径；两次增量不同，也不支持“每次固定泄漏30 GiB”。八分钟后的纠正指出，manager等待completed futures后没有清掉引用，future仍可能持有原buffer。[观察](https://github.com/marin-community/marin/issues/8506#issuecomment-5389681869) · [纠正](https://github.com/marin-community/marin/issues/8506#issuecomment-5389725084)

这与pinned BFC pool是两条不同链：引用仍在时，对象不能回收；引用清了以后，allocator仍可能保留内存作复用，RSS不一定立即降到启动值。因此“RSS没降”不能单独证明清引用无效，“RSS降了”也不能说明pool行为已弄清楚。

实际排查时，同步记四项：future/源buffer的可达量、allocator retained bytes、writer RSS、非writer RSS。先清引用，再单独改变allocator措施，才能区分措施解决哪一层。本报告没有在生产集群执行这组测量。

## 2. 排除解释，要指出它缺了哪一个必要条件

“刚过watch step就挂了”曾被解释为watch/plain executable交替。后续查部署commit，却是`WatchMode.INLINE`，每步一个训练编译路径，`step%10`只决定host日志；capacity-limited eval也早已禁用。原假说要求的交替根本不在这条路径上。[配置与排除过程](https://github.com/marin-community/marin/issues/8870#issuecomment-5531534090)

这种排除强于“试了几次没复现”：它否定了具体触发条件。单rack不复现则弱一些，因为11rack路径还有跨rack同步和704-rank通信。两者应分别记录：**必要条件缺失，可以撤回这条解释；测试环境没有触发，保留适用范围内的阴性结果。**

## 3. 200步通过以后，为什么还要留回退状态

NCCL headers与runtime不一致是明确的依赖缺陷，#9062把ARM64 PJRT依赖换为相应发布wheel。候选通过200步之后，生产在1590 clean steps后仍挂起。这同时支持两件事：版本偏差被修正；不能宣布silent hang已经全部消失。[依赖diff](https://github.com/marin-community/marin/pull/9062/files) · [复发](https://github.com/marin-community/marin/issues/8506#issuecomment-5614639876)

四次早期attempt与一次更长区间没有形成匹配条件的独立故障样本，不能直接算故障概率降低多少倍。观察窗也不应套一个通用“200步稳定即可上线”的数字：数值偏差可以很快暴露，稀有同步问题却可能需要较长窗口和相同故障特征的记录。

从这个案例可提炼两层验收。第一层在相同handoff和批次上查数值、路由与速度；第二层查生产稳定性和自身保存后恢复。每层有自己的终止条件。清洁推进了多少步、是否由preemption结束、是否出现同型hang、是否遇到独立硬件故障，不能挤成一个“通过”。

## 4. PDL机制解释与PDL-off缓解结果，要分开看

dump直接看到MMA/epilogue warps已退出、load warp仍等barrier。第二份dump有cluster在第一个tile就出现分歧，使“PDL下未等前驱的warp先读到旧`cu_seqlens`”成为作者更倾向的机制。确认测量仍是记录各warp实际读取的边界，而不是只观察退出状态。[第二份dump](https://github.com/marin-community/marin/issues/8870#issuecomment-5688372126)

#9183在相关QuACK launcher统一传`use_pdl=False`。部署还换了PJRT wheel。启动后194 paired steps的训练CE均差约−3.8e−5；随后2,052步没有hang/retry/watchdog。作者认为这与预期相符，也明确保留了“相同warp状态再现就改写判断”的条件。[diff](https://github.com/marin-community/marin/pull/9183/files) · [短窗](https://github.com/marin-community/marin/issues/8506#issuecomment-5690222518) · [后续窗口](https://github.com/marin-community/marin/issues/8506#issuecomment-5695342852)

可用于下一次排障的规则是：**dump提供机制约束，措施提供干预结果，观察窗提供适用时间；三者共同增加支持，但不要互相代填。** 代码注释把机制写得更肯定，也不能替代实际边界测量。整包吞吐+2.6%缺少固定wheel的PDL消融，不能归给关闭PDL一个参数。

## 5. 配额清理之后，又发生了另一种保存停滞

原事件索引把同一评论压成“超过quota，清理后正常保存”，漏掉了中间尚未确认的一次事故。现已修正为两段。[完整事故](https://github.com/marin-community/marin/issues/8506#issuecomment-5817400840)

| UTC时间 | 对应事件 | 有什么证据 | 能写到哪一步 |
|---|---|---|---|
| 09-23 23:43:47 | step146585 staged save | 104.56 TiB超过100 TiB；405拒写；rank0 manifest upload无timeout | 配额拒写和manifest等待有对应证据 |
| 09-24 00:39起 | 清理后写入恢复 | 使用量约80 TB，服务重新接受写入 | 配额措施恢复了服务写入 |
| 01:01 | step146582 background save | 448/704进程完成，256个分布在132hosts的进程停写，无报错 | **起因未确认**；不能自动归给仍超quota |
| 02:43 / 03:44 | 后续保存 | 146582、146817保存正常 | 恢复了保存进度，不填补上一段根因 |

异步保存的陷阱在这里很具体：训练还能继续，而commit从未完成；下一次save才被上一次任务阻塞。`stage成功`、`upload结束`、`commit可见`和`restore成功`必须分开。记录说146355–146732被训练两次，停滞/重启约两小时；这里只保留原文估计，不把步数差当全部wall-time成本。

删除checkpoint也改变回退能力：该次清理后保留108778/121638/146139等durable handoff，81716已删除，58014无durable副本。自己的管线应在容量预算里显式保留验证过的回退点；本报告没有执行任何生产删除。

## 6. 整包提速以后，最先要避免哪两种过度归因

146139的部署同时改FA4、ragged tail-mask与协调GC。200同批次步数里，MFU24.10%→26.75%，tokens/s2.83M→3.14M，CE均差约+3.6e−4。这支持整包在该窗口改善吞吐，同时公开数值偏移。[验收记录](https://github.com/marin-community/marin/issues/8506#issuecomment-5804146010)

第一种过度归因是把全部收益叫FA4收益。第二种是把200步CE接近叫模型质量相同。验收当时147000 eval、新run自己保存后恢复和首个permanent save尚未覆盖。本地复算了公开CE点，未取得token ID来独立证明字节级输入一致。

速度本身也有口径：median step是16.29s→14.67s，iteration是18.62s→15.53s。前者下降约9.94%，后者下降约16.60%；换成倒数速度比分别约+11.04%、+19.90%。这四个百分比描述的分母不同，不该混成“训练提升20%”。GC可能影响host停顿，kernel可能影响device执行；没有拆包对照时，这只是进一步测量的方向。

要归组件，可在同handoff/wheel上逐项引入FA4、mask、GC，并记录相互作用；要验收整包，则先完成固定评估、临时/永久保存和自己保存后恢复。两类目的使用不同实验，不必为接受一个有界整包候选先虚构组件收益。

## 7. 去mask的证据，应追到最后一个消费者

ragged QuACK grouped GEMM按`cu`读segment，return transport只读active rows，所以unused tail可以保持未定义值。local/FSDP combine读取完整buffer，仍保留mask。合法优化取决于这条消费契约，不取决于某一张buffer表面上看起来没用。[#9333 diff](https://github.com/marin-community/marin/pull/9333/files)

PR让unused tail为NaN，比较active输出、active输入梯度以及全部专家参数梯度，tail_rows覆盖0和127。这个测试针对的是“污染能否进入真实计算”，比要求unused tail归零更贴近机制。若以后新消费者读取完整buffer，就要重新审计，不能沿用原来通过的结论。本机只核对diff，未在SM100运行这些测试。

## 8. Teardown报错之前，系统已经停止了什么

9月27日先是gang停止推进，随后900秒watchdog结束，CUDA peer-memory errors首次出现在teardown。源记录明确说它们不能确立发起stall的GPU故障；重试从166006恢复，初始原因仍未知。[时序纠正](https://github.com/marin-community/marin/issues/8506#issuecomment-5856665151)

自己的故障表至少保留last progress、first abnormal event、watchdog、kill、restart五个时点。统一UTC，并记录采集端的时钟差和未采到的时间段。退出后的报错可用于解释退出，却需要先行独立证据才能解释最初停滞；“没见警报”也不是排除全部GPU问题的证明。

## 9. 把R13落实为一次排障的六个动作

本轮没有增加规则数量或修改1.1锚点。主要应用R13（根因/排除/缓解/未知）、R03（指标口径）、R14（实际状态）、R12（预定接受条件），按下面的执行顺序留下产物。这是一条工程检查支线，**不把现有配比管线P2的供体项机械用来批准kernel发布**。

| 动作 | 要留下的材料 | 下一步如何决定 |
|---|---|---|
| 固定事故身份 | run、commit、wheel、拓扑、handoff、UTC起止与已发生变化 | 身份缺失时先补；不要混两次save或不同attempt |
| 还原首个异常 | last progress与首个独立异常，区分退出后的报错 | 只有末端barrier/teardown时，保留发起原因未知 |
| 列必要条件 | 每个机制必须有哪些路径/状态；哪项已被原值排除 | 必要条件不存在就撤回该假说；阴性试验注明环境 |
| 选择一项诊断 | 能区分至少两种解释的测量，写预测与推翻条件 | 不写“再多跑一些”替代具体缺口 |
| 做有界验收 | 数值、性能、稳定窗、临时/永久save、自己save后restore与回退点分别打表 | 短窗只通过对应范围；整包不归组件 |
| 复盘并保留反例 | 原主张、后来的纠正、遗漏事件、资源损失和新版本 | 发现反证改正文；不要只向旧结论添加成功日志 |

空表见[工程排查模板](templates/engineering_incident.md)。页面可导出每案的预期/推翻条件和自己的备注；`execution_status`保持`not_executed`，理解结果与训练审批为空。真实集群测试需要生产环境，真实读者理解需要实际回答；这两项不能由报告作者编案例来完成。

## 10. 本轮实际改正了什么

1. 主报告把146k“质量近似对齐”改为这段训练CE比较，并补齐组合变更中的GC。
2. PDL章将dump观测与旧边界读取机制分开，保留确认测量与复发推翻条件。
3. 81716章说明同checkpoint/批次由公开部署记录支持，本地独立复算的是step对齐的指标点。
4. 58事件索引第54条补回清理后的256进程停滞，把已确认配额事故和未确认事件分开。

这些是已经实施的文字与证据范围修正，不是新增的训练结果。精确旧文、新文与所用规则见[语义审计](analysis/engineering_scope_audit.json)。

当前报告重建通过803项检查；浏览器验收包括八案24项陈述、实际按钮Blob、离线14图、旧题库拒绝和现有缓存不变。桌面与390px截图已逐张查看；实际OS下载落盘、GPU执行与外部理解仍未验证。完整交付范围见[审计](DELIVERY_AUDIT_ZH.md)。
