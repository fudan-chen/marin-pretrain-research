# #8506 生产运行时间线：58条记录逐条索引

这里按公开评论顺序保留全部58条事件。中文栏概括事件、处理和证据边界；完整英文与元数据保存在[sources/issue_8506_comments.json](sources/issue_8506_comments.json)。工程机制的详细解释见主报告第三部分。日期为UTC，避免把原记录时区隐式改掉。公开评论大量由agent生成，后续的人工纠正和反例优先；内部Echo/Iris链接未被当作已读取证据。

| 序号 | UTC时间 | 做了什么、解决到哪一步、原因是否确定 | 原评论 |
|---|---|---|---|
| 01 | 2026-08-20 23:22:04 | 约每3小时OOM。提高XLA memory fraction至0.83反而更差，slop 0.88无效，0.85仍在试；这是参数尝试，未证明根因。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5363258940) |
| 02 | 2026-08-21 04:31:59 | step约2359明确出现不可恢复NVLink错误，同rack其他任务连带失败。记录互连硬件证据，不能解释为Loss发散。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5365222655) |
| 03 | 2026-08-21 15:59:48 | 补记step164的手动恢复host OOM；由PR #8480的恢复内存修复后继续。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5372233783) |
| 04 | 2026-08-21 18:27:40 | 针对checkpoint后周期性host OOM，限制restore并发、offload cache为16GB并试malloc_trim。属于分阶段处理。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5373722833) |
| 05 | 2026-08-22 15:43:25 | malloc_trim仍不稳定，改用含jemalloc的main。#8584处理单worker restore stall；首次成功也可能受缓存影响，作者明确没有确认根治。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5381228200) |
| 06 | 2026-08-23 23:11:34 | step11252保存时OOM，取消root以避免两个704GPU gang重叠，再启动coord-memfix实例。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5389008642) |
| 07 | 2026-08-24 01:28:38 | 只有writer tasks 0–127的RSS最低值在两次保存后升11.7、4.4GiB，非writer稳定；最初归因pinned BFC pool。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5389681869) |
| 08 | 2026-08-24 01:36:39 | 纠正上一条：checkpoint manager等待后未清空_commit_futures，完成的future仍可能保留源buffer；pool问题与引用问题要分开。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5389725084) |
| 09 | 2026-08-24 11:03:59 | 训练task8在序列化时OOM，16秒后2GB coordinator也OOM；零重试预算使child结束。#8617增加coordinator至4GB并配置重试；不是训练数值修复。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5394314776) |
| 10 | 2026-08-24 21:29:34 | 1GB与125GB loader cache吞吐接近，低cap仍有约18.6倍训练所需余量；支持缩小缓存释放RAM。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5401613236) |
| 11 | 2026-08-27 17:16:28 | 节点失联导致process333–335 heartbeat消失，Kubernetes标NotReady；是节点/连接证据，重试恢复不等于硬件原因已确定。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5442658185) |
| 12 | 2026-09-02 00:25:58 | step50195后task83出现IB port error、IBFlapping；另一次恢复也卡住。把网络错误与恢复阶段问题分别记录。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5502471700) |
| 13 | 2026-09-03 00:19:44 | 首次从pooled-wave换ragged，旧run停于54262，新run从54000完整状态开始，使用独立run和checkpoint树。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5518372559) |
| 14 | 2026-09-03 01:16:59 | 新ragged两次silent hang，虽约70步Loss/速度/drop正常仍回退旧代码和旧checkpoint。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5518830230) |
| 15 | 2026-09-03 01:31:10 | 回退恢复54000，重放Loss与旧轨迹匹配；这验证回退状态，而不是验证ragged稳定性。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5518941160) |
| 16 | 2026-09-04 01:07:13 | 强制提交step58014再暂停，少丢约131steps。11rack调试235步未复现hang，随后部署gate/router WD=0.02；不把未复现当根治。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5534208221) |
| 17 | 2026-09-04 23:32:14 | 启动记录：run `hero-wd-gate-router-p02-step58k`，代码 `0908920c86`。这是部署身份记录，是否通过需看后面的结果。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5547644106) |
| 18 | 2026-09-05 18:19:13 | 补链9月5日IB port error导致的重试事件；内部事故文档未公开，报告仅使用这条公开说明。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5553842711) |
| 19 | 2026-09-09 17:26:25 | 提出9月9日ragged部署方案：先合并依赖，强制handoff，旧run续跑200步作为control，候选另run。计划条目需与后续结果配对。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5606002395) |
| 20 | 2026-09-09 18:31:06 | 强制checkpoint81716，约5.36TB、26795objects，验完整后保留为handoff；记录恢复前状态依据。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5606812293) |
| 21 | 2026-09-09 19:34:39 | 启动记录：run `hero-ragged_a2a-ep-step81k`，代码 `9ccc1bd5e4`。这是部署身份记录，是否通过需看后面的结果。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5607600274) |
| 22 | 2026-09-09 19:35:15 | 旧run完成control区间后取消，新run从81716启动；部署数据不跨run覆盖。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5607607490) |
| 23 | 2026-09-09 19:49:13 | 首5个paired steps符合预期数值/drop/速度签名；窗口短，只能初筛。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5607788308) |
| 24 | 2026-09-09 20:23:38 | 启动记录：run `hero-ragged_a2a-ep-step81k`，代码 `9ccc1bd5e4`。这是部署身份记录，是否通过需看后面的结果。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5608209836) |
| 25 | 2026-09-09 20:24:06 | 第一attempt仅43步后发生#8870型hang。Loss下降和提速不能代替稳定性验证。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5608215407) |
| 26 | 2026-09-09 20:39:01 | 启动记录：run `hero-ragged_a2a-ep-step81k`，代码 `9ccc1bd5e4`。这是部署身份记录，是否通过需看后面的结果。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5608404121) |
| 27 | 2026-09-09 20:48:56 | 启动记录：run `hero-ragged_a2a-ep-step81k`，代码 `9ccc1bd5e4`。这是部署身份记录，是否通过需看后面的结果。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5608541587) |
| 28 | 2026-09-09 20:49:14 | attempt2、3分别0步就hang，故障rack改变。每30秒用低于其他rack约0.6倍的CPU占用辅助检测，重启最多5次；检测信号不是根因。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5608545188) |
| 29 | 2026-09-09 21:07:43 | 启动记录：run `hero-ragged_a2a-ep-step81k`，代码 `9ccc1bd5e4`。这是部署身份记录，是否通过需看后面的结果。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5608763494) |
| 30 | 2026-09-09 21:11:18 | 启动记录：run `hero-ragged_a2a-ep-step81k`，代码 `50f42bc43a`。这是部署身份记录，是否通过需看后面的结果。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5608807722) |
| 31 | 2026-09-09 21:11:40 | 第四次同型hang，所有rank都在jitted train_step内。切换到NCCL2.30.7 headers编译wheel；其他配置与handoff保持相同。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5608811817) |
| 32 | 2026-09-09 22:24:57 | 新headers wheel通过200步无hang窗口，数值/drop通过；决定保留候选，但未来仍可能失败。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5609573154) |
| 33 | 2026-09-09 22:50:37 | 启动记录：run `hero-ragged_a2a-nccl2307-ep-step81k`，代码 `04fb348456`。这是部署身份记录，是否通过需看后面的结果。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5609812430) |
| 34 | 2026-09-09 22:51:07 | #9062合并，部署正式发布的同headers wheel与正确handoff。试验分支结果进入生产需核对依赖与配置。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5609817090) |
| 35 | 2026-09-09 23:59:35 | 正式新run完成200步且第一次临时checkpoint提交。CE−0.006498、drop大幅下降、MFU提高；后续1590步仍发生hang。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5610467489) |
| 36 | 2026-09-10 05:18:58 | 单rack验证新run自己保存的master-less checkpoint，恢复后跑20步正常。排除只验证旧格式迁移而遗漏新格式恢复。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5613579243) |
| 37 | 2026-09-10 07:13:36 | NCCL2.30.7 wheel在1590个干净steps后再次hang；说明200步通过不足以证实罕见故障解决。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5614639876) |
| 38 | 2026-09-10 07:17:12 | 恢复并重放83305–83307，Loss一致；上一条checkpoint步号83306被纠正为83305。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5614680529) |
| 39 | 2026-09-10 07:54:46 | 所有76communicators活着，65个AllReduce出现计数差，单rackEP路径未到归约。没有发现先行Kubernetes硬件警报；末端归约阻塞不是起因证据。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5615118768) |
| 40 | 2026-09-10 08:12:35 | 随后attempt1死亡是明确NVLink hardware fault，与silent hang分开；不能把两次失败混用同一根因。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5615346838) |
| 41 | 2026-09-15 17:15:14 | 启动记录：run `hero-mix-996f4891-step108k`，代码 `0cb3167f39`。这是部署身份记录，是否通过需看后面的结果。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5684706768) |
| 42 | 2026-09-15 17:27:10 | 108000切换到996f4891配比并开始训练，新分布CE上升。配置变化是数据事件，不能与81k执行路径事件混淆。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5684891157) |
| 43 | 2026-09-15 23:20:36 | 计划部署PDL-off QuACK和已验证的PJRT轮包，以108778为handoff；先留下旧run对照窗口。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5689462381) |
| 44 | 2026-09-15 23:20:44 | 启动记录：run `hero-nopdl-step108k`，代码 `8f6f33bebe`。这是部署身份记录，是否通过需看后面的结果。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5689464022) |
| 45 | 2026-09-15 23:34:49 | 候选恢复108778开始训练，初始两steps与旧control匹配。内核重新编译与整体XLA cache命中分别记录。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5689628437) |
| 46 | 2026-09-16 00:33:19 | 完整trial后接受：启动后194个paired steps，Loss均差−3.8e−5，吞吐约+2.6%。同时改PJRT轮包，不将提速全归PDL-off。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5690222518) |
| 47 | 2026-09-16 09:38:40 | PDL-off通过2052步无hang，超出旧run常见1–2k步故障间隔；这提高信心，仍不是任何长期故障都已消除的证明。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5695342852) |
| 48 | 2026-09-17 03:22:13 | 永久checkpoint metadata copy成功但删除临时源被DenyHeroCheckpointDeletion拒绝。临时checkpoint已通过，永久保存仍暴露；调查未修改policy。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5707969984) |
| 49 | 2026-09-17 03:26:53 | 追到时间关系：9月15日#8974改atomic_rename，8月22日policy早已存在；114000为此后首个永久里程碑。旧lock-release AccessDenied不是同一故障。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5708008076) |
| 50 | 2026-09-18 19:15:11 | 从121638完整handoff部署main，200步数值/路由通过，并验证新保存。checkpoint附近时间使平均step变慢，排除邻近窗口仅+0.10%是事后诊断。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5734989379) |
| 51 | 2026-09-23 21:39:45 | 计划从146139部署native SM100 FA4、去ragged tail masks、每100step协调GC；捆绑改动需整包对照。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5803401391) |
| 52 | 2026-09-23 21:40:58 | 启动记录：run `hero-fa4sm100-nomask-step146k`，代码 `ad754c6d67`。这是部署身份记录，是否通过需看后面的结果。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5803416767) |
| 53 | 2026-09-23 22:41:06 | 整包200步接受，MFU26.75%对24.10%，CE均差+3.6e−4；没有大偏离。eval、自己保存后恢复、首个永久保存当时仍未覆盖。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5804146010) |
| 54 | 2026-09-24 15:47:28 | 分两段：104.56TiB超100TiB、405拒写与rank0无timeout manifest阻塞有对应证据。清理恢复写入后，01:01的146582保存仍有256/704进程停写，起因未确认；后续两次保存正常。清理约49TB，保留3个durable handoff，81716已删除，58014无durable副本。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5817400840) |
| 55 | 2026-09-27 14:21:06 | 9月27日watchdog终止并重试，最初stall原因未知。CUDA peer-memory errors在teardown才出现，不能用来倒推GPU根因。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5856665151) |
| 56 | 2026-10-01 00:33:26 | 9月30日184731临时保存未完成，下次保存checkpoint barrier timeout。先行storage/worker/GPU原因未找到；恢复后继续。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5922288472) |
| 57 | 2026-10-02 15:57:11 | 10月2日多个启动重试，RegisterTask超时与multihost tracker初始化失败；起始原因未确认，attempt12恢复至193732。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5956153758) |
| 58 | 2026-10-03 11:27:19 | 10月3日198000保存读旧OCDBT object返回HTTP400，其他rank barrier超时。重启时task17四rank卡编译/cache读502，节点健康读取无权限；根因未确认，提出生产owner后续检查。 | [定位](https://github.com/marin-community/marin/issues/8506#issuecomment-5968715246) |
