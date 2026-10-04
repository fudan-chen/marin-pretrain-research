# -*- coding: utf-8 -*-
"""Generate precise mixture tables and a Chinese production-event index."""
import pathlib,json,csv,re
ROOT=pathlib.Path(__file__).resolve().parents[1]
names=['世界新闻与地缘政治','消费健康与补充剂','艺术、历史与文化','金融、保险与市场','消费科技与网络安全','政府公告与环境','公共政策讨论','游戏与赌博','学校与儿童活动','学习方法与学术写作','个人健康与行为','食品与食谱','旅游与目的地','命令行Agent记录','软件开发与Web代码','临床治疗与牙科','体育与竞赛','SEO与网络商业','商业软件与策略','家居维护与手工业','学校课程与教师培训','基督教信仰与灵修','应用与Web开发','消费设备与车辆','情绪、关系与身心健康','幻想与同人写作','议会与董事会会议记录','常识与问答','性能日志与底层代码','课程、辅导与学校服务','自然科学研究','小说、电影与电视','软件基础设施与安全','赛车、体育与娱乐记录','宠物、园艺与害虫防治','手工、设计与家居装饰','赌博与金融服务','法律、法院与监管','历史、文学与遗产','数学问题与证明']
domains=json.loads((ROOT/'analysis/domain_weights.json').read_text())
cells=json.loads((ROOT/'analysis/cell_weights.json').read_text())
for d in domains:d['name_zh']=names[d['id']]
(ROOT/'analysis/domain_weights_zh.json').write_text(json.dumps(domains,ensure_ascii=False,indent=2))
lines=['### 7.1 40个语义域的全部比例','','| ID | 中文导航（原始英文名保留在CSV/查询器） | 库存/T | 库存占比 | 原配比 | 新主阶段 | Cooldown计划 |','|---|---|---:|---:|---:|---:|---:|']
for d in domains:lines.append('| c%02d | %s | %.4f | %.3f%% | %.3f%% | %.3f%% | %.3f%% |'%(d['id'],d['name_zh'],d['store_tokens']/1e12,d['store_pct'],d['phase0_pct'],d['phase1_pct'],d['phase2_pct']))
lines+=['','### 7.2 计划曝光轮数最高的十个桶','','仅按当前三阶段计划计算；未来配比、长度与cursor改变会使实际曝光不同。极小桶的高轮数不能按绝对影响排名。','','| 桶 | 中文导航 | 库存/B tokens | 计划抽取/B tokens | 曝光轮数 | 原配比 | 新主阶段 | Cooldown计划 |','|---|---|---:|---:|---:|---:|---:|---:|']
for c in sorted(cells,key=lambda x:-x['planned_epochs'])[:10]:lines.append('| %s | %s | %.4f | %.4f | %.3f | %.5f%% | %.5f%% | %.5f%% |'%(c['cell'],names[c['domain_id']],c['available_tokens']/1e9,c['planned_tokens']/1e9,c['planned_epochs'],c['phase0_pct'],c['phase1_pct'],c['phase2_pct']))
lines+=['','### 7.3 d1536固定Paloma的全部子集BPB结果','','| 子集 | 旧配比BPB | 新配比BPB | 相对变化 |','|---|---:|---:|---:|']
for r in csv.DictReader((ROOT/'sources/mix_study_final-2026.09.15.1_final_results.csv').open()):
    if r['metric']=='bpb':lines.append('| %s | %.6f | %.6f | %+.3f%% |'%(r['subset'],float(r['baseline']),float(r['final']),float(r['change_pct'])))
p=ROOT/'DATA_GUIDE_ZH.md';text=p.read_text().split('<!-- GENERATED_TABLES -->')[0];p.write_text(text+'<!-- GENERATED_TABLES -->\n\n'+'\n'.join(lines)+'\n')

summaries={
1:'约每3小时OOM。提高XLA memory fraction至0.83反而更差，slop 0.88无效，0.85仍在试；这是参数尝试，未证明根因。',
2:'step约2359明确出现不可恢复NVLink错误，同rack其他任务连带失败。记录互连硬件证据，不能解释为Loss发散。',
3:'补记step164的手动恢复host OOM；由PR #8480的恢复内存修复后继续。',
4:'针对checkpoint后周期性host OOM，限制restore并发、offload cache为16GB并试malloc_trim。属于分阶段处理。',
5:'malloc_trim仍不稳定，改用含jemalloc的main。#8584处理单worker restore stall；首次成功也可能受缓存影响，作者明确没有确认根治。',
6:'step11252保存时OOM，取消root以避免两个704GPU gang重叠，再启动coord-memfix实例。',
7:'只有writer tasks 0–127的RSS最低值在两次保存后升11.7、4.4GiB，非writer稳定；最初归因pinned BFC pool。',
8:'纠正上一条：checkpoint manager等待后未清空_commit_futures，完成的future仍可能保留源buffer；pool问题与引用问题要分开。',
9:'训练task8在序列化时OOM，16秒后2GB coordinator也OOM；零重试预算使child结束。#8617增加coordinator至4GB并配置重试；不是训练数值修复。',
10:'1GB与125GB loader cache吞吐接近，低cap仍有约18.6倍训练所需余量；支持缩小缓存释放RAM。',
11:'节点失联导致process333–335 heartbeat消失，Kubernetes标NotReady；是节点/连接证据，重试恢复不等于硬件原因已确定。',
12:'step50195后task83出现IB port error、IBFlapping；另一次恢复也卡住。把网络错误与恢复阶段问题分别记录。',
13:'首次从pooled-wave换ragged，旧run停于54262，新run从54000完整状态开始，使用独立run和checkpoint树。',
14:'新ragged两次silent hang，虽约70步Loss/速度/drop正常仍回退旧代码和旧checkpoint。',
15:'回退恢复54000，重放Loss与旧轨迹匹配；这验证回退状态，而不是验证ragged稳定性。',
16:'强制提交step58014再暂停，少丢约131steps。11rack调试235步未复现hang，随后部署gate/router WD=0.02；不把未复现当根治。',
18:'补链9月5日IB port error导致的重试事件；内部事故文档未公开，报告仅使用这条公开说明。',
19:'提出9月9日ragged部署方案：先合并依赖，强制handoff，旧run续跑200步作为control，候选另run。计划条目需与后续结果配对。',
20:'强制checkpoint81716，约5.36TB、26795objects，验完整后保留为handoff；记录恢复前状态依据。',
22:'旧run完成control区间后取消，新run从81716启动；部署数据不跨run覆盖。',
23:'首5个paired steps符合预期数值/drop/速度签名；窗口短，只能初筛。',
25:'第一attempt仅43步后发生#8870型hang。Loss下降和提速不能代替稳定性验证。',
28:'attempt2、3分别0步就hang，故障rack改变。每30秒用低于其他rack约0.6倍的CPU占用辅助检测，重启最多5次；检测信号不是根因。',
31:'第四次同型hang，所有rank都在jitted train_step内。切换到NCCL2.30.7 headers编译wheel；其他配置与handoff保持相同。',
32:'新headers wheel通过200步无hang窗口，数值/drop通过；决定保留候选，但未来仍可能失败。',
34:'#9062合并，部署正式发布的同headers wheel与正确handoff。试验分支结果进入生产需核对依赖与配置。',
35:'正式新run完成200步且第一次临时checkpoint提交。CE−0.006498、drop大幅下降、MFU提高；后续1590步仍发生hang。',
36:'单rack验证新run自己保存的master-less checkpoint，恢复后跑20步正常。排除只验证旧格式迁移而遗漏新格式恢复。',
37:'NCCL2.30.7 wheel在1590个干净steps后再次hang；说明200步通过不足以证实罕见故障解决。',
38:'恢复并重放83305–83307，Loss一致；上一条checkpoint步号83306被纠正为83305。',
39:'所有76communicators活着，65个AllReduce出现计数差，单rackEP路径未到归约。没有发现先行Kubernetes硬件警报；末端归约阻塞不是起因证据。',
40:'随后attempt1死亡是明确NVLink hardware fault，与silent hang分开；不能把两次失败混用同一根因。',
42:'108000切换到996f4891配比并开始训练，新分布CE上升。配置变化是数据事件，不能与81k执行路径事件混淆。',
43:'计划部署PDL-off QuACK和已验证的PJRT轮包，以108778为handoff；先留下旧run对照窗口。',
45:'候选恢复108778开始训练，初始两steps与旧control匹配。内核重新编译与整体XLA cache命中分别记录。',
46:'完整trial后接受：启动后194个paired steps，Loss均差−3.8e−5，吞吐约+2.6%。同时改PJRT轮包，不将提速全归PDL-off。',
47:'PDL-off通过2052步无hang，超出旧run常见1–2k步故障间隔；这提高信心，仍不是任何长期故障都已消除的证明。',
48:'永久checkpoint metadata copy成功但删除临时源被DenyHeroCheckpointDeletion拒绝。临时checkpoint已通过，永久保存仍暴露；调查未修改policy。',
49:'追到时间关系：9月15日#8974改atomic_rename，8月22日policy早已存在；114000为此后首个永久里程碑。旧lock-release AccessDenied不是同一故障。',
50:'从121638完整handoff部署main，200步数值/路由通过，并验证新保存。checkpoint附近时间使平均step变慢，排除邻近窗口仅+0.10%是事后诊断。',
51:'计划从146139部署native SM100 FA4、去ragged tail masks、每100step协调GC；捆绑改动需整包对照。',
53:'整包200步接受，MFU26.75%对24.10%，CE均差+3.6e−4；没有大偏离。eval、自己保存后恢复、首个永久保存当时仍未覆盖。',
54:'分两段：104.56TiB超100TiB、405拒写与rank0无timeout manifest阻塞有对应证据。清理恢复写入后，01:01的146582保存仍有256/704进程停写，起因未确认；后续两次保存正常。清理约49TB，保留3个durable handoff，81716已删除，58014无durable副本。',
55:'9月27日watchdog终止并重试，最初stall原因未知。CUDA peer-memory errors在teardown才出现，不能用来倒推GPU根因。',
56:'9月30日184731临时保存未完成，下次保存checkpoint barrier timeout。先行storage/worker/GPU原因未找到；恢复后继续。',
57:'10月2日多个启动重试，RegisterTask超时与multihost tracker初始化失败；起始原因未确认，attempt12恢复至193732。',
58:'10月3日198000保存读旧OCDBT object返回HTTP400，其他rank barrier超时。重启时task17四rank卡编译/cache读502，节点健康读取无权限；根因未确认，提出生产owner后续检查。'
}
comments=json.loads((ROOT/'sources/issue_8506_comments.json').read_text())
out=['# #8506 生产运行时间线：58条记录逐条索引','','这里按公开评论顺序保留全部58条事件。中文栏概括事件、处理和证据边界；完整英文与元数据保存在[sources/issue_8506_comments.json](sources/issue_8506_comments.json)。工程机制的详细解释见主报告第三部分。日期为UTC，避免把原记录时区隐式改掉。公开评论大量由agent生成，后续的人工纠正和反例优先；内部Echo/Iris链接未被当作已读取证据。','','| 序号 | UTC时间 | 做了什么、解决到哪一步、原因是否确定 | 原评论 |','|---|---|---|---|']
for i,c in enumerate(comments,1):
    summary=summaries.get(i)
    if summary is None:
        run=re.search(r'Run ID:\s*`([^`]+)`',c['body']);commit=re.search(r'Commit:\s*`([^`]+)`',c['body'])
        assert run and commit,(i,c['body'][:100])
        summary='启动记录：run `%s`，代码 `%s`。这是部署身份记录，是否通过需看后面的结果。'%(run[1],commit[1][:10])
    out.append('| %02d | %s | %s | [定位](%s) |'%(i,c['created_at'].replace('T',' ').replace('Z',''),summary,c['html_url']))
(ROOT/'OPERATIONS_ZH.md').write_text('\n'.join(out)+'\n')
print('40-domain tables, top exposures, 17 eval rows, 58 production records generated.')
