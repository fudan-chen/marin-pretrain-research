# -*- coding: utf-8 -*-
"""Bind authored engineering interpretations to dated public comments and PR diffs."""
import hashlib, json, pathlib
R=pathlib.Path(__file__).resolve().parents[1]
PREFIX='sources/engineering_2026_10_05/'
episodes=[]
def ref(issue, comment, needle):
    return {'file':PREFIX+'issue_%d_comments.json'%issue,'comment_id':comment,'needle':needle}
def pull(number, needle):
    return {'file':PREFIX+'pull_%d_files.json'%number,'pull_number':number,'needle':needle}
def add(id,title,observation,mechanism,action,outcome,unknown,events,claims,next_test,rules,refs):
    episodes.append(dict(id=id,title=title,observation=observation,mechanism=mechanism,action=action,
        outcome=outcome,unknown=unknown,events=events,claims=claims,next_test=next_test,rules=rules,refs=refs))
def event(label, text, ref_index):return {'label':label,'text':text,'ref_index':ref_index}
def claim(text,status,reason):return {'text':text,'status':status,'reason':reason}

add('E01','保存后 RSS 抬高：引用和分配器不是一回事',
 'writer tasks 0–127 在两次保存后的 RSS 最低值分别抬高约11.7、4.4 GiB；非 writer 的128–175保持稳定。',
 '完成的 future 仍可持有源 buffer；pinned BFC pool 又可能保留已释放内存作复用。一个是对象仍可达，另一个是内存归分配器但未退还操作系统。',
 '排查 _commit_futures；修复方向是在等待后清理引用，但当时 #8599 的 post-join cleanup 尚未随 #8609 落地。恢复并发、host cache 上限和 allocator 措施另行尝试。',
 'writer/non-writer 差别把调查收窄到保存路径；这两次观测不支持固定每次泄漏30 GiB的模型。',
 '清引用后 RSS 下降多少、pool floor还剩多少，以及长时间生产是否不再OOM，不能由这条纠正代填。',
 [event('08-24 01:28 UTC','先把 writer 的 RSS 增量与 pinned pool 联系起来。',0),event('01:36 UTC','八分钟后补充 completed futures 的独立保留路径。',1)],
 [claim('保存路径值得优先调查，引用和 pool 需要分别测。','supported','writer/non-writer 对照与完成 future 的引用路径共同支持这一调查方向。'),claim('清空 futures 就证明所有 host OOM 根治。','unknown','缺少清理前后引用、pool和多次保存的生产比较。'),claim('两次记录证明每次固定泄漏30 GiB。','refuted','两次地板增量不相同，作者也纠正了固定增量解释。')],
 {'plan':'同一 writer 连续多次保存，分开记录 live buffer/future 数、pool retained bytes、RSS和非writer对照；清引用与allocator措施分别加。','prediction':'若引用保留主导，完成保存后的对象可达量应随清理下降；RSS可能仍有pool地板。','falsifier':'引用已释放而对象可达量不降，或非writer同样增长，应改查其他路径。','boundary':'这是本报告提出的诊断设计，未在535B集群执行。'},['R01','R02','R13'],
 [ref(8506,5389681869,'writer tasks 0'),ref(8506,5389725084,'_commit_futures')])

add('E02','日志旁边挂起：相邻时间不能替代执行路径',
 '#8870 首帖曾把 hang 与 watch/plain executable 交替联系起来。',
 '后续查到部署配置是 WatchMode.INLINE，每步同一训练编译路径，step%10只控制host日志；capacity-limited eval也已经禁用。原假说的触发条件不在这条生产路径上。',
 '核对实际commit和配置，排除这条解释；保留另一个确实存在但触发条件不同的 symmetric window 问题。',
 '能排除“这次生产由两个训练 executable 交替触发”；单rack多次未复现不排除11rack同步问题。',
 '最终发起挂起的rank/kernel仍需独立证据。不能把任何一个相似窗口bug直接搬成这次根因。',
 [event('早期首帖','watch step 相邻被作为解释线索。',0),event('09-03 20:13 UTC','INLINE与禁用eval推翻了这条生产触发链；单rack仍不复现。',0)],
 [claim('实际部署路径排除了 watch/plain 交替解释。','supported','这是代码/配置排除，不只是“没复现”。'),claim('单rack没有挂起，所以11rack不可能挂起。','refuted','源记录同时报告跨rack故障和单rack未复现，二者条件不同。'),claim('其余所有同步问题都已经排除。','unknown','只排除已核对的机制，剩余差异仍需调查。')],
 {'plan':'先画实际部署的 executable/collective 路径，再在同版本与同拓扑记录首个停止推进的位置。','prediction':'成立的交替假说需要实际出现两个对应executable及其窗口变化。','falsifier':'只有INLINE单路径时，这条假说缺必要条件，应直接撤回。','boundary':'不把单rack作为跨rack稳定性的等价替身。'},['R01','R02','R13'],
 [ref(8870,5531534090,'WatchMode.INLINE')])

add('E03','Headers 修复：短窗口通过，后来仍然复发',
 '旧headers构建的四次挂起出现在43、0、0、0步；换到NCCL2.30.7 headers的wheel后通过200步试跑。',
 '编译headers与runtime版本偏差是真实可修的缺陷。但“这次改动以后撑得更久”还没有单独定位之后所有hang的发起机制。',
 '#9062 固定ARM64 PJRT依赖，提交部署身份并从完整handoff恢复。',
 '新wheel后来在1590 clean steps后再挂；因此200步通过不能支持“silent hang全解决”。',
 '这些attempt依赖共同历史，配置/阶段也有差别，不能由4比1直接估计独立故障率或倍数。',
 [event('09-09 部署','一致headers的候选通过200步窗口。',0),event('09-10 07:13 UTC','生产1590 clean steps后再挂；后来checkpoint步号纠正为83305。',1)],
 [claim('修好了headers版本偏差，但hang仍需继续调查。','supported','PR diff支持依赖变化；后来复发限制了稳定性结论。'),claim('200步无hang证明所有silent hang已消除。','refuted','随后生产出现同类挂起。'),claim('旧新attempt能给出准确的故障概率下降倍数。','unknown','未形成匹配条件、独立样本和预定观察协议。')],
 {'plan':'冻结构建与handoff；记录清洁推进步数、结束原因、同型dump及基础设施故障，按预定观察窗继续。','prediction':'若只修版本偏差，仍可能在更长窗口出现另一个机制的挂起。','falsifier':'相同warp状态再现会推翻“该机制已经消失”；明确NVLink fault需另列。','boundary':'本文没有估计hazard rate，也不把200或5000步当通用阈值。'},['R01','R02','R12','R13'],
 [pull(9062,'b040736f2c4a'),ref(8506,5614639876,'1590 clean steps')])

add('E04','PDL：dump支持机制，干净区间支持缓解',
 'grouped backward GEMM中，MMA/epilogue warps已退休，load warp仍等barrier；第二份dump出现第一个tile就分歧的cluster。',
 '作者把PDL提前读到旧cu_seqlens放在其他解释之前：load/scheduler等前驱，MMA/epilogue首次解码没有相同wait，因而可能先判无工作退出。dump直接观测到warp状态；“读到哪份边界”仍需要确认测量。',
 '#9183在两个Python launcher定义里传 use_pdl=False（覆盖相关GEMM路径）；部署还同时换PJRT wheel。',
 '启动后194 paired steps，训练CE均差−3.8e−5；后续2,052步无hang/retry/watchdog，超过旧run常见1–2k区间。',
 '这段干净区间与机制相容，不构成对所有未来hang的证明；吞吐+2.6%属于PDL-off+wheel整包。代码注释写得较确定，不替代dump评论提出的确认测量。',
 [event('09-15 21:31 UTC','第二份dump：第一个tile分歧加强了旧边界读取解释。',0),event('09-16 00:33 UTC','194对齐点数值比较；整包接受。',2),event('09-16 09:38 UTC','2,052步干净区间，作者仍保留同型hang会推翻的条件。',3)],
 [claim('PDL-off是有机制支持且已有生产缓解证据的措施。','supported','把dump机制线索和生产区间分开，可同时保留两者。'),claim('关闭PDL单独带来2.6%吞吐收益。','unknown','同次部署变更了PJRT，缺组件消融。'),claim('2,052步证明任何未来hang都不会发生。','unknown','无故障区间不排除稀有或其他机制的故障。')],
 {'plan':'下一次同型hang保存各warp实际读取的cu_seqlens与前驱完成状态；另在同wheel比较PDL开/关或补wait。','prediction':'旧边界机制预测退休warps使用的边界与load/scheduler的更新后边界不同。','falsifier':'同型停滞时所有warp读取同一新边界，或PDL关闭仍出现相同早退特征，都要求改写解释。','boundary':'单独性能归因还需要固定wheel的消融，不能由microbenchmark代填生产。'},['R02','R03','R12','R13'],
 [ref(8870,5688372126,'cu_seqlens'),pull(9183,'use_pdl=_QUACK_USE_PDL'),ref(8506,5690222518,'194 paired steps'),ref(8506,5695342852,'2,052 steps')])

add('E05','同一夜两次保存卡住：一个有配额证据，一个未确认',
 '区域使用量104.56 TiB超过100 TiB，服务端405暂停写入；清理后恢复接受写入，但01:01 UTC的146582保存又停住。',
 '前一次有服务端配额拒写与rank0无timeout manifest上传阻塞的对应证据；后一次448/704进程完成，256个分布在132hosts的进程无报错停止写入，起因未确认。',
 '清理旧临时件与重复durable copies约49TB，保留108778/121638/146139等durable handoff；重试从146355恢复。',
 '后续02:43和03:44保存正常。146355–146732区间被训练两次，约两小时损失；81716已删，58014无durable副本。',
 '清理有效不能把清理后的256进程停滞也归给“已超过quota”。background save让训练还能继续，不代表checkpoint已经可恢复。',
 [event('09-23 23:43:47 UTC','146585 staged save：配额拒写，rank0 manifest阻塞。',0),event('09-24 00:39 UTC','清理后服务恢复接受写入。',0),event('01:01 UTC','146582：448完成，256停住；cause unconfirmed。',0),event('02:43 / 03:44 UTC','后续两次保存正常；保留未解释的第二次事故。',0)],
 [claim('配额事故与清理后的保存停滞应分开记。','supported','同一评论明确给了恢复写入时点与随后未确认事件。'),claim('清理完成证明两次保存停滞的根因都是quota。','unknown','第二次发生在重新接受写入之后，原文也没有确认起因。'),claim('训练还在推进，所以异步checkpoint必然完整提交。','refuted','源记录明确训练继续但commit未完成，下次保存会等待。')],
 {'plan':'为每次save分别记录stage/upload/commit/restore四个时点，附quota状态、最后成功object、worker停写时点和timeout。','prediction':'quota拒写事件应有同时的服务端拒写证据；无拒写的停滞需要另一条因果链。','falsifier':'清理后无quota拒写而同型停写持续，推翻“清理已解释全部”的主张。','boundary':'不执行删除策略；实际保留与容量预算由生产owner核对。'},['R01','R02','R13','R14'],
 [ref(8506,5817400840,'448 of 704 processes')])

add('E06','提速整包：CE窗口、step、iteration各自回答什么',
 '146139部署native SM100 FA4、去ragged tail masks和每100step协调GC；200同批次步数中MFU24.10%→26.75%、tokens/s2.83M→3.14M。',
 '更快的kernel、少一次全buffer处理与较少host停顿都可能参与收益；没有逐组件试验时，整包效果不能拆成FA4的因果份额。',
 '同handoff、相同step区间比较CE、routing、梯度、峰值内存与速度；接受记录仍列出eval/save/resume未覆盖。',
 'CE均差+3.6e−4、max abs约8.8e−4；median step16.29s→14.67s，iteration18.62s→15.53s。训练step提速与包含host开销的iteration提速分母不同。',
 '200步CE接近不等于长期质量或生成正确率相同；随后某次eval或save成功也只能补对应路径。',
 [event('09-23 22:41 UTC','整包200步接受：训练CE/路由/吞吐有记录。',0),event('验收当时','147000 eval、自己保存后恢复和首个permanent save仍未覆盖。',0)],
 [claim('整包改善了这一窗口的吞吐，训练CE差值已记录。','supported','有明确区间、指标和组合变更，正文可复算CE。'),claim('FA4单独造成全部吞吐提升。','unknown','FA4、tail-mask和GC同次变化。'),claim('这次CE比较确认长期模型质量和所有保存路径通过。','unknown','对应评估和恢复尚未覆盖，CE对象也不同于长期能力。')],
 {'plan':'同handoff与wheel先对整包做稳定验收；若要归组件，固定其他条件逐项加FA4/mask/GC并检查相互作用。冻结固定评估、临时/永久保存和自己保存后恢复清单。','prediction':'GC可能主要改善iteration尾延迟，kernel可能主要改变device step；分别记录分位数和同批次CE。','falsifier':'组件效应随另一组件存在而改变时，不能报告可直接相加的独立收益。','boundary':'消融方案未执行；step与iteration不能混用分母或都称“训练速度”。'},['R02','R03','R05','R12','R13','R14'],
 [ref(8506,5804146010,'Not yet exercised')])

add('E07','去mask的条件：unused tail不能被任何消费者读到',
 '#9333删ragged QuACK forward、output-grad、input-grad的三处tail处理；local/FSDP路径保留mask。',
 'grouped GEMM按cu segment边界读，ragged return只传active rows；unused tail可保持未定义值。安全性来自消费者范围，不能只靠“算子看起来没用到”。',
 'PR把unused receiver/input/dy tail填NaN；检查active输出、输入梯度active部分和专家权重梯度，覆盖tail_rows 0与127。',
 '公开diff提供针对泄露机制的测试设计和代码修改；本机没有SM100，未独立运行GPU测试。',
 '测试保留的是active语义，不证明整个padding buffer已归零；以后新增full-buffer消费者需要重审。',
 [event('09-23 PR #9333','segment契约与三处tail移除同时记录。',0),event('测试修改','unused tail=NaN，active值/梯度对reference；local/FSDP不外推。',0)],
 [claim('只有active消费者的路径可按该契约去tail mask。','supported','源diff明确边界和仍保留mask的路径。'),claim('同一改动可直接推广到所有full-buffer combine。','refuted','full-buffer路径会读取tail，PR明确保留mask。'),claim('本报告已在SM100上验证这些GPU测试通过。','refuted','本机只检查公开diff，未执行SM100训练或测试。')],
 {'plan':'为每个输出和梯度列出全部消费者的读取范围；NaN poison覆盖tail，比较active输出和全部参数梯度，并单测local/FSDP。','prediction':'任何越界读取都可能把NaN带进active结果或权重梯度。','falsifier':'active输出/梯度污染，或新增full-buffer消费者，即使吞吐更快也应恢复mask。','boundary':'本报告提炼测试原则，不声称复制了源作者的GPU执行结果。'},['R01','R02','R13','R14'],
 [pull(9333,'constant_values=jnp.nan')])

add('E08','退出时的CUDA错误：先核对它发生在什么时候',
 '09-27 gang先停止推进，900秒watchdog随后退出；CUDA peer-memory errors首次出现在watchdog teardown。',
 'teardown会破坏进程间协作，错误可能是退出的后果。晚于停止推进的日志无法单独确立最初GPU故障。',
 'watchdog终止并重试，从166006恢复；把initiating stall列为未知。',
 '有恢复进度与错误时序证据，没有发起stall的rank/kernel/fabric原因。',
 '不能因为没有先行GPU警报就证明GPU绝对无故障；也不能用后出的CUDA错误倒推根因。',
 [event('09-27 08:20:35 UTC','gang停止推进。',0),event('08:36:25–28 UTC','watchdog结束；peer-memory errors在teardown出现。',0),event('14:19 UTC','从166006恢复继续；初始原因仍未知。',0)],
 [claim('恢复进度成立，起因仍未确认。','supported','原文明确区分重试成功与initiating stall。'),claim('teardown的CUDA报错单独证明GPU导致最初stall。','unknown','证据晚于结果，缺独立先行故障链。'),claim('未见先行GPU警报，已证明GPU不可能有问题。','unknown','没有观察到不能排除所有未记录的故障。')],
 {'plan':'统一UTC时钟，记录last progressing step、first abnormal rank、设备/网络健康与watchdog/kill时间，优先保留退出前证据。','prediction':'若GPU故障发起stall，应能找到独立的先行设备事件或重复定位证据。','falsifier':'只有退出后的peer错误，则该证据不足以支持GPU起因。','boundary':'不把恢复成功或未见警报变成根因判决。'},['R01','R02','R13'],
 [ref(8506,5856665151,'watchdog teardown')])

def main():
    source_hashes={}
    for e in episodes:
        for s in e['refs']:
            path=R/s['file']; obj=json.loads(path.read_text())
            source_hashes[s['file']]=hashlib.sha256(path.read_bytes()).hexdigest()
            if 'comment_id' in s:
                c=next(c for c in obj if c['id']==s['comment_id'])
                assert s['needle'] in c['body'],(e['id'],s['needle'])
                s.update(url=c['html_url'],created_utc=c['created_at'],updated_utc=c['updated_at'])
            else:
                assert any(s['needle'] in f.get('patch','') for f in obj),(e['id'],s['needle'])
                metadata_file=PREFIX+'pull_%d.json'%s['pull_number']
                metadata_path=R/metadata_file
                p=json.loads(metadata_path.read_text())
                source_hashes[metadata_file]=hashlib.sha256(metadata_path.read_bytes()).hexdigest()
                s.update(url=p['html_url']+'/files',metadata_file=metadata_file,head_sha=p['head']['sha'],merge_sha=p['merge_commit_sha'],merged_utc=p['merged_at'])
        assert all(0<=v['ref_index']<len(e['refs']) for v in e['events'])
    result={'schema':'marin-engineering-evidence/1','author_method_version':'0.1',
        'scope':'Authored evidence atlas and explicit claim lookup, not automatic language grading, reader outcomes or GPU validation',
        'github_snapshot_directory':PREFIX,'production_curves_snapshot':'2026-10-04 retained; not refreshed',
        'rubric_framework_version':'1.1','episodes':episodes,'source_sha256':source_hashes,
        'actual_cluster_test':None,'external_reader_study':None}
    result['atlas_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    (R/'analysis/engineering_data.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('Engineering atlas: 8 episodes, 24 authored claims; no new cluster results.')

if __name__=='__main__': main()
