"""Cross-reference 15 reported cases and 6 proposed entry routes; not a result grader."""
import pathlib,json,re,hashlib
R=pathlib.Path(__file__).resolve().parents[1];spec=json.loads((R/'config/engineering_case_map.json').read_text());report=(R/'REPORT_ZH.md').read_text();rubrics=json.loads((R/'analysis/rubrics.json').read_text());known={r['id'] for r in rubrics['rules']}
sections={m.group(1):{'title':m.group(2),'body':m.group(3)} for m in re.finditer(r'^### (3\.\d+) ([^\n]+)\n(.*?)(?=^### |^## |\Z)',report,re.M|re.S)}
assert len(sections)==15 and len(spec['cases'])==15 and {c['report_section'] for c in spec['cases']}==set(sections)
assert len({c['id'] for c in spec['cases']})==15
inputs=['config/engineering_case_map.json','REPORT_ZH.md','analysis/rubrics.json']
for c in spec['cases']:
 section=sections[c['report_section']];c['report_title']=section['title'];c['original_evidence_urls']=list(dict.fromkeys(re.findall(r'\]\((https?://[^)]+)\)',section['body'])))
 assert c['original_evidence_urls'] and c['actual_new_cluster_execution'] is None and set(c['rubric_ids'])<=known
 for p in c['source_paths']+c['reading_paths']:assert (R/p).is_file(),p
 inputs.extend(c['source_paths'])
for route in spec['entry_routes']:
 assert set(route['rubric_ids'])<=known
 for p in route['first_read']:assert (R/p).is_file(),p
spec['source_sha256']={p:hashlib.sha256((R/p).read_bytes()).hexdigest() for p in sorted(set(inputs))}
spec['verification_scope']='Coverage/title/link/source-hash and rubric-ID consistency. Case judgments remain curated, not automatically proved.'
(R/'analysis/engineering_case_map.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n')
lines=['# 预训练排障总图：从症状选择证据，而不是从版本号读报告','',
 '本章把主报告15类工程问题接到后续源码深读，并给六种常见起点安排检查顺序。它是本报告作者的交接方法，不是Marin官方验收标准；对应的18条rubrics仍为1.1，没有另造一套总分。','',
 '第一张表回答“现在先查什么”，随后逐案分开作者已经做过的动作、本地到底核对了什么、以及下一项仍未执行的检查。链接与覆盖检查可以发现漏项和错指，但不能自动证明根因、GPU正确性或人的理解。','',
 '## 1. 六个入口，先留下一个能改变判断的产物','',
 '|你看到的症状|先读哪里|先留下什么|本轮能怎样决定|','|---|---|---|---|']
link=lambda p:'[%s](%s)'%(pathlib.Path(p).stem.replace('_ZH',''),p)
for rt in spec['entry_routes']:lines.append('|%s|%s|%s|%s|'%(rt['symptom'],' → '.join(link(p) for p in rt['first_read']),rt['first_artifact'],rt['hold_decision']))
lines+=['','这些是建议执行顺序，尚未在新的集群事故中验证。若首个产物缺少版本、时间或输入身份，先补身份；如果已有反证否定触发条件，撤回对应假说；不要用继续跑更多步替代能区分解释的测量。','',
 '## 2. 十五类问题的动作、源码与证据边界','',
 '阅读本地探针时，先看输出中的scope、substitutes与actual_*字段。CPU原辅助函数、模拟sharding、固定源码推断、公开指标复算和作者生产观察是不同证据，不能互相补齐。当前所有案例的“本轮新增集群执行”均为空。','']
for c in spec['cases']:
 lines+=['### %s · %s'%(c['report_section'],c['report_title']),'',
 '**症状与机制。** '+c['symptom']+'。'+c['mechanism'],'',
 '**作者做过什么，结果到哪一步。** '+c['reported_action_result'],'',
 '**本地核对的范围。** '+c['local_evidence_scope'],'',
 '**下一项检查（未执行）。** '+c['proposed_next_check'],'',
 '**接受或继续调查的边界。** '+c['decision_boundary'],'',
 '源码/patch入口：'+(' · '.join(link(p) for p in c['source_paths']) if c['source_paths'] else '该案没有定位为某段训练代码的根因，不强行补一个源码归因。'),'阅读：'+' · '.join(link(p) for p in c['reading_paths']),
 '原始证据：'+' · '.join('[来源%d](%s)'%(i+1,u) for i,u in enumerate(c['original_evidence_urls'])),
 '使用既有规则：'+', '.join(c['rubric_ids'])+'，见[18条判断规则](RUBRICS_ZH.md)。','']
lines+=['## 3. 配比与工程为什么必须共用一次变更记录','',
 '数据比例改变训练分布，router或drop改变前向，optimizer或pending state改变更新/评估状态，kernel和checkpoint改变执行与可恢复进度。只记录“这次改了什么配置”无法分开这些效应。','',
 '因此沿用[五个冻结对象](CHANGE_REVIEW_ZH.md)：输入与数据身份、模型/路由、optimizer与状态、评估口径、执行资源。一次试验先声明哪些对象改变，再选择对照；声明单变量，但其他对象也变了，就收窄为整包观察。','',
 '例如108k换配比之后、第一次固定评估之前，还发生PDL/PJRT接续；该生产窗口不能单独估计配比收益。15域均值、16域macro、token加权micro还可能给出不同方向；字段对应和预定保底项应先冻结。见[真实轨迹](MIX_TRAJECTORY_ZH.md)与[指标分母](EVAL_METRICS_ZH.md)。','',
 '研究可以继续提出候选和检查，但当前没有独立GPU重复、真实token流重放、完整checkpoint恢复或外部读者评分。缺这些证据不证明候选失败，只限制我们能推荐到哪一步。','',
 '## 4. 怎样维护这张总图','',
 '用[人工映射配置](config/engineering_case_map.json)维护15案和六入口，运行[构建脚本](scripts/build_engineering_map.py)生成[可机读索引](analysis/engineering_case_map.json)。脚本核对主报告15个标题完整覆盖、原文URL、规则ID和文件SHA，不自动给主张评分。','',
 '新增材料时先找它改变哪个机制、结果或退出条件，再修改该案；不同事故或不同attempt分别留时序。新增源代码不能自动将“历史执行版本未知”升级为已确认。旧release和纠正记录继续保留，网页构建日期不改变历史训练快照。','']
(R/'ENGINEERING_MAP_ZH.md').write_text('\n'.join(lines)+'\n')
print('Mapped 15 report cases and 6 proposed entry routes; no new cluster execution')
