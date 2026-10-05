# -*- coding: utf-8 -*-
"""Author source-bound comprehension cases. No reader responses or training results."""
import json,hashlib,pathlib
R=pathlib.Path(__file__).resolve().parents[1]
def read(f):return json.loads((R/f).read_text())
def sha(f):return hashlib.sha256((R/f).read_bytes()).hexdigest()
w=read('analysis/workbench_data.json');d=read('analysis/decision_data.json');t=read('analysis/transfer_data.json');o=read('analysis/order_workbench_data.json')
he=next(x for x in w['metrics'] if x['key']=='logprob_humaneval_10shot');he_pct=100*(he['means']['selected']/he['means']['proportional']-1)
c=next(x for x in d['candidates'] if '197c' in x['run']);ref=d['baselines']['selected']['values'];med_pct=100*(c['values']['medqa']/ref['medqa']-1);pm_pct=100*(c['values']['paloma']/ref['paloma']-1)
maxcell=max(read('analysis/cell_weights.json'),key=lambda x:x['planned_epochs'])
cases=[]
def add(id,title,prompt,allowed,reason,nextstep,critical,links,rules,figure,scope_anchor,mechanism_anchor,next_anchor):
 cases.append({'id':id,'title':title,'prompt':prompt,'reference_answer':allowed+' '+reason+' '+nextstep,'allowed_claim':allowed,'explanation':reason,'next_evidence':nextstep,'critical_errors':[{'id':id+'-E1','text':critical}],'links':[{'href':f,'label':label} for f,label in links],'rules':rules,'figure':figure,'anchors':{'scope':{'2':scope_anchor,'1':'对象或结论方向基本正确，但比较范围有遗漏。','0':'把题目中的范围改成了更强或不同的主张。'},'mechanism':{'2':mechanism_anchor,'1':'提到相关风险，但未说明如何影响本题比较。','0':'只背结论、混淆机制，或没有说明限制原因。'},'next':{'2':next_anchor,'1':'提出相关方向，但所需对照或证据不具体。','0':'没有下一步，或只写“更多数据/继续研究”。'}}})
add('Q01','CE跳升：究竟是谁变了',
 '108k配比切换附近，训练CE约1.210→1.235。有人据此要求立即回退，理由是“新配比让模型变差”。现有材料支持这句判断吗？写出原因，以及最少要补的可比证据。',
 '这只能描述该窗口训练CE上升，不能单独判定新配比损害能力。',
 '抽样分布变了，较难文本占比提高也会抬高训练CE；路由丢弃和恢复状态仍需核对。这个解释也不证明新配比更好。',
 '先比较共同状态和预算下、相同固定验证集与路由路径的结果，再决定是否回退。',
 '把训练CE跳升直接写成模型能力退步，或反过来写成更难数据必然更好。',
 [('analysis/summary.json','配对与窗口原值'),('REPORT_ZH.md','三个曲线案例')],['R02','R03','R14'],'assets/training_signals.png',
 '明确108k训练CE窗口和能力结论的区别，不反向保证收益。','解释批次分布改变平均CE，并保留路由/状态混杂。','指出同固定验证、路由、预算/状态的具体比较。')
add('Q02','BPB改善：能否换算正确率',
 f'三组d512续训的HumanEval目标文本BPB均值，相对库存比例基线约变化{he_pct:.2f}%。能否把它写成pass@1提升{abs(he_pct):.2f}%？这个候选现在可以支持什么决定？',
 '不能换算pass@1；目前支持代码目标文本预测改善的候选线索。',
 'BPB按byte度量给定目标文本的预测代价；生成、解码与执行错误不由这个均值直接给出。三seed续训也不是535B验证。',
 '冻结候选，补相同题目/解码设置的实际生成与执行评估，并记录波动和风险，再决定放大。',
 '把BPB百分比当成pass@1、正确率或535B已验证能力收益。',
 [('analysis/strong_baseline_comparison.csv','均值与参照'),('analysis/workbench_data.json','逐seedBPB')],['R02','R03','R07','R15'],'assets/paired_seed_tradeoffs.png',
 '正确说出d512、库存比例参照、目标文本BPB和候选层级。','解释给定文本概率与自主生成/执行的区别。','给出冻结候选后的实际生成/执行对照，不仅重测BPB。')
add('Q03','92.05%：是差值比例还是原因',
 f'旧、比例、选中方案的三seed Paloma宏平均BPB均值分别为{w["metrics"][0]["means"]["old"]:.6f}、{w["metrics"][0]["means"]["proportional"]:.6f}、{w["metrics"][0]["means"]["selected"]:.6f}。比例基线取得约{100*w["gap_fraction"]:.2f}%的旧→选中观察差。应怎样解读，不能怎样解读？',
 '这是三个续训终点均值的差值比例，说明简单参照已经取得大部分观察macro差。',
 '它没有隔离库存比例本身的因果贡献；不同配方同时改动多个桶。参照变强后，选中方案的额外macro改善很小，但代码等子任务仍需分别看。',
 '保留比例参照与退步项；若要归因某类数据，用明确供体、相同预算的局部干预与独立重复。',
 '把92.05%写成库存比例的因果贡献率、统计置信度或535B收益。',
 [('analysis/strong_baseline_comparison.csv','三参照均值'),('CONCLUSIONS_ZH.md','差值比例解释')],['R03','R04','R06','R08'],'assets/macro_contribution_findings.png',
 '说出三均值差值比例与d512观察范围，不改名为因果贡献。','解释强参照和多桶共同变化，允许不同子任务有不同方向。','保留强参照，并为归因提出具体供体和预算控制。')
add('Q04','597个候选：为什么不是597次复验',
 f'候选工作台有{len(d["candidates"])}个seed0终点，能够事后调整约束、主目标与排名。把第一名选出来后，是否已经得到稳定收益证据？应怎样进入确认阶段？',
 '这是一批不同配方的搜索终点，支持提出候选，不等于冻结候选的独立重复。',
 '搜索与事后调阈值会偏向乐观结果；当前材料没有恢复完整历史selector，也不能把新工作台的规则写成当时的选择方法。',
 '在看新结果前冻结候选、参照、主目标和风险阈值，用未参与选择的重复与确认评估检验；没有符合条件者应保留空集。',
 '把597个不同候选当同一候选的独立重复，或把事后筛选当独立确认。',
 [('analysis/decision_data.json','597候选与来源'),('analysis/selector_provenance_audit.json','历史selector检索范围'),('templates/selection_contract.json','未来确认契约')],['R01','R07','R12'],'assets/observed_frontier_findings.png',
 '区分搜索候选与同一冻结候选重复，保留selector未知。','说明选择偏差与事后阈值为何不能确认稳定收益。','具体冻结四项并使用新重复/确认评估，空集不静默放宽。')
add('Q05','平均改善：能否抵消关键退步',
 f'197c相对996f的seed0 Paloma macro BPB变化约{pm_pct:+.3f}%，MedQA目标文本BPB约{med_pct:+.2f}%。如果主目标是macro，能否直接说“全面更好”并使医学保底通过？',
 '可以把197c保留为有取舍的macro候选，不能说全面更好或医学保底已通过。',
 '平均与关键任务衡量不同目标；BPB越低越好，MedQA方向为退步。单seed事后比较也没有证明医学问答准确率下降同样百分比。',
 '按预先定义的风险阈值独立确认；若医学是硬约束，当前冲突要单列，不能由macro改善抵消。',
 '用macro改善抵消硬保底冲突，或把MedQA BPB变化换成医学答题准确率变化。',
 [('analysis/candidate_197c_counterexample.csv','197c/996f七指标'),('analysis/decision_data.json','同seed原值')],['R03','R06','R07','R12','R15'],'assets/observed_frontier_findings.png',
 '同时报告参照、seed、macro改善与MedQA BPB退步。','解释平均无法替代关键约束，BPB不等于准确率。','指出预定阈值、独立确认与不可抵消的关键冲突。')
add('Q06','名义曝光：能否当作文档重复次数',
 f'公开库存与全程计划中，{maxcell["cell"]}的计划量/库存约为{maxcell["planned_epochs"]:.2f}轮。是否说明其每篇文档实际重复了这么多次，或实际训练已经读完这份量？增量前先查什么？',
 '这是全程名义曝光比，不是每篇文档的真实重复次数，也不是已完成的读取量。',
 '库存、抽样权重和实际曝光有不同分母；去重覆盖、随机排列、有限库存取模和历史会影响具体重复。4K生产快照仍未完成18T目标。',
 '加量前核对有效去重库存、已有历史和剩余预算；若要报告实际重复，需实际token/文档ID与读取记录。',
 '把全程计划epochs当成逐文档实测重复次数或已完成生产量。',
 [('analysis/cell_weights.csv','计划库存与曝光'),('analysis/summary.json','生产快照和预算')],['R02','R05','R09','R14'],'assets/domain_mixture.png',
 '明确计划而非实测、桶而非每篇文档、全程而非当前完成量。','解释库存/权重/曝光不同分母，以及实际覆盖条件。','提出有效库存、历史、剩余预算或真实ID核对。')
add('Q07','数学加量：有没有同时改变时序',
 f'197c对996f的名义剩余计划：数学累计增量约{t["math_timing"]["cumulative_delta_tokens"]/1e9:.3f}B tokens；扣除同累计增量的常数方案后，前/后段余量约±{t["math_timing"]["timing_residual_tokens"][0]/1e9:.3f}B。能否用这两个候选的端点差证明“数学前置有效”？',
 '不能；原比较同时包含累计量与时序变化，且供体和其他桶也在变。',
 '两个阶段变化可拆成同累计量的常数加量与和为零的时序余量。账本分解识别了待检验因素，没有给各因素分配因果收益。',
 '先确认累计配方和供体，再在相同累计量、预算与状态下构造整数匹配的前置/后置比较。',
 '把候选端点差或约8.070B时序余量本身当成数学前置的因果收益。',
 [('analysis/transfer_data.json','数学累计与时序分解'),('TRANSFER_GUIDE_ZH.md','供体与因素草案')],['R05','R08','R10'],'assets/candidate_budget_decomposition.png',
 '保留候选整体比较、累计量和时序的范围区别。','解释常数加量/零和余量及供体共同变化，不归因端点。','提出确认累计配方后、共同状态下的匹配整数顺序对照。')
add('Q08','整数余数：配置只改两域就够了吗',
 'V7连续权重只声明数学与供体域变化，原加载器取整却改变了c28q4/c30q4。修正后24次整块计数匹配目标。现在能否说已经完成“仅两域的实际数据干预”和能力确认？',
 '修正支持声明域内的整块计数设计；不能由24次计数推出实际数据流或能力已经确认。',
 '全局余数分配可能改变未声明桶，权重支持与整数支持不一致。修正解决这个执行算术问题，但没有读取真实store、恢复内容或跑生成评估。',
 '保留原泄漏反例和逐桶整数残差；再核对实际库存、cursor、shuffle与token ID，按冻结方案独立训练确认。',
 '将加载器计数通过写成实际内容干预、训练完成或能力确认。',
 [('analysis/transfer_plan_quantization.csv','原取整泄漏'),('analysis/transfer_plan_integer_controlled.csv','修正逐桶计数'),('analysis/transfer_loader_probe_python312.json','原方法运行范围')],['R05','R08','R14','R15'],'assets/cell_weight_shifts.png',
 '只肯定24次整块计数范围，真实内容与能力保持未知。','解释全局取整余数导致未声明桶变化的机制。','保留反例/残差并提出真实映射和独立训练核对。')
add('Q09','整块零差：为何整个窗口还不同',
 '按公开窗口推导，前后为56/16个完整块，另有两端部分块。把数学每块改成前段+2k、后段−7k，完整块累计差为零。如果两端也随配方改变，是否足够声称纯顺序实验？',
 '不够；完整块匹配没有覆盖两端实际读取的槽位。',
 '部分块某桶的读数依赖块内排列，不能按权重期望代填。k10关闭shuffle的合法诊断仍有140序列L1差；这只是反例，不是Marin历史读取差。',
 '保留原总预算，让两端数组与累计游标共同不变，中间整数补偿；再核对实际恢复、key与映射。',
 '把整块残差零当完整窗口相同，或把140序列诊断当历史实际差。',
 [('analysis/order_integer_audit.json','窗口与整数计数'),('analysis/order_loader_probe_python312.json','部分块反例'),('ORDER_GUIDE_ZH.md','共同边缘设计')],['R05','R10','R14'],'assets/order_edge_ledger.png',
 '区分完整块、整个窗口与诊断，不把反例当历史值。','解释部分块排列和槽位读取为何破坏简单比例补偿。','保持预算/共同边缘和游标，补真实key/映射，而非裁窗口。')
add('Q10','条件索引证明：离实际token还缺什么',
 '共同边缘草案已核对480组阶段计数和20份两端数组/游标关系。有人准备把标题写成“已证明Marin各组看到了相同token”。这个标题应怎样改，哪项证据会推翻当前条件？',
 '应写成“共同状态条件下的累计逻辑索引匹配”；没有验证Marin实际token内容。',
 '计数、边缘数组和游标匹配提供条件逻辑关系；实际checkpoint、next index、混合/内部shuffle key、底层映射与packing仍未核对。任一共同状态不同都可能使推论失效。',
 '固定代码/环境和恢复内容，取得真实cursor、keys和store manifest，重放实际token ID；另做生成评估才能讨论能力。',
 '把条件索引证明写成实际Marin token流复现，或以它使能力/生产阶段通过。',
 [('analysis/order_loader_probe_python312.json','六段方法与条件证明'),('templates/order_edge_locked.json','空的执行状态'),('ORDER_GUIDE_ZH.md','四层相同')],['R01','R02','R10','R14','R15'],'assets/order_edge_ledger.png',
 '明确本地草案、条件索引与实际内容，改掉过强标题。','指出至少一种具体共同状态改变会破坏条件，并区分能力。','提出真实checkpoint/cursor/key/store与token重放，而非重跑计数。')
paths=sorted({l['href'] for c in cases for l in c['links']}|{'analysis/workbench_data.json','analysis/transfer_data.json','analysis/order_workbench_data.json','analysis/decision_data.json','analysis/rubrics.json'})
bank={'schema':'marin-reader-casebank/1','protocol_version':'0.1','rubric_framework_version':read('analysis/rubrics.json')['version'],'scope':'Authored source-bound comprehension practice; no external reader or scoring-validity study','snapshot':'2026-10-04 numerical archive; 2026-10-05 engineering and scale-comparison wording audit; questions and anchors unchanged','dimensions':[{'id':'scope','label':'结论范围'},{'id':'mechanism','label':'限制原因'},{'id':'next','label':'最小追加证据'}],'cases':cases,'source_sha256':{f:sha(f) for f in paths},'external_readers':None,'external_scoring_validity':None}
bank['bank_sha256']=hashlib.sha256(json.dumps(bank,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
(R/'analysis/assessment_data.json').write_text(json.dumps(bank,ensure_ascii=False,indent=2)+'\n')
blank={'schema':'marin-reader-assessment/1','protocol_version':'0.1','bank_sha256':bank['bank_sha256'],'record_kind':'local_unverified_reader_record','participant_label':'','cases':{c['id']:{'draft':'','reference_first_opened_at':None,'revisions':[]} for c in cases}}
(R/'templates/reader_assessment_blank.json').write_text(json.dumps(blank,ensure_ascii=False,indent=2)+'\n')
bad_answers={
 'Q01':'训练CE升高已经证明模型能力退步，可以据此直接回退新配比。',
 'Q02':'HumanEval BPB改善7.50%，所以pass@1也提升7.50%，可以写成代码正确率收益。',
 'Q03':'库存比例对最终能力提升的因果贡献率是92.05%，535B可以沿用。',
 'Q04':'597个终点就是同一候选597次独立复验，事后选第一名已确认稳定收益。',
 'Q05':'macro改善足以抵消MedQA退步，所以全面更好，医学硬保底也通过了。',
 'Q06':'计划6.74轮就是每篇文档实际重复6.74次，生产已经完成这些读取。',
 'Q07':'两候选端点的差证明数学前置有效，8.070B就是前置的因果收益。',
 'Q08':'24次计数通过说明真实数据干预和模型能力都已确认。',
 'Q09':'中间整块差为零就足够；140序列是Marin历史实际读取差。',
 'Q10':'480次计数和20个索引证明已复现Marin真实token流，生产切换可以直接通过。'}
replays=[]
for c in cases:
 for kind,answer,scores,critical in [('bounded',c['reference_answer'],{'scope':2,'mechanism':2,'next':2},[]),('generic','这个结果可以保留为线索，仍需要更多研究。',{'scope':1,'mechanism':0,'next':0},[]),('overreach',bad_answers[c['id']],{'scope':0,'mechanism':0,'next':0},[c['critical_errors'][0]['id']])]:
  replays.append({'case_id':c['id'],'kind':kind,'authored_answer':answer,'author_assigned_ratings':scores,'author_assigned_critical_errors':critical,'provenance':'Written calibration example; not a reader observation or automatic language judgment'})
(R/'analysis/assessment_authored_replays.json').write_text(json.dumps({'bank_sha256':bank['bank_sha256'],'scope':'30 authored calibration examples; exercises evaluation-state logic only; no external human agreement measured','examples':replays},ensure_ascii=False,indent=2)+'\n')
print('Reader cases:',len(cases),'source-bound; authored calibration examples:',len(replays))
