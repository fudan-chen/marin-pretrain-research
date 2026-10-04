# -*- coding: utf-8 -*-
"""Build the authored review framework and reader data from frozen research results."""
import json,pathlib,csv,statistics
R=pathlib.Path(__file__).resolve().parents[1];A=R/'analysis'
def save(name,obj):(A/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def rule(i,title,area,question,pass_,partial,fail,why,links):
    return {'id':'R%02d'%i,'title':title,'area':area,'question':question,'anchors':{'pass':pass_,'partial':partial,'fail':fail},'why':why,'evidence_links':[{'label':label,'href':href} for label,href in links]}
rules=[
rule(1,'来源和版本能回查','来源','这个数字来自哪个版本、时点和原文件？','原文件、版本/时间、计算入口可以互相对应。','只有原链接或截图，版本/计算入口缺一项。','不同快照混用，或原值无法对应。','线上run会变；没有版本就无法分清新事实与旧结论。',[('来源说明','PROVENANCE_ZH.md'),('源归档完整性清单','sources/archive_manifest.json')]),
rule(2,'结论范围写清','范围','这是哪个模型、训练阶段、任务和实验？','模型规模、恢复阶段、参照和适用范围明确。','信息可追溯，但正文省略关键范围。','把代理实验写成535B已验证结果。','同一个权重在不同历史和模型上不是同一干预。',[('代理与放大边界','CONCLUSIONS_ZH.md'),('swarm审计','analysis/swarm_audit.json')]),
rule(3,'指标含义和评估路径一致','评估','Loss、BPB、正确率与drop口径是否分清？','指标分母、方向、dropless/有drop路径和能力边界明确。','指标名称有了，分母或路由路径不明。','BPB当准确率，或训练CE与固定验证混作同一种证据。','度量对象变了，曲线就不能按原含义解读。',[('三种Loss误读','REPORT_ZH.md'),('配对窗口','analysis/summary.json')]),
rule(4,'包含有竞争力的简单参照','实验','候选优于旧方案，还是也优于简单方案？','旧方案与库存比例等简单基线均纳入相同范围对照。','只有旧方案对照，或简单基线条件不同。','把所有旧→新差值都归给复杂搜索。','92.05%的观察macro差值已由比例基线取得，参照影响结论。',[('强基线对照','analysis/strong_baseline_comparison.csv'),('第四轮判断','CONCLUSIONS_ZH.md')]),
rule(5,'预算从实际区间计算','实验','相同预算指token、更新、时间中的哪一种？','恢复起止、tokens/update和预算口径明确；比较使用匹配口径。','名义预算明确，但真实恢复边界或执行量未确认。','把13.125T/3.75T与2726/810更新口径直接混用。','一步差与模拟预算差是不同的问题。',[('三种顺序预算','analysis/order_budget_audit.csv'),('恢复窗口核对','analysis/findings_audit.json')]),
rule(6,'平均指标可拆回组成项','评估','总平均的收益由哪些子集贡献？','组成项、聚合方式和退步项全部列出，能重建平均。','只有平均与部分有利子集。','把整体汇总再当一个子集，或把重叠任务当独立投票。','16项macro里代码/学术收益和其他文本退步同时存在。',[('16项贡献','analysis/macro_contributions.csv'),('54项任务原值','analysis/strong_baseline_paired.csv')]),
rule(7,'选择后有独立确认','确认','这份结果是用来选候选，还是用来确认候选？','候选先冻结，再做未参与选择的重复与确认评估，报告波动。','有多seed，但参与选择，或共享历史的依赖未处理。','把597个单seed候选当成597次独立复验。','搜索极值容易含有噪声；确认预算不能被搜索预算替代。',[('候选seed分布','analysis/findings_audit.json'),('配对重复边界','CONCLUSIONS_ZH.md')]),
rule(8,'每次加量都写供体','数据','给B加量时，A失去了多少？','接收域、供体、转移量、其余变化和替代策略明确。','加量或删域已知，空出预算分给谁不明。','把删域再分配结果当成该域自身的纯贡献。','混合权重总和为1；收益包含替代关系。',[('80个删除干预','analysis/domain_ablation_audit.csv'),('第三轮实验方案','PRACTICAL_ZH.md')]),
rule(9,'库存、历史与重复分开','数据','增加比例后，独立覆盖和累计曝光各变多少？','有效库存、已有曝光、后续计划和上限有定义，并检查容量可行性。','只有公开名义库存与计划，真实去重/历史量未确认。','把库存占比当抽样权重，或把名义epochs当实际文档重复证明。','相同百分比在小库存上可能意味着高重复。',[('200桶库存与计划','analysis/cell_weights.csv'),('预算工具例子','analysis/planner_example.json')]),
rule(10,'顺序比较匹配累计量和游标','顺序','两种顺序真的只差先后吗？','逐桶累计量、完整块计数、恢复cursor和token流重放可对应。','连续补偿或完整块计算已做，token ID/内部shuffle未重放。','互换不同长度阶段却宣称曝光相同。','时序和总量一起变，就无法归因到先后。',[('完整块游标','analysis/cursor_audit.csv'),('顺序预算审计','analysis/order_budget_audit.csv')]),
rule(11,'跨规模结论有对应对照','确认','放大后是否仍保留强基线和关键风险对照？','更大模型同范围确认方向，并说明阶段/系统差异。','有放大结果，但缺简单基线或条件不同。','直接把d512收益幅度写成535B收益。','代码收益的方向在ladder上并非一致。',[('跨规模终点','analysis/proxy_scale_effects.csv'),('公开配比研究','https://github.com/marin-community/marin/issues/9126')]),
rule(12,'目标和能力保底预先冻结','决策','选配比究竟要优化什么，允许牺牲什么？','主目标、独立确认集、能力退步界限和停止条件在看结果前规定。','目标大致明确，约束或冻结时间缺失。','看完结果再换目标，把最终哈希当作全面最优证明。','观察前沿随任务组合改变；未知selector不能靠猜测补齐。',[('观察前沿','analysis/observed_pareto_frontiers.csv'),('197c反例','analysis/candidate_197c_counterexample.csv')]),
rule(13,'根因证据和缓解效果分开','工程','恢复运行能证明哪个假设，排除了什么？','根因证据、排除实验、缓解动作和残余未知分别归档。','有恢复和可行解释，但关键排除或底层证据不足。','恢复成功就写成根因已确认。','silent hang、存储故障和数值异常需要不同证据。',[('58条运行记录','OPERATIONS_ZH.md'),('15类工程机制','REPORT_ZH.md')]),
rule(14,'执行状态不只看配置路径','工程','实际加载的代码、checkpoint和数据状态是什么？','代码/环境、checkpoint内容、随机与数据状态可核对，回退点明确。','记录配置与fallback相同，实际恢复内容尚未确认。','相同路径就断言状态逐字节一致。','配置相同能减少混杂，但不能保证每次启动执行相同。',[('原始配置对照','analysis/strong_baseline_config_audit.csv'),('恢复与游标边界','DEEP_DIVE_ZH.md')]),
rule(15,'能力主张有实际任务验证','评估','目标文本概率的变化是否转化为任务表现？','生成正确率/执行测试与BPB并列，题目、解码和污染检查有记录。','只有BPB线索，能力主张保持为候选假设。','HumanEval BPB写成pass@1，GSM8K BPB写成正确率。','预测指定答案与自己生成正确答案不是同一个测试。',[('任务BPB边界','CONCLUSIONS_ZH.md'),('逐seed任务表','analysis/strong_baseline_paired.csv')]),
rule(16,'图表能被读者正确复述','表达','读者能说清横纵轴、参照、样本数和不能推出什么吗？','图包含单位、参照、范围、样本数、原值入口和解释；窄屏可读。','数值正确但解释/分母/可访问替代缺失。','截断差结果、混用参照，或把图中的相关性画成因果证明。','图要帮助判断，不应替代证据限定。',[('理解文档','LEARNING_GUIDE_ZH.md'),('图与交互复核','analysis/browser_validation_v5.json')]),
rule(17,'结论可离线复算','复现','分享包能从固定材料重建关键结果吗？','归档、计算脚本、数字检查、链接和离线阅读验证齐全。','报告可读，但部分计算需未知在线状态。','只有截图/结论，关键原值或公式丢失。','复算不等于重训；两者的验证范围都应写清。',[('离线构建','Makefile'),('计算验证','analysis/validation.json')]),
rule(18,'未解决问题有下一步和退出条件','改进','下一次工作会消除哪个不确定性？','缺口、下一步、所需资源、验收/推翻条件和变更记录明确。','有待办，但只写“继续深入”或缺验收。','抓取条数/美化数量当作研究收益，反例被删除。','持续改进应降低具体不确定性。',[('持续改进记录','IMPROVEMENT_LOG_ZH.md'),('决策管线','PIPELINE_ZH.md')])]
pipeline=[
{'id':'P0','title':'冻结来源与问题','question':'这次要解释哪一个变化？','required':['R01','R02','R03','R17'],'outputs':['source_manifest与版本','一个范围明确的研究问题','指标定义表'],'next':'缺来源先补来源；已有材料固定后进入比较。'},
{'id':'P1','title':'诊断与筛选候选','question':'可比的观察支持哪些候选？','required':['R04','R05','R06','R08','R16'],'outputs':['旧/比例/候选对照','退步与组成项表','候选名单和陈述边界'],'next':'缺参照先补参照；有取舍就保留多个候选。'},
{'id':'P2','title':'设计局部交换实验','question':'下一批token从谁移给谁？','required':['R09','R12','R14'],'outputs':['供体与接收域','库存/历史/容量账本','冻结目标和实验清单'],'next':'容量冲突先改方案；缺真实状态先核对恢复点。'},
{'id':'P3','title':'独立确认能力收益','question':'候选冻结以后，是否稳定且满足任务保底？','required':['R07','R12','R15'],'outputs':['未参与选择的确认结果','重复波动与退步限制','通过/重做/放弃决定'],'next':'只有BPB就保留线索；能力没确认先补任务评估。'},
{'id':'P4','title':'检验顺序与恢复','question':'累计量相同后，先后还有作用吗？','required':['R05','R09','R10','R14'],'outputs':['逐桶曝光匹配','整数混合块与cursor重放','中途及末期评估'],'next':'总量不匹配先修补偿；没做顺序实验可有理由跳过整个阶段。'},
{'id':'P5','title':'放大与生产切换','question':'更大模型确认后，实际切换状态是否可复验？','required':['R07','R11','R12','R14','R15'],'outputs':['更大代理与强基线','固定评估和回退点','切换前后状态清单'],'next':'方向不稳定返回候选；状态不可核对先处理工程问题。'},
{'id':'P6','title':'复盘与下一轮','question':'哪条判断被推翻，下一步最值得做什么？','required':['R01','R17','R18'],'outputs':['版本差异与反例','缺口优先级','下一轮可检验问题'],'next':'保留旧快照；新信息建立新版本，再回到P0。'}]
examples={'scope':'固定公开快照支持的结论范围；不是给Marin团队打分，也不是已完成的新训练。','schema':'marin-review/1','framework_version':'1.0','target':'用公开材料评估选中配比能被推荐到哪一步','judgments':{}}
partial={7:'三组continuation参与过候选研究，共同历史依赖仍在；无选择后独立确认。',9:'库存与名义计划已重算；去重后独立库存和真实历史token流未恢复。',10:'连续补偿与完整块cursor已算；桶内shuffle与token ID未重放。',11:'有ladder结果，但阶段/系统不同，未补上更大模型比例基线。',13:'公开时间线区分缓解与猜测，部分底层日志与根因证据不可访问。',14:'4组记录配置可比，fallback相同；实际恢复文件与启动环境未核对。',15:'54项任务为BPB；新535B生成正确率尚未验证。'}
missing={12:'完整selector目标和约束未知；不能替作者补造预先冻结的guardrail。'}
for r in rules:
    n=int(r['id'][1:]);status='partial' if n in partial else 'unassessed' if n in missing else 'pass'
    evidence=partial.get(n) or missing.get(n) or ('报告对应材料：'+r['evidence_links'][0]['href']+'。此通过项只覆盖其已声明的公开/本地复算范围。')
    examples['judgments'][r['id']]={'status':status,'evidence':evidence}
save('rubrics.json',{'version':'1.0','authorship':'本报告提炼的工作规则，不是Marin官方标准或经外部验证的评分量表。','status_labels':{'pass':'证据齐全','partial':'部分证据','fail':'有反证/口径冲突','unassessed':'未确认','na':'本阶段不适用'},'rules':rules,'pipeline':pipeline,'example_audit':examples})

findings=json.loads((A/'findings_audit.json').read_text());summary=json.loads((A/'summary.json').read_text());paired=list(csv.DictReader((A/'strong_baseline_paired.csv').open()))
metrics=[]
for key,label in [('eval_dropless/paloma/macro_bpb','通用文本平均'),('eval_dropless/paloma/dolma_100_programing_languages-llama3/bpb','编程语言文本'),('logprob_humaneval_10shot','HumanEval目标文本'),('logprob_gsm8k_5shot','GSM8K目标文本')]:
    rr=sorted([r for r in paired if r['metric']==key],key=lambda r:int(r['seed']))
    metrics.append({'key':key,'label':label,'means':{k:statistics.mean(float(r[k+'_bpb']) for r in rr) for k in ['old','proportional','selected']},'seeds':[{'seed':int(r['seed']),**{k:float(r[k+'_bpb']) for k in ['old','proportional','selected']}} for r in rr]})
ep=summary['paired_windows']['ep'];ke=summary['paired_windows']['kernels']
cases=[
{'id':'distribution','label':'108k：换配比后训练CE上升','nodes':[{'text':'配置权重切换','kind':'observed'},{'text':'抽到的文本分布改变','kind':'mechanism'},{'text':'训练CE约1.210→1.235','kind':'observed'}],
 'reading':'训练批次的题材和难度分布改变了。这个跳变本身无法判断同一固定任务上的预测变好还是变坏。','allowed':'配比切换附近，训练CE的观察值上升。','forbidden':'据此判断模型能力倒退，或用CE跳幅计算配比收益。','next':'比较固定验证并保留代码/数学等退步项；核对路由口径和参照条件。','source':'REPORT_ZH.md','rules':['R02','R03','R06','R08']},
{'id':'ep','label':'81k：drop减少，CE下降','nodes':[{'text':'Ragged EP与轮包切换','kind':'observed'},{'text':'容量丢弃显著减少','kind':'observed'},{'text':'同step CE均差 %.7f'%ep['train/cross_entropy_loss']['mean_new_minus_old'],'kind':'observed'}],
 'reading':'路由容量影响哪些token能经过专家。路径变了，训练目标的有效计算也可能改变。200个配对step显示CE下降，同时吞吐中位数比约%.3f。'%ep['throughput/tokens_per_second']['ratio_medians'],
 'allowed':'这组配对窗口中，drop、CE和吞吐同时变化。','forbidden':'把训练CE降幅全归给模型能力，或混用dropless评估。','next':'在固定评估路径上看效果；同时区分路由机制变化和系统吞吐收益。','source':'analysis/summary.json','rules':['R03','R05','R13']},
{'id':'kernel','label':'146k：吞吐提升，CE近似不变','nodes':[{'text':'FA4、mask、GC一起调整','kind':'observed'},{'text':'吞吐中位数比 %.3f'%ke['throughput/tokens_per_second']['ratio_medians'],'kind':'observed'},{'text':'同step CE均差 %+.7f'%ke['train/cross_entropy_loss']['mean_new_minus_old'],'kind':'observed'}],
 'reading':'这是组合部署窗口：可以核对整组调整的速度和数值表现，不能从这一组数据单独分出FA4的贡献。','allowed':'200个配对step里吞吐提高约10.82%，CE差很小。','forbidden':'声称FA4单独带来全部收益，或称能力提高10.82%。','next':'分别记录固定token与固定时间预算；需要单组件归因时做拆分对照。','source':'analysis/summary.json','rules':['R03','R05','R08','R13']},
{'id':'gate','label':'Gate接近零：早期层真的没用了吗？','nodes':[{'text':'部分gate范数异常增大','kind':'observed'},{'text':'怀疑早期层没作用','kind':'hypothesis'},{'text':'删除前10层：loss 2.093→6.584','kind':'observed'}],
 'reading':'小幅残差修改仍可能被后续层依赖。这组敲除使用dropless global macro loss，数值不是BPB。参数或门值的外观，不能直接替代功能敲除实验。','allowed':'敲除反例说明前10层仍影响该评估结果。','forbidden':'门接近零就断言整层已经无用，或把敲除loss当作实际部署结果。','next':'区分attention与FFN敲除、模型状态和评估路径，检验具体机制。','source':'REPORT_ZH.md','rules':['R02','R03','R13']},
{'id':'cursor','label':'4K换16K：配比相同，数据位置也相同吗？','nodes':[{'text':'序列/混合块设置改变','kind':'observed'},{'text':'完整块计数与阶段边界改变','kind':'mechanism'},{'text':'重算cursor可前进也可后退','kind':'computed'}],
 'reading':'step不是绝对的数据身份。权重、混合块、边界对齐和恢复计数一起决定读取位置。2.951B或108.907B是两种设置下的完整块L1游标差。','allowed':'本地复算证明这些设置下逻辑位置会不同。','forbidden':'把游标差直接叫作已经重复或遗漏的实际token数量。','next':'重放真实token ID与桶内顺序；先匹配累计量，再谈先后作用。','source':'DEEP_DIVE_ZH.md','rules':['R05','R09','R10','R14']}]
save('workbench_data.json',{'snapshot':'2026-10-04约05:38，曲线未刷新','scope':'d512续训指标为BPB；图中三点为seed0/1/2，均值点不是置信区间。','metrics':metrics,'cases':cases,'gap_fraction':findings['proportional_fraction_of_observed_old_to_selected_macro_gap']})

figure_rows=[
('training_signals','这次曲线变化发生在哪一层？','逐项看训练CE、固定评估、drop和吞吐，先核对各自分母。','跨run拼接和抽样窗口有边界；不据训练CE单独判断能力。','analysis/series.json','REPORT_ZH.md'),
('intervention_windows','系统、数据两类切换能否按同一方式比较？','81k和146k看200步同step配对；108k看分布切换。','组合部署不能给单组件归因；配比切换不是相同批次。','analysis/summary.json','REPORT_ZH.md'),
('domain_mixture','哪个语义域在什么阶段获得预算？','每个条形使用实际阶段抽样权重，同域五个Q档求和。','阶段比例不是全程累计量，也不是该域的因果价值。','analysis/domain_weights.csv','DATA_GUIDE_ZH.md'),
('quality_mixture','库存与抽样的质量档分布差多少？','库存和三阶段分别使用自己的分母；Q档内覆盖仍不同。','Q4比例不等于质量收益，未来cooldown还是计划。','analysis/quality_weights.csv','DATA_GUIDE_ZH.md'),
('mixture_eval','更大ladder的固定文本收益是否全面？','分别看规模和子集，保留代码/Wikipedia退步。','旧新设置存在混杂，不能直接当535B能力结论。','sources/mix_study_final-2026.09.15.1_final_results.csv','DATA_GUIDE_ZH.md'),
('paired_seed_tradeoffs','旧与选中在三组续训里有什么取舍？','看每个seed和不同任务，负BPB变化更好。','共享恢复历史且参与选择；BPB不是生成正确率。','analysis/paired_seed_tasks_bpb.csv','DEEP_DIVE_ZH.md'),
('cell_weight_shifts','同一语义域内部Q档怎样迁移？','200桶逐项看新主阶段减旧权重的百分点变化。','权重变动不是该桶自身贡献；还要乘阶段预算并除库存。','analysis/cell_main_shifts.csv','DEEP_DIVE_ZH.md'),
('cursor_differences','改序列与混合块后，逻辑位置偏到哪里？','比较两种设置的正负偏移与L1，整数块口径单列。','游标差不能直接称为实际重复/遗漏token，未重放token ID。','analysis/cursor_audit.csv','DEEP_DIVE_ZH.md'),
('domain_ablation_response','删除再分配对通用文本和数学是否一致？','两组用各自参照；每个删除干预只有seed0。','不是单域纯贡献，语义参照和比例参照不能混用。','analysis/domain_ablation_audit.csv','PRACTICAL_ZH.md'),
('math_q4_response','数学Q4加量是否单调改善？','横轴为恢复后名义模拟曝光，点为单seed终点。','连线不是拟合；其他桶预算同时减少，标签不等于权重倍数。','analysis/math_q4_response.csv','PRACTICAL_ZH.md'),
('macro_contribution_findings','相对强基线的净改善被哪些子集支撑？','16项绝对BPB差除以16；负贡献改善macro。','宏观分解不是训练桶因果归因；整体汇总不当第17项。','analysis/macro_contributions.csv','CONCLUSIONS_ZH.md'),
('observed_frontier_findings','为什么不同目标会选出不同候选？','上排全部597个seed0，下排局部放大；两个轴均越小越好。','观察前沿不是确认或部署推荐；197c还有其他任务退步。','analysis/observed_pareto_frontiers.csv','CONCLUSIONS_ZH.md'),
('candidate_budget_decomposition','看域净差会漏掉多少Q档变化？','四候选均相对996f；条形全长为桶累计半L1，蓝色为域净差半L1，棕色为域内相抵。','预算为13.125T/3.75T名义模拟；不指定实际供体路径，不估计Loss或桶因果贡献。','analysis/transfer_cell_differences.csv','TRANSFER_GUIDE_ZH.md'),
('order_edge_ledger','为什么时序变化只放进完整块？','三臂总窗口相同；两端灰块共同保留，前后完整部分按56/16做整数补偿。','本报告未执行草案；逻辑匹配要求共同历史、cursor、key与映射；不是Marin实际读取图。','templates/order_edge_locked.json','ORDER_GUIDE_ZH.md')]
save('figure_readings.json',[{'file':'assets/'+name+'.png','question':question,'reading':reading,'boundary':boundary,'values':values,'chapter':chapter} for name,question,reading,boundary,values,chapter in figure_rows])

doc=R/'RUBRICS_ZH.md'
if doc.exists():
    lines=['<!-- GENERATED_RULES -->','## 完整规则与判断锚点','','状态评的是所选范围的证据。缺证据不等于实验失败；有反证也不等于一个域永远无用。以下规则由本报告提炼，未经外部评分效度验证。','']
    for r in rules:
        lines+=['### '+r['id']+' · '+r['title'],'',r['question']+' '+r['why'],'','| 状态 | 判断锚点 |','|---|---|','| 证据齐全 | '+r['anchors']['pass']+' |','| 部分证据 | '+r['anchors']['partial']+' |','| 有反证/口径冲突 | '+r['anchors']['fail']+' |','', '依据：'+' · '.join('[%s](%s)'%(x['label'],x['href']) for x in r['evidence_links']),'']
    doc.write_text(doc.read_text().split('<!-- GENERATED_RULES -->')[0].rstrip()+'\n\n'+'\n'.join(lines))
print('Built 18 authored rubrics, seven pipeline stages and five interpretation cases from frozen results.')
