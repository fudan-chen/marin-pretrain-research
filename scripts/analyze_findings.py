# -*- coding: utf-8 -*-
"""Compare a strong stock baseline, decompose macro BPB, and audit observed frontiers."""
import pathlib,json,re,csv,statistics,math,copy,hashlib
import numpy as np
import pyarrow.parquet as pq
R=pathlib.Path(__file__).resolve().parents[1];S=R/'sources';A=R/'analysis';D=S/'findings_2026_10_04'
rows=pq.read_table(S/'deepening_2026_10_04/hf_observations.parquet').to_pylist();selected=json.loads((S/'mix_study_swarm-2026.09.14.1_selected_runs.json').read_text())
def seed(r):return int(re.search(r'seed(\d+)',r['run_name'])[1])
groups={k:sorted(v,key=seed) for k,v in selected.items()};prop=sorted([r for r in rows if r['group']=='proportional_baseline'],key=seed);groups['proportional']=prop[:3]
PM='eval_dropless/paloma/macro_bpb';PC='eval_dropless/paloma/dolma_100_programing_languages-llama3/bpb';GM='logprob_gsm8k_5shot';HE='logprob_humaneval_10shot'
def read(p):return json.loads(p.read_text())
def save(name,obj):(A/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def table(name,rs):
    with (A/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]),lineterminator='\n');w.writeheader();w.writerows(rs)
def value(r,k):return r['training_eval_metrics' if k.startswith('eval') else 'grouped_bpb'][k]
keys=sorted(k for k in groups['new'][0]['training_eval_metrics'] if k.startswith('eval_dropless/') and (k.endswith('/bpb') or k.endswith('/macro_bpb')))+sorted(groups['new'][0]['grouped_bpb'])
comp=[];paired=[]
for k in keys:
    vals={label:[value(r,k) for r in rs] for label,rs in groups.items()};means={label:statistics.mean(v) for label,v in vals.items()}
    ds=[a-b for a,b in zip(vals['new'],vals['proportional'])];mean=statistics.mean(ds);halfwidth=4.302652729749462*statistics.stdev(ds)/math.sqrt(3)
    comp.append({'metric':k,'old_mean':means['old'],'proportional_mean':means['proportional'],'selected_mean':means['new'],
                 'proportional_vs_old_pct':100*(means['proportional']/means['old']-1),'selected_vs_old_pct':100*(means['new']/means['old']-1),'selected_vs_proportional_pct':100*(means['new']/means['proportional']-1),
                 'mean_selected_minus_proportional':mean,'conditional_t95_low':mean-halfwidth,'conditional_t95_high':mean+halfwidth,
                 'all_three_selected_better':all(d<0 for d in ds),'all_three_selected_worse':all(d>0 for d in ds)})
    for i in range(3):paired.append({'metric':k,'seed':i,'old_bpb':vals['old'][i],'proportional_bpb':vals['proportional'][i],'selected_bpb':vals['new'][i],'selected_minus_proportional':ds[i],'selected_vs_proportional_pct':100*(vals['new'][i]/vals['proportional'][i]-1)})
table('strong_baseline_comparison.csv',comp);table('strong_baseline_paired.csv',paired);lookup={r['metric']:r for r in comp}
# The overall /paloma/bpb is a token-weighted aggregate, not one of the 16 macro constituents.
subsets=sorted(k for k in groups['new'][0]['training_eval_metrics'] if k.startswith('eval_dropless/paloma/') and k.endswith('-llama3/bpb'))
assert len(subsets)==16
macro_errors=[statistics.mean(value(r,k) for k in subsets)-value(r,PM) for rs in groups.values() for r in rs]
assert max(map(abs,macro_errors))<2e-7
contributions=[{'metric':k,'proportional_mean':lookup[k]['proportional_mean'],'selected_mean':lookup[k]['selected_mean'],
                'selected_minus_proportional_bpb':lookup[k]['mean_selected_minus_proportional'],
                'contribution_to_macro_bpb':lookup[k]['mean_selected_minus_proportional']/16,'relative_change_pct':lookup[k]['selected_vs_proportional_pct']} for k in subsets]
table('macro_contributions.csv',contributions)
macro=lookup[PM];gap_fraction=(macro['old_mean']-macro['proportional_mean'])/(macro['old_mean']-macro['selected_mean'])
noncode={label:statistics.mean(value(r,k) for r in rs for k in subsets if k!=PC) for label,rs in groups.items()}
# An observed frontier only: all seed0 endpoints, no seed averaging or hidden selection model.
candidates=[r for r in rows if r['group']=='mixprior_candidate' and seed(r)==0]
matrix=np.array([[value(r,k) for k in [PM,HE,GM]] for r in candidates]);si=next(i for i,r in enumerate(candidates) if r['run_name']==groups['new'][0]['run_name'])
pareto={};candidate_rows=[]
for label,dims in [('paloma_humaneval',[0,1]),('paloma_gsm8k',[0,2]),('paloma_humaneval_gsm8k',[0,1,2])]:
    v=matrix[:,dims];dominates=(v[:,None,:]<=v[None,:,:]).all(axis=2)&(v[:,None,:]<v[None,:,:]).any(axis=2)
    counts=dominates.sum(axis=0);front=counts==0;pareto[label]={'dimensions':dims,'frontier_count':int(front.sum()),'selected_seed0_dominator_count':int(counts[si]),'selected_seed0_dominators':[candidates[i]['run_name'] for i in np.where(dominates[:,si])[0]]}
    for i,r in enumerate(candidates):candidate_rows.append({'objective':label,'run_name':r['run_name'],'paloma_bpb':matrix[i,0],'humaneval_bpb':matrix[i,1],'gsm8k_bpb':matrix[i,2],'dominator_count':int(counts[i]),'on_observed_frontier':bool(front[i])})
table('observed_pareto_frontiers.csv',candidate_rows)
dominator=next(r for r in candidates if '197c9f5ceff6b9ee-' in r['run_name']);counterexample=[]
for k in keys:counterexample.append({'metric':k,'selected_seed0':value(groups['new'][0],k),'alternative_seed0':value(dominator,k),'alternative_vs_selected_pct':100*(value(dominator,k)/value(groups['new'][0],k)-1)})
table('candidate_197c_counterexample.csv',counterexample)
# Freeze source configuration checks independently of the old swarm analysis.
def raw_config(r):
    folder=D if 'mixprior' in r['run_name'] else S/'practical_2026_10_04'
    run=read(folder/('config_'+r['run_name']+'.json'))['data']['project']['run']
    return json.loads(run['config']),json.loads(run['summaryMetrics'])
def canonical(c):
    c=copy.deepcopy(c);c.pop('_wandb',None);c['data']['value'].pop('train_weights');t=c['trainer']['value']['trainer'];t.pop('id');t['load_checkpoint_path']=[t['load_checkpoint_path'][-1]]
    for k in ['base_path','temporary_base_path']:t['checkpointer'].pop(k,None)
    for k in ['name','id','replicate_path']:t['tracker'].pop(k,None)
    return c
def differences(a,b,p=''):
    if isinstance(a,dict) and isinstance(b,dict):return [x for k in sorted(set(a)|set(b)) for x in differences(a.get(k),b.get(k),p+'.'+k)]
    return [] if a==b else [p]
configs=[]
for a,b in list(zip(groups['new'],groups['proportional']))+[(dominator,groups['new'][0])]:
    ca,ma=raw_config(a);cb,mb=raw_config(b);ds=differences(canonical(ca),canonical(cb))
    configs.append({'run_a':a['run_name'],'run_b':b['run_name'],'differences_after_declared_path_and_weight_exclusions':ds,
                    'restore_fallback_a':ca['trainer']['value']['trainer']['load_checkpoint_path'][-1],'restore_fallback_b':cb['trainer']['value']['trainer']['load_checkpoint_path'][-1],
                    'phase_boundaries_a':[x[0] for x in ca['data']['value']['train_weights']],'phase_boundaries_b':[x[0] for x in cb['data']['value']['train_weights']],
                    'endpoint_error_a':ma[PM]-value(a,PM),'endpoint_error_b':mb[PM]-value(b,PM)})
windows={}
for r in groups['new']+groups['proportional']:
    h=read(D/('window_'+r['run_name']+'.json'))['data']['project']['run']['sampledHistory'];hh=[[json.loads(x) if isinstance(x,str) else x for x in p] for p in h]
    steps=[x['_step'] for x in hh[0]];assert steps==list(range(393,431))
    windows[r['run_name']]={'first_logged_step':steps[0],'last_logged_step':steps[-1],'points':len(steps),'first_total_tokens':hh[1][0]['throughput/total_tokens']}
table('strong_baseline_config_audit.csv',[{'run_a':r['run_a'],'run_b':r['run_b'],'critical_config_differences':';'.join(r['differences_after_declared_path_and_weight_exclusions']),'endpoint_error_a':r['endpoint_error_a'],'endpoint_error_b':r['endpoint_error_b']} for r in configs])
# Recalculate the larger ladder's endpoint effect from frozen CSV, not from plotted labels.
scale=[{'size':'d512_proxy','metric':k,'change_pct':lookup[k]['selected_vs_old_pct']} for k in [PM,PC]]
for r in csv.DictReader((S/'mix_study_final-2026.09.15.1_point_speedups.csv').open()):
    if r['metric']=='bpb' and r['subset'] in ['macro','dolma_100_programing_languages']:
        scale.append({'size':r['size'],'metric':PM if r['subset']=='macro' else PC,'change_pct':100*(float(r['variant_loss'])/float(r['baseline_loss'])-1)})
table('proxy_scale_effects.csv',scale)
# Inventory-proportional sampling is almost uniform exposure, including its tiny-bucket floor.
stock={b['cell']:b['available_tokens'] for b in pq.read_table(S/'deepening_2026_10_04/hf_buckets.parquet').to_pylist()[0]['cells']};budgets=pq.read_table(S/'deepening_2026_10_04/hf_swarm.parquet').to_pylist()[0]['phase_budgets']
weights=[]
for label,rs in groups.items():
    r=rs[0]
    for q in range(5):weights.append({'baseline':label,'kind':'quality','name':'Q%d'%q,'nominal_remaining_pct':100*sum(sum(T*r['phase%d_weights'%i][k] for i,T in enumerate(budgets))/sum(budgets) for k in stock if k.endswith('q%d'%q))})
    weights.append({'baseline':label,'kind':'domain','name':'c39_math','nominal_remaining_pct':100*sum(sum(T*r['phase%d_weights'%i][k] for i,T in enumerate(budgets))/sum(budgets) for k in stock if k.startswith('c39q'))})
table('strong_baseline_exposure_shares.csv',weights)
summary={'three_seed_metrics':{k:lookup[k] for k in [PM,PC,GM,HE]},'all_metric_comparisons':comp,
         'proportional_fraction_of_observed_old_to_selected_macro_gap':gap_fraction,'macro_reconstruction_max_error':max(map(abs,macro_errors)),
         'sixteen_macro_constituents':subsets,'macro_contributions':contributions,'noncode_15_subset_means':noncode,
         'noncode_selected_vs_proportional_pct':100*(noncode['new']/noncode['proportional']-1),
         'candidate_seed_counts':{str(i):sum(r['group']=='mixprior_candidate' and seed(r)==i for r in rows) for i in [0,1,2]},
         'observed_seed0_candidates':len(candidates),'pareto':pareto,'counterexample_run':dominator['run_name'],
         'counterexample_task_directions':{'better':sum(value(dominator,k)<value(groups['new'][0],k) for k in groups['new'][0]['grouped_bpb']),'worse':sum(value(dominator,k)>value(groups['new'][0],k) for k in groups['new'][0]['grouped_bpb'])},
         'configs':configs,'resume_windows':windows,'scale_effects':scale,'exposure_shares':weights,
         'assumptions':['Three paired continuation seed labels, not three independently initialized checkpoints.','Conditional t intervals assume iid normal paired differences, exclude selection effects, and are not guarantees.','Gap fraction is arithmetic on endpoints, not a causal mediation estimate.','Pareto frontiers are observed single-seed endpoint frontiers, not deployment recommendations.','Proxy and larger ladders differ in stage timing and some recorded settings.']}
save('findings_audit.json',summary)
chapter=R/'CONCLUSIONS_ZH.md'
if chapter.exists():
    lines=['<!-- GENERATED_FINDINGS_TABLES -->','## 附录：54项任务相对比例基线的逐seed结果','','全部为BPB。每个Δ按同seed的比例基线为分母，均值列使用三seed均值之比。语言平均与组成项不能当成独立的54次投票。','','| 指标 | 比例基线均值 | 选中配比均值 | 均值Δ | seed0 | seed1 | seed2 |','|---|---:|---:|---:|---:|---:|---:|']
    bymetric={k:[r for r in paired if r['metric']==k] for k in groups['new'][0]['grouped_bpb']}
    for k in sorted(bymetric):
        r=lookup[k];v=bymetric[k];lines.append('| %s | %.6f | %.6f | %+.3f%% | %+.3f%% | %+.3f%% | %+.3f%% |'%(k,r['proportional_mean'],r['selected_mean'],r['selected_vs_proportional_pct'],*(x['selected_vs_proportional_pct'] for x in v)))
    lines+=['','[全部固定文本与任务均值CSV](analysis/strong_baseline_comparison.csv) · [逐seed原值CSV](analysis/strong_baseline_paired.csv)','']
    chapter.write_text(chapter.read_text().split('<!-- GENERATED_FINDINGS_TABLES -->')[0].rstrip()+'\n\n'+'\n'.join(lines))
print(json.dumps({k:v for k,v in summary.items() if k not in ['all_metric_comparisons','macro_contributions','sixteen_macro_constituents','resume_windows','configs']},ensure_ascii=False,indent=2))
