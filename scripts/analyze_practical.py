# -*- coding: utf-8 -*-
"""Audit ablation controls and the order experiment's two token-budget conventions."""
import json,pathlib,csv,re,math,statistics,copy,hashlib
import pyarrow.parquet as pq
R=pathlib.Path(__file__).resolve().parents[1];A=R/'analysis';D=R/'sources/deepening_2026_10_04'
rows=pq.read_table(D/'hf_observations.parquet').to_pylist();swarm=pq.read_table(D/'hf_swarm.parquet').to_pylist()[0]
stock={r['cell']:r['available_tokens'] for r in pq.read_table(D/'hf_buckets.parquet').to_pylist()[0]['cells']}
names=json.loads((A/'domain_weights_zh.json').read_text());names={d['id']:d['name_zh'] for d in names}
core=[r for r in rows if r['group']=='core_design'];old=next(r for r in core if 'baseline-seed0-' in r['run_name'])
semantic=next(r for r in core if 'broad-technical-capped-' in r['run_name'])
prop=[r for r in rows if r['group']=='proportional_baseline'];prop0=next(r for r in prop if '-seed0-' in r['run_name'])
PM='eval_dropless/paloma/macro_bpb';GM='logprob_gsm8k_5shot';HM='logprob_humaneval_10shot'
def metric(r,k):return (r['training_eval_metrics'] if k==PM else r['grouped_bpb'])[k]
P=R/'sources/practical_2026_10_04'
def config(r):
    run=json.loads((P/('config_'+r['run_name']+'.json')).read_text())['data']['project']['run']
    return json.loads(run['config']),json.loads(run['summaryMetrics'])
def controlled_config(c):
    c=copy.deepcopy(c);c.pop('_wandb',None);c['data']['value'].pop('train_weights')
    t=c['trainer']['value']['trainer'];t.pop('id')
    t['load_checkpoint_path']=[t['load_checkpoint_path'][-1]]
    for k in ['base_path','temporary_base_path']:t['checkpointer'].pop(k,None)
    for k in ['name','id','replicate_path']:t['tracker'].pop(k,None)
    return c
def paths(a,b,p=''):
    if isinstance(a,dict) and isinstance(b,dict):return [x for k in sorted(set(a)|set(b)) for x in paths(a.get(k),b.get(k),p+'.'+k)]
    return [] if a==b else [p]
def write_csv(name,rs):
    with (A/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]),lineterminator='\n');w.writeheader();w.writerows(rs)
def error(r,b,removed):
    rest=[k for k in stock if k not in removed];diff=[]
    for p in ['phase0_weights','phase1_weights']:
        z=math.fsum(b[p][k] for k in rest)
        diff.extend(abs(r[p][k]-b[p][k]/z) for k in rest)
    return max(diff)
ab=[]
for r in rows:
    if r['group'] not in ['semantic_domain_ablation','proportional_domain_ablation']:continue
    c=int(re.search(r'zero-c(\d\d)',r['run_name'])[1]);removed={k for k in stock if k.startswith('c%02dq'%c)}
    b=semantic if r['group'].startswith('semantic') else prop0
    assert all(r[p][k]==0 for p in ['phase0_weights','phase1_weights'] for k in removed)
    ab.append({'group':r['group'],'domain':c,'domain_zh':names[c],'run_name':r['run_name'],'control_run':b['run_name'],
               'control_removed_phase0_pct':100*math.fsum(b['phase0_weights'][k] for k in removed),'control_removed_phase1_pct':100*math.fsum(b['phase1_weights'][k] for k in removed),
               'max_remaining_renormalization_error':error(r,b,removed),
               'paloma_bpb':metric(r,PM),'paloma_change_pct':100*(metric(r,PM)/metric(b,PM)-1),
               'gsm8k_bpb':metric(r,GM),'gsm8k_change_pct':100*(metric(r,GM)/metric(b,GM)-1),
               'humaneval_bpb':metric(r,HM),'humaneval_change_pct':100*(metric(r,HM)/metric(b,HM)-1)})
    C,M=config(r);BC,BM=config(b)
    ab[-1]['controlled_config_differences']=';'.join(paths(controlled_config(C),controlled_config(BC)))
    ab[-1]['paloma_registry_matches_wandb_endpoint']=abs(M[PM]-metric(r,PM))<1e-12
ab.sort(key=lambda x:(x['group'],x['domain']));write_csv('domain_ablation_audit.csv',ab)
# Export every endpoint without pretending the 54 overlapping metrics form an independent macro.
long=[]
for r in rows:
    if r['group'] not in ['semantic_domain_ablation','proportional_domain_ablation','semantic_quality_ablation','proportional_quality_isolation','core_design','proportional_baseline']:continue
    for k,v in r['grouped_bpb'].items():long.append({'run_name':r['run_name'],'group':r['group'],'metric':k,'bpb':v})
write_csv('ablation_task_endpoints.csv',long)
qualities=[]
for r in rows:
    if r['group'] not in ['semantic_quality_ablation','proportional_quality_isolation']:continue
    b=semantic if r['group'].startswith('semantic') else prop0
    removed={k for k in stock if r['phase0_weights'][k]==r['phase1_weights'][k]==0 and b['phase0_weights'][k]+b['phase1_weights'][k]>0}
    qualities.append({'group':r['group'],'run_name':r['run_name'],'paloma_bpb':metric(r,PM),'paloma_change_pct':100*(metric(r,PM)/metric(b,PM)-1),
                      'gsm8k_bpb':metric(r,GM),'gsm8k_change_pct':100*(metric(r,GM)/metric(b,GM)-1),'humaneval_bpb':metric(r,HM),
                      'removed_cells':len(removed),'max_remaining_renormalization_error':error(r,b,removed)})
write_csv('quality_ablation_audit.csv',qualities)
def get(fragment,seed=0):return next(r for r in core if '-'+fragment+'-seed%d-'%seed in r['run_name'])
tpb=1024*4096
conventions={'registry_nominal_simulated':swarm['phase_budgets'],'weight_compensation_2726_810_physical':[2726*tpb,810*tpb],'observed_2727_810_physical':[2727*tpb,810*tpb]}
orders=[];cell_differences=[]
def compare_order(a,b,label):
    ca,ma=config(a);cb,mb=config(b)
    control_diffs=paths(controlled_config(ca),controlled_config(cb))
    for convention,budgets in conventions.items():
        deltas={k:sum(T*(b['phase%d_weights'%i][k]-a['phase%d_weights'%i][k]) for i,T in enumerate(budgets)) for k in stock}
        orders.append({'pair':label,'run_a':a['run_name'],'run_b':b['run_name'],'budget_convention':convention,'phase0_tokens':budgets[0],'phase1_tokens':budgets[1],
                       'l1_exposure_difference_tokens':sum(abs(x) for x in deltas.values()),'net_token_difference':sum(deltas.values()),
                       'paloma_a':metric(a,PM),'paloma_b':metric(b,PM),'paloma_b_minus_a':metric(b,PM)-metric(a,PM),
                       'gsm8k_a':metric(a,GM),'gsm8k_b':metric(b,GM),'gsm8k_b_minus_a':metric(b,GM)-metric(a,GM),
                       'controlled_config_differences':';'.join(control_diffs)})
        for k,v in deltas.items():cell_differences.append({'pair':label,'convention':convention,'cell':k,'b_minus_a_exposure_tokens':v})
for family in ['r-stem','u-math']:
    for level in ['high','low']:compare_order(get(family+'-pre-'+level),get(family+'-cool-'+level),family+' '+level+' pre_vs_cool seed0')
    for seed in [0,1]:compare_order(get(family+'-high-pre-swap',seed),get(family+'-low-pre-swap',seed),family+' swap_high_vs_low seed%d'%seed)
write_csv('order_budget_audit.csv',orders);write_csv('order_cell_exposures.csv',cell_differences)
q4=[]
for r in [old]+[r for r in core if '-math-q4-' in r['run_name']]:
    q4.append({'run_name':r['run_name'],'phase0_c39q4_pct':100*r['phase0_weights']['c39q4'],'phase1_c39q4_pct':100*r['phase1_weights']['c39q4'],
               'nominal_remaining_c39q4_epochs':sum(T*r['phase%d_weights'%i]['c39q4'] for i,T in enumerate(swarm['phase_budgets']))/stock['c39q4'],
               'paloma_bpb':metric(r,PM),'gsm8k_bpb':metric(r,GM),'humaneval_bpb':metric(r,HM)})
write_csv('math_q4_response.csv',q4)
summary={'domain_ablations':len(ab),'semantic_control':semantic['run_name'],'proportional_control_seed0':prop0['run_name'],
         'semantic_control_name_suffix_differs_from_ablation':True,'semantic_max_weight_error':max(r['max_remaining_renormalization_error'] for r in ab if r['group'].startswith('semantic')),
         'proportional_max_weight_error':max(r['max_remaining_renormalization_error'] for r in ab if r['group'].startswith('proportional')),
         'proportional_baseline_paloma_mean':statistics.mean(metric(r,PM) for r in prop),'proportional_baseline_paloma_sd':statistics.stdev(metric(r,PM) for r in prop),
         'proportional_baseline_gsm8k_mean':statistics.mean(metric(r,GM) for r in prop),'proportional_baseline_gsm8k_sd':statistics.stdev(metric(r,GM) for r in prop),
         'quality_ablations':qualities,'math_q4_response':q4,'order_pair_count':len(orders)//3,'order_conventions':conventions,
         'domain_control_config_matches':sum(not r['controlled_config_differences'] for r in ab),'domain_endpoints_match_wandb':sum(r['paloma_registry_matches_wandb_endpoint'] for r in ab),
         'config_fields_removed':['data.train_weights','_wandb runtime metadata','trainer.id','run-specific checkpoint base paths','own-run checkpoint fallback paths (shared last fallback retained)','tracker name/id/replicate_path'],
         'order_claim_scope':'All compared launch configs archived. Two order histories begin at logged step393; boundary3120, final3929 imply2727/810 updates. Weight compensation instead fits2726/810 exactly. No original token-ID or partial block replay.',
         'order_comparisons':orders}
# Freeze all 125 configs into a compact cross-check table; source bytes remain unchanged.
config_audit=[]
for r in rows:
    if not (P/('config_'+r['run_name']+'.json')).exists():continue
    c,m=config(r);d=c['data']['value'];t=c['trainer']['value']['trainer']
    phase_errors=[max(abs(weights[k]-r['phase%d_weights'%i][k]) for k in stock) for i,(_,weights) in enumerate(d['train_weights'][1:])]
    assert len(phase_errors)==2
    config_audit.append({'run_name':r['run_name'],'group':r['group'],'run_seed':t['seed'],'max_seq_len':c['model']['value']['max_seq_len'],'train_batch_size':t['train_batch_size'],'num_train_steps':t['num_train_steps'],
                         'phase_steps':';'.join(str(s) for s,_ in d['train_weights']),'last_restore_path':t['load_checkpoint_path'][-1],
                         'max_phase_weight_error':max(phase_errors),'paloma_registry_minus_wandb':metric(r,PM)-m[PM],
                         'controlled_config_sha256':hashlib.sha256(json.dumps(controlled_config(c),sort_keys=True,separators=(',',':')).encode()).hexdigest()})
write_csv('experiment_config_audit.csv',config_audit);summary['archived_configs']=len(config_audit)
windows={}
for short in ['r-stem-pre-high','r-stem-cool-high']:
    h=json.loads((P/(short+'_step_windows.json')).read_text())['data']['project']['run']['sampledHistory']
    hh=[[json.loads(x) if isinstance(x,str) else x for x in points] for points in h]
    windows[short]={'initial_logged_steps':[x['_step'] for x in hh[0]],'boundary_logged_steps':[x['_step'] for x in hh[2]],
                    'first_tokens':hh[1][0]['throughput/total_tokens'],'first_ce':hh[0][0]['train/cross_entropy_loss']}
    assert windows[short]['initial_logged_steps']==list(range(393,431))
    assert windows[short]['boundary_logged_steps']==list(range(3100,3141))
summary['observed_history_windows']=windows
(A/'practical_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
lines=['<!-- GENERATED_PRACTICAL_TABLES -->','## 附录：80个删域实验的对照与终点评估','','每行是一个seed0实验。语义配比组的参照是broad-technical-capped；比例配比组的参照是proportional seed0。Δ以各自参照为分母，负值表示BPB降低。80项与对应参照的记录配置在去掉配比和运行路径后相同；代码SHA、实际加载的checkpoint及token ID未逐一验证。这是删掉领域并把预算分给其余领域的联合干预。','','| 配比组 | 删除领域 | 前段删掉的比例 | 后段删掉的比例 | Paloma Δ | GSM8K Δ | HumanEval Δ |','|---|---|---:|---:|---:|---:|---:|']
for r in ab:lines.append('| %s | c%02d %s | %.3f%% | %.3f%% | %+.3f%% | %+.3f%% | %+.3f%% |'%('语义' if r['group'].startswith('semantic') else '比例',r['domain'],r['domain_zh'],r['control_removed_phase0_pct'],r['control_removed_phase1_pct'],r['paloma_change_pct'],r['gsm8k_change_pct'],r['humaneval_change_pct']))
lines+=['','[80行权重与参照审计CSV](analysis/domain_ablation_audit.csv) · [全部54项任务终点CSV](analysis/ablation_task_endpoints.csv) · [逐桶顺序曝光差CSV](analysis/order_cell_exposures.csv)','']
chapter=R/'PRACTICAL_ZH.md'
if chapter.exists():chapter.write_text(chapter.read_text().split('<!-- GENERATED_PRACTICAL_TABLES -->')[0].rstrip()+'\n\n'+'\n'.join(lines))
print(json.dumps({k:v for k,v in summary.items() if k not in ['quality_ablations','order_comparisons']},ensure_ascii=False,indent=2))
