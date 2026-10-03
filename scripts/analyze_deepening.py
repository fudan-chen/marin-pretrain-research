# -*- coding: utf-8 -*-
"""Audit the pinned swarm, paired seeds and deterministic complete-block cursors offline."""
import json,pathlib,csv,hashlib,statistics,math,re,collections,itertools,sys
import pyarrow.parquet as pq
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];S=ROOT/'sources';D=S/'deepening_2026_10_04';A=ROOT/'analysis'
def read(p):return json.loads(p.read_text())
def save(name,obj):(A/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2))
def table(name,rows):
    with (A/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
rows=pq.read_table(D/'hf_observations.parquet').to_pylist();swarm=pq.read_table(D/'hf_swarm.parquet').to_pylist()[0]
buckets=pq.read_table(D/'hf_buckets.parquet').to_pylist()[0];content=pq.read_table(D/'hf_content.parquet').to_pylist()[0]
selected=read(S/'mix_study_swarm-2026.09.14.1_selected_runs.json');byname={r['run_name']:r for r in rows}
assert len(rows)==len(byname)==934
assert all(byname[r['run_name']]==r for group in selected.values() for r in group)
assert hashlib.sha256((D/'hf_observations.parquet').read_bytes()).hexdigest()==swarm['observations']['sha256']
weight_hashes=[digest([r['phase0_weights'],r['phase1_weights']]) for r in rows]
counts=collections.Counter(weight_hashes)
inventory={'rows':len(rows),'unique_run_names':len(byname),'exact_distinct_weight_pairs':len(counts),'group_counts':dict(collections.Counter(r['group'] for r in rows)),
           'weight_pair_multiplicities':dict(collections.Counter(counts.values())),
           'task_metrics_per_row':dict(collections.Counter(len(r['grouped_bpb']) for r in rows)),
           'selected_rows_match_v1_exactly':True,'observations_manifest_sha_matches':True,
           'physical_training_tokens':swarm['physical_training_tokens'],'simulated_training_tokens':swarm['simulated_training_tokens'],
           'slice_ratio':swarm['physical_training_tokens']/swarm['simulated_training_tokens'],
           'phase_budgets':swarm['phase_budgets'],'active_parameters':swarm['model_active_parameters'],'total_parameters':swarm['model_total_parameters'],
           'bucket_semantics':buckets['bucket_semantics'],'content_matrix_shape':[len(content['matrix']),len(content['matrix'][0])],
           'content_row_sum_bounds':[float(np.min(np.sum(content['matrix'],axis=1))),float(np.max(np.sum(content['matrix'],axis=1)))],
           'content_provenance':content['provenance']}
table('swarm_inventory.csv',[{'run_name':r['run_name'],'group':r['group'],'weight_pair_sha256':h,'paloma_macro_bpb':r['training_eval_metrics'].get('eval_dropless/paloma/macro_bpb'),'gsm8k_bpb':r['grouped_bpb']['logprob_gsm8k_5shot'],'humaneval_bpb':r['grouped_bpb']['logprob_humaneval_10shot']} for r,h in zip(rows,weight_hashes)])
# Pair by the name's seed label. This is not a claim about independent checkpoints or identical PRNG streams.
paired={label:{int(re.search(r'seed(\d+)',r['run_name'])[1]):r for r in group} for label,group in selected.items()}
assert all(set(p)=={0,1,2} for p in paired.values())
TCRIT=4.302652729749462  # 97.5th percentile of t with 2 df; conditional on iid normal paired deltas.
def compare(source,key):
    old=[paired['old'][i][source][key] for i in range(3)];new=[paired['new'][i][source][key] for i in range(3)]
    delta=[b-a for a,b in zip(old,new)];relative=[100*(b/a-1) for a,b in zip(old,new)]
    avg=statistics.mean(delta);se=statistics.stdev(delta)/math.sqrt(3)
    sign_values=[abs(statistics.mean(s*d for s,d in zip(signs,delta))) for signs in itertools.product([-1,1],repeat=3)]
    return {'metric':key,'old_mean':statistics.mean(old),'new_mean':statistics.mean(new),'change_pct':100*(statistics.mean(new)/statistics.mean(old)-1),
            **{'seed%d_delta'%i:delta[i] for i in range(3)},**{'seed%d_change_pct'%i:relative[i] for i in range(3)},
            'mean_paired_delta':avg,'t95_low':avg-TCRIT*se,'t95_high':avg+TCRIT*se,
            'sign_flip_two_sided_p':sum(v>=abs(avg)-1e-15 for v in sign_values)/8,
            'all_three_improve':all(d<0 for d in delta),'all_three_worsen':all(d>0 for d in delta)}
keys=sorted(k for k in selected['old'][0]['training_eval_metrics'] if k.startswith('eval_dropless/') and (k.endswith('/bpb') or k.endswith('/macro_bpb')))
fixed=[compare('training_eval_metrics',k) for k in keys];tasks=[compare('grouped_bpb',k) for k in sorted(selected['old'][0]['grouped_bpb'])]
assert len(tasks)==54
save('paired_seed_metrics.json',{'fixed_text':fixed,'grouped_tasks_bpb':tasks,'assumptions':['Same seed label does not establish identical random streams.','t95 intervals assume iid normal paired differences and ignore candidate selection.','Sign-flip p is a sensitivity calculation assuming exchangeable symmetric differences, not an actual randomized trial.','Task BPB is not accuracy or pass@k.']})
table('paired_seed_fixed.csv',fixed);table('paired_seed_tasks_bpb.csv',tasks)
inventory['selected_task_directions']={'all_three_improve':sum(r['all_three_improve'] for r in tasks),'all_three_worsen':sum(r['all_three_worsen'] for r in tasks),'mixed':sum(not r['all_three_improve'] and not r['all_three_worsen'] for r in tasks)}
# Historical stock is used for production weights; registry stock can be audited independently.
config=read(S/'wandb/hero-fa4sm100-nomask-step146k_meta.json')['config'];data=config['data']['value'];stages=data['train_weights'];stock=read(S/'harrier_spec_12d8b6f0.json')['available_tokens'];names=[k for k in data['components'] if any(w.get(k,0)>0 for _,w in stages)]
assert len(names)==200 and set(names)==set(stock)
registry_stock={c['cell']:c['available_tokens'] for c in buckets['cells']}
inventory['registry_stock_total']=sum(registry_stock.values());inventory['production_stock_total']=sum(stock.values());inventory['registry_stock_differences']=[{'cell':k,'production_tokens':stock[k],'registry_tokens':registry_stock[k]} for k in stock if registry_stock[k]!=stock[k]]
# Accurate-sum scenario is separately checked against native Python 3.12 in reproduce_loader_probe.py.
# Explicit sequential sum keeps the old-runtime sensitivity example stable on newer Python versions.
def sequential_sum(values):
    z=0.
    for x in values:z+=x
    return z
def counts_for(weights,K,mode='accurate'):
    total=(math.fsum if mode=='accurate' else sequential_sum)(weights.values());normalized={k:v/total for k,v in weights.items() if v>0}
    counts=np.zeros(len(names),dtype=np.int64)
    for i,k in enumerate(names):counts[i]=normalized.get(k,0)*K
    recipient=int(counts.argmax());remainder=K-int(counts.sum());counts[recipient]+=remainder
    return counts,{'block_size':K,'sum_mode':mode,'normalization_total':total,'remainder':remainder,'recipient':names[recipient],
                  'positive_but_zero':[k for k,n in zip(names,counts) if n==0 and weights.get(k,0)>0],
                  'max_absolute_shift_pp':max(abs(n/K-weights.get(k,0))*100 for k,n in zip(names,counts))}
quant=[];detail=[]
for K in [49152,12288]:
    for phase,(_,weights) in enumerate(stages):
        for mode in ['accurate','legacy_sequential']:
            c,q=counts_for(weights,K,mode);q['phase']=phase;quant.append(q)
            for k,n in zip(names,c):detail.append({'cell':k,'phase':phase,'block_size':K,'sum_mode':mode,'configured_pct':weights[k]*100,'count_per_block':int(n),'effective_pct':100*n/K,'delta_pp':100*(n/K-weights[k])})
save('loader_quantization.json',quant);table('loader_quantization.csv',detail)
# Complete blocks only: no source modulo, inner shuffle, documents or partial mix-block permutation.
def cursor(step,seq_len,batch_size,K,boundary):
    assert boundary*batch_size%K==0
    phase0_blocks=boundary*batch_size//K;blocks=step*batch_size//K
    c0=counts_for(stages[0][1],K)[0];c1=counts_for(stages[1][1],K)[0]
    return (c0*phase0_blocks+c1*(blocks-phase0_blocks))*seq_len,(step*batch_size%K)*seq_len
cursor_summary=[];cursor_rows=[]
for step in [195125,216000]:
    old,old_tail=cursor(step,4096,11264,49152,108000)
    for K,boundary in [(49152,108096),(12288,108000)]:
        new,new_tail=cursor(step,16384,2816,K,boundary);difference=new-old
        result={'step':step,'new_block_size':K,'new_historical_boundary':boundary,'l1_cursor_difference_tokens':int(np.abs(difference).sum()),
                'forward_cursor_tokens':int(difference[difference>0].sum()),'backward_cursor_tokens':int(-difference[difference<0].sum()),
                'old_partial_block_tokens':old_tail,'new_partial_block_tokens':new_tail,'largest_difference_cell':names[int(np.abs(difference).argmax())]}
        cursor_summary.append(result)
        for k,a,b,d in zip(names,old,new,difference):cursor_rows.append({'step':step,'new_block_size':K,'cell':k,'old_complete_block_cursor_tokens':int(a),'new_complete_block_cursor_tokens':int(b),'new_minus_old_tokens':int(d)})
save('cursor_audit.json',{'scope':'complete mixture blocks, physical token offsets before finite-source modulo and inner shuffling','scenarios':cursor_summary});table('cursor_audit.csv',cursor_rows)
assert cursor_summary[0]['l1_cursor_difference_tokens']==2951348224
assert cursor_summary[1]['l1_cursor_difference_tokens']==108907479040
cells=read(A/'cell_weights.json');table('cell_main_shifts.csv',[{'cell':r['cell'],'domain':r['domain'],'quality':r['quality'],'old_pct':r['phase0_pct'],'main_pct':r['phase1_pct'],'change_pp':r['phase1_pct']-r['phase0_pct']} for r in cells])
save('swarm_audit.json',inventory)
# Deterministic appendix consumed by the prose chapter.
lines=['<!-- GENERATED_DEEP_TABLES -->','## 附录：54项任务BPB完整对照','', '这里全部是registry里的`grouped_bpb`，不是答题准确率。部分行是语言平均与其组成项，不能再把54行平均当独立macro。列中变化分别按每个seed的旧值作分母，汇总列使用三seed均值之比。','', '| 指标 | 旧均值 | 新均值 | 均值变化 | seed0 | seed1 | seed2 |','|---|---:|---:|---:|---:|---:|---:|']
for r in tasks:lines.append('| %s | %.6f | %.6f | %+.3f%% | %+.3f%% | %+.3f%% | %+.3f%% |'%(r['metric'],r['old_mean'],r['new_mean'],r['change_pct'],r['seed0_change_pct'],r['seed1_change_pct'],r['seed2_change_pct']))
lines+=['','[完整逐seed差值与条件区间CSV](analysis/paired_seed_tasks_bpb.csv) · [固定文本逐seedCSV](analysis/paired_seed_fixed.csv)','']
p=ROOT/'DEEP_DIVE_ZH.md'
if p.exists():p.write_text(p.read_text().split('<!-- GENERATED_DEEP_TABLES -->')[0].rstrip()+'\n\n'+'\n'.join(lines))
print(json.dumps({'swarm_rows':len(rows),'distinct_weight_pairs':len(counts),'task_directions':inventory['selected_task_directions'],'stock_differences':len(inventory['registry_stock_differences']),'cursor':cursor_summary,'quantization_accurate':[r for r in quant if r['sum_mode']=='accurate']},ensure_ascii=False,indent=2))
