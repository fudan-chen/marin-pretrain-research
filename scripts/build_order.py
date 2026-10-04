# -*- coding: utf-8 -*-
"""Audit historical order blocks and compile edge-locked prospective schedules."""
import json,pathlib,csv,math,hashlib
from transfer_core import quantize
from order_core import window_ledger,partial_difference_bounds,compile_order
R=pathlib.Path(__file__).resolve().parents[1];A=R/'analysis';S=R/'sources/practical_2026_10_04';K=49152
pairs=[r for r in csv.DictReader((A/'order_budget_audit.csv').open()) if r['budget_convention']=='observed_2727_810_physical']
def config(name):return json.loads(json.loads((S/('config_'+name+'.json')).read_text())['data']['project']['run']['config'])
def save(path,x):path.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def table(name,rows):
    with (A/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
historical=[];runs={};cells=[]
for pair in pairs:
    counts=[]
    for field in ['run_a','run_b']:
        name=pair[field];c=config(name);d=c['data']['value'];t=c['trainer']['value']['trainer'];batch=t['train_batch_size']
        assert d['mixture_block_size']==K and batch==1024 and t['num_train_steps']==3930
        phases=d['train_weights'];assert [s for s,_ in phases]==[0,384,3120]
        cc=[quantize({k:v for k,v in w.items() if len(k)==5 and k.startswith('c')},K)['counts'] for _,w in phases]
        runs[name]={'source':'sources/practical_2026_10_04/config_'+name+'.json','source_sha256':hashlib.sha256((S/('config_'+name+'.json')).read_bytes()).hexdigest(),
                    'phase_weights':[{k:v for k,v in w.items() if k in cc[0]} for _,w in phases],
                    'expected_counts':cc,'sequence_boundaries':[s*batch for s,_ in phases]}
        counts.append(cc)
    ca,cb=counts;ledger=window_ledger(393*1024,3930*1024,[0,384*1024,3120*1024],K)
    complete={k:sum(p['complete_blocks']*(cb[p['phase']][k]-ca[p['phase']][k]) for p in ledger) for k in ca[0]}
    lows=complete.copy();highs=complete.copy()
    for p in ledger:
        for edge in p['partials']:
            bounds=partial_difference_bounds(ca[p['phase']],cb[p['phase']],edge['read_sequences'],K)
            for k in complete:lows[k]+=bounds[k][0];highs[k]+=bounds[k][1]
    conventions=[r for r in csv.DictReader((A/'order_budget_audit.csv').open()) if r['pair']==pair['pair']]
    historical.append({'pair':pair['pair'],'run_a':pair['run_a'],'run_b':pair['run_b'],'window_ledger':ledger,
       'continuous_conventions':[{k:r[k] for k in ['budget_convention','l1_exposure_difference_tokens','paloma_b_minus_a','gsm8k_b_minus_a']} for r in conventions],
       'complete_block_difference_sequences':complete,'complete_block_l1_tokens':sum(abs(v) for v in complete.values())*4096,
       'complete_block_math_net_tokens':sum(v for k,v in complete.items() if k.startswith('c39q'))*4096,
       'complete_block_changed_cells':sum(v!=0 for v in complete.values()),
       'whole_window_difference_bounds_sequences':{k:[lows[k],highs[k]] for k in complete},
       'bound_scope':'Per-bucket loose bounds for arbitrary edge permutations; not jointly tight or actual token stream',
       'whole_window_exact_difference_tokens':None})
    cells.extend({'pair':pair['pair'],'cell':k,'complete_block_delta_sequences':complete[k],
                  'whole_window_delta_lower_sequences':lows[k],'whole_window_delta_upper_sequences':highs[k]} for k in complete)
table('order_integer_cell_audit.csv',cells)
base=json.loads((R/'templates/math_factorial_donor_c26.json').read_text())['arms'][-1]['integer_repair']['target_phase_counts']
initial=next(iter(runs.values()))['expected_counts'][0]
assert all(r['expected_counts'][0]==initial for r in runs.values())
locked=compile_order(base,initial,K,10);unlocked=compile_order(base,initial,K,10,lock_edges=False)
save(R/'templates/order_edge_locked.json',locked);save(R/'templates/order_edges_changed_counterexample.json',unlocked)
data={'schema':'marin-order-workbench/1','snapshot':'Frozen 2026-10-04 source archive; no online training refresh',
      'historical_pairs':historical,'historical_runs':runs,'baseline_counts':base,'initial_counts':initial,
      'locked_default':locked,'unlocked_default':unlocked,'native_probe':'analysis/order_loader_probe_python312.json',
      'limitations':['Historical start index is derived from logged step393 and batch1024, not restored live cursor',
                     'Whole historical window depends on actual edge permutations and shared data state',
                     'New schedule is an authored conditional experiment using unconfirmed V7 baseline',
                     'No GPU training, generation evaluation or token-store replay']}
save(A/'order_workbench_data.json',data)
save(A/'order_integer_audit.json',{'historical_pairs':len(historical),'historical_runs':len(runs),'historical_complete_block_l1_tokens':[p['complete_block_l1_tokens'] for p in historical],
 'start_sequence_index':393*1024,'end_sequence_index':3930*1024,'common_edge_read_sequences':[39936,43008],
 'interior_blocks':[56,16],'symmetric_max_k':locked['capacity']['symmetric_max_k'],'default_k':10,
 'default_receiver_forward_tokens':locked['arms'][1]['receiver_forward_sequences']*4096,
 'conditional_count_match':'Edge-locked construction only, provided same keys, mapping, history and cursor',
 'actual_stream_verified':False})
print('Built',len(historical),'historical order pairs and two prospective count contracts.')
