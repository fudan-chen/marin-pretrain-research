"""Scientific ledger invariants, negative controls and conditional scope checks."""
import json,pathlib,math
from order_core import compile_order,window_ledger,partial_difference_bounds
R=pathlib.Path(__file__).resolve().parents[1];d=json.loads((R/'analysis/order_workbench_data.json').read_text());passed=[]
def check(name,test):
    if not test:raise AssertionError(name)
    passed.append(name)
def raises(fn):
    try:fn()
    except ValueError:return True
    return False
base=d['baseline_counts'];initial=d['initial_counts'];K=49152;cells=sorted(initial)
ledger=window_ledger(402432,4024320,[0,393216,3194880],K)
check('Observed-index window covers 3537 updates with two partial edges',sum(x['complete_blocks']*K+sum(p['read_sequences'] for p in x['partials']) for x in ledger)==3537*1024 and [x['complete_blocks'] for x in ledger]==[56,16] and [p['read_sequences'] for x in ledger for p in x['partials']]==[39936,43008])
check('Single partial-block window is never counted twice',window_ledger(10,50,[0],100)[0]['partials']==[{'block':0,'slot_begin':10,'slot_end':50,'read_sequences':40}])
check('Unaligned phase boundary fails rather than silently snapping',raises(lambda:window_ledger(402432,4024320,[0,393217,3194880],K)))
check('Historical complete blocks retain nonzero differences for every pair',len(d['historical_pairs'])==8 and all(p['complete_block_l1_tokens']>0 and p['whole_window_exact_difference_tokens'] is None for p in d['historical_pairs']))
for k in range(20):
    p=compile_order(base,initial,K,k)
    assert p['capacity']['early_units_per_k']==2 and p['capacity']['late_units_per_k']==7 and p['capacity']['symmetric_max_k']==19
    for a in p['arms']:
        assert all(sum(s['counts'].values())==K and min(s['counts'].values())>=0 for s in a['stages'])
        assert not any(a['interior_cumulative_difference_sequences'].values())
        assert all(a['interior_counts'][i][c]==base[i][c] for i in [0,1] for c in cells if c not in [p['receiver'],p['donor']])
        assert a['stages'][1]['counts']==base[0] and a['stages'][-1]['counts']==base[1]
check('All 20 amplitudes conserve counts and declared support',True)
check('Integer compensation is 2k early and minus 7k late',all(56*a['receiver_early_delta_units']+16*a['receiver_late_delta_units']==0 for a in d['locked_default']['arms']))
check('All outer blocks keep common weights and absolute boundaries',[s['sequence_begin'] for s in d['locked_default']['arms'][1]['stages']]==[0,393216,442368,3194880,3981312])
check('Magnitude, type and capacity violations cannot be clipped',all(raises(lambda k=k:compile_order(base,initial,K,k)) for k in [20,-1,1.5,True]))
check('Unlocked interior equality cannot assert whole-window equality',all(a['interior_l1_difference_sequences']==0 for a in d['unlocked_default']['arms']) and d['unlocked_default']['whole_window_count_match_by_construction'] is False)
check('Zero amplitude is a no-op even without edge locking',compile_order(base,initial,K,0,lock_edges=False)['whole_window_count_match_by_construction'] and len({json.dumps(a['interior_counts'],sort_keys=True) for a in compile_order(base,initial,K,0)['arms']})==1)
for n in range(9):
    for m in range(9):
        for read in range(9):
            lo,hi=partial_difference_bounds({'x':n},{'x':m},read,8)['x']
            assert all(lo<=b-a<=hi for a in range(max(0,n-(8-read)),min(n,read)+1) for b in range(max(0,m-(8-read)),min(m,read)+1))
check('Partial-bucket difference bounds contain all feasible counts',True)
check('Matching proposal keeps live state, keys and outcomes unknown',all(p['status']=='planned_not_executed' and p['launchable'] is False and p['actual_token_stream_verified'] is False and all(p[k] is None for k in ['actual_checkpoint_digest','actual_data_cursor','actual_shuffle_keys','token_store_manifest','independent_eval_manifest','generation_results']) for p in [d['locked_default'],d['unlocked_default']]))
(R/'analysis/order_validation.json').write_text(json.dumps({'test_groups_passed':len(passed),'tests':passed,'amplitudes_checked':20,'scope':'Integer-ledger designs and assumptions; no actual stream or GPU outcome validation'},indent=2)+'\n')
print('Order ledger:',len(passed),'boundary groups passed')
