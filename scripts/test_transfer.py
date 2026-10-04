# -*- coding: utf-8 -*-
"""Check the scientific contrasts and budget invariants on prospective plans."""
import pathlib,json,math,copy
from transfer_core import exposure,factorial_plan,difference,quantize
R=pathlib.Path(__file__).resolve().parents[1];data=json.loads((R/'analysis/transfer_data.json').read_text());tests=[]
def check(name,fn):fn();tests.append(name)
def assert_(v):
    if not v:raise AssertionError('Transfer contract invariant failed')
def raises(fn):
    try:fn()
    except ValueError:return True
    return False
base=data['base_weights'];alt=data['profiles'][0]['weights'];T=data['budgets']['nominal'];ea=exposure(base,T)
big_alt=[]
for w in alt:
    mass=math.fsum(v for k,v in w.items() if k.startswith('c39q'))
    big_alt.append({k:v*(.8/mass if k.startswith('c39q') else .2/(1-mass)) for k,v in w.items()})
check('Cumulative shift conserves budget and separates cancellation',lambda:assert_(all(abs(p['comparisons']['nominal']['net_difference_tokens'])<.01 and abs(p['comparisons']['nominal']['cross_domain_net_tokens']+p['comparisons']['nominal']['within_domain_cancellation_tokens']-p['comparisons']['nominal']['moved_tokens'])<.001 for p in data['profiles'])))
check('Math cumulative gain and time redistribution are distinct',lambda:assert_(data['math_timing']['stage_delta_tokens'][0]>0>data['math_timing']['stage_delta_tokens'][1] and data['math_timing']['cumulative_delta_tokens']>0 and abs(math.fsum(data['math_timing']['timing_residual_tokens']))<.001))
for p in data['plans']:
    rkeys=[k for k in ea if k.startswith('c39q')];dkeys=[k for k in ea if k.startswith('c%02dq'%p['donor_domain'])]
    for a in p['arms']:
        e=exposure(a['weights'],T)
        assert all(w[k]>=0 for w in a['weights'] for k in ea)
        assert all(abs(math.fsum(w.values())-1)<1e-12 for w in a['weights'])
        assert abs(math.fsum(e[k] for k in rkeys)-a['target_receiver_tokens'])<.001
        assert max(abs(e[k]/a['target_receiver_tokens']-a['target_conditional_quality'][k]) for k in rkeys)<1e-14
        assert abs(math.fsum(e[k]-ea[k] for k in dkeys)-a['donor_exposure_delta_tokens'])<.001
        assert all(a['weights'][i][k]==base[i][k] for i in [0,1] for k in ea if k not in rkeys+dkeys)
        assert all(abs(a['weights'][0][k]-base[0][k]-(a['weights'][1][k]-base[1][k]))<1e-16 for k in ea)
        assert all(sum(counts.values())==49152 for counts in a['full_block_audit']['phase_counts'])
check('All twelve arms preserve continuous normalization and exact factor targets',lambda:assert_(sum(len(p['arms']) for p in data['plans'])==12))
check('Donor losses match receiver amount gains and other 38 domains stay fixed',lambda:assert_(all(a['donor_exposure_delta_tokens']==-a['amount_factor']*p['increment_tokens'] for p in data['plans'] for a in p['arms'])))
check('Every changed bucket receives same additive delta in both phases',lambda:assert_(all(all(abs(a['weights'][0][k]-base[0][k]-(a['weights'][1][k]-base[1][k]))<1e-16 for k in ea) for p in data['plans'] for a in p['arms'])))
check('Invalid donor or negative capacity fails without clipping',lambda:assert_(raises(lambda:factorial_plan(base,alt,T,39,39)) and raises(lambda:factorial_plan(base,big_alt,T,39,26))))
check('Full-block totals conserve units but leave explicit extra-bucket audit',lambda:assert_(all(sum(c.values())==49152 for p in data['plans'] for a in p['arms'] for c in a['full_block_audit']['phase_counts']) and all('extra_changed_cells_vs_baseline' in a['full_block_audit'] for p in data['plans'] for a in p['arms'])))
check('Compiled weight contracts cannot claim experiment results or launch readiness',lambda:assert_(all(p['status']=='planned_not_executed' and p['launchable'] is False and p['generation_results'] is None and p['effective_inventory_and_history'] is None for p in data['plans'])))
check('Prospective math experiment does not reproduce the global 197c recipe',lambda:assert_(any(data['plans'][0]['arms'][-1]['weights'][0][k]!=alt[0][k] for k in ea if not k.startswith(('c39q','c26q')))))
check('Integer repair confines changes to declared receiver and donor',lambda:assert_(all(not a['integer_repair']['extra_changed_cells_vs_baseline'] and all(c[k]==quantize(base[i],49152)['counts'][k] for i,c in enumerate(a['integer_repair']['target_phase_counts']) for k in ea if not k.startswith(('c39q','c%02dq'%p['donor_domain']))) for p in data['plans'] for a in p['arms'])))
check('Baseline and Q-only preserve every domain total after integer compilation',lambda:assert_(all(sum(c[k] for k in ea if k.startswith('c%02dq'%d))==sum(quantize(base[i],49152)['counts'][k] for k in ea if k.startswith('c%02dq'%d)) for p in data['plans'] for a in p['arms'] if a['amount_factor']==0 for i,c in enumerate(a['integer_repair']['target_phase_counts']) for d in range(40))))
check('Integer amount transfer is 61 units and all 24 encoded blocks roundtrip',lambda:assert_(all(a['integer_repair']['transferred_units_per_block']==[61*a['amount_factor']]*2 and all(quantize(w,49152)['counts']==c and sum(c.values())==49152 for w,c in zip(a['integer_controlled_weights'],a['integer_repair']['target_phase_counts'])) for p in data['plans'] for a in p['arms'])))
check('Repeated controls are eight unique recipes and finite-stream status stays false',lambda:assert_(data['unique_integer_recipes']==len({json.dumps(a['integer_repair']['target_phase_counts'],sort_keys=True) for p in data['plans'] for a in p['arms']})==8 and all(a['integer_repair']['actual_finite_stream_verified'] is False for p in data['plans'] for a in p['arms'])))
(R/'analysis/transfer_validation.json').write_text(json.dumps({'test_groups_passed':len(tests),'tests':tests,'scope':'Prospective continuous contrasts and full-block arithmetic, not model training.'},indent=2)+'\n')
print('Transfer contrast checks:',len(tests),'groups passed')
