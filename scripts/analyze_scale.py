# -*- coding: utf-8 -*-
"""Recompute ladder contrasts, fit assumptions and a synthetic batch-normalization counterexample."""
import csv
import hashlib
import json
import math
import pathlib
import re
import statistics
import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
S, A = R / 'sources', R / 'analysis'
D = S / 'scale_2026_10_05'
SIZES = ['d768', 'd1024', 'd1536']
OLD = ['rav-ladder-d768-v2', 'rav-ladder-d1024', 'rav-ladder-d1536']


def save(name, obj):
    (A / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def table(name, rows):
    with (A / name).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


raw = list(csv.DictReader((S / 'mix_study_final-2026.09.15.1_point_speedups.csv').open()))
bpb = [r for r in raw if r['metric'] == 'bpb']
subsets = sorted({r['subset'] for r in bpb})
endpoints, means, normalization = [], [], []
for size in SIZES:
    rs = [r for r in bpb if r['size'] == size]
    assert len(rs) == 16
    for r in rs:
        ce = next(x for x in raw if x['size'] == size and x['metric'] == 'loss' and x['subset'] == r['subset'])
        old, new = float(r['baseline_loss']), float(r['variant_loss'])
        oc, nc = float(ce['baseline_loss']), float(ce['variant_loss'])
        endpoints.append({'size': size, 'subset': r['subset'], 'old_bpb': old, 'new_bpb': new,
                          'bpb_change_pct': 100 * (new / old - 1), 'old_ce': oc, 'new_ce': nc,
                          'ce_change_pct': 100 * (nc / oc - 1), 'direction_disagrees': (new-old)*(nc-oc) < 0,
                          'compute_csv': float(r['baseline_compute']), 'source': 'mix_study_final-2026.09.15.1_point_speedups.csv'})
        normalization.append({'size': size, 'subset': r['subset'], 'old_bpb_over_ce': old/oc,
                              'new_bpb_over_ce': new/nc, 'ratio_change_pct': 100*((new/nc)/(old/oc)-1)})
    means.append({'size': size, 'old_bpb': statistics.mean(float(r['baseline_loss']) for r in rs),
                  'new_bpb': statistics.mean(float(r['variant_loss']) for r in rs),
                  'compute': float(rs[0]['baseline_compute']), 'improved': sum(float(r['variant_loss']) < float(r['baseline_loss']) for r in rs)})
table('scale_endpoints.csv', endpoints)
table('scale_normalization.csv', normalization)
flips = [s for s in subsets if len({np.sign(r['new_bpb']-r['old_bpb']) for r in endpoints if r['subset'] == s}) > 1]

# Exact logged configuration values; a config is not proof of the actual token stream.
contrasts, early_cells, runtimes = [], [], []
fields = {'train_batch': 'trainer.trainer.train_batch_size', 'updates': 'trainer.trainer.num_train_steps',
          'expert_axis': 'trainer.expert_axis_size', 'eval_batch': 'eval.eval_batch_size',
          'dropless_eval': 'eval.dropless_eval', 'gpu': 'resources.device.variant',
          'learning_rate': 'optimizer.learning_rate', 'adam_lr': 'optimizer.adam_lr',
          'beta2': 'optimizer.beta2', 'epsilon': 'optimizer.epsilon', 'tokenizer': 'data.tokenizer',
          'block_sequences': 'data.mixture_block_size', 'permutation': 'data.permutation_type'}


def at(c, path):
    v = c
    for part in path.split('.'):
        v = v[part]
    return v


for size, old_name in zip(SIZES, OLD):
    records = []
    for name in [old_name, 'h100-mix25-20260912-' + size]:
        p = D / ('config_' + name + '.json')
        run = json.loads(p.read_text())['data']['project']['run']
        c = {k: v['value'] for k, v in json.loads(run['config']).items() if 'value' in v}
        sm = json.loads(run['summaryMetrics'])
        records.append((c, sm))
        runtimes.append({'size': size, 'run': name, 'state': run['state'], 'final_step': sm['_step'],
                         'logged_total_tokens': sm['throughput/total_tokens'],
                         'configured_total_tokens': c['trainer']['trainer']['num_train_steps'] * c['trainer']['trainer']['train_batch_size'] * c['model']['max_seq_len'],
                         'weight_starts': [x[0] for x in c['data']['train_weights']],
                         'weight_start_fractions': [x[0]/c['trainer']['trainer']['num_train_steps'] for x in c['data']['train_weights']],
                         'config_file': str(p.relative_to(R)), 'config_sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    (oc, os), (nc, ns) = records
    for label, path in fields.items():
        contrasts.append({'size': size, 'field': label, 'path': path, 'old': at(oc,path), 'new': at(nc,path), 'equal': at(oc,path)==at(nc,path)})
    wa, wb = oc['data']['train_weights'][0][1], nc['data']['train_weights'][0][1]
    cells = sorted(k for k in wa.keys() | wb.keys() if re.fullmatch(r'c\d{2}q\d', k))
    assert len(cells) == 200
    for cell in cells:
        early_cells.append({'size': size, 'cell': cell, 'old_initial_weight': wa.get(cell,0),
                            'new_initial_weight': wb.get(cell,0), 'new_minus_old': wb.get(cell,0)-wa.get(cell,0)})
    half_l1 = sum(abs(wa.get(k,0)-wb.get(k,0)) for k in wa.keys() | wb.keys())/2
    contrasts.append({'size': size, 'field': 'initial_weight_half_l1', 'path': 'data.train_weights[0][1]', 'old': 0, 'new': half_l1, 'equal': half_l1==0})
    valkeys = [k for k in oc['data']['components'] if k.startswith('paloma/')]
    contrasts.append({'size': size, 'field': 'sixteen_validation_component_configs', 'path': 'data.components[paloma/*]',
                      'old': len(valkeys), 'new': len([k for k in nc['data']['components'] if k.startswith('paloma/')]),
                      'equal': len(valkeys)==16 and all(oc['data']['components'][k]==nc['data']['components'][k] for k in valkeys)})
table('scale_config_contrasts.csv', contrasts)
table('scale_initial_weights.csv', early_cells)
save('scale_run_metadata.json', runtimes)

# Refit alpha and intercept jointly at each *chosen* floor. These are diagnostic assumptions, not estimated floors.
profiles, crossvalidation = [], []
for subset in ['macro', 'dolma_100_programing_languages']:
    points = means if subset == 'macro' else [{'size': r['size'], 'old_bpb':r['old_bpb'], 'new_bpb':r['new_bpb'], 'compute':r['compute_csv']} for r in endpoints if r['subset']==subset]
    points = sorted(points, key=lambda r:SIZES.index(r['size']))
    C = np.array([r['compute'] for r in points]); L = np.array([r['old_bpb'] for r in points]); x=np.log(C/C[0])
    zero_slope, zero_intercept = np.polyfit(x, np.log(L), 1); zero_alpha=-zero_slope
    for frac in [0,.1,.25,.5,.75,.9]:
        floor=frac*min(L); slope,intercept=np.polyfit(x,np.log(L-floor),1);alpha=-slope
        predicted=floor+np.exp(intercept+slope*x)
        ratio=(L[-1]-floor)/(points[-1]['new_bpb']-floor)
        profiles.append({'subset':subset,'floor_fraction':frac,'floor_bpb':floor,'refit_alpha':alpha,
                         'refit_recentered_speedup':float(ratio**(1/alpha)),
                         'fixed_zero_floor_alpha_speedup':float(ratio**(1/zero_alpha)),
                         'max_in_sample_error_bpb':float(max(abs(predicted-L))),
                         'target_old_bpb':L[-1],'target_new_bpb':points[-1]['new_bpb'],
                         'observations':3,'free_fit_parameters_at_fixed_floor':2,
                         'scope':'Assumption profile; not confidence interval, wall-clock saving or a fitted floor'})
    for held in range(3):
        keep=[i for i in range(3) if i!=held];slope,intercept=np.polyfit(x[keep],np.log(L[keep]),1)
        predicted=float(np.exp(intercept+slope*x[held]))
        crossvalidation.append({'subset':subset,'held_out_size':SIZES[held],'alpha':float(-slope),
                                'predicted_bpb':predicted,'observed_bpb':L[held],
                                'error_bpb':predicted-L[held],'relative_error_pct':100*(predicted/L[held]-1),
                                'scope':'Two-rung zero-floor fit predicts the third baseline rung; three dependent diagnostics, not independent trials'})
table('scale_fit_sensitivity.csv', profiles)
table('scale_fit_leave_one_rung.csv', crossvalidation)

# Invented records, exactly the token-weighted batch-BPB formula seen in pinned public eval.py.
records=[{'record':'A','tokens':100,'bytes':100,'old_nll':100,'new_nll':110},
         {'record':'B','tokens':100,'bytes':1000,'old_nll':100,'new_nll':50}]
toy=[]
for batch_size in [1,2]:
    result={'batch_size':batch_size}
    for model in ['old','new']:
        batches=[records[i:i+batch_size] for i in range(0,len(records),batch_size)]
        result[model+'_ce']=sum(r[model+'_nll'] for r in records)/sum(r['tokens'] for r in records)
        result[model+'_batch_token_weighted_bpb']=sum(sum(r[model+'_nll'] for r in b)/sum(r['bytes'] for r in b)*math.log2(math.e)*sum(r['tokens'] for r in b) for b in batches)/sum(r['tokens'] for r in records)
        result[model+'_global_ratio_bpb']=sum(r[model+'_nll'] for r in records)/sum(r['bytes'] for r in records)*math.log2(math.e)
    result['logged_formula_change_pct']=100*(result['new_batch_token_weighted_bpb']/result['old_batch_token_weighted_bpb']-1)
    result['global_ratio_change_pct']=100*(result['new_global_ratio_bpb']/result['old_global_ratio_bpb']-1)
    toy.append(result)
save('scale_batch_counterexample.json',{'scope':'Synthetic arithmetic replay; no JAX/GPU/checkpoint or historical metric root cause verified',
                                     'source':'sources/scale_2026_10_05/eval.py:583-590',
                                     'records':records,'results':toy,'unchanged_predictions_between_batchings':True})

# Proposed finite confirmation budget. Run-level slots do not imply independent initializations.
plan=[]
for size,total in [('d768',11420),('d1536',90767)]:
    for fraction in [.1,.25]:
        start=math.ceil(total*fraction/48)*48
        for arm in ['old_frozen','proportional_frozen','selected_996f_frozen','risk_197c_frozen']:
            for continuation_seed in [3,4,5]:
                plan.append({'size':size,'target_start_fraction':fraction,'total_updates':total,'start_update':start,
                             'actual_start_fraction':start/total,'arm':arm,'continuation_seed':continuation_seed,
                             'remaining_tokens':(total-start)*1024*4096,'status':'planned_not_executed'})
save('scale_confirmation_plan.json',{'schema':'marin-scale-confirmation-draft/1','status':'planned_not_executed',
    'launchable':False,'purpose':'Candidate-vs-baseline within a declared common checkpoint; scale-by-history interaction remains conditional on those checkpoints',
    'arms':plan,'continuation_slots':len(plan),'total_continuation_tokens':sum(r['remaining_tokens'] for r in plan),
    'start_rounding':'ceil(requested progress / 48 updates) * 48; block 49152 sequences, batch 1024',
    'initialization_tokens_in_budget':False,'shared_checkpoint_cost_tokens':None,'generation_eval_cost':None,
    'checkpoint_digests':None,'actual_data_state':None,'frozen_weight_digests':None,'independent_eval_manifest':None,
    'primary_metric_and_guardrails':None,'actual_launch':None,'actual_results':None,
    'order_effect_claim_permitted':False,'reader_efficacy_study':None,
    'stopping_contract':'Freeze objectives, paired-difference tolerance and risk limits before new outcomes; no candidate passes automatically; missing execution bindings prevent launch'})
audit={'scope':'Frozen public endpoint recomputation plus logged configuration contrast; no controlled scale-effect estimate',
       'endpoint_rows':len(endpoints),'macro_rows':means,'direction_changes':flips,
       'ce_bpb_direction_disagreements':[{'size':r['size'],'subset':r['subset']} for r in endpoints if r['direction_disagrees']],
       'initial_weight_half_l1':[r['new'] for r in contrasts if r['field']=='initial_weight_half_l1'],
       'fit_assumptions':{'compute_source':'point_speedups.csv baseline_compute; not nominal launcher constants',
                          'fixed_floor_then_refit':True,'recenter_at_measured_target_endpoint':True,
                          'throughput_ratio':1,'confidence_interval':None,'estimated_irreducible_floor':None},
       'sources_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(D.glob('*'))},
       'actual_scale_causal_experiment':None,'actual_gpu_confirmation':None}
save('scale_audit.json',audit)
print('Ladder directions:',flips,'CE/BPB disagreements:',audit['ce_bpb_direction_disagreements'])
print('Initial mixture half-L1:',audit['initial_weight_half_l1'])
print('Planned continuation slots/tokens:',len(plan),sum(r['remaining_tokens'] for r in plan))
