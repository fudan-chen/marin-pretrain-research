# -*- coding: utf-8 -*-
"""Audit candidate shifts and build prospective two-factor transfer contracts."""
import pathlib,json,csv,math,re,hashlib
import pyarrow.parquet as pq
from transfer_core import exposure,difference,factorial_plan,quantize,controlled_integer_plan
R=pathlib.Path(__file__).resolve().parents[1];A=R/'analysis';S=R/'sources/deepening_2026_10_04'
rows=pq.read_table(S/'hf_observations.parquet').to_pylist()
candidates={re.search(r'mixprior-([0-9a-f]+)-',r['run_name'])[1]:r for r in rows if r['group']=='mixprior_candidate' and '-seed0-' in r['run_name']}
ids=['197c9f5ceff6b9ee','b100ad1935c71002','4602cace15312bd4','44bfd83cf7199b40'];base=candidates['996f489106c7b922']
names={int(d['id']):d['name_zh'] for d in json.loads((A/'domain_weights_zh.json').read_text())}
budgets={'nominal':[13125000000000,3750000000000],'recorded_physical':[2727*1024*4096,810*1024*4096]}
def weights(r):return [r['phase%d_weights'%p] for p in [0,1]]
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def table(name,rs):
    with (A/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]),lineterminator='\n');w.writeheader();w.writerows(rs)
metrics=json.loads((A/'decision_data.json').read_text())
profiles=[];long=[];domain_rows=[]
for id_ in ids:
    target=candidates[id_];comparisons={}
    for convention,T in budgets.items():
        d=difference(weights(base),weights(target),T);ea=exposure(weights(base),T);eb=exposure(weights(target),T)
        comparisons[convention]={'budget_tokens':T,'total_tokens':sum(T),'moved_tokens':d['moved'],
          'cross_domain_net_tokens':d['cross_domain_net'],'within_domain_cancellation_tokens':d['within_domain_cancellation'],
          'phase_moved_sum_tokens':d['phase_moved_sum'],'temporal_cancellation_tokens':d['temporal_cancellation'],
          'net_difference_tokens':d['net'],'cells':[{'cell':k,'domain':int(k[1:3]),'quality':int(k[-1]),'base_tokens':ea[k],'target_tokens':eb[k],'delta_tokens':d['cells'][k]} for k in ea],
          'domains':[{'id':c,'label':names[c],'delta_tokens':d['domains'][c]} for c in range(40)],
          'quality_net_tokens':[math.fsum(v for k,v in d['cells'].items() if k.endswith('q%d'%q)) for q in range(5)]}
        for r in comparisons[convention]['cells']:long.append({'candidate':id_,'budget_convention':convention,**r})
        for r in comparisons[convention]['domains']:domain_rows.append({'candidate':id_,'budget_convention':convention,**r})
    profiles.append({'id':id_,'run':target['run_name'],'changed_phase_cells':sum(any(base['phase%d_weights'%p][k]!=target['phase%d_weights'%p][k] for p in [0,1]) for k in base['phase0_weights']),
                     'weights':weights(target),'comparisons':comparisons,'metric_changes_vs_selected_seed0':
                     {m['id']:100*(next(c for c in metrics['candidates'] if c['id']==id_)['values'][m['id']]/metrics['baselines']['selected']['values'][m['id']]-1) for m in metrics['metrics']}})
table('transfer_cell_differences.csv',long);table('transfer_domain_differences.csv',domain_rows)
T=budgets['nominal'];alt=candidates[ids[0]];plans=[];quantrows=[];fixedrows=[]
for donor in [26,37,14]:
    p=factorial_plan(weights(base),weights(alt),T,39,donor)
    p.update({'schema':'marin-prospective-factorial-transfer/1','status':'planned_not_executed',
              'base_candidate':'996f489106c7b922','comparison_candidate':ids[0],'donor_label':names[donor],
              'budget_convention':'nominal_simulated_for_teaching','phase_budget_tokens':T,
              'physical_training_budget':None,'generation_results':None,'checkpoint_content_digest':None,
              'effective_inventory_and_history':None,'floors_and_caps':None,'independent_eval_manifest':None,
              'launchable':False,'phase_boundaries_unchanged_in_design':True,
              'block_units':'sequences','integer_encoding_scope':'fixed block_size=49152 and checked Python3.12/numpy2 loader methods; not a portable trainer change',
              'limitations':['Only weight arithmetic is compiled. No GPU training or actual token-ID replay.',
                             'Same additive delta in both phases deliberately excludes alternative candidate timing changes.',
                             'Positive weights do not establish exposure capacity, data-loader offsets or task safety.',
                             'Math budget and quality proportions are observed anchors, not optimal values.']})
    qb=[quantize(w,49152) for w in weights(base)]
    for arm in p['arms']:
        qs=[quantize(w,49152) for w in arm['weights']]
        continuous=exposure(arm['weights'],T);effective=exposure([q['weights'] for q in qs],T)
        arm['full_block_audit']={'block_units':49152,'scope':'full-block allocation arithmetic; not actual finite token stream',
           'phase_counts':[q['counts'] for q in qs],'recipients':[q['recipient'] for q in qs],
           'l1_nominal_exposure_error_tokens':math.fsum(abs(effective[k]-continuous[k]) for k in continuous),
           'extra_changed_cells_vs_baseline':sorted({k for k in continuous if arm['constant_phase_delta'][k]==0 and any(qs[i]['counts'][k]!=qb[i]['counts'][k] for i in [0,1])})}
        repaired=[controlled_integer_plan(w,arm['weights'][i],39,donor,arm['amount_factor']*p['increment_tokens']/sum(T),49152) for i,w in enumerate(weights(base))]
        if arm['id']=='M0Q0':
            # The baseline is preserved exactly; do not reconstruct a no-op condition.
            repaired=[{'target_counts':q['counts'],'weights':w,'transferred_units_per_block':0} for w,q in zip(weights(base),qb)]
        arm['integer_controlled_weights']=[x['weights'] for x in repaired]
        effective_fixed=exposure([{k:n/49152 for k,n in x['target_counts'].items()} for x in repaired],T)
        arm['integer_repair']={'method':'within-domain fixed totals plus explicit floor-interval encoding',
          'target_phase_counts':[x['target_counts'] for x in repaired],
          'transferred_units_per_block':[x['transferred_units_per_block'] for x in repaired],
          'receiver_tokens':math.fsum(v for k,v in effective_fixed.items() if k.startswith('c39q')),
          'l1_nominal_exposure_error_tokens':math.fsum(abs(effective_fixed[k]-continuous[k]) for k in continuous),
          'extra_changed_cells_vs_baseline':sorted({k for k in continuous if not k.startswith(('c39q','c%02dq'%donor)) and any(repaired[i]['target_counts'][k]!=qb[i]['counts'][k] for i in [0,1])}),
          'actual_finite_stream_verified':False}
        for k in continuous:quantrows.append({'donor':donor,'arm':arm['id'],'cell':k,'continuous_nominal_tokens':continuous[k],
                                           'full_block_nominal_tokens':effective[k],'error_tokens':effective[k]-continuous[k]})
        for k in continuous:fixedrows.append({'donor':donor,'arm':arm['id'],'cell':k,'continuous_nominal_tokens':continuous[k],
          'integer_controlled_nominal_tokens':effective_fixed[k],'error_tokens':effective_fixed[k]-continuous[k],
          'phase0_count':repaired[0]['target_counts'][k],'phase1_count':repaired[1]['target_counts'][k]})
    path=R/'templates'/('math_factorial_donor_c%02d.json'%donor);save(path,p);p['download']=str(path.relative_to(R));plans.append(p)
table('transfer_plan_quantization.csv',quantrows)
table('transfer_plan_integer_controlled.csv',fixedrows)
unique_recipes=len({json.dumps(a['integer_repair']['target_phase_counts'],sort_keys=True) for p in plans for a in p['arms']})
mathkeys=['c39q%d'%q for q in range(5)];b0=exposure(weights(base),T);b1=exposure(weights(alt),T)
stage_math=[t*math.fsum(alt['phase%d_weights'%i][k]-base['phase%d_weights'%i][k] for k in mathkeys) for i,t in enumerate(T)]
total_increment=math.fsum(b1[k]-b0[k] for k in mathkeys);constant_shift=[t/sum(T)*total_increment for t in T]
data={'schema':'marin-transfer-workbench/1','evidence_version':metrics['evidence_version'],
      'source_sha256':metrics['source_sha256'],'base_weights':weights(base),'base_candidate':'996f489106c7b922',
      'budgets':budgets,'profiles':profiles,'plans':plans,'unique_integer_recipes':unique_recipes,'math_timing':{'stage_delta_tokens':stage_math,
      'cumulative_delta_tokens':total_increment,'constant_same_total_stage_delta_tokens':constant_shift,
      'timing_residual_tokens':[x-y for x,y in zip(stage_math,constant_shift)]}}
save(A/'transfer_data.json',data)
audit={'profiles':[{'id':p['id'],'changed_phase_cells':p['changed_phase_cells'],**{k:v for k,v in p['comparisons']['nominal'].items() if k not in ['cells','domains']}} for p in profiles],
       'math_timing':data['math_timing'],'plan_count':len(plans),'arm_records':sum(len(p['arms']) for p in plans),
       'unique_integer_recipes':unique_recipes,'plan_status':'planned_not_executed',
       'not_verified':['causal bucket contributions','GPU factorial outcomes','actual training token stream','historical exposure capacity','optimal math share or order']}
save(A/'transfer_audit.json',audit)
print(json.dumps(audit,ensure_ascii=False,indent=2))
