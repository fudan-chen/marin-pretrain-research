"""Descriptive seed-label paired endpoint audit; not independent post-selection inference."""
import csv,hashlib,itertools,json,pathlib,re
import numpy as np
R=pathlib.Path(__file__).resolve().parents[1];P=R/'sources/mix_study_swarm-2026.09.14.1_selected_runs.json';C=R/'sources/mix_study_swarm-2026.09.14.1_comparison.csv';I=R/'sources/issue_9126.json'
def main():
 d=json.loads(P.read_text());groups={g:{int(re.search(r'seed(\d+)',r['run_name']).group(1)):r for r in rows} for g,rows in d.items()};published={r['metric']:r for r in csv.DictReader(C.open())};checks=[]
 def check(n,v):assert v,n;checks.append(n)
 check('Six unique observations have matching parsed seed labels',set(groups)=={'old','new'} and all(set(x)=={0,1,2} for x in groups.values()) and len({r['observation_id'] for rows in d.values() for r in rows})==6)
 check('Within-group phase weights identical across seed-labelled runs',all(all(groups[g][s][phase]==groups[g][0][phase] for s in [1,2]) for g in groups for phase in ['phase0_weights','phase1_weights']))
 keys=sorted(k for k in groups['old'][0]['training_eval_metrics'] if k.startswith('eval_dropless/') and (k.endswith('/macro_bpb') or (k.endswith('/bpb') and len(k.split('/'))==4)))
 check('25 dropless leaf/macro endpoints present on all six records',len(keys)==25 and all(all(k in r['training_eval_metrics'] for k in keys) for rows in d.values() for r in rows))
 rows=[]
 for k in keys:
  old=np.array([groups['old'][s]['training_eval_metrics'][k] for s in [0,1,2]]);new=np.array([groups['new'][s]['training_eval_metrics'][k] for s in [0,1,2]]);delta=new-old
  rows.append({'metric':k,'old_mean':float(old.mean()),'new_mean':float(new.mean()),'relative_mean_change_pct':float((new.mean()/old.mean()-1)*100),'paired_deltas':delta.tolist(),'paired_change_pct':((new/old-1)*100).tolist(),'paired_delta_std_ddof1':float(delta.std(ddof=1)),'improved_seed_labels':int(sum(delta<0)),'regressed_seed_labels':int(sum(delta>0)),'leave_one_pair_out_mean_delta':[float(np.delete(delta,s).mean()) for s in [0,1,2]],'old_run_names':[groups['old'][s]['run_name'] for s in [0,1,2]],'new_run_names':[groups['new'][s]['run_name'] for s in [0,1,2]]})
 check('All means reconstruct author comparison CSV',all(abs(r['old_mean']-float(published[r['metric']]['old_mean']))<1e-12 and abs(r['new_mean']-float(published[r['metric']]['new_mean']))<1e-12 for r in rows))
 by={r['metric']:r for r in rows};macro=by['eval_dropless/paloma/macro_bpb'];code=by['eval_dropless/paloma/dolma_100_programing_languages-llama3/bpb'];un=by['eval_dropless/uncheatable_eval/macro_bpb']
 check('Paloma/code/Uncheatable macro improve in all three parsed pairs',all(r['improved_seed_labels']==3 for r in [macro,code,un]))
 bad=[r for r in rows if r['regressed_seed_labels']==3];check('Two endpoints consistently regress across parsed pairs',len(bad)==2 and set(r['metric'] for r in bad)=={'eval_dropless/paloma/manosphere_meta_sep-llama3/bpb','eval_dropless/uncheatable_eval/bbc_news-llama3/bpb'})
 check('Leave-one-pair-out macro differences remain negative',all(x<0 for r in [macro,un] for x in r['leave_one_pair_out_mean_delta']))
 # Synthetic positive magnitudes: resolution example only, no selected-mixture p-value.
 magnitudes=np.array([1.,2.,3.]);signs=list(itertools.product([-1,1],repeat=3));stats=[abs(float(np.mean(magnitudes*np.array(s)))) for s in signs];floor=sum(x>=max(stats) for x in stats)/len(stats)
 check('Three-pair synthetic two-sided sign-flip resolution is one quarter',len(signs)==8 and floor==.25)
 body=json.loads(I.read_text())['body'];check('Author explicitly records checkpoint and nonindependent selection','step-393' in body and 'not independent validation' in body)
 summary={'metrics':25,'all_three_improve':sum(r['improved_seed_labels']==3 for r in rows),'all_three_regress':len(bad),'mixed_direction':sum(r['improved_seed_labels']>0 and r['regressed_seed_labels']>0 for r in rows)}
 with (R/'analysis/swarm_seed_pairs.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['metric','seed_label','old_bpb','new_bpb','delta','relative_change_pct']);w.writeheader()
  for r in rows:
   for s in [0,1,2]:w.writerow({'metric':r['metric'],'seed_label':s,'old_bpb':groups['old'][s]['training_eval_metrics'][r['metric']],'new_bpb':groups['new'][s]['training_eval_metrics'][r['metric']],'delta':r['paired_deltas'][s],'relative_change_pct':r['paired_change_pct'][s]})
 o={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'pairing':'Parsed seed labels in run names; actual common random stream and complete execution identity not verified.','summary':summary,'endpoints':rows,'synthetic_sign_flip_resolution':{'magnitudes':[1,2,3],'patterns':8,'two_sided_min_probability':floor,'not_a_selected_mixture_p_value':True},'post_selection_p_value':None,'actual_independent_confirmation':None,'actual_Hero_causal_effect':None,'actual_matched_runtime_savings':None,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [P,C,I]}}
 (R/'analysis/swarm_seed_pairs.json').write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'checks_passed':len(checks),'summary':summary,'key_results':[{k:r[k] for k in ['metric','relative_mean_change_pct','paired_change_pct']} for r in [macro,code,un]+bad]},indent=2))
if __name__=='__main__':main()
