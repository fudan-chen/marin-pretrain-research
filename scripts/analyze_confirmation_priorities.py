"""Archived three-seed endpoint sensitivity; descriptive, selected candidate, no new training."""
import csv,hashlib,json,pathlib,statistics
R=pathlib.Path(__file__).resolve().parents[1];P=R/'analysis/strong_baseline_paired.csv';C=R/'analysis/strong_baseline_comparison.csv'
rows=list(csv.DictReader(P.open()));summary={r['metric']:r for r in csv.DictReader(C.open())}
metrics={'macro':'eval_dropless/paloma/macro_bpb','code_text':'eval_dropless/paloma/dolma_100_programing_languages-llama3/bpb','humaneval_target':'logprob_humaneval_10shot','gsm8k_target':'logprob_gsm8k_5shot'}
checks=[];out={}
def ck(name,c):
 if not c:raise RuntimeError(name)
 checks.append({'name':name,'passed':True})
def contrast(rs,base):
 selected=statistics.mean(float(r['selected_bpb']) for r in rs);baseline=statistics.mean(float(r[base+'_bpb']) for r in rs)
 return {'n':len(rs),'selected_mean_bpb':selected,'baseline_mean_bpb':baseline,'absolute_delta':selected-baseline,'relative_pct':100*(selected/baseline-1)}
for label,key in metrics.items():
 rs=sorted([r for r in rows if r['metric']==key],key=lambda r:int(r['seed']))
 ck(label+' three distinct archived continuation labels',len(rs)==3 and [int(r['seed']) for r in rs]==[0,1,2])
 allpairs=contrast(rs,'proportional');ck(label+' aggregate independently matches archived summary',abs(allpairs['relative_pct']-float(summary[key]['selected_vs_proportional_pct']))<1e-10)
 loo={str(drop):{base:contrast([r for r in rs if int(r['seed'])!=drop],base) for base in ['proportional','old']} for drop in [0,1,2]}
 out[label]={'metric':key,'all_pairs':{base:contrast(rs,base) for base in ['proportional','old']},'per_seed_pct':{r['seed']:float(r['selected_vs_proportional_pct']) for r in rs},'leave_one_seed_out':loo,'loo_proportional_pct_range':[min(x['proportional']['relative_pct'] for x in loo.values()),max(x['proportional']['relative_pct'] for x in loo.values())]}
ck('math pooled direction changes under leave-one-out against proportional',out['gsm8k_target']['loo_proportional_pct_range'][0]<0<out['gsm8k_target']['loo_proportional_pct_range'][1])
ck('math against old remains worse in all selected leave-one-out subsets',all(x['old']['relative_pct']>0 for x in out['gsm8k_target']['leave_one_seed_out'].values()))
ck('macro and code targets improve in all selected leave-one-out subsets',all(v['loo_proportional_pct_range'][1]<0 for k,v in out.items() if k!='gsm8k_target'))
result={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'metrics':out,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [P,C]},'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'selection_bias_corrected':False,'confidence_intervals_computed':False,'actual_new_training_runs':0,'actual_generated_task_accuracy':None,'actual_535B_transfer_effect':None}
(R/'analysis/confirmation_priorities.json').write_text(json.dumps(result,indent=2)+'\n');print('Archived sensitivity controls:',len(checks))
