"""Census declared configuration differences; no historical equivalence claim."""
import hashlib,json,pathlib
R=pathlib.Path(__file__).resolve().parents[1];S=R/'sources/scale_2026_10_05'
PAIRS=[('rav-ladder-d768-v2','h100-mix25-20260912-d768'),('rav-ladder-d1024','h100-mix25-20260912-d1024'),('rav-ladder-d1536','h100-mix25-20260912-d1536')]
def digest(obj):return hashlib.sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def differences(a,b,path=''):
 if isinstance(a,dict) and isinstance(b,dict):
  rows=[]
  for k in sorted(set(a)|set(b)):
   p=path+'.'+k if path else k
   if k not in a or k not in b:rows.append(dict(path=p,old_present=k in a,new_present=k in b,old=a.get(k),new=b.get(k)))
   else:rows+=differences(a[k],b[k],p)
  return rows
 if type(a) is not type(b) or a!=b:return [dict(path=path,old_present=True,new_present=True,old=a,new=b)]
 return []
records=[];source_hash={}
for old,new in PAIRS:
 configs=[];observations=[]
 for name in [old,new]:
  p=S/('config_'+name+'.json');raw=p.read_bytes();source_hash[str(p.relative_to(R))]=hashlib.sha256(raw).hexdigest();run=json.loads(raw)['data']['project']['run'];configs.append({k:v['value'] for k,v in json.loads(run['config']).items() if 'value' in v});summary=json.loads(run['summaryMetrics']);observations.append(dict(run=name,state=run['state'],logged_step=summary.get('_step'),logged_total_tokens=summary.get('throughput/total_tokens')))
 a,b=configs;groups={}
 for key in ['model','optimizer','eval','trainer','resources']:
  d=differences(a[key],b[key],key);groups[key]=dict(old_sha256=digest(a[key]),new_sha256=digest(b[key]),declared_equal=not d,differences=d)
 cache_roots=[]
 for c in configs:
  roots=sorted({v['cache_dir'].rsplit('/cluster=',1)[0] for k,v in c['data']['components'].items() if k.startswith('c') and '/cluster=' in (v.get('cache_dir') or '')});cache_roots.append(roots)
 fields=['block_cross_document_attention','enforce_eos','experiment_budget','mixture_block_size','permutation_type','shuffle','shuffle_before_trainval_split','target_budget','tokenizer']
 data_diff=[dict(field=k,old=a['data'].get(k),new=b['data'].get(k),old_present=k in a['data'],new_present=k in b['data']) for k in fields if a['data'].get(k)!=b['data'].get(k) or (k in a['data'])!=(k in b['data'])]
 records.append(dict(old_run=old,new_run=new,groups=groups,cache_roots=cache_roots,mixture_sha256=[digest(c['data']['train_weights']) for c in configs],selected_data_differences=data_diff,observations=observations,assessment='observed_complete_recipe_comparison; pure_mixture_effect_not_identified',actual_code_sha=None,actual_start_state_digest=None,actual_cache_content_equivalence=None,actual_evaluation_equivalence=None))
assert len(records)==3 and all(r['groups']['model']['declared_equal'] for r in records)
out=dict(scope='all declared model/optimizer/eval/trainer/resources fields; selected data settings and cell cache roots; lists compared as whole values',pairs=records,source_sha256=source_hash,configuration_equal_is_execution_equal=False)
(R/'analysis/ladder_contract_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print('Audited 3 paired configurations; no actual execution equivalence asserted')
