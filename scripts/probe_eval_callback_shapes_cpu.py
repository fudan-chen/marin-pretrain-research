"""Original evaluator on malformed synthetic callback arrays plus offline gate.
Uses V74 original accumulation/result construction with explicit hax/loader adapters.
No evidence of malformed Hero callbacks, GPU behavior or production fix.
"""
import hashlib,json
import numpy as np
import probe_tagged_eval_accumulator_cpu as base
from check_eval_callback_arrays import check_callback_arrays
R=base.R

def main():
 cases=[];checks=[]
 def check(name,condition):
  assert condition,(name,cases)
  checks.append(name)
 standard={'losses':[[1,1],[3,3]],'weights':[[1,1],[1,1]],'ids':[[0,1],[0,1]],'tags':[[1,0],[0,1]]}
 variants=[('valid',{}),('loss_column_broadcast',{'losses':[[1],[3]]}),('weight_column_broadcast',{'weights':[[1],[1]]}),('token_column_broadcast',{'ids':[[0],[0]]}),('tag_row_broadcast',{'tags':[[1,0]]}),('rank_one_loss',{'losses':[1,3]}),('incompatible_time_shape',{'losses':[[1,1,1],[3,3,3]]}),('multi_membership',{'tags':[[1,1],[0,1]]}),('positive_untagged_row',{'tags':[[1,0],[0,0]]}),('negative_token_ids',{'ids':[[-1,-1],[-1,-1]]}),('oversized_token_ids',{'ids':[[9,9],[9,9]]}),('swapped_tag_columns',{'tags':[[0,1],[1,0]]})]
 for name,patch in variants:
  x=standard|patch;b=base.batch(x['losses'],x['weights'],x['ids'],x['tags'])
  row={'case':name,'shapes':{k:list(np.asarray(v).shape) for k,v in x.items()},'source_result':None,'source_error':None,'offline_gate':None,'offline_gate_error':None}
  try:row['source_result']=base.run([b])
  except Exception as e:row['source_error']=type(e).__name__+': '+str(e)
  try:row['offline_gate']=check_callback_arrays(*b[0],b[1],[1.,3.],num_tags=2,tag_policy='exclusive_leaf')
  except ValueError as e:row['offline_gate_error']=str(e)
  cases.append(row)
 by={x['case']:x for x in cases};v=by['valid']['source_result']
 check('Exact valid arrays preserve source mean',v['micro_CE']==v['parent_micro_CE']==2 and by['valid']['offline_gate_error'] is None)
 for n in ['loss_column_broadcast','weight_column_broadcast','token_column_broadcast','tag_row_broadcast']:
  check(n+' silently accepted by source but rejected offline',by[n]['source_error'] is None and by[n]['offline_gate_error'] is not None)
 check('Weight broadcast repeats numerator without expanding original denominator',by['weight_column_broadcast']['source_result']['micro_CE']==4)
 check('Token-column broadcast changes BPB with unchanged CE',by['token_column_broadcast']['source_result']['micro_CE']==2 and abs(by['token_column_broadcast']['source_result']['micro_BPB']-2*v['micro_BPB'])<1e-6)
 check('Rank-only source check rejects rank-one callback',by['rank_one_loss']['source_error'].startswith('ValueError: Expected batched eval tensors with rank 2'))
 check('Nonbroadcastable dimensions fail in backend',by['incompatible_time_shape']['source_error'] is not None)
 # Original domain tag construction explicitly supports multiple membership.
 cl=next(n for n in base.tree.body if getattr(n,'name',None)=='DomainTaggedDataset')
 method=next(n for n in cl.body if getattr(n,'name',None)=='_compute_tag_arrays');base.execute([method],base.E)
 fixture=base.types.SimpleNamespace(datasets=[(None,['paloma/A','paloma/B']),(None,['paloma/B'])],tag_to_index={'paloma/A':0,'paloma/B':1},num_tags=2)
 tags=np.asarray(base._compute_tag_arrays(fixture));allowed=check_callback_arrays(*base.batch(standard['losses'],standard['weights'],standard['ids'],tags)[0],tags,[1.,3.],num_tags=2,tag_policy='overlapping_tags')
 check('Original tag builder deliberately emits multihot membership',tags.tolist()==[[1,1],[0,1]] and allowed['tag_mass']==[2,4])
 check('Multitag parent pool differs from root pooled targets',abs(by['multi_membership']['source_result']['parent_micro_CE']-5/3)<1e-6 and by['multi_membership']['source_result']['micro_CE']==2)
 check('Exclusive exporter gate must reject multi membership',by['multi_membership']['offline_gate_error'] is not None)
 check('Positive untagged row hides from parent but remains in root',by['positive_untagged_row']['source_result']['micro_CE']==2 and by['positive_untagged_row']['source_result']['parent_micro_CE']==1 and by['positive_untagged_row']['offline_gate_error'] is not None)
 for n in ['negative_token_ids','oversized_token_ids']:
  check(n+' gather yields finite wrong-byte denominator without source error',by[n]['source_error'] is None and np.isfinite(by[n]['source_result']['micro_BPB']) and by[n]['offline_gate_error'] is not None)
 check('Shape gate cannot prove ordered tag identity',by['swapped_tag_columns']['offline_gate_error'] is None and by['swapped_tag_columns']['source_result']['leaf_CE']['paloma/A']==3 and v['leaf_CE']['paloma/A']==1)
 files=[base.E,base.S,R/'scripts/probe_tagged_eval_accumulator_cpu.py',R/'scripts/check_eval_callback_arrays.py']
 o={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'cases':cases,'original_multitag_arrays':tags.tolist(),'overlap_gate_receipt':allowed,'explicit_substitutions':['V74 original-method extraction and same CPU loader/loss/hax adapters','synthetic malformed callback arrays; no model forward','offline structural gate is report-side code, not upstream patch'],'runtime':{'jax':base.jax.__version__,'numpy':np.__version__},'actual_Hero_malformed_callback':None,'actual_GPU_execution':None,'actual_upstream_fix':None,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
 (R/'analysis/eval_callback_shapes_cpu.json').write_text(json.dumps(o,indent=2,allow_nan=False,default=lambda x:x.item())+'\n');print('Original callback controls:',len(cases),'checks:',len(checks))
if __name__=='__main__':main()
