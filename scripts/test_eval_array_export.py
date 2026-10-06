"""Synthetic supplied-array tests; no production forward/export or Hero arrays."""
import copy,json,pathlib,subprocess,sys,tempfile
import numpy as np
from export_eval_arrays import export,array_digest
from compare_eval_replays import compare
R=pathlib.Path(__file__).resolve().parents[1];checks=[]
def ck(n,v):assert v,n;checks.append(n)
identity={'checkpoint_sha256':'a'*64,'execution_sha':'b'*40,'parameter_view':'stored_params','pending_beta_policy':'stored_not_applied','compute_dtype':'bf16','backend':'synthetic','tokenizer_sha256':'c'*64}
a={'input_tokens':np.array([[1,2],[2,1]],np.int32),'scoring_token_ids':np.array([[2,0],[1,0]],np.int32),'losses':np.array([[2.,9.],[3.,8.]],np.float32),'weights':np.array([[1.,0.],[1.,0.]],np.float32),'tags':np.array([[1,0],[0,1]],np.int32),'bytes_per_token':np.array([0.,1.,3.],np.float32),'mask_doc_ids':np.array([[0,0],[1,1]],np.int32)}
with tempfile.TemporaryDirectory() as temp:
 p=pathlib.Path(temp);m={'identity':identity,'array_scope':'global','domains':['A','B'],'mask_spec':'synthetic document ids; explicit leaves, not GPU AttentionMask','batches':[{'batch':0,'path':'one.npz'}]}
 def run(arr,manifest=None):np.savez(p/'one.npz',**arr);return export(manifest or m,p)
 first=run(a);ck('Exporter computes expected global leaf N/T/B',[(r['weighted_loss_sum'],r['loss_weight_sum'],r['weighted_byte_sum']) for r in first['records']]==[(2.,1.,3.),(3.,1.,1.)])
 ck('Byte-table identity is computed from supplied typed bytes',len(first['byte_table_sha256'])==64 and first['array_export']['input_hash_verified_against_supplied_arrays'])
 same=run(a);ck('Identical arrays compare under matching identities',compare(first,same)['status']=='within_tolerance_under_matching_declarations')
 changed={k:v.copy() for k,v in a.items()};changed['losses'][0,0]=2.5;second=run(changed);ck('Changing loss keeps input digest but changes numeric observation',first['records'][0]['input_sha256']==second['records'][0]['input_sha256'] and compare(first,second)['status']=='numeric_difference_under_matching_declarations')
 for name in ['input_tokens','scoring_token_ids','weights','tags','mask_doc_ids']:
  changed={k:v.copy() for k,v in a.items()}
  if name=='tags':changed[name]=changed[name][::-1].copy()
  elif name=='weights':changed[name][0,0]=.5
  elif name=='mask_doc_ids':changed[name][0,1]=9
  else:changed[name][0,0]=1 if changed[name][0,0]==2 else 2
  second=run(changed);ck('Changing '+name+' changes supplied input digest',first['records'][0]['input_sha256']!=second['records'][0]['input_sha256'])
 ck('C versus Fortran memory layout hashes identically',array_digest({'x':a['weights']},{})==array_digest({'x':np.asfortranarray(a['weights'])},{}))
 ck('Byte order normalization preserves typed value digest',array_digest({'x':a['input_tokens']},{})==array_digest({'x':a['input_tokens'].astype('>i4')},{}))
 ck('Dtype width remains part of identity',array_digest({'x':a['input_tokens']},{})!=array_digest({'x':a['input_tokens'].astype(np.int64)},{}))
 ck('Shape remains part of identity',array_digest({'x':a['weights']},{})!=array_digest({'x':a['weights'].reshape(1,4)},{}))
 for label,change in [('out of range',lambda x:x['scoring_token_ids'].__setitem__((0,0),99)),('overlap tags',lambda x:x['tags'].__setitem__((0,1),1)),('untagged weights',lambda x:x['tags'].__setitem__(0,[0,0])),('NaN',lambda x:x['losses'].__setitem__((0,0),np.nan)),('negative weight',lambda x:x['weights'].__setitem__((0,0),-1)),('object mask',lambda x:x.update(mask_doc_ids=np.array([object()],object)))]:
  changed={k:v.copy() for k,v in a.items()};change(changed)
  try:run(changed)
  except (ValueError,TypeError):ck('Reject '+label,True)
  else:ck('Reject '+label,False)
 bad=copy.deepcopy(m);bad['array_scope']='local'
 try:run(a,bad)
 except ValueError:ck('Reject local scope declaration',True)
 else:ck('Reject local scope declaration',False)
 bad=copy.deepcopy(m);bad['identity']['byte_table_sha256']='0'*64
 try:run(a,bad)
 except ValueError:ck('Reject false byte table digest',True)
 else:ck('Reject false byte table digest',False)
 run(a);ip=p/'manifest.json';op=p/'output.json';ip.write_text(json.dumps(m));cmd=[sys.executable,str(R/'scripts/export_eval_arrays.py'),str(ip),str(op)];done=subprocess.run(cmd,capture_output=True);result=json.loads(op.read_text());ck('Actual CLI writes manifest and NPZ digests',done.returncode==0 and len(result['manifest_sha256'])==64 and len(result['array_export']['sources'][0]['npz_sha256'])==64);before=op.read_bytes();again=subprocess.run(cmd,capture_output=True);ck('Actual CLI refuses output overwrite',again.returncode!=0 and op.read_bytes()==before)
(R/'analysis/eval_array_export_validation.json').write_text(json.dumps({'scope':__doc__,'checks_passed':len(checks),'checks':checks,'synthetic_export':first,'actual_Hero_arrays':None,'actual_GPU_forward':None,'actual_rank_gather':None},ensure_ascii=False,indent=2)+'\n');print('Synthetic array-export checks:',len(checks))
