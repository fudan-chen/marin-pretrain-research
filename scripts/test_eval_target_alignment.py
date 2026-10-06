"""Synthetic optional causal target checks, not real loss or Hero arrays."""
import ast,asyncio,copy,hashlib,json,pathlib,tempfile,types
import numpy as np
from export_eval_arrays import export
R=pathlib.Path(__file__).resolve().parents[1];checks=[]
def ck(n,v):assert v,n;checks.append(n)
m={'identity':{'checkpoint_sha256':'a'*64,'execution_sha':'b'*40,'parameter_view':'stored_params','pending_beta_policy':'stored_not_applied','compute_dtype':'bf16','backend':'synthetic','tokenizer_sha256':'c'*64},'array_scope':'global','domains':['A'],'mask_spec':'synthetic EOS segments only','target_contract':'causal_next_token_v1','batches':[{'batch':0,'path':'one.npz'}]}
a={'input_tokens':np.array([[1,2,0,3]],np.int32),'scoring_token_ids':np.array([[2,0,3,1]],np.int32),'losses':np.array([[2,3,4,9]],np.float32),'weights':np.array([[1,1,1,0]],np.float32),'tags':np.array([[1]],np.int32),'bytes_per_token':np.array([0,1,2,3],np.float32),'mask_doc_ids':np.array([[0,0,0,1]],np.int32)}
with tempfile.TemporaryDirectory() as temp:
 p=pathlib.Path(temp)
 def run(x=a,j=m):np.savez(p/'one.npz',**x);return export(j,p)
 result=run();ck('Valid active next-token alignment is explicitly verified',result['array_export']['active_next_token_alignment_verified'] is True)
 ck('Synthetic leaf N T B uses shifted targets and active weights',[(x['weighted_loss_sum'],x['loss_weight_sum'],x['weighted_byte_sum']) for x in result['records']]==[(9.,3.,5.)])
 x=copy.deepcopy(a);x['scoring_token_ids'][0,-1]=0;other=run(x);ck('Zero-weight wrapped id may differ without changing N T B',[(r['weighted_loss_sum'],r['loss_weight_sum'],r['weighted_byte_sum']) for r in other['records']]==[(9.,3.,5.)])
 def reject(name,x,j=m):
  try:run(x,j)
  except ValueError:ck(name,True)
  else:ck(name,False)
 x=copy.deepcopy(a);x['scoring_token_ids'][0,0]=1;reject('Reject active input-id rather than successor-id export',x)
 x=copy.deepcopy(a);x['weights'][0,-1]=1;reject('Reject positive wrap-around final weight',x)
 x=copy.deepcopy(a);x['scoring_token_ids']=x['scoring_token_ids'][:,:3];x['weights']=x['weights'][:,:3];x['losses']=x['losses'][:,:3];reject('Reject unequal input/scored lengths under causal contract',x)
 j=copy.deepcopy(m);j['target_contract']='unknown';reject('Reject unknown target contract',a,j)
 j=copy.deepcopy(m);j['target_contract']=True;reject('Reject boolean target contract',a,j)
 x=copy.deepcopy(a);x['weights'][0,0]=.25;ck('Fractional positive weights still require next-token alignment',run(x)['records'][0]['loss_weight_sum']==2.25)
 x['scoring_token_ids'][0,0]=1;reject('Reject shifted error even at fractional positive weight',x)
 x=copy.deepcopy(a);x['weights'][0,0]=0;x['scoring_token_ids'][0,0]=1;ck('Zero-weight positions need no active-target alignment',run(x)['array_export']['active_next_token_alignment_verified'])
 j=copy.deepcopy(m);j.pop('target_contract');x=copy.deepcopy(a);x['scoring_token_ids'][0,0]=1;ck('Generic export remains explicit about unverified alignment',run(x,j)['array_export']['active_next_token_alignment_verified'] is False)
 # Static source binding: actual backend and model computation remain unexecuted.
 src=R/'sources/scale_2026_10_05/eval.py';t=ast.parse(src.read_text());f=next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name=='_default_lm_eval_loss_fn');roll=next(x for x in ast.walk(f) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='roll');ck('Archived default evaluator explicitly rolls targets left by one',ast.literal_eval(roll.args[1])==-1 and ast.literal_eval(next(k.value for k in roll.keywords if k.arg=='axis'))==-1)
# Execute the original mask and length bodies using NumPy/fake length metadata.
env={'jnp':np}
def install(path,cls,name):
 t=ast.parse(path.read_text());c=next(x for x in t.body if isinstance(x,ast.ClassDef) and x.name==cls);f=next(x for x in c.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name==name);f.decorator_list=[]
 mod=ast.Module(body=ast.parse('from __future__ import annotations').body+[f],type_ignores=[]);exec(compile(ast.fix_missing_locations(mod),str(path),'exec'),env);return env[name]
mask=install(R/'sources/boundaries_2026_10_05/examples.py','GrugLmExample','causal_loss_mask');length=install(R/'sources/deepening_2026_10_04/datasets_production.py','TokenSeqDataset','async_len')
coverage=[]
for M,L in [(3,4),(9,4),(8,4),(409617,4096),(7,1)]:
 class Cache:
  async def async_flat_field_length(self,key):return M
 n=asyncio.run(length(types.SimpleNamespace(doc_cache=Cache(),seq_len=L)));T=int(n*mask(L).sum());r=M%L
 ck('Original default target coverage M=%d L=%d'%(M,L),T==n*(L-1) and M-T==r+n)
 coverage.append({'cached_input_positions':M,'window_length':L,'full_windows':n,'remainder':r,'default_unit_weight_targets':T,'positions_not_scored_as_targets':M-T})
(R/'analysis/eval_target_alignment_validation.json').write_text(json.dumps({'checks_passed':len(checks),'checks':checks,'synthetic_export':result,'synthetic_coverage':coverage,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'sources/scale_2026_10_05/eval.py',R/'sources/boundaries_2026_10_05/examples.py',R/'sources/deepening_2026_10_04/datasets_production.py']},'actual_Hero_arrays':None,'actual_GPU_forward':None,'actual_loss_array_alignment':None},ensure_ascii=False,indent=2)+'\n');print('Synthetic target alignment checks:',len(checks))
