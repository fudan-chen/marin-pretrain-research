"""Execute original restore_weights consumer with real temporary JSON IO and explicit
layout/array-loader/digest/template/tree stubs. No actual checkpoint arrays or OCDBT.
"""
import ast,hashlib,json,logging,pathlib,tempfile,types
from typing import cast
import jax,jax.numpy as jnp
from probe_pending_router_view_cpu import model,apply
R=pathlib.Path(__file__).resolve().parents[1];P=R/'sources/state_2026_10_05/weights.py';CP=R/'sources/contracts_2026_10_05/checkpointing.py'
node=next(n for n in ast.parse(P.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='restore_weights')
master_node=next(n for n in ast.parse(CP.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='checkpoint_stores_master')
class Transformer:
 init=None
def diagnostic_digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def run(name,temporary=False,manifest_master=False,no_manifest=False,marker=False,digest_mismatch=False,missing_load=False,bad_manifest=False):
 calls=[]
 with tempfile.TemporaryDirectory(prefix='marin-weight-consumer-') as tmp:
  root=pathlib.Path(tmp);md={} if temporary=='missing' else {'is_temporary':temporary};(root/'metadata.json').write_text(json.dumps(md))
  if marker:(root/'master_params').mkdir()
  def read_manifest(_):
   calls.append('manifest')
   if bad_manifest:raise ValueError('synthetic malformed manifest')
   return None if no_manifest else types.SimpleNamespace(array_paths=['master_params/token_embed'] if manifest_master else ['params/token_embed'])
  mn={'read_manifest':read_manifest,'MASTER_PARAMS_KEY':'master_params','LEGACY_STATE_KEY':'state'};exec(compile(ast.Module(body=[master_node],type_ignores=[]),str(CP),'exec'),mn)
  def load(**kwargs):
   calls.append({'requested_keys':sorted(kwargs['state']),'allow_partial':kwargs['allow_partial'],'candidate_matches':kwargs['candidate']==tmp})
   if missing_load:raise FileNotFoundError('synthetic missing array')
   key=next(k for k in kwargs['state'] if k!='pending_qb_betas')
   return {key:model(jnp.zeros((1,3))), 'pending_qb_betas':jnp.array([[0.,3.,0.]])}
  cp=types.SimpleNamespace(checkpoint_stores_master=mn['checkpoint_stores_master'],MASTER_PARAMS_KEY='master_params',LEGACY_STATE_KEY='state',load_grug_checkpoint=load)
  env={'StoragePath':pathlib.Path,'json':json,'digest':diagnostic_digest,'eqx':types.SimpleNamespace(filter_eval_shape=lambda *a,**kw:model(jnp.zeros((1,3)))),'Transformer':Transformer,'GrugModelConfig':object,'jax':jax,'jnp':jnp,'checkpointing':cp,'PENDING_QB_BETAS_KEY':'pending_qb_betas','cast':cast,'apply_qb_betas':apply,'logger':logging.getLogger('consumer_probe')}
  exec(compile(ast.Module(body=[node],type_ignores=[]),str(P),'exec'),env)
  try:
   result=env['restore_weights'](tmp,'wrong' if digest_mismatch else diagnostic_digest(md),types.SimpleNamespace(num_layers=1,num_experts=3),None)
   bias=result.stacked_blocks.stacked.mlp.router_bias.tolist();error=None
  except (ValueError,FileNotFoundError) as e:bias=None;error=type(e).__name__
  return {'name':name,'metadata':md,'calls':calls,'error':error,'returned_bias':bias}
def main():
 cases=[run('permanent_master',manifest_master=True),run('permanent_params'),run('temporary',temporary=True),run('missing_permanent_flag',temporary='missing'),run('integer_zero_is_not_false',temporary=0),run('digest_mismatch',digest_mismatch=True),run('legacy_local_master_marker',no_manifest=True,marker=True),run('manifest_params_overrides_stale_local_master_marker',marker=True),run('load_missing_array',missing_load=True),run('malformed_manifest',bad_manifest=True)]
 checks=[]
 def check(n,v):assert v,n;checks.append(n)
 for i in [0,1,6,7]:check(cases[i]['name']+' requests expected authoritative view',cases[i]['error'] is None and cases[i]['calls'][1]['requested_keys']==sorted(['master_params' if i in [0,6] else 'params','pending_qb_betas']) and cases[i]['returned_bias']==[[1.,-2.,1.]])
 check('Temporary/missing/integer-zero metadata rejected before layout read',all(cases[i]['error']=='ValueError' and not cases[i]['calls'] for i in [2,3,4]))
 check('Digest mismatch rejected before layout read',cases[5]['error']=='ValueError' and not cases[5]['calls'])
 check('Every reached load forbids partial and binds original candidate',all(c['allow_partial'] is False and c['candidate_matches'] for r in cases for c in r['calls'] if isinstance(c,dict)))
 check('Consumer does not hide propagated missing-array failure',cases[8]['error']=='FileNotFoundError' and len(cases[8]['calls'])==2)
 check('Malformed manifest error is not treated as manifest absence',cases[9]['error']=='ValueError' and cases[9]['calls']==['manifest'])
 out={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'cases':cases,'stubs':['metadata digest algorithm is diagnostic SHA256, not original digest','read_manifest/layout metadata','load_grug_checkpoint array loader','filter_eval_shape Transformer template','eqx tree_at from prior explicit stub'],'actual_checkpoint_array_IO':None,'actual_OCDBT_execution':None,'actual_production_recovery':None,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [P,CP,R/'sources/contracts_2026_10_05/model.py']}}
 (R/'analysis/weights_consumer_faults.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('Weight consumer fault-path checks:',len(checks),'cases:',len(cases))
if __name__=='__main__':main()
