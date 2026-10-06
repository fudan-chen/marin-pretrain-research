"""Original Grug state schema and restore policy with small synthetic payloads.
Real local IO and Optax state; not a Transformer, GPU state, or production restore.
"""
import ast,dataclasses,hashlib,json,pathlib,tempfile
from jax.tree_util import register_dataclass,SequenceKey,DictKey,GetAttrKey,FlattenedIndexKey
import jax,jax.numpy as jnp,numpy as np,optax
import probe_tree_restore_contracts as base
R=base.R;ns=base.ns;T=R/'sources/scale_2026_10_05/train_hero_ep.py';G=R/'sources/contracts_2026_10_05/checkpointing.py';U=R/'sources/state_restore_2026_10_07/tree_utils.py'
ns.update(globals());ns['StrEnum']=base.base.enum.StrEnum;ns['fields']=dataclasses.fields
base.extract(U,['tree_flatten_one_level_with_keys','key_path_to_str'])
base.extract(T,['GrugTrainState','MasterParamMode','template_for_candidate_layout','take_master_as_params'])
base.extract(base.base.P,['_flatten_serializable_leaves'])
ns.update(LEGACY_STATE_KEY='train_state',MASTER_PARAMS_KEY='master_params',RESTORE_COMPLETE_BARRIER='grug_checkpoint_restore_complete',RESTORE_BARRIER_TIMEOUT=2400)
barriers=[];ns['barrier_sync_named']=lambda *args,**kwargs:barriers.append({'args':args,'kwargs':kwargs})
base.extract(G,['_get_fs_and_plain_path','_scan_checkpoint_root','_checkpoint_candidates','checkpoint_stores_master','load_grug_checkpoint','restore_grug_state_from_checkpoint'])
State=ns['GrugTrainState'];actual_load=ns['load_checkpoint']
def main():
 checks=[];rows={};calls=[];w=jnp.array([1.25,-2.5,3.75],dtype=jnp.float32);tx=optax.scale_by_adam();opt=tx.init({'w':w});_,opt=tx.update({'w':w*.1},opt,{'w':w})
 def state(step,master=None):return State(step=jnp.array(step,dtype=jnp.int32),params={'w':w},master_params=master,opt_state=opt,ema_params=None,pending_qb_betas=jnp.array([[1.,2.,3.],[4.,5.,6.]]))
 def check(n,c):assert c,n;checks.append(n)
 def save(root,s,omit=None,absent=None):
  root.mkdir(parents=True);paths,arrays=ns['_flatten_serializable_leaves'](s);entries=[ns['CheckpointArray'](path=p,shape=tuple(a.shape),dtype=np.dtype(a.dtype).name,chunk_shape=tuple(a.shape)) for p,a in zip(paths,arrays) if p!=omit];ns['write_manifest'](str(root),ns['build_manifest'](entries,array_driver='zarr3',kvstore_driver='ocdbt'))
  manager=base.base.writer.array_ser.GlobalAsyncCheckpointManager();included=[(p,a) for p,a in zip(paths,arrays) if p not in [omit,absent]];ns['_serialize_arrays']([np.array(a) for _,a in included],[ns['_create_ocdbt_spec'](str(root),p,entry=next(e for e in entries if e.path==p)) for p,_ in included],[None]*len(included),manager,ns['TensorStoreWriteConfig'](max_staged_host_bytes=128),lambda:None,None,None);manager.wait_until_finished();(root/'metadata.json').write_text(json.dumps({'step':int(s.step),'timestamp':'2026-10-07T00:00:00+00:00','is_temporary':False}));return paths
 def load(exemplar,path,**kwargs):
  event={'candidate':pathlib.Path(path).name,'wrapped':isinstance(exemplar,dict) and 'train_state' in exemplar};calls.append(event)
  try:return actual_load(exemplar,path,read_config=ns['TensorStoreReadConfig'](replica_mode=ns['ReplicaRestoreMode'].EVERY_REPLICA),**kwargs)
  except Exception as e:event['error']=type(e).__name__+': '+str(e);raise
 def restore(root,template):return ns['restore_grug_state_from_checkpoint'](template,checkpoint_search_paths=[str(root)],load_checkpoint_setting=True,mesh=None,allow_partial=False,_load_fn=load)
 with tempfile.TemporaryDirectory(prefix='marin-grug-state-') as tmp:
  root=pathlib.Path(tmp);template=state(0);complete=root/'complete';paths=save(complete/'step10',state(10));loaded=restore(complete,template)
  check('Original Grug schema roundtrips all serialized leaves',all(np.array_equal(a,b) for a,b in zip(jax.tree.leaves(loaded),jax.tree.leaves(state(10)))));rows['complete']={'step':int(loaded.step),'adam_count':int(loaded.opt_state.count),'paths':paths,'calls':calls.copy()}
  check('Original Grug schema includes raw step pending and optimizer count',{'step','pending_qb_betas','opt_state/count','opt_state/mu/w','opt_state/nu/w','params/w'}==set(paths))
  calls.clear();fallback=root/'fallback';save(fallback/'step10',state(10));save(fallback/'step20',state(20),omit='opt_state/nu/w');loaded=restore(fallback,template);rows['missing_manifest_leaf']={'step':int(loaded.step),'calls':calls.copy()}
  check('Leaf absent from manifest triggers original older-candidate fallback',int(loaded.step)==10 and [x['candidate'] for x in calls]==['step20','step20','step10'])
  check('Original missing-leaf policy attempts legacy wrapper at same candidate',calls[1]['wrapped'])
  check('Legacy retry error differs from first missing-leaf evidence','Missing 1 arrays' in calls[0]['error'] and 'Missing 6 arrays' in calls[1]['error'])
  calls.clear();failure=root/'failure';save(failure/'step10',state(10));save(failure/'step20',state(20),absent='opt_state/nu/w')
  try:restore(failure,template);error=None
  except Exception as e:error=type(e).__name__+': '+str(e)
  rows['manifest_listed_array_absent']={'error':error,'calls':calls.copy()}
  check('Real listed-array NOT_FOUND remains ValueError and aborts fallback',error is not None and error.startswith('ValueError:') and 'NOT_FOUND' in error and len(calls)==1 and calls[0]['candidate']=='step20')
  calls.clear();migration=root/'migration';master={'w':w+.125};source=dataclasses.replace(state(20,master),params={'w':w+1000});save(migration/'step20',source)
  raw=restore(migration,template);check('Without layout hook raw device template reads compute params',raw.master_params is None and np.array_equal(raw.params['w'],w+1000))
  loaded=ns['restore_grug_state_from_checkpoint'](template,checkpoint_search_paths=[str(migration)],load_checkpoint_setting=True,mesh=None,allow_partial=False,_load_fn=load,template_for_candidate=lambda p:ns['template_for_candidate_layout'](template,p,ns['MasterParamMode'].DEVICE));loaded=ns['take_master_as_params'](loaded)
  check('Original layout hook plus migration selects stored master',loaded.master_params is None and np.array_equal(loaded.params['w'],master['w']) and int(loaded.step)==20)
  rows['migration']={'compute_values':np.asarray(raw.params['w']).tolist(),'migrated_values':np.asarray(loaded.params['w']).tolist(),'adam_count':int(loaded.opt_state.count)}
  check('Barrier adapter only reached successful policy restores',len(barriers)==4)
 model=R/'sources/contracts_2026_10_05/model.py'
 def declarations(p,name):
  c=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.ClassDef) and n.name==name);return {n.target.id:ast.unparse(n.annotation) for n in c.body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name)}
 o={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'observations':rows,'declared_fields_not_runtime_inventory':{'GrugTrainState':declarations(T,'GrugTrainState'),'Transformer':declarations(model,'Transformer')},'runtime':{'jax':jax.__version__,'optax':optax.__version__},'original_GrugTrainState_class':True,'original_restore_policy':True,'explicit_substitutions':{'params':'one-array dictionary instead of Transformer','opt_state':'real scale_by_adam fixture, not Hero optimizer groups','barrier':'recording adapter, not collective','save_fixture':'original flatten and host writer; arrays converted to NumPy, markers manually written','sharding_and_StoragePath':'inherited single-device/local adapters'},'actual_Hero_state_inventory':None,'actual_production_missing_array_event':None,'actual_multirank_restore':None,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [T,G,U,base.base.P,base.base.M,model,base.D/'jax_utils.py']}}
 (R/'analysis/grug_state_restore_real_io.json').write_text(json.dumps(o,indent=2)+'\n');print('Grug state restore real IO checks:',len(checks))
if __name__=='__main__':main()
