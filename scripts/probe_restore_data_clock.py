"""Real original checkpoint policy joined to original loader/mixture methods.
Synthetic checkpoint markers and identity child stores; no Hero token stream or training step.
"""
import asyncio,dataclasses,hashlib,json,pathlib,tempfile,types
import jax,jax.numpy as jnp,numpy as np,optax
import probe_grug_state_restore_real_io as state_probe
import probe_loader_resume as loader
import probe_mixture_identity_cpu as mixture
R=state_probe.R;ns=state_probe.ns;P=mixture.P
state_probe.base.extract(P,['rescale_mixture_schedule_for_batch_schedule'])
def main():
 checks=[];rows={};w=jnp.array([1.,2.,3.]);opt=optax.scale_by_adam().init({'w':w});schedule=loader.Schedule([loader.Step(0,4),loader.Step(3,8)])
 def check(n,c):assert c,(n,rows);checks.append(n)
 def state(step):return state_probe.State(step=jnp.array(step,dtype=jnp.int32),params={'w':w},master_params=None,opt_state=opt,ema_params=None,pending_qb_betas=jnp.zeros((2,3)))
 def save(path,s,marker_step):
  path.mkdir(parents=True);paths,arrays=ns['_flatten_serializable_leaves'](s);entries=[ns['CheckpointArray'](path=p,shape=tuple(a.shape),dtype=np.dtype(a.dtype).name,chunk_shape=tuple(a.shape)) for p,a in zip(paths,arrays)];ns['write_manifest'](str(path),ns['build_manifest'](entries,array_driver='zarr3',kvstore_driver='ocdbt'))
  manager=state_probe.base.base.writer.array_ser.GlobalAsyncCheckpointManager();ns['_serialize_arrays']([np.asarray(a) for a in arrays],[ns['_create_ocdbt_spec'](str(path),p,entry=e) for p,e in zip(paths,entries)],[None]*len(arrays),manager,ns['TensorStoreWriteConfig'](max_staged_host_bytes=128),lambda:None,None,None);manager.wait_until_finished();(path/'metadata.json').write_text(json.dumps({'step':marker_step,'timestamp':'2026-10-07T00:00:00+00:00','is_temporary':False}))
 class Harness(loader.Harness):
  def __init__(self,bs,start):
   super().__init__(bs,start)
   stages=ns['rescale_mixture_schedule_for_batch_schedule']([(0,{'A':1.}),(21,{'B':1.})],bs)
   self.dl.data_store=mixture.Mix({n:mixture.Identity(n,10000) for n in ['A','B']},stages,4,key=7)
  def _batchify_local_data(self,b):return {'step':b.index,'offset':b.global_data_offset,'size':b.global_size,'identities':[b.data_by_local_index[i] for i in sorted(b.data_by_local_index)]}
 def batch(bs,step):return asyncio.run(loader.first(Harness(bs,step)))
 with tempfile.TemporaryDirectory(prefix='marin-clock-join-') as tmp:
  root=pathlib.Path(tmp);save(root/'old',state(90),90);save(root/'candidate',state(20),100);calls=[]
  def load(exemplar,path,**kwargs):calls.append(pathlib.Path(path).name);return state_probe.actual_load(exemplar,path,read_config=ns['TensorStoreReadConfig'](replica_mode=ns['ReplicaRestoreMode'].EVERY_REPLICA),**kwargs)
  restored=ns['restore_grug_state_from_checkpoint'](state(0),checkpoint_search_paths=[str(root)],load_checkpoint_setting=True,mesh=None,allow_partial=False,_load_fn=load)
  rows['checkpoint']={'marker_step':100,'loaded_state_step':int(restored.step),'other_candidate_marker_and_state_step':90,'selected_calls':calls}
  check('Original candidate selection follows marker rank not loaded state progress',calls==['candidate'] and int(restored.step)==20)
  actual=batch(schedule,int(restored.step));marker=batch(schedule,100);rewritten=batch(loader.Schedule(8),int(restored.step));boundary=batch(schedule,21)
  rows.update(restored_step_batch=actual,marker_step_counterfactual_batch=marker,rewritten_history_batch=rewritten,mixture_boundary_batch=boundary)
  check('Restored-state clock feeds original loader cumulative offset',actual['offset']==148 and actual['size']==8 and set(actual['identities'])=={'A:'+str(i) for i in range(148,156)})
  check('Marker-clock counterfactual changes phase and sample identities',marker['offset']==788 and all(x.startswith('B:') for x in marker['identities']))
  check('Original schedule conversion aligns phase at restored next boundary',boundary['offset']==156 and set(boundary['identities'])=={'B:'+str(i) for i in range(8)})
  check('Rewriting history changes sample identity at same loaded step',rewritten['offset']==160 and set(rewritten['identities'])=={'A:'+str(i) for i in range(160,168)} and rewritten['identities']!=actual['identities'])
  check('Same original schedule and mixture key reproduce ordered identities',batch(schedule,20)==actual)
  check('Host-state marker mismatch should fail independent identity audit',rows['checkpoint']['marker_step']!=rows['checkpoint']['loaded_state_step'])
 o={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'observations':rows,'metadata_state_guard':'audit mismatch demonstrated; upstream policy not modified','training_entry_source_binding':{'file':str(state_probe.T.relative_to(R)),'line':1101,'expression':'train_loader.iter_from_step(int(state.step))','executed_full_training_entry':False},'explicit_substitutions':{'metadata':'deliberately mismatched manual marker','params':'synthetic small dictionary','data':'finite identity child stores, not tokens','loader':'three original async methods, host layout/batchify/watchdog adapters','collective':'barrier recording adapter','save':'original flatten+host writer, NumPy payload conversion'},'actual_Hero_marker_step_mismatch':None,'actual_Hero_next_tokens':None,'actual_full_training_step':None,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [state_probe.T,state_probe.G,loader.LP,loader.SP,P]}}
 (R/'analysis/restore_data_clock.json').write_text(json.dumps(o,indent=2)+'\n');print('Restore/data clock joined checks:',len(checks))
if __name__=='__main__':main()
