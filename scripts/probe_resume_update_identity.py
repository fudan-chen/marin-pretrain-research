"""Assembled single-CPU state snapshot, real local IO restore, next-update identity.
Original Grug state and restore functions; synthetic 3-parameter quadratic and real Optax Adam.
Not original Hero train_step, full Transformer/optimizer, checkpoint publisher or collective.
"""
import ast,asyncio,dataclasses,functools,hashlib,json,pathlib,tempfile,threading,traceback
import jax,jax.numpy as jnp,numpy as np,optax
import probe_grug_state_restore_real_io as previous
R=previous.R;ns=previous.ns;State=previous.State;base=previous.base
P=R/'sources/donation_snapshot_2026_10_07/lib/levanter/src/levanter/tensorstore_serialization.py'
tree=ast.parse(P.read_text());names={'_transfer_shard_to_pageable_host','_slice_shard_on_device'}
ns.update(asyncio=asyncio)
for n in tree.body:
 if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in {'_HOST_MEMORY_KIND','_GPU_PLATFORM','_PAGEABLE_HOST_MEMORY_KIND'}:ns[n.targets[0].id]=ast.literal_eval(n.value)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names],type_ignores=[]),str(P),'exec'),ns)
base.extract(previous.G,['init_weights_only_from_checkpoint'])
ns['cast']=__import__('typing').cast;ns['Sequence']=__import__('typing').Sequence
# Resolve postponed type names only used in annotations by original helper extraction environment.
ns['GrugStateT']=ns.get('GrugStateT',object)
# Keep Levanter discovery globals separate from Grug's same-named filesystem helper.
discovery=dict(ns)
discovery_names={'_load_metadata','_get_fs_and_plain_path','_discover_checkpoint_paths_single','_discover_checkpoint_candidates_single','_checkpoint_candidate_sort_key','_is_path_under_any','discover_checkpoint_candidates','discover_latest_checkpoint','latest_checkpoint_path'}
discovery_nodes=[n for n in ast.parse(base.base.C.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in discovery_names]
assert len(discovery_nodes)==len(discovery_names)
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+discovery_nodes,type_ignores=[])),str(base.base.C),'exec'),discovery)
ns['latest_checkpoint_path']=discovery['latest_checkpoint_path']
actual_load=ns['load_checkpoint']
def load(exemplar,path,**kwargs):
 return actual_load(exemplar,path,read_config=ns['TensorStoreReadConfig'](replica_mode=ns['ReplicaRestoreMode'].EVERY_REPLICA),**kwargs)
ns['load_checkpoint']=load
tx=optax.adam(learning_rate=.01);checks=[];rows={};target=jnp.array([.5,-.75,1.25],dtype=jnp.float32)
def loss(params,batch):return jnp.mean((params['w']-batch)**2)
@functools.partial(jax.jit,donate_argnums=(0,))
def advance(s,batch):
 value,grad=jax.value_and_grad(loss)(s.params,batch);updates,opt=tx.update(grad,s.opt_state,s.params)
 return dataclasses.replace(s,step=s.step+1,params=optax.apply_updates(s.params,updates),opt_state=opt),value
def check(n,c):assert c,n;checks.append({'name':n,'passed':True})
def host(s):return jax.tree.map(lambda x:np.array(x,copy=True),s)
def device(s):return jax.tree.map(lambda x:jnp.array(x,copy=True),s)
def equal(a,b):return all(np.array_equal(x,y) for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b))) and jax.tree.structure(a)==jax.tree.structure(b)
def fresh():
 params={'w':jnp.array([1.25,-2.5,3.75],dtype=jnp.float32)}
 return State(step=jnp.array(0,dtype=jnp.int32),params=params,master_params=None,opt_state=tx.init(params),ema_params=None,pending_qb_betas=jnp.zeros((2,3),dtype=jnp.float32))
async def stage(s):
 leaves,structure=jax.tree.flatten(s)
 snapshots=[]
 for leaf in leaves:snapshots.append(await ns['_transfer_shard_to_pageable_host'](leaf.addressable_shards[0]))
 return jax.tree.unflatten(structure,snapshots)
def save(root,s):
 root.mkdir(parents=True);paths,arrays=ns['_flatten_serializable_leaves'](s)
 entries=[ns['CheckpointArray'](path=p,shape=tuple(a.shape),dtype=np.dtype(a.dtype).name,chunk_shape=tuple(a.shape)) for p,a in zip(paths,arrays)]
 ns['write_manifest'](str(root),ns['build_manifest'](entries,array_driver='zarr3',kvstore_driver='ocdbt'))
 manager=base.base.writer.array_ser.GlobalAsyncCheckpointManager()
 ns['_serialize_arrays'](arrays,[ns['_create_ocdbt_spec'](str(root),p,entry=e) for p,e in zip(paths,entries)],[None]*len(arrays),manager,ns['TensorStoreWriteConfig'](max_staged_host_bytes=128),lambda:None,None,None)
 manager.wait_until_finished();(root/'metadata.json').write_text(json.dumps({'step':int(s.step),'timestamp':'2026-10-07T00:00:00+00:00','is_temporary':False}));return paths
state=fresh()
for batch in [target+3,target-2,target+1]:state,_=advance(state,batch)
state=dataclasses.replace(state,pending_qb_betas=jnp.arange(6,dtype=jnp.float32).reshape(2,3))
snapshot=asyncio.run(stage(state));before=host(state)
check('Original helper snapshots every concrete state leaf',equal(snapshot,before))
with tempfile.TemporaryDirectory(prefix='marin-resume-update-') as tmp:
 root=pathlib.Path(tmp)/'step3';started=threading.Event();release=threading.Event();out={}
 def writer():
  started.set()
  if not release.wait(30):out['error']='event timeout';return
  try:out['paths']=save(root,snapshot)
  except Exception:out['error']=traceback.format_exc()
 thread=threading.Thread(target=writer);thread.start();assert started.wait(5)
 old_params=state.params['w'];old_moment=state.opt_state[0].mu['w']
 try:
  expected_next,expected_loss=advance(state,target);jax.block_until_ready(expected_next)
  check('Actual state parameter and optimizer moment inputs donated',old_params.is_deleted() and old_moment.is_deleted())
  expected_next_host=host(expected_next)
 finally:release.set();thread.join(60)
 assert not thread.is_alive(), 'writer still live'
 assert 'error' not in out,out.get('error')
 restored=ns['restore_grug_state_from_checkpoint'](fresh(),checkpoint_search_paths=[str(root)],load_checkpoint_setting=True,mesh=None,allow_partial=False,_load_fn=load)
 check('Real local restore exactly matches every staged state leaf',equal(host(restored),before))
 rows['serialized_paths']=out['paths'];rows['restored_step']=int(restored.step);rows['restored_adam_count']=int(restored.opt_state[0].count)
 # Original weights-only initializer preserves fresh optimizer/clock/pending while replacing params.
 weights_only=ns['init_weights_only_from_checkpoint'](fresh(),str(root),mesh=None,allow_partial=False)
 check('Original weights-only initializer reads same parameters',np.array_equal(np.asarray(weights_only.params['w']),before.params['w']))
 check('Original weights-only initializer retains fresh optimizer and clock',int(weights_only.step)==0 and int(weights_only.opt_state[0].count)==0 and np.array_equal(np.asarray(weights_only.pending_qb_betas),np.zeros((2,3))))
 full_loss=float(loss(restored.params,target));weight_loss=float(loss(weights_only.params,target))
 check('Same pre-update quadratic loss cannot identify optimizer restoration',full_loss==weight_loss==float(expected_loss))
 next_restored,_=advance(restored,target);jax.block_until_ready(next_restored)
 check('Full-state restore gives exact same next-update state',equal(host(next_restored),expected_next_host))
 next_weights,_=advance(weights_only,target);jax.block_until_ready(next_weights)
 delta_full=np.asarray(next_restored.params['w'])-before.params['w'];delta_weights=np.asarray(next_weights.params['w'])-before.params['w']
 check('Weights-only restore diverges on first parameter update',not np.array_equal(delta_full,delta_weights))
 rows.update(pre_update_loss=full_loss,weights_only_pre_update_loss=weight_loss,full_update=delta_full.tolist(),weights_only_update=delta_weights.tolist(),max_update_difference=float(np.max(np.abs(delta_full-delta_weights))),next_full_loss=float(loss(next_restored.params,target)),next_weights_only_loss=float(loss(next_weights.params,target)),pending_applied_in_quadratic=False)
paths=[pathlib.Path(__file__),P,previous.T,previous.G,previous.U,base.base.P,base.base.M,base.base.C,base.D/'jax_utils.py',R/'scripts/probe_grug_state_restore_real_io.py',R/'scripts/probe_tree_restore_contracts.py',R/'scripts/probe_restore_candidate_real_io.py',R/'scripts/probe_serialize_arrays_real_io.py',R/'analysis/resume_source_binding.json']
paths.extend(sorted(base.D.glob('*.py')));paths.append(base.base.writer.B)
record={'checks_passed':len(checks),'checks':checks,'observations':rows,'scope':__doc__,'runtime':{'jax':jax.__version__,'jaxlib':__import__('jaxlib').__version__,'optax':optax.__version__,'numpy':np.__version__,'tensorstore':__import__('importlib.metadata').metadata.version('tensorstore')},'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'explicit_adapters':['AST extraction, archived core bytes verified identical to frozen eee467 revision','three-vector quadratic instead of Transformer and original Hero train_step','real Optax Adam instead of Hero optimizer groups','pending stored/restored but deliberately not applied to quadratic','local StoragePath and single-device sharding adapters inherited from V69','original host writer with manual metadata publication after wait','barrier recording adapter, no collective','event-gated writer thread is constructed interleaving, not production scheduling','EVERY_REPLICA CPU reader; Levanter discovery namespace isolated from Grug filesystem helper'], 'actual_Hero_next_update_identity':None,'actual_GPU_roundtrip':None,'actual_causal_mixture_effect':None}
(R/'analysis/resume_update_identity.json').write_text(json.dumps(record,indent=2)+'\n');print('Resume next-update identity:',len(checks),'passed')
