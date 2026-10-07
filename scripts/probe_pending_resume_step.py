"""Original Grug train-step closure + original QB setter/route + real Adam/local OCDBT restore.
Synthetic one-layer three-expert scalar model, fixed logits, injected beta estimator and squared loss.
No Transformer, real QB estimation, capacity/dispatch, hardware mesh or Hero causal result.
"""
import ast,dataclasses,functools,hashlib,json,pathlib,tempfile,types,contextlib,typing,importlib.metadata
import jax,jax.numpy as jnp,numpy as np,optax,equinox as eqx,jmp
import probe_grug_state_restore_real_io as prior
from probe_router_weight_path_cpu import route,recipe,P as ROUTER
R=prior.R;ns=prior.ns;T=R/'sources/main_incident_2026_10_07/train.py';M=R/'sources/main_incident_2026_10_07/model.py';E=R/'sources/scale_2026_10_05/eval.py'
def extract(p,names,env):
 nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 assert len(nodes)==len(names)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])),str(p),'exec'),env)
class MLP(eqx.Module):
 router_bias:jax.Array
class Block(eqx.Module):
 mlp:MLP
class Stack(eqx.Module):
 stacked:Block
class Tiny(eqx.Module):
 stacked_blocks:Stack
 w:jax.Array

def fresh_model():return Tiny(Stack(Block(MLP(jnp.zeros((1,3),jnp.float32)))),jnp.array([3.,1.,-2.],jnp.float32))
ns.update(jax=jax,jnp=jnp,eqx=eqx,optax=optax,jmp=jmp,functools=functools,dataclasses=dataclasses)
extract(M,['apply_qb_betas'],ns);apply=ns['apply_qb_betas'];State=prior.State
logits=jnp.array([[.1,.2,0.]],jnp.float32)
def forward(p):
 _,ids,weights,alpha=route(recipe,jnp.ones((1,1),jnp.float32),logits,p.stacked_blocks.stacked.mlp.router_bias[0])
 return jnp.sum(weights*p.w[ids]),ids,weights

def objective(p,target):return (forward(p)[0]-target)**2
# Deliberate fixture substitutes only the full model loss/gradient and beta estimator.
def loss_and_grads(p,batch,mp,z_loss):
 loss,grad=jax.value_and_grad(objective)(p,batch['target'])
 return (loss,{'qb_beta_per_layer':batch['beta']}),grad
ns['_loss_and_grads']=loss_and_grads
extract(T,['_make_train_step','initial_state'],ns)
tx=optax.adam(.01);mp=jmp.Policy(jnp.float32,jnp.float32,jnp.float32)
step=ns['_make_train_step'](tx,mp,z_loss_weight=0.,ema_beta=.9,watch_config=None,offload_opt_state=False,master_param_mode=ns['MasterParamMode'].DEVICE)
def fresh():
 p=fresh_model();return State(step=jnp.array(0,jnp.int32),params=p,master_params=None,opt_state=tx.init(p),ema_params=jax.tree.map(lambda x:jnp.array(x,copy=True),p),pending_qb_betas=jnp.zeros((1,3),jnp.float32))
def copy(s):return jax.tree.map(lambda x:jnp.array(x,copy=True),s)
def host(s):return jax.tree.map(lambda x:np.array(x,copy=True),s)
def equal(a,b):return jax.tree.structure(a)==jax.tree.structure(b) and all(np.array_equal(x,y) for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)))
checks=[]
def check(n,v):assert v,n;checks.append({'name':n,'passed':True})
def describe(s):
 view=apply(s.params,s.pending_qb_betas)
 return {'step':int(s.step),'adam_count':int(s.opt_state[0].count),'params_w':s.params.w.tolist(),'stored_bias':s.params.stacked_blocks.stacked.mlp.router_bias.tolist(),'pending':s.pending_qb_betas.tolist(),'next_forward_bias':view.stacked_blocks.stacked.mlp.router_bias.tolist(),'stored_ids':forward(s.params)[1].tolist(),'next_forward_ids':forward(view)[1].tolist(),'stored_prediction':float(forward(s.params)[0]),'next_forward_prediction':float(forward(view)[0])}
# Actual original initializer with a Tiny.init adapter preserves the source EMA alias.
from jax.sharding import Mesh,PartitionSpec as P,AxisType
ns['Transformer']=types.SimpleNamespace(init=lambda config,key:fresh_model());ns['P']=P
with jax.set_mesh(Mesh(np.array(jax.devices()),('data',),axis_types=(AxisType.Explicit,))):
 aliased=ns['initial_state'](types.SimpleNamespace(num_layers=1,num_experts=3),optimizer=tx,mp=mp,key=jax.random.PRNGKey(0),ema_beta=.9)
 alias_identity=all(a is b for a,b in zip(jax.tree.leaves(aliased.params),jax.tree.leaves(aliased.ema_params)))
 try:
  alias_out=step(aliased,{'target':jnp.array(2.),'beta':jnp.array([[0.,3.,0.]])});jax.block_until_ready(alias_out);alias_error=None
 except Exception as exc:alias_error=type(exc).__name__+': '+str(exc)
check('Original initializer aliases EMA leaves and this CPU donation path rejects repeated buffer',alias_identity and alias_error is not None and 'donate the same buffer twice' in alias_error)
with jax.set_mesh(Mesh(np.array(jax.devices()),('data',),axis_types=(AxisType.Explicit,))):
 independent=copy(ns['initial_state'](types.SimpleNamespace(num_layers=1,num_experts=3),optimizer=tx,mp=mp,key=jax.random.PRNGKey(0),ema_beta=.9))
 no_ema=ns['initial_state'](types.SimpleNamespace(num_layers=1,num_experts=3),optimizer=tx,mp=mp,key=jax.random.PRNGKey(0),ema_beta=None)
independent_input_distinct=all(a is not b for a,b in zip(jax.tree.leaves(independent.params),jax.tree.leaves(independent.ema_params)))
independent_result=step(independent,{'target':jnp.array(2.),'beta':jnp.array([[0.,3.,0.]])});jax.block_until_ready(independent_result)
no_ema_step=ns['_make_train_step'](tx,mp,z_loss_weight=0.,ema_beta=None,watch_config=None,offload_opt_state=False,master_param_mode=ns['MasterParamMode'].DEVICE)
no_ema_result=no_ema_step(no_ema,{'target':jnp.array(2.),'beta':jnp.array([[0.,3.,0.]])});jax.block_until_ready(no_ema_result)
check('Independent EMA buffers allow same original donated train closure to advance',independent_input_distinct and int(independent_result[0].step)==1)
check('Original initializer with EMA disabled advances with original donated train closure',no_ema_result[0].ema_params is None and int(no_ema_result[0].step)==1)
meta=json.loads((R/'sources/operational_2026_10_07/wandb_meta.json').read_text());cfg=json.loads(meta['data']['project']['run']['config']);hero_ema_beta=cfg['trainer']['value']['ema_beta']
check('Archived Hero EMA disabled; aliased-EMA CPU failure not claimed as production incident',hero_ema_beta is None)

# Warm up to a nonzero Adam count, EMA and pending state using the original train closure.
s=fresh()
for target,beta in [(2.,[[0.,3.,0.]]),(-1.,[[0.,0.,3.]]),(1.,[[0.,3.,0.]])]:
 s,metrics,_=step(s,{'target':jnp.array(target),'beta':jnp.array(beta)})
jax.block_until_ready(s);before=host(s)
check('Real CPU original train closure produces nonempty optimizer EMA and pending',jax.default_backend()=='cpu' and int(s.step)==3 and int(s.opt_state[0].count)==3 and s.ema_params is not None and bool(jnp.any(s.pending_qb_betas!=0)))
check('Actual Equinox QB replacement preserves input and is idempotent',equal(apply(apply(s.params,s.pending_qb_betas),s.pending_qb_betas),apply(s.params,s.pending_qb_betas)) and equal(host(s),before))
# Real original host serializer/tree reader/Grug restore. Metadata and barrier remain explicit adapters.
def save(root,state):
 root.mkdir(parents=True);paths,arrays=ns['_flatten_serializable_leaves'](host(state))
 entries=[ns['CheckpointArray'](path=p,shape=tuple(a.shape),dtype=np.dtype(a.dtype).name,chunk_shape=tuple(a.shape)) for p,a in zip(paths,arrays)]
 ns['write_manifest'](str(root),ns['build_manifest'](entries,array_driver='zarr3',kvstore_driver='ocdbt'))
 manager=prior.base.base.writer.array_ser.GlobalAsyncCheckpointManager()
 ns['_serialize_arrays'](arrays,[ns['_create_ocdbt_spec'](str(root),p,entry=e) for p,e in zip(paths,entries)],[None]*len(arrays),manager,ns['TensorStoreWriteConfig'](max_staged_host_bytes=128),lambda:None,None,None)
 manager.wait_until_finished();(root/'metadata.json').write_text(json.dumps({'step':int(state.step),'timestamp':'2026-10-07T00:00:00+00:00','is_temporary':False}));return paths
actual_load=ns['load_checkpoint']
def load(exemplar,path,**kwargs):return actual_load(exemplar,path,read_config=ns['TensorStoreReadConfig'](replica_mode=ns['ReplicaRestoreMode'].EVERY_REPLICA),**kwargs)
with tempfile.TemporaryDirectory(prefix='marin-pending-resume-') as tmp:
 paths=save(pathlib.Path(tmp)/'step3',s)
 restored=ns['restore_grug_state_from_checkpoint'](fresh(),checkpoint_search_paths=[tmp],load_checkpoint_setting=True,mesh=None,allow_partial=False,_load_fn=load)
check('Original local full-state restore preserves all leaves including pending and EMA',equal(s,restored))
check('Manifest includes pending and EMA optimizer leaves', 'pending_qb_betas' in paths and any(p.startswith('ema_params/') for p in paths) and any(p.startswith('opt_state/') for p in paths))
batch={'target':jnp.array(.5),'beta':jnp.array([[3.,0.,0.]],jnp.float32)}
materialized=apply(restored.params,restored.pending_qb_betas)
variants={'uninterrupted':s,'full_restore':restored,'clear_pending':dataclasses.replace(restored,pending_qb_betas=jnp.zeros_like(restored.pending_qb_betas)), 'materialize_keep_pending':dataclasses.replace(restored,params=materialized),'materialize_clear_pending':dataclasses.replace(restored,params=materialized,pending_qb_betas=jnp.zeros_like(restored.pending_qb_betas))}
rows={};outputs={}
for name,state in variants.items():
 row=describe(state);out,metrics,_=step(copy(state),batch);jax.block_until_ready(out)
 row.update(train_loss=float(metrics['train/loss']),next_state=describe(out));rows[name]=row;outputs[name]=out
check('Full restored state matches next original update and reported loss exactly',equal(outputs['full_restore'],outputs['uninterrupted']) and rows['full_restore']['train_loss']==rows['uninterrupted']['train_loss'])
check('Clearing pending preserves current params and count but changes next route and update',equal(variants['clear_pending'].params,restored.params) and int(variants['clear_pending'].opt_state[0].count)==int(restored.opt_state[0].count) and rows['clear_pending']['next_forward_ids']!=rows['full_restore']['next_forward_ids'] and not equal(outputs['clear_pending'],outputs['full_restore']))
check('Materializing params while keeping pending is idempotent at next original step',equal(outputs['materialize_keep_pending'],outputs['full_restore']))
check('Materializing then zeroing pending is not equivalent because zero replaces bias',rows['materialize_clear_pending']['next_forward_bias']==[[0.,0.,0.]] and equal(outputs['materialize_clear_pending'],outputs['clear_pending']) and not equal(outputs['materialize_clear_pending'],outputs['full_restore']))
check('Output pending is current injected estimator result not input pending',all(r['next_state']['pending']==batch['beta'].tolist() for r in rows.values()))
check('Stored output bias follows old pending while next forward follows new pending',rows['full_restore']['next_state']['stored_bias']==rows['full_restore']['next_forward_bias'] and rows['full_restore']['next_state']['stored_bias']!=rows['full_restore']['next_state']['next_forward_bias'])
# Execute exact model accessors captured by the training runner constructor.
call=next(n for n in ast.walk(ast.parse(T.read_text())) if isinstance(n,ast.Call) and isinstance(n.func,ast.Subscript) and isinstance(n.func.value,ast.Name) and n.func.value.id=='StateCallbackRunner')
getters={k.arg:eval(compile(ast.Expression(k.value),str(T),'eval'),{}) for k in call.keywords if k.arg in ['model_getter','eval_model_getter']}
getters_rows={name:forward(fn(restored))[1].tolist() for name,fn in getters.items()}
check('Original callback accessors read stored current and stored EMA without pending application',equal(getters['model_getter'](restored),restored.params) and equal(getters['eval_model_getter'](restored),restored.ema_params))
# Execute original tagged callback using recording evaluator/tracker and no-op event/PGLE scopes.
seen=[];logs=[]
e={'Callable':typing.Callable,'StepInfo':object,'ProgressEvent':types.SimpleNamespace(EVALUATION_STARTED='start',EVALUATION_FINISHED='end'),'progress_event_scope':lambda *a:contextlib.nullcontext(),'jax_config':types.SimpleNamespace(enable_pgle=lambda x:contextlib.nullcontext()),'levanter':types.SimpleNamespace(tracker=types.SimpleNamespace(log=lambda d,step:logs.append({'step':step,'metrics':d}))), '_join_prefix':lambda a,b:a+'/'+b}
def eval_model(evaluator,model,prefix):
 seen.append({'prefix':prefix,'ids':forward(model)[1].tolist(),'prediction':float(forward(model)[0])});return {prefix+'/loss':float(objective(model,jnp.array(.5)))}
e['eval_model']=eval_model;extract(E,['cb_tagged_evaluate'],e)
cb=e['cb_tagged_evaluate'](None,eval_current=True,eval_ema=True)
info=types.SimpleNamespace(step=int(restored.step)-1,model=getters['model_getter'](restored),eval_model=getters['eval_model_getter'](restored),emit_event=lambda *a:None)
cb(info);cb(info,force=True)
check('Original tagged callback evaluates stored current EMA once and suppresses forced duplicate',len(seen)==2 and len(logs)==2 and seen[0]['ids']==rows['full_restore']['stored_ids'])
check('Stored current callback and next train forward can select different experts',seen[0]['ids']!=rows['full_restore']['next_forward_ids'])
check('All variants retain finite loss; finite guard cannot establish resume equivalence',all(np.isfinite(r['train_loss']) for r in rows.values()) and rows['clear_pending']['train_loss']!=rows['full_restore']['train_loss'])
check('Inherited original Grug state class AST equals current frozen training state schema',ast.dump(next(n for n in ast.parse(prior.T.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='GrugTrainState'))==ast.dump(next(n for n in ast.parse(T.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='GrugTrainState')))
files=[pathlib.Path(__file__),prior.T,R/'scripts/probe_router_weight_path_cpu.py',R/'sources/operational_2026_10_07/wandb_meta.json',T,M,E,ROUTER,R/'scripts/probe_grug_state_restore_real_io.py',R/'scripts/probe_tree_restore_contracts.py',R/'scripts/probe_restore_candidate_real_io.py',R/'scripts/probe_serialize_arrays_real_io.py',prior.G,prior.U,prior.base.base.P,prior.base.base.M,prior.base.base.C,R/'sources/checkpoint_memory_2026_10_05/byte_budget.py']+sorted((R/'sources/tree_restore_2026_10_07').glob('*.py'))
out={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'runtime':{n:importlib.metadata.version(n) for n in ['jax','jaxlib','numpy','optax','equinox','tensorstore','jmp']},'backend':jax.default_backend(),'cases':rows,'serialized_paths':paths,'ema_alias_control':{'original_initializer_executed':True,'independent_EMA_buffers_succeed':independent_input_distinct,'EMA_disabled_original_step_succeeds':int(no_ema_result[0].step)==1,'parameter_and_ema_leaf_identity_shared':alias_identity,'error':alias_error,'archived_Hero_ema_beta':hero_ema_beta,'actual_Hero_incident':None,'independent_buffer_continuation':'fixture allocates EMA leaves separately; not a merged upstream fix'},'callback_views':seen,'callback_logs':logs,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'substitutions':['Tiny Eqx hierarchy and Transformer.init adapter replace full Transformer; three trainable scalar expert outputs and fixed logits; main continuation fixture allocates independent EMA leaves after recording original initializer alias failure','Squared loss uses original route block; injected batch beta replaces full model loss and QB estimator','Identity reshard and partition-spec placeholder; no capacity drops or transport','Manual local metadata; recording restore barrier, single-CPU sharding/local StoragePath and no transfer accounting inherited from prior probes','Callback evaluator/tracker and event/PGLE contexts are recording adapters, not real evaluator execution'],'actual_Hero_view_error_magnitude':None,'actual_Hero_pending_loss_event':None,'actual_GPU_execution':None,'actual_distributed_restore':None,'actual_causal_mixture_effect':None}
(R/'analysis/pending_resume_step_cpu.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print('Original-step pending/restore controls',len(checks)); print(json.dumps(rows,indent=2))
