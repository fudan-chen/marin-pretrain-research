"""Original scheduler and Hero build, real Optax injection/Adam in selected Adam group.
Synthetic single-group parameters; unused MuonH/AdamH transforms are identities.
In-memory state transfer, not disk restore, full Hero optimizer, or training experiment.
"""
import ast,dataclasses,hashlib,json,pathlib,types,warnings
import jax,jax.numpy as jnp,numpy as np,optax
R=pathlib.Path(__file__).resolve().parents[1];S=R/'sources/optimizer_schedule_2026_10_08/config.py';O=R/'sources/optimizer_2026_10_05/optimizer.py'
ns={'jax':jax,'jnp':jnp,'np':np,'optax':optax,'warnings':warnings,'dataclass':dataclasses.dataclass}
def compile_nodes(nodes):
 exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])),'<original>','exec'),ns)
roots=ast.parse(S.read_text()).body
ctx=next(x for x in roots if isinstance(x,ast.ClassDef) and x.name=='LrScheduleContext')
lin=next(x for x in roots if isinstance(x,ast.ClassDef) and x.name=='LinearLrSchedule')
# Registry removal only; preserve original builder method and context fields.
lin.decorator_list=[];lin.bases=[]
compile_nodes([ctx,lin,next(x for x in roots if isinstance(x,ast.FunctionDef) and x.name=='_convert_frac_or_steps')])
cfg=next(x for x in roots if isinstance(x,ast.ClassDef) and x.name=='OptimizerConfig')
compile_nodes([x for x in cfg.body if isinstance(x,ast.FunctionDef) and x.name in ['lr_scheduler','_get_cycle_minima']])
optroot=ast.parse(O.read_text()).body;hero=next(x for x in optroot if isinstance(x,ast.ClassDef) and x.name=='GrugMoeMuonHConfig')
compile_nodes([x for x in optroot if isinstance(x,ast.FunctionDef) and x.name in ['_is_gate_or_router_weight','_gate_router_decay_mask','_scale_by_adam_gate_router_decay']])
compile_nodes([x for x in hero.body if isinstance(x,ast.FunctionDef) and x.name in ['build','create_mask']])
ns.update(leaf_key_paths=lambda p:{'mlp.router':'mlp.router'},scale_with_grug_muonh=lambda **kw:optax.identity(),_match_named_update_sharding=lambda:optax.identity(),scale_by_adamh=lambda *a:optax.identity())
class Config:
 lr_scheduler=ns['lr_scheduler'];_get_cycle_minima=ns['_get_cycle_minima'];build=ns['build'];create_mask=ns['create_mask']
 cooldown=None;cycle_length=None;cycles=None;haps=None;warmup=.1;rewarmup=0.;decay=None
 learning_rate=.02;adam_lr=.01;min_lr_ratio=.1;max_grad_norm=None
 momentum=.95;nesterov=True;backend_steps=5;muon_epsilon=1e-8;coefficient_type='quintic';use_syrk=True
 beta1=.9;beta2=.95;epsilon=1e-8;gate_router_weight_decay=.02
 def _resolve_lr_schedule(self):return ns['LinearLrSchedule']()

def flat(t):return [np.asarray(x).tolist() for x in jax.tree.leaves(t)]
def equal(a,b):return jax.tree.structure(a)==jax.tree.structure(b) and all(np.array_equal(x,y) for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)))
params={'mlp.router':jnp.array([2.,4.])};grad={'mlp.router':jnp.array([.25,.25])};conf=Config();old=conf.build(100);state=old.init(params)
for _ in range(50):_,state=old.update(grad,state,params)
checks=[];cases={}
def ck(name,value):
 if not value:raise RuntimeError(name)
 checks.append({'name':name,'passed':True})
ck('original mask selects only Adam group',conf.create_mask(params)=={'mlp.router':'adam'})
old_state_before=flat(state)
def run(name,n,scheduler_n=None,input_state=state):
 c=Config()
 if scheduler_n is not None:
  c.lr_scheduler=lambda ignored,override_lr=None:Config.lr_scheduler(c,scheduler_n,override_lr)
 tx=c.build(n);updates,nextstate=tx.update(grad,input_state,params)
 cases[name]={'build_N':n,'schedule_N':scheduler_n or n,'input_schedule_counts':{k:int(v.count) for k,v in input_state.hyperparams_states.items()},'output_schedule_counts':{k:int(v.count) for k,v in nextstate.hyperparams_states.items()},'input_outer_count':int(input_state.count),'output_outer_count':int(nextstate.count),'cached_input_hyperparams':{k:float(v) for k,v in input_state.hyperparams.items()},'used_output_hyperparams':{k:float(v) for k,v in nextstate.hyperparams.items()},'update':np.asarray(updates['mlp.router']).tolist(),'next_state_leaves':flat(nextstate)}
 return updates,nextstate
u0,s0=run('same_N',100);u1,s1=run('new_N',200)
ud,sd=run('decay_only_counterfactual',200,100);ul,sl=run('lr_only_counterfactual',100,200)
short,ss=run('shortened_N',40)
corrupt=state._replace(hyperparams={k:jnp.array(123.) for k in state.hyperparams})
uc,sc=run('cached_hyperparams_only_changed',100,input_state=corrupt)
outer_reset=state._replace(count=jnp.array(0,jnp.int32))
uo,so=run('outer_count_only_reset',100,input_state=outer_reset)
schedule_reset=state._replace(hyperparams_states={k:v._replace(count=jnp.array(0,jnp.int32)) for k,v in state.hyperparams_states.items()})
us,ssr=run('schedule_counts_only_reset',100,input_state=schedule_reset)
jit_u,jit_s=jax.jit(conf.build(100).update)(grad,state,params)
ck('complete injected state carries count50 with distinct cached pre-update schedule',int(state.count)==50 and abs(float(state.hyperparams['adam_lr'])-.0061)<1e-7)
ck('same state with rebuilt horizon uses new LR on next update',abs(cases['same_N']['used_output_hyperparams']['adam_lr']-.006)<1e-7 and abs(cases['new_N']['used_output_hyperparams']['adam_lr']-.0085)<1e-7)
ck('unchanged input state preserved by all branches',flat(state)==old_state_before)
ck('same gradients preserve all next inner Adam states across rebuilt plans',equal(s0.inner_state,s1.inner_state) and equal(s0.inner_state,sd.inner_state) and equal(s0.inner_state,sl.inner_state) and all(equal(s0.inner_state,x.inner_state) for x in [ss,sc,so,ssr]))
ck('changing cached hyperparams alone does not bind scheduled next value',equal(uc,u0) and equal(sc,s0))
ck('isolated LR multiplies identical direction by known schedule ratio',np.allclose(np.asarray(ul['mlp.router'])/np.asarray(u0['mlp.router']),.0085/.006,rtol=2e-6))
ck('isolated horizon decay changes direction independently of LR',not equal(ud,u0) and cases['decay_only_counterfactual']['used_output_hyperparams']==cases['same_N']['used_output_hyperparams'])
ck('shortened horizon remains at minimum LR not frozen',abs(cases['shortened_N']['used_output_hyperparams']['adam_lr']-.001)<1e-7 and bool(jnp.any(short['mlp.router']!=0)))
ck('eager and JIT updates agree within float tolerance',np.allclose(jit_u['mlp.router'],u0['mlp.router'],rtol=1e-6) and equal(jit_s.inner_state,s0.inner_state))
ck('all branches advance outer clock exactly once',all(x['output_outer_count']==x['input_outer_count']+1 for x in cases.values()))
ck('outer count reset alone does not reset wrapped schedule or Adam direction',equal(uo,u0) and equal(so.inner_state,s0.inner_state) and cases['outer_count_only_reset']['output_outer_count']==1 and cases['outer_count_only_reset']['output_schedule_counts']['adam_lr']==51)
ck('schedule clock reset alone zeroes initial warmup LR and update',cases['schedule_counts_only_reset']['used_output_hyperparams']['adam_lr']==0. and bool(jnp.all(us['mlp.router']==0)))
ck('zero LR update still advances unchanged inner Adam state',equal(ssr.inner_state,s0.inner_state) and cases['schedule_counts_only_reset']['output_schedule_counts']['adam_lr']==1 and int(ssr.count)==51)
points=[0,5,10,20,49,50,90,100,150,200]
curves={str(n):[{'count':p,'adam_lr':float(conf.lr_scheduler(n,override_lr=conf.adam_lr)(p))} for p in points] for n in [40,100,200]}
result={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'cases':cases,'schedule_values':curves,'input_state_leaves':old_state_before,'selected_group':'adam','parameters_fixed_during_moment_prewarm':True,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [S,O]},'probe_script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'runtime':{'jax':jax.__version__,'optax':optax.__version__},'actual_disk_checkpoint_restore':None,'actual_Hero_schedule_rebuild':None,'actual_full_Hero_optimizer_execution':None,'actual_training_loss_effect':None}
(R/'analysis/optimizer_schedule_resume_cpu.json').write_text(json.dumps(result,indent=2)+'\n');print('Optimizer schedule resume controls:',len(checks))
