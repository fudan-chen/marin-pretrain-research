# -*- coding: utf-8 -*-
"""Original train-step/layout helper probes with declared substitutes; no autodiff/GPU/I/O."""
import ast
import dataclasses
import functools
import hashlib
import json
import logging
import pathlib
import types
import numpy as np

R=pathlib.Path(__file__).resolve().parents[1]
T=R/'sources/scale_2026_10_05/train_hero_ep.py'
M=R/'sources/contracts_2026_10_05/model.py'
D=R/'sources/state_2026_10_05'
checks=[];extracted={}

def check(name,value):
 if not value:raise AssertionError(name)
 checks.append(name)

def extract(path,names,ns):
 nodes=[x for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef) and x.name in names]
 assert {x.name for x in nodes}==set(names)
 future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
 module=ast.fix_missing_locations(ast.Module(body=[future]+nodes,type_ignores=[]))
 exec(compile(module,str(path),'exec'),ns)
 extracted[str(path.relative_to(R))]=[{'function':n.name,'first_line':n.lineno,'last_line':n.end_lineno} for n in nodes]

def bf16(value):
 """Finite FP32 round-to-nearest-even BF16 emulation, stored as FP32 numbers."""
 x=np.asarray(value,dtype=np.float32)
 assert np.all(np.isfinite(x))
 u=x.view(np.uint32)
 rounded=(u+np.uint32(0x7fff)+((u>>16)&1)) & np.uint32(0xffff0000)
 return rounded.view(np.float32)

@dataclasses.dataclass(frozen=True)
class Model:
 weight:object
 router_bias:object

@dataclasses.dataclass(frozen=True)
class State:
 step:object
 params:object
 master_params:object
 opt_state:object
 ema_params:object
 pending_qb_betas:object

class Policy:
 def __init__(self,low):self.low=low
 def cast_to_param(self,x):
  cast=bf16 if self.low else lambda a:np.asarray(a,dtype=np.float32)
  return Model(cast(x.weight),cast(x.router_bias))

class Optimizer:
 def update(self,grads,state,params):
  return Model(-np.float32(.1)*grads.weight,np.zeros_like(params.router_bias)),state+1

def updates(p,u):
 return Model(p.weight+u.weight,p.router_bias+u.router_bias)

# Eqx replacement adapter runs the original beta negation and centering arithmetic.
def tree_at(selector,model,new_bias):
 return dataclasses.replace(model,router_bias=new_bias)
ns={'jnp':np,'eqx':types.SimpleNamespace(tree_at=tree_at)}
extract(M,['apply_qb_betas'],ns)
apply_beta=ns['apply_qb_betas']
applied=apply_beta(Model(np.array(1.),np.zeros((1,2))),np.array([[1.,3.]]))
check('Original beta helper negates and centers each layer',np.array_equal(applied.router_bias,[[1.,-1.]]))
check('Adding constant to all beta entries leaves centered bias invariant',np.array_equal(apply_beta(applied,np.array([[101.,103.]])).router_bias,applied.router_bias))

memory_moves=[];gradient_inputs=[]
def memory(tree,kind):memory_moves.append(kind);return tree
def grads(params,batch,mp,z):
 gradient_inputs.append(params.router_bias.tolist())
 beta=np.zeros((1,2)) if batch is None else np.asarray(batch,dtype=np.float32)
 return (np.float32(0),{'qb_beta_per_layer':beta}),Model(bf16(.01),np.zeros_like(params.router_bias))
Mode=types.SimpleNamespace(DEVICE='device',FP32_PINNED_HOST='fp32_pinned_host')
ns.update(functools=functools,dataclasses=dataclasses,MasterParamMode=Mode,
          jax=types.SimpleNamespace(jit=lambda fn,**kw:fn),
          optax=types.SimpleNamespace(apply_updates=updates),_loss_and_grads=grads,
          _FP32_POLICY=Policy(False),_tree_to_memory_kind=memory)
extract(T,['_make_train_step','template_for_candidate_layout','take_master_as_params'],ns)

def start(master,low=True,pending=None,ema=False):
 base=Model(np.asarray(1,dtype=np.float32),np.zeros((1,2),dtype=np.float32))
 return State(np.int32(0),Policy(low).cast_to_param(base),base if master else None,0,
              Policy(low).cast_to_param(base) if ema else None,
              np.zeros((1,2)) if pending is None else np.asarray(pending))

precision=[]
for name,mode,low in [('host_master_bf16',Mode.FP32_PINNED_HOST,True),('device_bf16_counterfactual',Mode.DEVICE,True),('device_fp32',Mode.DEVICE,False)]:
 mp=Policy(low)
 # The DEVICE BF16 scenario deliberately rounds apply_updates back to BF16.
 # Actual Optax/JAX dtype behavior is not executed; this is a declared finite-arithmetic illustration.
 ns['optax']=types.SimpleNamespace(apply_updates=(lambda p,u:mp.cast_to_param(updates(p,u))) if mode==Mode.DEVICE else updates)
 step=ns['_make_train_step'](Optimizer(),mp,z_loss_weight=0,ema_beta=None,offload_opt_state=True,master_param_mode=mode)
 state=start(mode==Mode.FP32_PINNED_HOST,low)
 for i in range(10):
  state,_,_=step(state,None)
  precision.append({'case':name,'completed_updates':i+1,'compute_weight':float(state.params.weight),
                    'master_weight':None if state.master_params is None else float(state.master_params.weight),
                    'optimizer_counter':int(state.opt_state)})
 last=precision[-1]
 check('Original train step increments optimizer metadata '+name,last['optimizer_counter']==10 and int(state.step)==10)
 if name=='device_bf16_counterfactual':check('Illustrative repeated BF16 updates erase sub-ULP increments',last['compute_weight']==1)
 else:check('FP32 authoritative path accumulates ten increments '+name,abs((last['master_weight'] if state.master_params else last['compute_weight'])-(1-10*.1*float(bf16(.01))))<1e-6)
check('Host authoritative parameters recast to finite BF16 replica',precision[9]['compute_weight']==float(bf16(precision[9]['master_weight'])))

ns['optax']=types.SimpleNamespace(apply_updates=updates)
gradient_inputs.clear();memory_moves.clear()
step=ns['_make_train_step'](Optimizer(),Policy(True),z_loss_weight=0,ema_beta=None,offload_opt_state=True,master_param_mode=Mode.FP32_PINNED_HOST)
original=start(True,pending=[[0.,0.]])
after_one,_,_=step(original,[[1.,3.]])
before_second=apply_beta(after_one.params,after_one.pending_qb_betas)
after_two,_,_=step(after_one,[[2.,4.]])
check('Original step stores newly observed betas as pending',np.array_equal(after_one.pending_qb_betas,[[1.,3.]]))
check('Stored step-one params retain beta used on step one',np.array_equal(after_one.params.router_bias,[[0.,0.]]))
check('Next step uses pending betas before substituted gradient computation',gradient_inputs==[[[0.,0.]],[[1.,-1.]]])
check('Master and compute views share centered beta at step two',np.array_equal(after_two.master_params.router_bias,after_two.params.router_bias))
check('Offload ordering transfers optimizer and master in and out',memory_moves==['device','device','pinned_host','pinned_host']*2)
qb={'after_first_stored_bias':after_one.params.router_bias.tolist(),'after_first_pending_beta':after_one.pending_qb_betas.tolist(),
    'next_training_bias':before_second.router_bias.tolist(),'after_second_stored_bias':after_two.params.router_bias.tolist(),
    'gradient_input_biases':gradient_inputs.copy(),'raw_logits':[.1,.2],
    'stored_bias_top1':int(np.argmax(np.array([.1,.2])+after_one.params.router_bias[0])),
    'pending_applied_top1':int(np.argmax(np.array([.1,.2])+before_second.router_bias[0]))}
check('Synthetic near-tie routing changes expert under pending bias',qb['stored_bias_top1']==1 and qb['pending_applied_top1']==0)

# The same original step, with the full modeled state, resumes exactly under these substitutes.
clone=dataclasses.replace(after_one)
resumed,_,_=step(clone,[[2.,4.]])
check('Full modeled-state continuation equals uninterrupted two-step result',all(np.array_equal(getattr(resumed.params,k),getattr(after_two.params,k)) for k in ['weight','router_bias']) and resumed.opt_state==after_two.opt_state and np.array_equal(resumed.pending_qb_betas,after_two.pending_qb_betas))
without_pending=dataclasses.replace(after_one,pending_qb_betas=np.zeros((1,2)))
bad,_,_=step(without_pending,[[2.,4.]])
check('Same weights and step with missing pending beta changes next router state',not np.array_equal(bad.params.router_bias,after_two.params.router_bias))

# Migration helper executes against fabricated manifest-presence responses, never actual storage.
ns.update(checkpoint_stores_master=lambda c:c=='master',logger=logging.getLogger('state-probe'))
migrations=[]
for candidate in ['master','no-master']:
 for mode in [Mode.DEVICE,Mode.FP32_PINNED_HOST]:
  state=start(mode==Mode.FP32_PINNED_HOST,low=mode!=Mode.DEVICE)
  try:
   template=ns['template_for_candidate_layout'](state,candidate,mode)
   migrations.append({'checkpoint_has_master':candidate=='master','run_mode':mode,'error':None,
                      'template_params_is_none':template.params is None,'same_template':template is state})
   if candidate=='master' and mode==Mode.DEVICE:
    check('Migration template selects master slot instead of rounded params slot',template.params is None and template.master_params is state.params)
    migrated=ns['take_master_as_params'](template)
    check('Original post-restore helper moves master into params',migrated.params is state.params and migrated.master_params is None)
  except ValueError:
   check('Synthesizing host master from masterless layout is rejected',candidate=='no-master' and mode==Mode.FP32_PINNED_HOST)
   migrations.append({'checkpoint_has_master':False,'run_mode':mode,'error':'ValueError'})

configs=[]
for p in sorted((R/'sources/scale_2026_10_05').glob('config_*.json')):
 raw=json.loads(p.read_text())['data']['project']['run'];c=json.loads(raw['config']);t=c['trainer']['value']
 configs.append({'run':raw['name'],'master_param_mode':t['master_param_mode'],'offload_opt_state':t['offload_opt_state'],
                 'ema_beta':t['ema_beta'],'policy':t['trainer']['mp'],'load_checkpoint':t['trainer']['load_checkpoint'],
                 'source':str(p.relative_to(R)),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
check('Six archived ladder configs choose BF16 replica and FP32 host master',len(configs)==6 and all(x['master_param_mode']=='FP32_PINNED_HOST' and x['policy']['param_dtype']=='jax.numpy.bfloat16' and x['offload_opt_state'] is True and x['ema_beta'] is None for x in configs))

# Read exact original StepInfo property lambdas without importing its dependencies.
core_tree=ast.parse((D/'callback_core.py').read_text());info=next(n for n in core_tree.body if isinstance(n,ast.ClassDef) and n.name=='StepInfo')
properties={n.targets[0].id:n.value.args[0] for n in info.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['step','next_step']}
getter={k:eval(compile(ast.Expression(v),str(D/'callback_core.py'),'eval')) for k,v in properties.items()}
step_labels=[{'completed_updates':s,'log_step':getter['step'](types.SimpleNamespace(state=types.SimpleNamespace(step=s))),
              'next_step':getter['next_step'](types.SimpleNamespace(state=types.SimpleNamespace(step=s)))} for s in [1,393,394]]
check('Original callback labels first completed update as log step zero',step_labels[0]=={'completed_updates':1,'log_step':0,'next_step':1})
check('Original callback log labels differ from completed-update count by one',all(x['log_step']+1==x['completed_updates']==x['next_step'] for x in step_labels))
tree=ast.parse(T.read_text());fields=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GrugTrainState')
state_fields=[n.target.id for n in fields.body if isinstance(n,ast.AnnAssign)]
check('Actual training dataclass has six fields and no standalone RNG leaf',state_fields==['step','params','master_params','opt_state','ema_params','pending_qb_betas'])
source_text=T.read_text();weights_text=(D/'weights.py').read_text()
check('Inference weight consumer explicitly applies pending beta','return apply_qb_betas(cast(Transformer, state[weights_key]), pending)' in weights_text)
check('Callback supplies stored params without pending-beta helper','model_getter=lambda s: s.params' in source_text and 'eval_model_getter=lambda s: s.ema_params if s.ema_params is not None else s.params' in source_text)
check('Dataset cursor reconstructed from completed state step','train_loader.iter_from_step(int(state.step))' in source_text)
check('Checkpoint writer stores completed-update count','checkpointer.on_step(tree=state, step=current_step)' in source_text)
all_sources=[T,M]+sorted(D.glob('*.py'))
result={'status':'original_function_control_flow_with_substitutes','checks_passed':len(checks),'checks':checks,
        'pinned_revision':'84869ae8c91ffe64e9f761c5bd714542eb1876e0','source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in all_sources},
        'extracted_original_functions':extracted,'train_state_fields':state_fields,'precision_rows':precision,'qb_example':qb,
        'migration_cases':migrations,'ladder_configs':configs,'step_labels':step_labels,
        'substitutions':['JIT replaced with identity; no compilation or donation','NumPy arrays and finite BF16 RNE emulation',
                        'synthetic constant gradients and SGD update, not actual model/Adam','model and Eqx tree replacement adapters',
                        'memory transfer records only; no device transfers','fabricated manifest-presence values'],
        'actual_hardware_result':None,'actual_checkpoint_restore':None,'historical_execution_binding':None,
        'limitations':['No JAX autodiff, BF16 accelerator arithmetic, scheduling, or multi-host I/O reproduced',
                       'Toy top-1 differs from Hero top-k; illustrates decision sensitivity only',
                       'Stored-vs-pending evaluation timing is code-derived; magnitude and intended preferred semantics unmeasured',
                       'Control-flow resume equivalence does not prove full real token/state replay']}
(R/'analysis/train_state_probe.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('Original state-control-flow checks:',len(checks))
