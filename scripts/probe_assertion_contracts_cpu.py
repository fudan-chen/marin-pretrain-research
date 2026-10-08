"""Execute unchanged uploaded sampler/config bodies at Python compile optimization 0/1/2.
Synthetic finite identities, dependency adapters, CPU JAX; no full launcher or training.
Probe acceptance checks deliberately use explicit raises, not removable asserts.
"""
import ast, asyncio, collections, contextlib, functools, hashlib, json, logging, pathlib, platform, sys, types, typing, warnings
import jax, numpy as np
R=pathlib.Path(__file__).resolve().parents[1]
M=R/'sources/run_code_2026_10_08/mixture.py'
D=R/'sources/run_code_2026_10_08/datasets.py'
T=typing.TypeVar('T')
class Base(typing.Generic[T]): pass
class Stops:
 FIRST_STOP_STRATEGY='first_exhausted'; ALL_STOP_STRATEGY='all_exhausted'; RESTART_STRATEGY='restart'
 def __contains__(self,x): return x in [self.FIRST_STOP_STRATEGY,self.ALL_STOP_STRATEGY,self.RESTART_STRATEGY]
async def future(x): return x
class Identity:
 def __init__(self,name): self.name=name
 def is_finite(self): return True
 async def async_len(self): return 100
 async def get_batch(self,indices):
  if any(i<0 or i>=100 for i in indices): raise IndexError(indices)
  return [self.name+':'+str(i) for i in indices]
 async def getitem_async(self,i): return (await self.get_batch([i]))[0]
def module(nodes):
 return ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[]))
class ExplicitContracts(ast.NodeTransformer):
 def visit_Assert(self,node):
  message=node.msg if node.msg is not None else ast.Constant(value="Invalid stage configuration")
  return ast.copy_location(ast.If(test=ast.UnaryOp(op=ast.Not(),operand=node.test),body=[ast.Raise(exc=ast.Call(func=ast.Name(id="ValueError",ctx=ast.Load()),args=[message],keywords=[]),cause=None)],orelse=[]),node)
def environment(level,explicit=False):
 ns={'AsyncDataset':Base,'T':T,'StopStrategy':Stops(),'local_cpu_mesh':contextlib.nullcontext,'future_from_value':future,'jax':jax,'np':np,'asyncio':asyncio,'functools':functools,'warnings':warnings,'logger':logging.getLogger(__name__)}
 nodes=[n for n in ast.parse(M.read_text()).body if getattr(n,'name',None) in ['MixtureDataset','_compute_block_assignment']]
 if explicit:
  mix_class=next(n for n in nodes if isinstance(n,ast.ClassDef))
  init=next(n for n in mix_class.body if getattr(n,'name',None)=='__init__')
  ExplicitContracts().visit(init)
 exec(compile(module(nodes),str(M),'exec',optimize=level),ns)
 cls=next(n for n in ast.parse(D.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='LmDataConfig')
 post=next(n for n in cls.body if getattr(n,'name',None)=='__post_init__')
 if explicit: ExplicitContracts().visit(post)
 exec(compile(module([post]),str(D),'exec',optimize=level),ns)
 return ns
async def sample(ns,stages):
 try:
  m=ns['MixtureDataset']({'A':Identity('A'),'B':Identity('B')},stages,8,key=7,randomize_blocks=False)
  ids=await m.get_batch(list(range(32)))
  raw=[]
  for i in range(32):
   ds,idx=m._index_into_dataset_for_id(m._get_block(i//8)[i%8],i//8)
   raw.append([m.dataset_index[ds],idx])
  return {'accepted':True,'stream':ids,'logical_unwrapped':raw,'counts_after_stage':[v.tolist() for v in m._counts_after_stage],'stages_for_first_four_blocks':[m._get_stage_for_block(i) for i in range(4)],'repeated_identity_slots':len(ids)-len(set(ids))}
 except Exception as e: return {'accepted':False,'error_type':type(e).__name__,'error':str(e)}
def config(ns,budget,split,weights=None):
 c=types.SimpleNamespace(components={'A':object()},train_weights={'A':1.0} if weights is None else weights,max_train_batches=None,num_validation_sequences=split,experiment_budget=budget,target_budget=None if budget is None else 1)
 try: ns['__post_init__'](c); return {'accepted':True}
 except Exception as e:return {'accepted':False,'error_type':type(e).__name__,'error':str(e)}
async def main():
 checks=[]
 def check(name,condition):
  if not condition: raise RuntimeError(name)
  checks.append({'name':name,'passed':True})
 cases={}
 for level in [0,1,2]:
  ns=environment(level)
  cases[str(level)]={
   'aligned':await sample(ns,[(0,{'A':1}),(8,{'B':1}),(16,{'A':1})]),
   'unaligned':await sample(ns,[(0,{'A':1}),(9,{'B':1}),(17,{'A':1})]),
   'nonzero_first':await sample(ns,[(8,{'A':1})]),
   'mixed_budget_split':config(ns,.5,{'A':4}),
   'valid_config':config(ns,None,{'A':4}),
   'explicit_weight_error':config(ns,None,None,{'missing':1.0})}
  try:ns['MixtureDataset']({}, {'A':1},8,key=7)
  except Exception as e:cases[str(level)]['explicit_empty_error']=type(e).__name__
 check('Actual CPU backend',jax.default_backend()=='cpu')
 check('Aligned accepted stream identical across all three compile levels',all(cases[str(l)]['aligned']==cases['0']['aligned'] for l in [1,2]))
 check('Unaligned stages rejected in optimization0',not cases['0']['unaligned']['accepted'] and cases['0']['unaligned']['error_type']=='AssertionError')
 check('Unaligned optimization1 and2 run original sampler and repeat eight identities',all(cases[str(l)]['unaligned']['accepted'] and cases[str(l)]['unaligned']['repeated_identity_slots']==8 for l in [1,2]))
 check('Nonzero first stage rejected normally but produces negative logical offsets when optimized',not cases['0']['nonzero_first']['accepted'] and all(cases[str(l)]['nonzero_first']['logical_unwrapped'][0]==['A',-8] for l in [1,2]))
 check('Restart modulo maps optimized negative logical offset to finite tail identity',all(cases[str(l)]['nonzero_first']['stream'][0]=='A:92' for l in [1,2]))
 check('Budget split exclusivity removed by optimization1 and2',not cases['0']['mixed_budget_split']['accepted'] and all(cases[str(l)]['mixed_budget_split']['accepted'] for l in [1,2]))
 check('Legal config accepted and explicit ValueError protections survive all levels',all(cases[str(l)]['valid_config']['accepted'] and cases[str(l)]['explicit_weight_error']['error_type']=='ValueError' and cases[str(l)]['explicit_empty_error']=='ValueError' for l in [0,1,2]))
 check('Optimized unaligned stream contains A8 through A15 twice',all(all(collections.Counter(cases[str(l)]['unaligned']['stream'])['A:'+str(i)]==2 for i in range(8,16)) for l in [1,2]))
 fixed={}
 for level in [0,1,2]:
  ns=environment(level,explicit=True)
  fixed[str(level)]={'aligned':await sample(ns,[(0,{'A':1}),(8,{'B':1}),(16,{'A':1})]),'unaligned':await sample(ns,[(0,{'A':1}),(9,{'B':1}),(17,{'A':1})]),'nonzero_first':await sample(ns,[(8,{'A':1})]),'mixed_budget_split':config(ns,.5,{'A':4})}
 check('Counterfactual explicit guards preserve aligned stream at all compile levels',all(fixed[str(l)]['aligned']==cases['0']['aligned'] for l in [0,1,2]))
 check('Counterfactual explicit guards reject three invalid configurations at all compile levels',all(fixed[str(l)][k].get('error_type')=='ValueError' for l in [0,1,2] for k in ['unaligned','nonzero_first','mixed_budget_split']))
 assertion_inventory={str(p.relative_to(R)):[{'line':n.lineno,'condition':ast.unparse(n.test)} for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,ast.Assert)] for p in [M,D]}
 out={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'runtime':{'python':platform.python_version(),'probe_interpreter_optimize':sys.flags.optimize,'jax':jax.__version__,'numpy':np.__version__,'backend':jax.default_backend()},'probe_script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'compiled_source_optimization_levels':[0,1,2],'cases':cases,'counterfactual_explicit_guard_cases':fixed,'counterfactual_patch_applied_upstream':False,'assertion_inventory':assertion_inventory,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [M,D]},'substitutions':['AsyncDataset generic base','StopStrategy constants','local CPU mesh null context','value future','finite strict identity stores A and B length100','compile optimize explicitly0/1/2 for original selected bodies, not full module import or launcher'], 'actual_historical_PYTHONOPTIMIZE':None,'actual_Hero_assertion_bypass':None,'actual_training_loss_effect':None,'actual_GPU_execution':None}
 (pathlib.Path(sys.argv[1]) if len(sys.argv)>1 else R/'analysis/assertion_contracts_cpu.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
 print('Original source optimization controls:',len(checks),'passed')
 print('Optimized unaligned stream:',cases['1']['unaligned']['stream'])
if __name__=='__main__':asyncio.run(main())
