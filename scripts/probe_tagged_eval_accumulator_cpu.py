"""Original evaluator accumulator/result construction on CPU fixture arrays.
Original Equinox state classes; loader/loss, hax JIT/shard/where and mesh adapters explicit.
No model forward, production input, GPU or historical execution binding.
"""
import ast,dataclasses,hashlib,json,pathlib,types,typing
import equinox as eqx,jax,jax.numpy as jnp,numpy as np
R=pathlib.Path(__file__).resolve().parents[1];E=R/'sources/scale_2026_10_05/eval.py';S=R/'sources/eval_metrics_2026_10_05/stat_utils.py'
Arrayish=typing.Any
hax=types.SimpleNamespace(where=jnp.where,named_jit=lambda **kw:lambda f:f,shard=lambda x:x)
class LoadingTimeTrackerIterator:
 def __init__(self,loader):self.loader=loader;self.total_time=0.
 def __iter__(self):return iter(self.loader)
def tqdm(it,*args,**kw):return it

def execute(nodes,p):
 exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])),str(p),'exec'),globals())
execute([next(n for n in ast.parse(S.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='RunningMean')],S)
tree=ast.parse(E.read_text());cl=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TaggedEvaluator')
execute([n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ['_EvalRunningMeans','EvalResult']],E)
execute([n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name in ['_make_accum_for_batch','evaluate','_construct_tag_hierarchy']],E)

def run(batches,tag_names=('paloma/A','paloma/B'),byte_table=(1.,3.)):
 ev=types.SimpleNamespace(bytes_per_token=jnp.array(byte_table),device_mesh=None,per_pos_out_sharding=None,axis_mapping=None,dataset=types.SimpleNamespace(num_tags=len(tag_names),tag_to_index={t:i for i,t in enumerate(tag_names)}),loss_fn=lambda model,batch:batch,loader=batches)
 ev.hierarchy=_construct_tag_hierarchy(ev);ev.accum_for_batch=_make_accum_for_batch(ev)
 result=evaluate(ev,None)
 return {'micro_CE':result.micro_avg_loss,'root_macro_CE':result.macro_avg_loss,'parent_macro_CE':float(result.tag_macro_losses['paloma']),'parent_micro_CE':float(result.tag_micro_losses['paloma']),'micro_BPB':result.micro_bpb,'root_macro_BPB':result.macro_bpb,'parent_macro_BPB':float(result.tag_macro_bpb['paloma']),'leaf_CE':result.tag_micro_losses,'leaf_BPB':result.tag_micro_bpb}

def batch(losses,weights,ids,tags):return ((jnp.array(losses,dtype=jnp.float32),jnp.array(weights,dtype=jnp.float32),jnp.array(ids,dtype=jnp.int32)),jnp.array(tags,dtype=jnp.float32))
def main():
 checks=[];obs={}
 def check(n,c):assert c,(n,obs);checks.append(n)
 # Same two target losses/weights/ids, only batch partition changes.
 separate=[batch([[2]],[[1]],[[0]],[[1,0]]),batch([[2]],[[1]],[[1]],[[1,0]])]
 joined=[batch([[2],[2]],[[1],[1]],[[0],[1]],[[1,0],[1,0]])]
 obs['separate']=run(separate);obs['joined']=run(joined)
 check('Original finite root CE unchanged by repartition',abs(obs['separate']['micro_CE']-obs['joined']['micro_CE'])<1e-6)
 check('Original BPB changes under same targets regrouped',abs(obs['separate']['micro_BPB']-obs['joined']['micro_BPB'])>.4)
 check('Root macro includes zero-weight declared leaf',abs(obs['separate']['root_macro_CE']-1)<1e-6 and obs['separate']['leaf_CE']['paloma/B']==0)
 check('Parent macro filters zero-weight leaf',abs(obs['separate']['parent_macro_CE']-2)<1e-6)
 check('Root and parent macro BPB differ by zero leaf',abs(obs['separate']['parent_macro_BPB']-2*obs['separate']['root_macro_BPB'])<1e-6)
 obs['fractional']=run([batch([[2]],[[.25]],[[0]],[[1,0]])])
 check('Root CE clamp differs from positive leaf mean when mass below one',abs(obs['fractional']['micro_CE']-.5)<1e-6 and obs['fractional']['leaf_CE']['paloma/A']==2)
 check('Fractional byte clamp remains explicit',abs(obs['fractional']['micro_BPB']-.5*np.log2(np.e))<1e-6)
 # Inactive but finite target does not poison the accumulating scalar.
 good=[batch([[2]],[[1]],[[0]],[[1,0]])]
 obs['finite_zero_after']=run(good+[batch([[99]],[[0]],[[1]],[[0,1]])])
 check('Finite all-zero batch preserves preceding finite scalar',obs['finite_zero_after']['micro_CE']==2 and abs(obs['finite_zero_after']['micro_BPB']-2*np.log2(np.e))<1e-6)
 obs['nan_zero_after']=run(good+[batch([[float('nan')]],[[0]],[[1]],[[0,1]])])
 check('Zero weight times NaN poisons original global CE',np.isnan(obs['nan_zero_after']['micro_CE']))
 check('Original safe per-tag CE masks only empty-tag division',obs['nan_zero_after']['leaf_CE']['paloma/A']==2 and obs['nan_zero_after']['leaf_CE']['paloma/B']==0)
 check('BPB zero-weight update can poison leaf accumulator',all(np.isnan(v) for v in obs['nan_zero_after']['leaf_BPB'].values()))
 obs['root_fractional_repartition']=run([batch([[2]],[[.25]],[[0]],[[1,0]]),batch([[2]],[[.25]],[[0]],[[1,0]])])
 obs['root_fractional_joined']=run([batch([[2],[2]],[[.25],[.25]],[[0],[0]],[[1,0],[1,0]])])
 check('Root CE fractional clamp breaks repartition invariance',obs['root_fractional_repartition']['micro_CE']==.5 and obs['root_fractional_joined']['micro_CE']==1)
 # JSON must not emit nonstandard NaN tokens.
 def clean(x):
  if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
  if isinstance(x,np.generic):x=x.item()
  if isinstance(x,float) and not np.isfinite(x):return {'nonfinite':str(x)}
  return x
 o={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'observations':clean(obs),'runtime':{'jax':jax.__version__,'equinox':eqx.__version__,'numpy':np.__version__},'explicit_substitutions':['loss callback returns synthetic unweighted per-position arrays, no forward','loader list and loading/progress tracker adapters','hax.named_jit/shard identity; hax.where jnp.where','mesh=None; no distributed reduction'],'actual_Hero_affected':None,'actual_GPU_execution':None,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [E,S]}}
 (R/'analysis/tagged_eval_accumulator_cpu.json').write_text(json.dumps(o,indent=2,allow_nan=False)+'\n');print(json.dumps(o,indent=2,allow_nan=False))
if __name__=='__main__':main()
