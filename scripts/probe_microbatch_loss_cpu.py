"""Original microbatched body with true JAX/Equinox autodiff and CPU reference loss.
Plain arrays only; hax fold uses lax.scan, shape/physical/sharding adapters explicit.
No Hero accumulation-path binding, model forward, custom gradient state or GPU.
"""
import ast,functools,hashlib,json,pathlib,types,typing
from typing import Any,cast
import equinox as eqx,jax,jax.numpy as jnp,numpy as np
from jax.sharding import PartitionSpec
import probe_loss_composition_cpu as loss_base
R=loss_base.R;F=R/'sources/accumulation_2026_10_07/grad_accum.py';U=R/'sources/tree_restore_2026_10_07/jax_utils.py';HU=R/'sources/tree_restore_2026_10_07/haliax_jax_utils.py'
class Axis:
 def __init__(self,name,size):self.name=name;self.size=size
 def resize(self,size):return Axis(self.name,size)
class NamedArray:pass
class CustomGradientAccumulation:pass
class Metric:pass
hax=types.SimpleNamespace(NamedArray=NamedArray,partitioning=types.SimpleNamespace(physical_axis_size=lambda *a:1,physical_axis_name=lambda *a:'data'),fold=lambda fn,axis:lambda acc,xs:jax.lax.scan(lambda c,x:(fn(c,x),None),acc,xs)[0],shard_with_axis_mapping=lambda x,m:x)
hq=types.SimpleNamespace(CustomGradientAccumulation=CustomGradientAccumulation,accumulate_gradients=lambda a,b:jax.tree_util.tree_map(lambda x,y:x+y,a,b))
is_named_array=lambda x:isinstance(x,NamedArray)
with_sharding_constraint=lambda x,p:x
ResourceAxis=types.SimpleNamespace(DATA='data')
reshape_array_into_microbatches=lambda x,n:x.reshape((n,x.shape[0]//n)+x.shape[1:])
fold_metric=lambda a,b:a+b

def extract(p,names):
 nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names];assert len(nodes)==len(names)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])),str(p),'exec'),globals())
extract(HU,['is_jax_array_like']);extract(U,['zeros_like_tree','_zeros_like']);extract(F,['microbatched','_reshape_for_microbatch'])
class Provider(loss_base.Provider):
 def __call__(self,tokens,mask=None):return jnp.ones(tokens.shape+(1,)),{'router_z_loss_per_layer':jnp.array([0.])}
def model_loss(h,t,w):return loss_base.ns['next_token_loss'](Provider(h),t,w)
def numerator_loss(h,t,w):
 labels=jnp.pad(t[:,1:],((0,0),(0,1)))
 return loss_base.ns['fused_linear_softmax_cross_entropy_loss'](jnp.ones(t.shape+(1,)),h,labels,weight=w,reduction='sum')
def vg(h,t,w,mode='mean',total=None,k=2):
 def objective(head):
  if mode=='sum_global':return numerator_loss(head,t,w)*k/total
  out=model_loss(head,t,w)
  return out if mode=='mean' else out*w.sum()*k/total
 value,grad=jax.value_and_grad(objective)(h)
 return ((value,{}),grad)
def result(r):return {'loss':float(r[0][0]),'gradient':np.asarray(r[1]).tolist()}
def accumulated(h,t,w,mode='mean'):
 fn=functools.partial(vg,h,mode=mode,total=w.sum())
 return microbatched(fn,Axis('batch',4),2,{},{})(t,w)
def main():
 checks=[]
 def check(n,c):assert c,n;checks.append(n)
 h=jnp.array([[-.4,0.]])
 t=jnp.array([[0,0,0,0],[0,0,0,0],[1,1,1,1],[1,1,1,1]])
 w=jnp.array([[1.,0,0,0],[0.,0,0,0],[1.,1,0,0],[1.,0,0,0]])
 full=vg(h,t,w);micro=accumulated(h,t,w);weighted=accumulated(h,t,w,'weighted_mean');summed=accumulated(h,t,w,'sum_global')
 check('True autodiff original mean-of-means reverses head gradient',float(full[1][0,0])>0>float(micro[1][0,0]))
 check('Denominator-weighted ordinary gradient matches full positive blocks',np.allclose(weighted[1],full[1],atol=1e-6) and np.allclose(weighted[0][0],full[0][0],atol=1e-6))
 check('Original numerator reduction normalized once matches full',np.allclose(summed[1],full[1],atol=1e-6) and np.allclose(summed[0][0],full[0][0],atol=1e-6))
 q=jnp.array([0,2,1,3]);reordered=accumulated(h,t[q],w[q]);full_reordered=vg(h,t[q],w[q])
 check('Repartition leaves full gradient unchanged but changes accumulated gradient',np.allclose(full_reordered[1],full[1],atol=1e-6) and not np.allclose(reordered[1],micro[1],atol=1e-6))
 equal=w.at[1,0].set(1).at[2,1].set(0)
 check('Equal effective denominators recover full objective',np.allclose(accumulated(h,t,equal)[1],vg(h,t,equal)[1],atol=1e-6))
 empty=w.at[:2].set(0);ef=vg(h,t,empty);em=accumulated(h,t,empty);ew=accumulated(h,t,empty,'weighted_mean');es=accumulated(h,t,empty,'sum_global')
 check('Whole batch positive denominator remains finite',np.isfinite(np.asarray(ef[1])).all())
 check('One empty microbatch yields finite loss but nonfinite ordinary gradients',np.isfinite(float(em[0][0])) and not np.isfinite(np.asarray(em[1])).all())
 check('Multiplying bad local mean by zero does not repair NaN gradient',not np.isfinite(np.asarray(ew[1])).all())
 check('Numerator-first avoids local zero-mean derivative and recovers full',np.isfinite(np.asarray(es[1])).all() and np.allclose(es[1],ef[1],atol=1e-6) and np.allclose(es[0][0],ef[0][0],atol=1e-6))
 update_full=h-.1*full[1];update_micro=h-.1*micro[1]
 check('Measured one-step SGD head moves in opposite directions',float(update_full[0,0])<float(h[0,0])<float(update_micro[0,0]))
 observations={'positive_blocks':{'mass':[1,3],'full':result(full),'ordinary':result(micro),'weighted_mean':result(weighted),'numerator_global':result(summed),'repartitioned':result(reordered),'sgd_head_full':np.asarray(update_full).tolist(),'sgd_head_ordinary':np.asarray(update_micro).tolist()},'one_empty_block':{'mass':[0,3],'full':result(ef),'ordinary':result(em),'weighted_mean':result(ew),'numerator_global':result(es)}}
 def clean(x):
  if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
  if isinstance(x,list):return [clean(v) for v in x]
  if isinstance(x,float) and not np.isfinite(x):return {'nonfinite':str(x)}
  return x
 files=[F,U,HU,loss_base.D/'api.py',loss_base.D/'reference.py',loss_base.G,loss_base.M,R/'scripts/probe_loss_composition_cpu.py']
 o={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'observations':clean(observations),'runtime':{'jax':jax.__version__,'equinox':eqx.__version__,'numpy':np.__version__},'explicit_substitutions':['synthetic fixed hidden/router provider; no Transformer forward','V72 original CPU reference delegate replacing GPU dispatch','Axis metadata/physical size=1; sharding identity','hax.fold replaced by real lax.scan; plain-array reshape adapter','ordinary gradient add adapter; no custom gradient or metrics; RNG absent'],'actual_Hero_accumulation_bug':None,'actual_GPU_execution':None,'actual_upstream_fix':None,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
 (R/'analysis/microbatch_loss_cpu.json').write_text(json.dumps(o,indent=2,allow_nan=False)+'\n');print(json.dumps(o['observations'],indent=2));print('Original microbatch/JAX checks:',len(checks))
if __name__=='__main__':main()
