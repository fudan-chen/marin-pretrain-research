"""Original generic microbatched bodies with serial NumPy adapters, not JAX execution."""
import ast,contextlib,functools,hashlib,json,pathlib,types
import numpy as np
R=pathlib.Path(__file__).resolve().parents[1];F=R/'sources/accumulation_2026_10_07/grad_accum.py';checks=[]
def ck(n,v):assert v,n;checks.append(n)
def tree(fn,*xs):
 x=xs[0]
 if isinstance(x,dict):return {k:tree(fn,*(z[k] for z in xs)) for k in x}
 if isinstance(x,(tuple,list)):return type(x)(tree(fn,*(z[i] for z in xs)) for i in range(len(x)))
 return fn(*xs)
class Axis:
 def __init__(self,name,size):self.name=name;self.size=size
 def resize(self,size):return Axis(self.name,size)
class NamedArray:pass
class CustomGradientAccumulation:pass
def partition(x,pred,**kwargs):return tree(lambda z:z if pred(z) else None,x),tree(lambda z:None if pred(z) else z,x)
def combine(x,y,**kwargs):return tree(lambda a,b:b if a is None else a,x,y)
def fold(fn,axis):
 def run(acc,xs):
  for i in range(axis.size):acc=fn(acc,tree(lambda x:x[i] if isinstance(x,np.ndarray) else x,xs))
  return acc
 return run
hax=types.SimpleNamespace(NamedArray=NamedArray,partitioning=types.SimpleNamespace(physical_axis_size=lambda *x:1,physical_axis_name=lambda *x:'data'),fold=fold,shard_with_axis_mapping=lambda x,m:x)
hq=types.SimpleNamespace(CustomGradientAccumulation=CustomGradientAccumulation,accumulate_gradients=lambda a,b:tree(lambda x,y:x+y,a,b))
eqx=types.SimpleNamespace(filter_eval_shape=lambda fn,*a,**kw:fn(*a,**kw),partition=partition,combine=combine)
class Metric:pass
ns={'np':np,'jnp':np,'hax':hax,'hq':hq,'eqx':eqx,'jax':types.SimpleNamespace(named_scope=lambda *x:contextlib.nullcontext(),tree_util=types.SimpleNamespace(tree_map=lambda fn,*xs,**kw:tree(fn,*xs))),'Axis':Axis,'functools':functools,'cast':lambda t,x:x,'Any':object,'R':object,'Metric':Metric,'fold_metric':lambda a,b:a+b,'is_named_array':lambda x:isinstance(x,NamedArray),'zeros_like_tree':lambda x,*a:tree(lambda v:np.zeros_like(v),x),'reshape_array_into_microbatches':lambda x,n:x.reshape((n,x.shape[0]//n)+x.shape[1:]),'with_sharding_constraint':lambda x,p:x,'PartitionSpec':lambda *x:x,'ResourceAxis':types.SimpleNamespace(DATA='data')}
module=ast.parse(F.read_text());selected=[n for n in module.body if isinstance(n,ast.FunctionDef) and n.name in ['microbatched','_reshape_for_microbatch']]
class Strip(ast.NodeTransformer):
 def visit_FunctionDef(self,n):
  n.returns=None
  for a in n.args.args+n.args.kwonlyargs:a.annotation=None
  return self.generic_visit(n)
selected=[Strip().visit(n) for n in selected];exec(compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),str(F),'exec'),ns)
# Scalar squared-error objective; exact analytic derivatives, no numerical/autodiff claim.
def value_grad(y,w,theta=0.,mode='effective'):
 n=np.sum(.5*(theta-y)**2*w);g=np.sum((theta-y)*w);t=w.sum() if mode=='effective' else w.size
 return ((np.float64(n/t if t else 0.),{}),np.float64(g/t if t else 0.))
def accumulated(y,w,micro=2,mode='effective'):
 return ns['microbatched'](lambda yy,ww:value_grad(yy,ww,mode=mode),Axis('batch',len(y)),micro,{},{})(y,w)
def scalar(r):return [float(r[0][0]),float(r[1])]
y=np.array([[3.,99.],[99.,99.],[-2.,-2.],[-2.,99.]])
w=np.array([[1.,0.],[0.,0.],[1.,1.],[1.,0.]])
full=scalar(value_grad(y,w));micro=scalar(accumulated(y,w));ck('Variable effective denominators change objective and reverse analytic gradient',np.allclose(full,[2.625,.75]) and np.allclose(micro,[3.25,-.5]))
q=np.array([0,2,1,3]);reordered=scalar(accumulated(y[q],w[q]));ck('Repartitioning same weighted targets changes mean-of-means',np.allclose(scalar(value_grad(y[q],w[q])),full) and not np.allclose(reordered,micro))
fixed=scalar(accumulated(y,w,mode='positions'));fixedfull=scalar(value_grad(y,w,mode='positions'));ck('Equal fixed position denominators preserve full-batch objective',np.allclose(fixed,fixedfull))
equal_w=np.ones_like(w);ck('Equal effective denominators preserve objective and derivative',np.allclose(scalar(accumulated(y,equal_w)),scalar(value_grad(y,equal_w))))
# A denominator-weighted aggregate of per-microbatch effective means recovers full result.
pieces=[scalar(value_grad(y[i:i+2],w[i:i+2])) for i in [0,2]];ts=[float(w[i:i+2].sum()) for i in [0,2]];correct=np.average(pieces,axis=0,weights=ts);ck('Effective-denominator weighting recovers full objective and derivative',np.allclose(correct,full))
# Empty chunk contributes zero but retains a step in the ordinary arithmetic average.
w_empty=w.copy();w_empty[:2]=0;empty=scalar(accumulated(y,w_empty));emptyfull=scalar(value_grad(y,w_empty));ck('Empty microbatch can halve ordinary mean result',np.allclose(empty,np.array(emptyfull)/2))
ck('All-empty artificial batches return declared zero policy',scalar(accumulated(y,np.zeros_like(w)))==[0.,0.])
ck('Original large microbatch shortcut returns original function',ns['microbatched'](value_grad,Axis('batch',4),4,{}, {}) is value_grad)
for size in [0,-1,3]:
 try:ns['microbatched'](value_grad,Axis('batch',4),size,{},{});rejected=False
 except ValueError:rejected=True
 ck('Original invalid microbatch size rejected '+str(size),rejected)
# Static call binding: generic trainer closes over model; specialized Hero path computes once on batch.
T=R/'sources/accumulation_2026_10_07/trainer.py';H=R/'sources/scale_2026_10_05/train_hero_ep.py';G=R/'sources/scale_2026_10_05/grug_loss.py'
ck('Pinned generic trainer closes model before wrapping microbatched','grad_fn = partial(eqx.filter_value_and_grad(loss_fn, has_aux=True), model)' in T.read_text())
ck('Pinned Hero train step has no generic microbatched call','microbatched(' not in H.read_text() and '(loss, summarized_metrics), grads = _loss_and_grads(qb_params, batch, mp, z_loss)' in H.read_text())
ck('Pinned grug reducer globally sums numerator and effective denominator','total_sum = _psum_over_axes(local_sum, token_sharding_axes)' in G.read_text() and 'total_denom = _psum_over_axes(local_denom, token_sharding_axes)' in G.read_text())
out={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'synthetic_variable_denominator':{'effective_targets_by_microbatch':ts,'full_loss_gradient':full,'ordinary_accumulated_loss_gradient':micro,'repartitioned_loss_gradient':reordered,'denominator_weighted_loss_gradient':correct.tolist()},'synthetic_fixed_positions':{'full_loss_gradient':fixedfull,'accumulated_loss_gradient':fixed},'synthetic_empty_microbatch':{'full_loss_gradient':emptyfull,'accumulated_loss_gradient':empty},'adapters':['NumPy scalar/array operations','serial tree partition/combine and fold','shape inference replaced by numerical pure function call','ordinary gradients only; no custom quantized gradient state','single-device physical axis stub','reshape adapter and identity sharding constraints','no RNG key supplied','analytic scalar squared-error derivatives, no model or autodiff'],'actual_JAX_execution':None,'actual_distributed_collectives':None,'actual_Hero_accumulation_bug':None,'actual_training_loss_function_binding':None,'actual_optimizer_trajectory':None,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [F,T,H,G]}}
(R/'analysis/gradient_accumulation_probe.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('Original body / artificial objective checks:',len(checks),full,micro)
# Synthetic visualization: analytic gradient, not optimizer/model measurements.
import matplotlib,re
matplotlib.use('Agg');matplotlib.rcParams['svg.hashsalt']='marin-gradient-accumulation-20261007'
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(8,4));values=[full[1],micro[1],reordered[1]];labels=['Global effective mean','Mean of micro means','After repartition']
ax.barh(labels,values,color=['#245987','#b85137','#6a8075']);ax.axvline(0,color='#333',lw=1);ax.set_xlabel('Analytic gradient dL/dtheta at theta=0');ax.set_title('Same weighted targets, different reduction\nSynthetic squared-error objective; no JAX/model/distributed execution',fontsize=11)
for i,x in enumerate(values):ax.text(x+.025 if x>=0 else x-.025,i,f'{x:+.4f}',va='center',ha='left' if x>=0 else 'right')
ax.set_xlim(-.8,1.5);fig.tight_layout();fig.savefig(R/'assets/gradient_accumulation.png',dpi=150);fig.savefig(R/'assets/gradient_accumulation.svg',metadata={'Date':None});plt.close(fig)
p=R/'assets/gradient_accumulation.svg';s=p.read_text();ids=re.findall(r'id="([^"]+)"',s)
for old in sorted(set(ids),key=len,reverse=True):s=s.replace('id="'+old+'"','id="gradacc-'+old+'"').replace('#'+old+'"','#gradacc-'+old+'"').replace('#'+old+')','#gradacc-'+old+')')
p.write_text('\n'.join(line.rstrip() for line in s.splitlines())+'\n')
