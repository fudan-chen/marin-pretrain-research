"""Real four-local-CPU original loss shard_map/psum; CPU CE reference substitution."""
import os
os.environ['XLA_FLAGS']='--xla_force_host_platform_device_count=4'
os.environ['JAX_PLATFORMS']='cpu'
import pathlib,json,hashlib
R=pathlib.Path(__file__).resolve().parents[1]
base=R/'scripts/probe_loss_denominator_dtype_cpu.py'
nscope={'__file__':str(base)}
exec(compile(base.read_text().split("bf=run('bf16_257'")[0],str(base),'exec'),nscope)
jax=nscope['jax'];jnp=nscope['jnp'];np=nscope['np'];ns=nscope['ns'];extract=nscope['extract'];wrapper=nscope['wrapper'];ref=nscope['ref'];reduce=nscope['reduce']
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
mesh=None
ns.update(P=P,_current_mesh=lambda:mesh,partition_spec_of=lambda x:x.sharding.spec if isinstance(x.sharding,NamedSharding) else None,_axis_names=lambda x:() if x is None else x if isinstance(x,tuple) else (x,),_reshard_for_shard_map=lambda x,m,s:jax.device_put(x,NamedSharding(m,s)))
extract(R/'sources/scale_2026_10_05/grug_loss.py',['_token_dim_specs'])
checks=[];cases={}
def ck(name,value):
 if not value:raise RuntimeError(name)
 checks.append({'name':name,'passed':True})
def scalar(x):
 x=float(x)
 return x if np.isfinite(x) else 'inf' if x>0 else '-inf' if x<0 else 'nan'
def run(name,shape,dtype,scale=1.,counts=None,two_axes=False,replicated_axis=False):
 global mesh
 mesh=Mesh(np.array(jax.devices()).reshape((2,2) if two_axes or replicated_axis else (4,)),('data','sequence') if two_axes else ('data','model') if replicated_axis else ('data',))
 spec=P('data','sequence') if two_axes else P('data',None)
 weight=jnp.full(shape,scale,dtype)
 labels=jnp.zeros(shape,jnp.int32)
 if counts is not None:
  weight=(jnp.arange(shape[1])[None,:]<jnp.array(counts)[:,None]).astype(dtype)*scale
  labels=jnp.broadcast_to(jnp.array([0,1,0,1])[:,None],shape)
 hidden=jax.device_put(jnp.ones((*shape,1),jnp.float32),NamedSharding(mesh,P(*spec, None)))
 labels=jax.device_put(labels,NamedSharding(mesh,spec));weight=jax.device_put(weight,NamedSharding(mesh,spec));head=jax.device_put(jnp.zeros((1,2),jnp.float32),NamedSharding(mesh,P()))
 def fn(h):return wrapper(hidden,h,labels,weight=weight,dtype=jnp.float32,implementation='xla_fast_bwd')
 loss,grad=jax.value_and_grad(fn)(head)
 flat_hidden=jnp.ones((int(np.prod(shape)),1),jnp.float32);flat_labels=jnp.asarray(np.asarray(labels).reshape(-1));flat_weight=jnp.asarray(np.asarray(weight).reshape(-1)).astype(jnp.float32)
 def expected(h):
  per,_=ref(flat_hidden,flat_labels,h,dtype=jnp.float32)
  return reduce(per,'mean',flat_weight)
 el,eg=jax.value_and_grad(expected)(jnp.zeros((1,2),jnp.float32))
 denom=jax.shard_map(lambda w:jax.lax.psum(jnp.sum(w),tuple(x for entry in tuple(spec) for x in ns['_axis_names'](entry))),mesh=mesh,in_specs=spec,out_specs=P(),check_vma=False)(weight)
 row={'mesh_shape':dict(mesh.shape),'shape':list(shape),'weight_dtype':str(dtype),'local_denominators':[scalar(jnp.sum(s.data)) for s in weight.addressable_shards],'global_denominator':scalar(denom),'FP32_represented_weight_sum':scalar(jnp.sum(flat_weight)),'loss':scalar(loss),'loss_finite':bool(jnp.isfinite(loss)),'head_gradient':np.asarray(grad).tolist(),'head_gradient_finite':bool(jnp.all(jnp.isfinite(grad))),'reference_loss':scalar(el),'reference_head_gradient':np.asarray(eg).tolist()}
 # JSON does not encode NaN numeric constants.
 row['head_gradient']=[[scalar(v) for v in rr] for rr in np.asarray(grad)]
 row['reference_head_gradient']=[[scalar(v) for v in rr] for rr in np.asarray(eg)]
 cases[name]=row
 return row
ck('Four real local CPU devices available',len(jax.devices())==4 and jax.default_backend()=='cpu')
a=run('unequal_targets_fp32',(4,258),jnp.float32,counts=[257,1,0,258])
b=run('bf16_four_shards',(4,257),jnp.bfloat16)
c=run('fp16_global_overflow',(4,16384),jnp.float16)
d=run('fp16_uniform_half',(4,16384),jnp.float16,.5)
e=run('fp32_global_baseline',(4,16384),jnp.float32)
f=run('fp16_two_axes',(256,256),jnp.float16,two_axes=True)
g=run('fp32_two_axes',(256,256),jnp.float32,two_axes=True)
h=run('all_zero_fp32',(4,8),jnp.float32,0.)
i=run('fp32_replicated_model_axis',(4,16384),jnp.float32,replicated_axis=True)
ck('Unequal targets including empty shard globally weighted mean agrees',a['local_denominators']==[257,1,0,258] and a['global_denominator']==516 and np.isclose(a['loss'],a['reference_loss'],rtol=2e-6))
ck('Unequal targets including empty shard gradient agrees and is finite',a['head_gradient_finite'] and np.allclose(a['head_gradient'],a['reference_head_gradient'],atol=2e-7))
ck('BF16 local rounding persists through psum',b['local_denominators']==[256]*4 and b['global_denominator']==1024 and b['FP32_represented_weight_sum']==1028 and np.isclose(b['loss']/b['reference_loss'],1028/1024,rtol=2e-6))
ck('FP16 global overflow despite finite local denominators',c['local_denominators']==[16384]*4 and c['global_denominator']=='inf')
ck('FP16 globally overflowed mean has finite zero loss and zero gradient',c['loss']==0 and c['loss_finite'] and c['head_gradient']==[[0,0]])
ck('Uniform half scaling violates invariance on overflow path',d['global_denominator']==32768 and np.isclose(d['loss'],d['reference_loss'],rtol=2e-6) and np.allclose(d['head_gradient'],[[-.5,.5]]))
ck('FP32 same global token count restores expected loss and gradient',e['global_denominator']==65536 and np.isclose(e['loss'],e['reference_loss'],rtol=2e-6) and np.allclose(e['head_gradient'],[[-.5,.5]]))
ck('Original multi token-axis psum also exposes global overflow',f['mesh_shape']=={'data':2,'sequence':2} and f['global_denominator']=='inf' and f['loss']==0 and f['head_gradient']==[[0,0]] and f['local_denominators']==[16384]*4)
ck('FP32 two token axes agree with reference',g['global_denominator']==65536 and np.isclose(g['loss'],g['reference_loss'],rtol=2e-6) and np.allclose(g['head_gradient'],[[-.5,.5]]))
ck('All zero global weights have zero loss but nonfinite head gradient',h['global_denominator']==0 and h['loss']==0 and not h['head_gradient_finite'])
ck('Replicated non-token model axis is not counted twice',i['mesh_shape']=={'data':2,'model':2} and i['global_denominator']==65536 and i['local_denominators']==[32768]*4 and np.isclose(i['loss'],i['reference_loss'],rtol=2e-6) and np.allclose(i['head_gradient'],[[-.5,.5]]))
ck('Original shard_map and collective run with explicit CPU reference backend substitution',all(x['executed']=='original CPU reference' for x in nscope['backend_calls']))
paths=list(nscope['paths']) if 'paths' in nscope else [R/'sources/scale_2026_10_05/grug_loss.py',R/'sources/boundaries_2026_10_05/examples.py',R/'sources/contracts_2026_10_05/api.py',R/'sources/contracts_2026_10_05/reference.py']
out={'scope':'Original wrapper shard_map and psum on four virtual local CPU devices, original CPU CE reference; actual collectives, no GPU or multi-host network. Synthetic tensors and sharding helper adapters.','checks_passed':len(checks),'checks':checks,'cases':cases,'runtime':{'jax':jax.__version__,'numpy':np.__version__,'devices':[str(x) for x in jax.devices()]},'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'base_probe_sha256':hashlib.sha256(base.read_bytes()).hexdigest(),'probe_script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'adapters':['NamedSharding spec lookup instead of partition_spec_of','_axis_names tuple normalization','_current_mesh explicit test mesh','_reshard_for_shard_map real device_put with NamedSharding','original CPU reference and reducer instead of fused backend dispatcher'],'actual_GPU_execution':None,'actual_multi_host_execution':None,'actual_Hero_weight_dtype':None,'actual_training_loss_effect':None,'upstream_patch_applied':False}
(R/'analysis/loss_denominator_mesh_cpu.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print('Loss denominator four-CPU controls:',len(checks))
