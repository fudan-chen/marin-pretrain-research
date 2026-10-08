"""Original streaming custom VJP registration, forward and both backwards on CPU."""
import ast,pathlib,json,hashlib,functools,typing
import jax,jax.numpy as jnp,numpy as np
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/contracts_2026_10_05';ns={'jax':jax,'jnp':jnp,'partial':functools.partial,'cast':typing.cast}
def load(path,names,registration=False):
 allnodes=ast.parse(path.read_text()).body
 nodes=[n for n in allnodes if isinstance(n,ast.FunctionDef) and n.name in names]
 assert len(nodes)==len(names)
 if registration:
  nodes += [n for n in allnodes if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='defvjp' and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='_linear_softmax_cross_entropy_loss_streaming_custom_vjp']
 exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])),str(path),'exec'),ns)
load(D/'reference.py',['_default_logsumexp','_apply_logit_soft_cap','linear_softmax_cross_entropy_loss_reference','linear_softmax_cross_entropy_loss_streaming'])
ns['_cross_entropy_logsumexp']=ns['_default_logsumexp'];ns['_cross_entropy_logaddexp']=jnp.logaddexp;ns['_cross_entropy_exp']=jnp.exp
load(D/'xla.py',['_materialize_cotangent','_linear_softmax_cross_entropy_loss_streaming_fwd','_linear_softmax_cross_entropy_loss_streaming_bwd','_linear_softmax_cross_entropy_loss_streaming_bwd_scan','_linear_softmax_cross_entropy_loss_streaming_custom_vjp','_linear_softmax_cross_entropy_loss_streaming_custom_vjp_fwd','_linear_softmax_cross_entropy_loss_streaming_custom_vjp_bwd'],True)
ref=ns['linear_softmax_cross_entropy_loss_reference'];stream=ns['linear_softmax_cross_entropy_loss_streaming'];custom=ns['_linear_softmax_cross_entropy_loss_streaming_custom_vjp'];checks=[];cases={}
def ck(name,v):
 if not v:raise RuntimeError(name)
 checks.append({'name':name,'passed':True})
def scalar(v):
 v=float(v);return v if np.isfinite(v) else 'nan' if np.isnan(v) else 'inf' if v>0 else '-inf'
def arr(a):return [[scalar(v) for v in row] for row in np.asarray(a)]
def run(name,backend,dtype,values,block=2,z=0.,jit=False):
 x=jnp.ones((1,1),jnp.float32);w=jnp.array([values],jnp.float32);labels=jnp.array([2],jnp.int32)
 def fn(x,w):
  if backend=='full':loss,lse=ref(x,labels,w,dtype=dtype)
  elif backend=='streaming_autodiff':loss,lse=stream(x,labels,w,block_size=block,dtype=dtype)
  else:loss,lse=custom(block,1,dtype,None,None,backend=='custom_scan',None,None,x,labels,w)
  return jnp.sum(loss+z*lse**2)
 f=jax.value_and_grad(fn,argnums=(0,1))
 if jit:f=jax.jit(f)
 loss,(gx,gw)=f(x,w)
 row={'backend':backend,'logits_dtype':str(dtype),'vocab_block_size':block,'z_coefficient':z,'jit':jit,'input_finite':bool(jnp.all(jnp.isfinite(w))),'represented_logits':arr(w.astype(dtype)),'loss':scalar(loss),'loss_finite':bool(jnp.isfinite(loss)),'hidden_gradient':arr(gx),'head_gradient':arr(gw),'gradient_finite':bool(jnp.all(jnp.isfinite(gx)) & jnp.all(jnp.isfinite(gw)))};cases[name]=row;return row
normal=[-1.,0.,1.];extreme=[-1e5,-1e5,0.]
for backend in ['full','streaming_autodiff','custom_slow','custom_scan']:
 run('normal_'+backend,backend,jnp.float32,normal)
 run('fp16_extreme_'+backend,backend,jnp.float16,extreme)
 run('normal_z_'+backend,backend,jnp.float32,normal,z=.1)
run('fp16_extreme_stream_block3','streaming_autodiff',jnp.float16,extreme,block=3)
run('bf16_extreme_stream','streaming_autodiff',jnp.bfloat16,extreme)
run('fp32_extreme_stream','streaming_autodiff',jnp.float32,extreme)
run('fp16_extreme_custom_scan_jit','custom_scan',jnp.float16,extreme,jit=True)
def close(a,b):
 return np.isclose(a['loss'],b['loss'],rtol=2e-6,atol=2e-7) and np.allclose(a['head_gradient'],b['head_gradient'],rtol=2e-6,atol=2e-7) and np.allclose(a['hidden_gradient'],b['hidden_gradient'],rtol=2e-6,atol=2e-7)
ck('Normal FP32 full and direct streaming value and both gradients agree',close(cases['normal_full'],cases['normal_streaming_autodiff']))
ck('Original slow custom VJP normal FP32 agrees with full reference',close(cases['normal_full'],cases['normal_custom_slow']))
ck('Original scan custom VJP normal FP32 agrees with full reference',close(cases['normal_full'],cases['normal_custom_scan']))
ck('Extreme inputs finite before requested FP16 conversion',cases['fp16_extreme_streaming_autodiff']['input_finite'] and cases['fp16_extreme_streaming_autodiff']['represented_logits']==[['-inf','-inf',0.0]])
ck('Direct streaming FP16 finite zero forward has nonfinite gradient',cases['fp16_extreme_streaming_autodiff']['loss']==0 and not cases['fp16_extreme_streaming_autodiff']['gradient_finite'])
ck('Full reference selected FP16 extreme gradient is finite zero',cases['fp16_extreme_full']['gradient_finite'] and cases['fp16_extreme_full']['head_gradient']==[[0,0,0]])
ck('Changing only direct streaming block size removes selected gradient failure',cases['fp16_extreme_stream_block3']['loss']==0 and cases['fp16_extreme_stream_block3']['gradient_finite'])
ck('Original slow custom VJP avoids selected direct streaming differentiation failure',close(cases['fp16_extreme_full'],cases['fp16_extreme_custom_slow']) and cases['fp16_extreme_custom_slow']['gradient_finite'])
ck('Original scan custom VJP avoids selected direct streaming differentiation failure',close(cases['fp16_extreme_full'],cases['fp16_extreme_custom_scan']) and cases['fp16_extreme_custom_scan']['gradient_finite'])
ck('Original scan custom VJP selected extreme JIT matches eager',close(cases['fp16_extreme_custom_scan'],cases['fp16_extreme_custom_scan_jit']))
ck('BF16 and FP32 direct streaming selected controls remain finite',all(cases[n]['loss']==0 and cases[n]['gradient_finite'] for n in ['bf16_extreme_stream','fp32_extreme_stream']))
ck('Nonzero lse cotangent propagates through original slow custom VJP',close(cases['normal_z_full'],cases['normal_z_custom_slow']))
ck('Nonzero lse cotangent propagates through original scan custom VJP',close(cases['normal_z_full'],cases['normal_z_custom_scan']))
ck('Nonzero z objective differs from pure CE while all selected normal paths agree',cases['normal_z_full']['loss']>cases['normal_full']['loss'] and close(cases['normal_z_full'],cases['normal_z_streaming_autodiff']))
paths=[D/'reference.py',D/'xla.py'];out={'scope':'Original custom_vjp decorator and defvjp registration, fwd, slow backward and scan backward on CPU synthetic tensors; original reference comparison, not public dispatcher or GPU kernel.','checks_passed':len(checks),'checks':checks,'cases':cases,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'probe_script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'runtime':{'jax':jax.__version__,'numpy':np.__version__,'backend':jax.default_backend()},'substitutions':['Select original default CPU logsumexp','Select jnp.logaddexp and jnp.exp default CPU branches'],'actual_backend_dispatcher_execution':False,'actual_Hero_FP16_logits_event':None,'actual_GPU_execution':None,'actual_training_loss_effect':None,'upstream_patch_applied':False}
(R/'analysis/ce_custom_vjp_cpu.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print('Original CE custom VJP CPU controls:',len(checks))
