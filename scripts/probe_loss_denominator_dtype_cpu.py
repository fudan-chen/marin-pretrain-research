"""Original Grug wrapper + original CPU CE reference/reducer, direct weight dtype controls.
Original causal constructor protects dtype with an attention-mask adapter. No fused GPU or production incident.
"""
import ast,hashlib,json,pathlib,types
import jax,jax.numpy as jnp,numpy as np
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/contracts_2026_10_05';G=R/'sources/scale_2026_10_05/grug_loss.py';E=R/'sources/boundaries_2026_10_05/examples.py'
ns={'jax':jax,'jnp':jnp,'named_call':lambda f:f,'_current_mesh':lambda:None}
def extract(path,names,container=None):
 nodes=ast.parse(path.read_text()).body
 if container:nodes=next(n for n in nodes if isinstance(n,ast.ClassDef) and n.name==container).body
 nodes=[n for n in nodes if isinstance(n,ast.FunctionDef) and n.name in names]
 if len(nodes)!=len(names):raise RuntimeError('missing AST functions')
 for n in nodes:n.decorator_list=[]
 exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])),str(path),'exec'),ns)
extract(D/'api.py',['_apply_reduction']);reduce=ns['_apply_reduction']
extract(D/'reference.py',['_default_logsumexp','_apply_logit_soft_cap','linear_softmax_cross_entropy_loss_reference']);ns['_cross_entropy_logsumexp']=ns['_default_logsumexp'];ref=ns['linear_softmax_cross_entropy_loss_reference']
backend_calls=[]
def delegate(x,labels,w,**kw):
 backend_calls.append({'requested':kw.get('implementation'),'executed':'original CPU reference','requested_logits_dtype':str(kw['dtype']),'input_weight_dtype':str(kw['weight'].dtype)})
 losses,_=ref(x,labels,w,dtype=kw['dtype'])
 return reduce(losses,kw['reduction'],kw['weight'])
ns['fused_cross_entropy_loss_and_logsumexp_penalty']=delegate
extract(G,['_psum_over_axes','fused_linear_softmax_cross_entropy_loss']);wrapper=ns['fused_linear_softmax_cross_entropy_loss']
class Example:
 def __init__(self,**kw):self.__dict__.update(kw)
class Mask:
 @staticmethod
 def causal(**kw):return Mask()
ns.update(GrugLmExample=Example,GrugAttentionMask=Mask)
extract(E,['causal','causal_loss_mask'],'GrugLmExample');Example.causal_loss_mask=staticmethod(ns['causal_loss_mask']);Example.causal=staticmethod(ns['causal'])
checks=[];cases={}
def ck(name,value):
 if not value:raise RuntimeError(name)
 checks.append({'name':name,'passed':True})
def scalar(v):
 x=float(v)
 return x if np.isfinite(x) else 'inf' if x>0 else '-inf' if x<0 else 'nan'
def run(name,n,dtype,scale=1.,promote=False,constructor=False):
 weight=jnp.full((n,),scale,dtype=dtype)
 if constructor:weight=Example.causal(jnp.ones(n,jnp.int32),loss_weight=weight,block_cross_document_attention=False).loss_weight
 if promote:weight=weight.astype(jnp.float32)
 hidden=jnp.ones((n,1),jnp.float32);labels=jnp.zeros(n,jnp.int32);head=jnp.zeros((1,2),jnp.float32)
 def fn(h):return wrapper(hidden,h,labels,weight=weight,dtype=jnp.float32,implementation='xla_fast_bwd')
 loss,grad=jax.value_and_grad(fn)(head)
 perloss,_=ref(hidden,labels,head,dtype=jnp.float32)
 api=reduce(perloss,'mean',weight)
 row={'positions':n,'weight_dtype':str(weight.dtype),'weight_scale':scale,'weight_sum_dtype':str(jnp.sum(weight).dtype),'original_weight_sum':scalar(jnp.sum(weight)),'FP32_sum_of_represented_weights':float(jnp.sum(weight.astype(jnp.float32))),'wrapper_loss':scalar(loss),'wrapper_loss_finite':bool(jnp.isfinite(loss)),'head_gradient':np.asarray(grad).tolist(),'shared_API_mean_loss':scalar(api),'constructor_applied':constructor,'weight_promotion_before_wrapper':promote};cases[name]=row
 return row
bf=run('bf16_257',257,jnp.bfloat16);fp=run('fp32_257',257,jnp.float32)
over=run('fp16_65536',65536,jnp.float16);half=run('fp16_65536_half_scale',65536,jnp.float16,.5)
large=run('fp32_65536',65536,jnp.float32)
prom=run('promoted_fp16_65536',65536,jnp.float16,promote=True)
cbf=run('causal_constructor_bf16_258',258,jnp.bfloat16,constructor=True)
cfp=run('causal_constructor_fp16_65537',65537,jnp.float16,constructor=True)
ck('BF16 sum of 257 unit weights rounds to256',bf['original_weight_sum']==256 and bf['FP32_sum_of_represented_weights']==257)
ck('BF16 denominator changes normalized loss and gradient',np.isclose(bf['wrapper_loss']/fp['wrapper_loss'],257/256,rtol=2e-6) and np.allclose(np.asarray(bf['head_gradient'])/np.asarray(fp['head_gradient']),257/256,rtol=2e-6))
ck('shared API FP32 loss promotes weights and avoids selected BF16 denominator gap',np.isclose(bf['shared_API_mean_loss'],fp['wrapper_loss'],rtol=2e-6))
ck('FP16 represented unit weights sum to inf at65536',over['original_weight_sum']=='inf' and over['FP32_sum_of_represented_weights']==65536)
ck('finite zero loss and zero head gradient pass loss finiteness after denominator overflow',over['wrapper_loss']==0 and over['wrapper_loss_finite'] and np.array_equal(over['head_gradient'],[[0.,0.]]))
ck('uniform half scaling restores nonzero normalized loss and gradient in overflow control',half['original_weight_sum']==32768 and np.isclose(half['wrapper_loss'],large['wrapper_loss'],rtol=2e-6) and np.allclose(half['head_gradient'],large['head_gradient'],rtol=2e-6))
ck('shared API with FP32 per-token loss remains finite for original FP16 weights',np.isclose(over['shared_API_mean_loss'],large['wrapper_loss'],rtol=2e-6))
ck('promoting represented weights before wrapper restores selected loss and gradient',prom['weight_dtype']=='float32' and np.isclose(prom['wrapper_loss'],large['wrapper_loss'],rtol=2e-6) and np.allclose(prom['head_gradient'],large['head_gradient'],rtol=2e-6))
ck('original causal constructor promotes BF16 weights and removes final target',cbf['weight_dtype']=='float32' and cbf['original_weight_sum']==257 and np.isclose(cbf['wrapper_loss'],fp['wrapper_loss'],rtol=2e-6))
ck('original causal constructor protects selected FP16 input with65536 valid targets',cfp['weight_dtype']=='float32' and cfp['original_weight_sum']==65536 and np.isclose(cfp['wrapper_loss'],large['wrapper_loss'],rtol=2e-6))
# Promotion after an earlier lossy cast is not reconstruction of original weights.
small=jnp.full((4,),2.**-26,jnp.float32);quantized=small.astype(jnp.float16).astype(jnp.float32)
ck('promotion cannot recover small weights already rounded tozero',bool(jnp.all(small>0)) and bool(jnp.all(quantized==0)))
ck('explicit backend substitution rather than GPU dispatcher execution',len(backend_calls)>0 and all(c['executed']=='original CPU reference' and c['requested']=='xla_fast_bwd' for c in backend_calls))
result={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'cases':cases,'backend_calls':backend_calls,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [G,E,D/'api.py',D/'reference.py']},'probe_script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'runtime':{'jax':jax.__version__,'numpy':np.__version__,'backend':jax.default_backend()},'actual_Hero_low_precision_weight_denominator':None,'actual_fused_GPU_execution':None,'actual_distributed_reduction':None,'actual_training_loss_effect':None,'upstream_patch_applied':False}
(R/'analysis/loss_denominator_dtype_cpu.json').write_text(json.dumps(result,indent=2)+'\n');print('Loss denominator dtype controls:',len(checks))
