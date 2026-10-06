"""JAX 0.7.2 / jaxlib 0.7.2 CPU checks of pinned numerical function bodies.
Not an executed Hero checkpoint, GPU kernel, distributed collective or complete training step.
"""
import ast,hashlib,json,math,pathlib,platform,types
import jax,jax.numpy as jnp,jaxlib,numpy as np
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/contracts_2026_10_05';checks=[];rows={};functions=[]
def ck(n,b):assert b,n;checks.append(n)
def extract(p,names,ns):
 nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names];assert {n.name for n in nodes}==set(names)
 future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[future]+nodes,type_ignores=[])),str(p),'exec'),ns)
 functions.extend({'file':str(p.relative_to(R)),'function':n.name,'first_line':n.lineno,'last_line':n.end_lineno} for n in nodes)
ns={'jax':jax,'jnp':jnp};extract(D/'api.py',['_apply_reduction'],ns);reduce=ns['_apply_reduction']
extract(D/'reference.py',['_default_logsumexp','_apply_logit_soft_cap','linear_softmax_cross_entropy_loss_reference'],ns)
ns['_cross_entropy_logsumexp']=ns['_default_logsumexp'] # explicit original CPU branch, not TPU dispatch
ns['_cross_entropy_exp']=jnp.exp # original default CPU operation; not TPU accuracy branch
extract(D/'xla.py',['_linear_softmax_cross_entropy_loss_streaming_bwd_scan'],ns);scan=ns['_linear_softmax_cross_entropy_loss_streaming_bwd_scan'];ref=ns['linear_softmax_cross_entropy_loss_reference']
ck('Pinned CPU-only probe runtime',jax.__version__==jaxlib.__version__=='0.7.2' and jax.default_backend()=='cpu')
x=jnp.array([2.,4.]);zero=jnp.zeros(2);active=jnp.array([1.,0.]);f=lambda a:reduce(a,'mean',zero)
for name,fn in [('eager',f),('jit',jax.jit(f))]:
 val,grad=jax.value_and_grad(fn)(x);rows['zero_mean_'+name]={'value':val,'gradient':grad};ck('Original zero-weight mean returns zero '+name,float(val)==0);ck('Original zero-weight mean produces NaN loss cotangents '+name,np.isnan(np.asarray(grad)).all())
def safe_reduce(loss,w):
 weighted=loss*w;t=jnp.sum(w);safe_t=jnp.where(t!=0,t,jnp.ones_like(t));return jnp.where(t!=0,jnp.sum(weighted)/safe_t,jnp.zeros_like(t))
for name,fn in [('eager',lambda a:safe_reduce(a,zero)),('jit',jax.jit(lambda a:safe_reduce(a,zero)))]:
 val,grad=jax.value_and_grad(fn)(x);rows['safe_zero_'+name]={'value':val,'gradient':grad};ck('Proposed safe denominator has zero finite cotangents '+name,float(val)==0 and np.array_equal(np.asarray(grad),np.zeros(2)))
ck('Safe denominator preserves positive-weight mean',np.allclose(np.asarray(safe_reduce(x,active)),np.asarray(reduce(x,'mean',active))))
for name,bad in [('NaN',jnp.nan),('Inf',jnp.inf)]:
 val=reduce(jnp.array([2.,bad]),'mean',active);rows['masked_'+name]={'value':val};ck('Original multiplication does not exclude inactive '+name,bool(jnp.isnan(val)))
unsafe=lambda a:safe_reduce(jnp.where(active>0,jnp.log(a),0.),active)
safe=lambda a:safe_reduce(jnp.log(jnp.where(active>0,a,1.)),active)
for name,fn in [('post_select_log',unsafe),('safe_log_operand',safe)]:
 val,grad=jax.value_and_grad(fn)(jnp.array([1.,0.]));rows[name]={'value':val,'gradient':grad}
ck('Post-loss selection can hide forward undefinedness but not its derivative',float(rows['post_select_log']['value'])==0 and np.isnan(np.asarray(rows['post_select_log']['gradient'])[1]))
ck('Making log operand safe yields inactive zero derivative',np.array_equal(np.asarray(rows['safe_log_operand']['gradient']),[1.,0.]))
hidden=jnp.array([[.2,.3],[.1,-.4]],dtype=jnp.float32);head=jnp.array([[.1,-.2,.5],[.3,.2,-.1]],dtype=jnp.float32);labels=jnp.array([0,2])
def ce(h,w,weight,reduction='mean'):
 loss,lse=ref(h,labels,w);return reduce(loss,reduction,weight)
for name,fn in [('eager',lambda h,w:ce(h,w,zero)),('jit',jax.jit(lambda h,w:ce(h,w,zero)))]:
 val,(gh,gw)=jax.value_and_grad(fn,argnums=(0,1))(hidden,head);rows['reference_zero_'+name]={'value':val,'hidden_gradient':gh,'head_gradient':gw};ck('Original CPU reference CE zero mean yields zero forward and nonfinite gradients '+name,float(val)==0 and np.isnan(np.asarray(gh)).all() and np.isnan(np.asarray(gw)).all())
val,(gh,gw)=jax.value_and_grad(lambda h,w:ce(h,w,zero,'sum'),argnums=(0,1))(hidden,head);rows['reference_zero_sum']={'value':val,'hidden_gradient':gh,'head_gradient':gw};ck('Original zero-weight sum has finite zero parameter derivatives',float(val)==0 and np.all(np.asarray(gh)==0) and np.all(np.asarray(gw)==0))
val,(gh,gw)=jax.value_and_grad(lambda h,w:ce(h,w,active),argnums=(0,1))(hidden,head);rows['reference_positive']={'value':val,'hidden_gradient':gh,'head_gradient':gw};ck('Finite inactive CE row has zero hidden gradient with positive denominator',np.isfinite(np.asarray(gh)).all() and np.isfinite(np.asarray(gw)).all() and np.all(np.asarray(gh)[1]==0))
loss,lse=ref(hidden,labels,head);bad_dout=jax.grad(lambda v:reduce(v,'mean',zero))(loss);good_dout=jax.grad(lambda v:safe_reduce(v,zero))(loss)
for name,dout in [('unsafe_zero',bad_dout),('safe_zero',good_dout)]:
 gh,gw=scan(hidden,labels,head,lse,dout,jnp.zeros_like(lse),block_size=2,dtype=jnp.float32,batch_block_size=2,logit_soft_cap=None,precision=None);rows['original_scan_'+name]={'dout_loss':dout,'hidden_gradient':gh,'head_gradient':gw}
ck('Original fast backward scan propagates NaN cotangents on CPU',np.isnan(np.asarray(rows['original_scan_unsafe_zero']['hidden_gradient'])).all() and np.isnan(np.asarray(rows['original_scan_unsafe_zero']['head_gradient'])).all())
ck('Original fast backward scan returns finite zeros for safe zero cotangents',np.all(np.asarray(rows['original_scan_safe_zero']['hidden_gradient'])==0) and np.all(np.asarray(rows['original_scan_safe_zero']['head_gradient'])==0))
bad_hidden=hidden.at[1,0].set(jnp.nan);bad_loss,bad_lse=ref(bad_hidden,labels,head);gh,gw=scan(bad_hidden,labels,head,bad_lse,active,jnp.zeros_like(bad_lse),block_size=2,dtype=jnp.float32,batch_block_size=2,logit_soft_cap=None,precision=None);rows['scan_inactive_nonfinite']={'hidden_gradient':gh,'head_gradient':gw};ck('Zero row cotangent does not sanitize nonfinite operands in original scan',np.isnan(np.asarray(gh)[1]).all() and np.isnan(np.asarray(gw)).any())
# Positive control: original tail-row and vocab-block padding, finite float32 data.
h3=jnp.concatenate([hidden,jnp.array([[-.3,.9]],dtype=jnp.float32)]);l3=jnp.array([0,2,1]);wt3=jnp.array([1.,.5,0.]);loss3,lse3=ref(h3,l3,head)
gh3,gw3=scan(h3,l3,head,lse3,wt3/wt3.sum(),jnp.zeros_like(lse3),block_size=2,dtype=jnp.float32,batch_block_size=2,logit_soft_cap=None,precision=None)
rg3=jax.grad(lambda h,w:reduce(ref(h,l3,w)[0],'mean',wt3),argnums=(0,1))(h3,head)
rows['scan_positive_tail_padding']={'hidden_gradient':gh3,'head_gradient':gw3,'reference_hidden_gradient':rg3[0],'reference_head_gradient':rg3[1]}
ck('Original scan tail-row and vocab padding match finite float32 reference gradients',np.allclose(np.asarray(gh3),np.asarray(rg3[0]),atol=1e-6,rtol=1e-6) and np.allclose(np.asarray(gw3),np.asarray(rg3[1]),atol=1e-6,rtol=1e-6))
# Original single-device Grug wrapper with an explicitly substituted original CPU reference delegate.
def reference_delegate(h,l,w,**kw):
 loss,lse=ref(h,l,w,dtype=kw['dtype']);z=kw.get('logsumexp_weight');loss=loss if z is None else loss+z*lse**2;return reduce(loss,kw['reduction'],kw['weight'])
ns.update({'named_call':lambda f:f,'_current_mesh':lambda:None,'fused_cross_entropy_loss_and_logsumexp_penalty':reference_delegate})
G=R/'sources/scale_2026_10_05/grug_loss.py';extract(G,['_psum_over_axes','fused_linear_softmax_cross_entropy_loss'],ns)
val,(gh,gw)=jax.value_and_grad(lambda h,w:ns['fused_linear_softmax_cross_entropy_loss'](h,w,labels,weight=zero),argnums=(0,1))(hidden,head);rows['grug_reference_delegate_zero']={'value':val,'hidden_gradient':gh,'head_gradient':gw};ck('Original Grug reducer plus CPU reference delegate reproduces zero forward and nonfinite derivatives',float(val)==0 and np.isnan(np.asarray(gh)).all() and np.isnan(np.asarray(gw)).all())
def encode(v):
 if isinstance(v,dict):return {k:encode(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [encode(x) for x in v]
 if isinstance(v,(jax.Array,np.ndarray)):return encode(np.asarray(v).tolist())
 if isinstance(v,float) and not math.isfinite(v):return 'NaN' if math.isnan(v) else ('+Inf' if v>0 else '-Inf')
 return v
out={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'runtime':{'python':platform.python_version(),'jax':jax.__version__,'jaxlib':jaxlib.__version__,'numpy':np.__version__,'backend':jax.default_backend(),'devices':[str(x) for x in jax.devices()],'jax_enable_x64':bool(jax.config.jax_enable_x64)},'observations':rows,'original_functions':functions,'adaptations':['AST-selected original bodies and future annotations','CPU logsumexp branch explicitly selects original _default_logsumexp','CPU exponential explicitly jnp.exp, no TPU AccuracyMode branch','Grug wrapper mesh=None and reference delegate replaces backend dispatcher','Direct original backward scan invocation with supplied cotangents, not complete custom_vjp entry'],'proposed_safe_denominator_is_upstream_patch':False,'actual_Hero_zero_denominator_batch':None,'actual_Hero_inactive_nonfinite_operands':None,'actual_historical_runtime_binding':None,'actual_GPU_or_TPU_kernel_execution':None,'actual_distributed_collectives':None,'actual_optimizer_step':None,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [D/'api.py',D/'reference.py',D/'xla.py',G]}}
(R/'analysis/masked_numerics_cpu.json').write_text(json.dumps(encode(out),ensure_ascii=False,indent=2,allow_nan=False)+'\n');print('Genuine JAX CPU checks:',len(checks))
