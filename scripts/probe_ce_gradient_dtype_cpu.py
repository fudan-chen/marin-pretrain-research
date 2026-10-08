"""Original registered CE VJP dtype and batch-chunk accumulation CPU controls."""
import pathlib,json,hashlib
R=pathlib.Path(__file__).resolve().parents[1];base=R/'scripts/probe_ce_custom_vjp_cpu.py';env={'__file__':str(base)}
exec(compile(base.read_text().split('normal=[-1.')[0],str(base),'exec'),env)
jax=env['jax'];jnp=env['jnp'];np=env['np'];custom=env['custom'];checks=[];cases={}
def ck(name,v):
 if not v:raise RuntimeError(name)
 checks.append({'name':name,'passed':True})
def run(name,n,xdtype,wdtype,T,fast,forward_chunk=1,backward_chunk=None,scale=1.,xvalue=1.):
 x=jnp.full((n,1),xvalue,xdtype);w=jnp.zeros((1,2),wdtype);labels=jnp.zeros(n,jnp.int32)
 def f(a,b):
  loss,_=custom(2,forward_chunk,jnp.float32,None,None,fast,backward_chunk,None,a,labels,b)
  return jnp.sum(loss)*scale/T
 value,(gx,gw)=jax.value_and_grad(f,argnums=(0,1))(x,w)
 unscaled=gw.astype(jnp.float32)/scale
 row={'rows':n,'x_dtype':str(xdtype),'w_dtype':str(wdtype),'backward_gemm_dtype':str(jnp.result_type(xdtype,wdtype)),'logits_dtype':'float32','fast_backward':fast,'forward_batch_chunk':forward_chunk,'backward_batch_chunk':backward_chunk,'denominator':T,'loss_scale':scale,'hidden_value':xvalue,'scaled_loss':float(value),'loss_before_scale':float(value/scale),'head_gradient_unscaled':np.asarray(unscaled).tolist(),'head_gradient_dtype':str(gw.dtype),'gradient_finite':bool(jnp.all(jnp.isfinite(gx))&jnp.all(jnp.isfinite(gw))),'FP32_analytic_head_gradient':[[-n*xvalue/(2*T),n*xvalue/(2*T)]],'per_row_precast_dlogit_magnitude':.5*scale/T};cases[name]=row;return row
for name,fast,chunk,bwd,dtype in [('bf16_slow_chunk1',False,1,None,jnp.bfloat16),('bf16_slow_chunk32',False,32,None,jnp.bfloat16),('bf16_slow_chunk512',False,512,None,jnp.bfloat16),('bf16_scan_chunk1',True,1,1,jnp.bfloat16),('bf16_scan_chunk512',True,1,512,jnp.bfloat16),('fp32_slow_chunk1',False,1,None,jnp.float32)]:
 run(name,512,dtype,dtype,512,fast,chunk,bwd)
for name,xd,wd,t,fast,scale in [('fp16_slow_power',jnp.float16,jnp.float16,2**26,False,1),('fp16_scan_power',jnp.float16,jnp.float16,2**26,True,1),('fp16_scan_power_scale8',jnp.float16,jnp.float16,2**26,True,8),('fp16_slow_declared_T',jnp.float16,jnp.float16,46126080,False,1),('fp16_scan_declared_T',jnp.float16,jnp.float16,46126080,True,1),('fp16_scan_declared_T_scale128',jnp.float16,jnp.float16,46126080,True,128),('bf16_scan_declared_T',jnp.bfloat16,jnp.bfloat16,46126080,True,1),('mixed_scan_declared_T',jnp.float16,jnp.float32,46126080,True,1)]:
 run(name,1,xd,wd,t,fast,scale=scale,xvalue=4096.)
def grad(name):return np.array(cases[name]['head_gradient_unscaled'])
ck('BF16 slow 512 one-row chunks lose half of selected accumulated head gradient',np.array_equal(grad('bf16_slow_chunk1'),[[-.25,.25]]))
ck('BF16 slow single full batch GEMM retains selected head gradient',np.array_equal(grad('bf16_slow_chunk512'),[[-.5,.5]]))
ck('BF16 slow larger 32-row chunks retain selected head gradient',np.array_equal(grad('bf16_slow_chunk32'),[[-.5,.5]]))
ck('Original scan FP32 accumulation retains selected BF16 gradient across both backward chunks',all(np.array_equal(grad(k),[[-.5,.5]]) for k in ['bf16_scan_chunk1','bf16_scan_chunk512']))
ck('FP32 slow one-row chunks retain selected gradient',np.array_equal(grad('fp32_slow_chunk1'),[[-.5,.5]]))
ck('Identical logits target and normalization retain equal forward loss across batch chunk controls',all(np.isclose(cases[k]['loss_before_scale'],np.log(2),rtol=2e-6) for k in list(cases)[:6]))
ck('All selected dtype gradients finite including inaccurate and zero controls',all(c['gradient_finite'] for c in cases.values()))
ck('FP16 early dlogit conversion zeros selected representable final gradient at power denominator',np.array_equal(grad('fp16_scan_power'),[[0,0]]) and np.array_equal(grad('fp16_slow_power'),[[-2**-15,2**-15]]))
ck('FP16 uniform scale8 and FP32 unscale restores selected power-of-two gradient',np.array_equal(grad('fp16_scan_power_scale8'),grad('fp16_slow_power')))
ck('FP16 declared-sized denominator same selected early conversion failure',np.array_equal(grad('fp16_scan_declared_T'),[[0,0]]) and np.abs(grad('fp16_slow_declared_T')).min()>0)
ck('Scaling128 at declared-sized denominator yields nonzero approximate not exact recovery',np.abs(grad('fp16_scan_declared_T_scale128')).min()>0 and np.allclose(grad('fp16_scan_declared_T_scale128'),cases['fp16_scan_declared_T']['FP32_analytic_head_gradient'],rtol=.02,atol=0) and not np.array_equal(grad('fp16_scan_declared_T_scale128'),grad('fp16_slow_declared_T')))
ck('BF16 and mixed FP16 activation FP32 head avoid selected all-zero conversion',all(np.allclose(grad(k),cases[k]['FP32_analytic_head_gradient'],rtol=.005,atol=0) for k in ['bf16_scan_declared_T','mixed_scan_declared_T']))
meta=R/'sources/wandb/hero-fa4sm100-nomask-step146k_meta.json';meta_obj=json.loads(meta.read_text());trainer=meta_obj['config']['trainer']['value']['trainer'];mp=trainer['mp'];batch=trainer['train_batch_size'];seq=meta_obj['config']['model']['value']['max_seq_len'];derived_T=batch*(seq-1)
ck('Archived Hero declaration uses BF16 compute and FP32 params not pure FP16 control',mp=={'compute_dtype':'jax.numpy.bfloat16','output_dtype':'jax.numpy.bfloat16','param_dtype':'jax.numpy.float32'} and derived_T==46126080)
model=R/'sources/contracts_2026_10_05/model.py'
ck('Pinned model explicitly documents backward numerical change and FP32 loss default','2.170e-03 -> 1.445e-03' in model.read_text() and 'loss_dtype: jnp.dtype = jnp.float32' in model.read_text())
paths=[R/'sources/contracts_2026_10_05/reference.py',R/'sources/contracts_2026_10_05/xla.py',model,meta]
out={'scope':'Original registered streaming CE VJP on CPU synthetic BF16/FP16/FP32 operands. Batch accumulation and dlogit cast counterexamples, symbolic large denominator without allocating global batch.','checks_passed':len(checks),'checks':checks,'cases':cases,'archived_Hero_mixed_precision_declaration':mp,'normalization_control':{'declared_batch':batch,'declared_max_seq_len':seq,'assumption':'one masked final target per full-length sequence; no other mask','derived_symbolic_T':derived_T,'actual_measured_global_weight_sum':None},'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'base_probe_sha256':hashlib.sha256(base.read_bytes()).hexdigest(),'probe_script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'runtime':{'jax':jax.__version__,'numpy':np.__version__,'backend':jax.default_backend()},'actual_global_batch_allocated':False,'actual_Hero_operand_dtypes':None,'actual_Hero_gradient_loss_event':None,'actual_GPU_execution':None,'actual_training_loss_effect':None,'upstream_patch_applied':False}
(R/'analysis/ce_gradient_dtype_cpu.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print('Original CE gradient dtype CPU controls:',len(checks))
