"""Real Optax clipping and original AdamH on synthetic finite gradient matrices.
Optional clipping branch; archived Hero max_grad_norm=None, no production attribution.
"""
import ast,hashlib,importlib.util,inspect,json,pathlib,sys
import jax,jax.numpy as jnp,numpy as np,optax
R=pathlib.Path(__file__).resolve().parents[1];A=R/'sources/clipping_precision_2026_10_07/adamh.py';O=R/'sources/clipping_precision_2026_10_07/optimizer.py';T=R/'sources/runtime_defaults_2026_10_07/train.py';M=R/'sources/live_2026_10_07/meta.json'
spec=importlib.util.spec_from_file_location('clip_precision_original_adamh',A);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
def clip(g,limit):
 tx=optax.clip_by_global_norm(limit);return tx.update(g,tx.init(g))[0]
def matrix(x):return np.asarray(x.astype(jnp.float32)).tolist()
def main():
 checks=[];cases=[]
 def check(n,c):assert c,n;checks.append(n)
 check('Fixed CPU runtime',jax.default_backend()=='cpu' and jax.__version__=='0.7.2' and optax.__version__=='0.2.5')
 for dtype in [jnp.float16,jnp.bfloat16,jnp.float32]:
  for label,value,limit in [('large',300.,1.),('small',.0001,.00001)]:
   g=jnp.full((2,2),value,dtype);gn=optax.global_norm(g);direct=clip(g,limit);fp32=clip(g.astype(jnp.float32),limit).astype(dtype)
   norm64=float(np.linalg.norm(np.asarray(g,dtype=np.float64)));dn64=float(np.linalg.norm(np.asarray(direct,dtype=np.float64)));fn64=float(np.linalg.norm(np.asarray(fp32,dtype=np.float64)))
   record={'dtype':str(jnp.dtype(dtype)),'case':label,'threshold':limit,'gradient':matrix(g),'source_norm':float(gn) if np.isfinite(float(gn)) else 'Infinity','source_norm_dtype':str(gn.dtype),'independent_input_norm_float64':norm64,'direct_clipped':matrix(direct),'direct_clipped_norm_float64':dn64,'fp32_norm_control_cast_back':matrix(fp32),'fp32_control_norm_float64':fn64,'direct_output_finite':bool(jnp.all(jnp.isfinite(direct)))}
   cases.append(record)
   check(str(dtype)+' '+label+' finite original gradients and clipped outputs',bool(jnp.all(jnp.isfinite(g))) and record['direct_output_finite'])
   check(str(dtype)+' '+label+' direct eager and JIT agree',bool(jnp.array_equal(direct,jax.jit(lambda gg:clip(gg,limit))(g))))
 large=next(c for c in cases if c['dtype']=='float16' and c['case']=='large');small=next(c for c in cases if c['dtype']=='float16' and c['case']=='small')
 check('Finite float16 squaring overflows norm and erases clipped gradient',large['independent_input_norm_float64']==600 and large['source_norm']=='Infinity' and large['direct_clipped_norm_float64']==0 and large['fp32_control_norm_float64']==1)
 check('Float16 small squaring underflows norm and fails to trigger clipping',small['source_norm']==0 and small['direct_clipped_norm_float64']>19*small['threshold'] and small['fp32_control_norm_float64']<1.01*small['threshold'])
 p=jnp.array([[2.,0.],[0.,1.]],jnp.float32);tx=mod.scale_by_adamh(b1=.9,b2=.95,eps=1e-8,learning_rate=.1);s0=tx.init(p);g0=jnp.array([[1.,4.],[2.,3.]],jnp.float32);u0,s1=tx.update(g0,s0,p);p1=optax.apply_updates(p,u0)
 bad_g=clip(jnp.full((2,2),300.,jnp.float16),1.).astype(jnp.float32);good_g=clip(jnp.full((2,2),300.,jnp.float32),1.)
 ub,sb=tx.update(bad_g,s1,p1);ug,sg=tx.update(good_g,s1,p1)
 warm={'direct_clipped_gradient_norm':float(jnp.linalg.norm(bad_g)),'fp32_control_gradient_norm':float(jnp.linalg.norm(good_g)),'direct_warm_update_norm':float(jnp.linalg.norm(ub)),'fp32_warm_update_norm':float(jnp.linalg.norm(ug)),'update_difference_norm':float(jnp.linalg.norm(ub-ug)),'count_before':int(s1.count),'count_after_direct':int(sb.count),'count_after_control':int(sg.count),'original_mu_decay_error':float(jnp.max(jnp.abs(sb.mu-.9*s1.mu))),'original_nu_decay_error':float(jnp.max(jnp.abs(sb.nu-.95*s1.nu)))}
 check('Collapsed clipped gradient still advances original warm AdamH and moves parameters',warm['direct_clipped_gradient_norm']==0 and warm['direct_warm_update_norm']>1e-4 and warm['count_after_direct']==2 and warm['original_mu_decay_error']<1e-7 and warm['original_nu_decay_error']<1e-7)
 check('Same warm state distinguishes true zeroed path from FP32 norm control',warm['count_after_control']==2 and warm['update_difference_norm']>1e-4)
 meta=json.loads(M.read_text());declared=meta['config']['optimizer']['value']['max_grad_norm'];check('Frozen Hero declaration keeps optional clipping disabled',declared is None)
 ot=ast.parse(O.read_text());clipcalls=[n for n in ast.walk(ot) if isinstance(n,ast.If) and ast.unparse(n.test)=='self.max_grad_norm' and 'optax.clip_by_global_norm' in ast.unparse(n)]
 check('Three optional group clip branches and master casting source order retained',len(clipcalls)==3 and '_FP32_POLICY.cast_to_param(grads)' in T.read_text() and 'optimizer.update(master_grads, opt_state_in, master_params_in)' in T.read_text())
 library={'optax.global_norm':inspect.getsource(optax.global_norm),'optax.clip_by_global_norm':inspect.getsource(optax.clip_by_global_norm)}
 out={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'cases':cases,'warm_AdamH':warm,'archived_Hero_max_grad_norm':declared,'runtime':{'jax':jax.__version__,'optax':optax.__version__,'backend':jax.default_backend()},'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [A,O,T,M]},'library_source':library,'library_source_sha256':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in library.items()},'explicit_substitutions':['manual 2x2 gradients; optional clip transform not full Hero build','FP32 norm control casts entire gradient to FP32 then clipped result back; not upstream fix','original AdamH warm state is artificial and FP32; both compared gradients cast to FP32 before AdamH','master policy inspected statically, not full train-step execution'],'actual_Hero_clipping_incident':None,'actual_Hero_gradient_dtype':None,'actual_GPU_execution':None,'actual_training_benefit':None}
 (R/'analysis/clipping_precision_cpu.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps({'checks_passed':len(checks),'cases':cases,'warm_AdamH':warm},indent=2))
if __name__=='__main__':main()
