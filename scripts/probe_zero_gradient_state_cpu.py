"""Full original AdamH module with real JAX/Optax CPU transforms on artificial matrices.
No Hero loss, checkpoint, multi-transform, GPU/TPU, collective or complete train-step replay.
"""
import importlib.util,sys,pathlib,json,hashlib,platform,ast
import jax,jax.numpy as jnp,jaxlib,optax,chex,numpy as np
R=pathlib.Path(__file__).resolve().parents[1];F=R/'sources/adamh_2026_10_05/adamh.py';checks=[]
def ck(n,b):assert b,n;checks.append(n)
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('marin_original_adamh_cpu',F);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
ck('Observed fixed CPU runtime',jax.__version__==jaxlib.__version__=='0.7.2' and optax.__version__=='0.2.5' and chex.__version__=='0.1.90' and jax.default_backend()=='cpu')
p0=jnp.array([[2.,0.],[0.,1.]],dtype=jnp.float32);g1=jnp.array([[1.,4.],[2.,3.]],dtype=jnp.float32);g2=jnp.array([[-2.,.2],[3.,-1.]],dtype=jnp.float32);z=jnp.zeros_like(p0);tx=mod.scale_by_adamh(b1=.9,b2=.95,eps=1e-8,learning_rate=.1);s0=tx.init(p0)
norm=lambda x:float(jnp.linalg.norm(x));arr=lambda x:np.asarray(x).tolist()
u_fresh,s_fresh=tx.update(z,s0,p0);ck('Fresh zero gradient advances AdamH count with near-zero parameter update',int(s_fresh.count)==1 and norm(u_fresh)<1e-6 and np.all(np.asarray(s_fresh.mu)==0) and np.all(np.asarray(s_fresh.nu)==0))
u1,s1=tx.update(g1,s0,p0);p1=optax.apply_updates(p0,u1);u_zero,s_zero=tx.update(z,s1,p1);p_zero=optax.apply_updates(p1,u_zero)
ck('Warm zero gradient decays original moments and advances count',int(s_zero.count)==2 and np.allclose(s_zero.mu,.9*s1.mu,atol=1e-7) and np.allclose(s_zero.nu,.95*s1.nu,atol=1e-7))
ck('Warm original AdamH zero-gradient update is nonzero and norm-preserving',norm(u_zero)>1e-4 and abs(norm(p_zero)/norm(p1)-1)<1e-6)
try:
 jax.jit(tx.update)(z,s1,p1);bare_jit_error=None
except ValueError as e:bare_jit_error=str(e)
ck('Bare CPU jit exposes empty-mesh reshard compatibility boundary',bare_jit_error is not None and 'nonempty mesh' in bare_jit_error)
mesh=jax.sharding.Mesh(np.array(jax.devices()),('cpu',));named=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
with jax.set_mesh(mesh):
 u_jit,s_jit=jax.jit(tx.update)(jax.device_put(z,named),jax.tree.map(lambda v:jax.device_put(v,named),s1),jax.device_put(p1,named))
ck('Original AdamH named one-device CPU jit agrees with eager warm-zero update',np.allclose(u_jit,u_zero,atol=1e-6,rtol=1e-6) and int(s_jit.count)==2)
# Drop only the final update: parameters remain p1, but the optimizer state is s_zero.
u_after_discard,s_after_discard=tx.update(g2,s_zero,p1);u_after_freeze,s_after_freeze=tx.update(g2,s1,p1)
ck('Discarding update is not freezing state on the next nonzero gradient',norm(u_after_discard-u_after_freeze)>1e-5 and int(s_after_discard.count)==3 and int(s_after_freeze.count)==2)
rows=[];p=p1;s=s1
for k in range(1,21):
 u,s=tx.update(z,s,p);p=optax.apply_updates(p,u);rows.append({'zero_steps':k,'count':int(s.count),'moment_mu_norm':norm(s.mu),'moment_nu_norm':norm(s.nu),'update_norm':norm(u),'parameter_norm':norm(p),'parameter_distance_from_p1':norm(p-p1)})
ck('Twenty zero gradients preserve original moment decay and count clocks',int(s.count)==21 and np.allclose(s.mu,(.9**20)*s1.mu,rtol=2e-5,atol=1e-7) and np.allclose(s.nu,(.95**20)*s1.nu,rtol=2e-5,atol=1e-7))
ck('Twenty zero gradients preserve parameter norm while changing direction',all(abs(x['parameter_norm']/norm(p1)-1)<2e-6 for x in rows) and norm(p-p1)>.1)
# Independent ordinary Optax controls. They are not the Hero grouped optimizer.
a=optax.adam(.1,b1=.9,b2=.95);a0=a.init(p0);au1,a1=a.update(g1,a0,p0);ap1=optax.apply_updates(p0,au1);au0,a2=a.update(z,a1,ap1);ck('Ordinary Optax Adam has a nonzero warm-zero update',norm(au0)>1e-4)
af,afs=a.update(z,a0,p0);ck('Ordinary fresh Adam has zero update but increments count',norm(af)==0 and int(afs[0].count)==1)
aw=optax.adamw(.1,b1=.9,b2=.95,weight_decay=.1);awu,aws=aw.update(z,aw.init(p0),p0);ck('Independent AdamW control moves fresh parameters by declared decay',np.allclose(awu,-.01*p0,atol=1e-7))
# Training envelope is inspected statically, not executed on a model.
H=R/'sources/scale_2026_10_05/train_hero_ep.py';t=H.read_text();ck('Pinned train envelope updates optimizer and advances step','optimizer.update(grads, opt_state_in, qb_params)' in t and 'step=state.step + one' in t)
ck('Pinned EMA envelope uses new parameters and carries QB pending metrics','ema_beta * old + (1.0 - ema_beta) * new' in t and 'pending_qb_betas=metrics["qb_beta_per_layer"]' in t)
M=R/'sources/live_2026_10_07/meta.json';declared_decay=json.loads(M.read_text())['config']['optimizer']['value']['gate_router_weight_decay']
record={'declared_gate_router_weight_decay':declared_decay,'scope':__doc__,'checks_passed':len(checks),'checks':checks,'runtime':{'python':platform.python_version(),'jax':jax.__version__,'jaxlib':jaxlib.__version__,'optax':optax.__version__,'chex':chex.__version__,'numpy':np.__version__,'backend':jax.default_backend(),'devices':[str(x) for x in jax.devices()]},'artificial_recipe':{'p0':arr(p0),'g1':arr(g1),'next_nonzero_g':arr(g2),'b1':.9,'b2':.95,'eps':1e-8,'learning_rate':.1},'fresh_zero':{'update_norm':norm(u_fresh),'count':int(s_fresh.count)},'warm_zero':{'p1':arr(p1),'update':arr(u_zero),'update_norm':norm(u_zero),'parameter_norm_before':norm(p1),'parameter_norm_after':norm(p_zero),'count_before':int(s1.count),'count_after':int(s_zero.count)},'discard_vs_freeze':{'next_update_difference_norm':norm(u_after_discard-u_after_freeze),'next_update_discard':arr(u_after_discard),'next_update_freeze':arr(u_after_freeze),'discard_count_after_next':int(s_after_discard.count),'freeze_count_after_next':int(s_after_freeze.count)},'twenty_zero_steps':rows,'ordinary_Optax_controls':{'warm_Adam_update_norm':norm(au0),'fresh_Adam_update_norm':norm(af),'fresh_AdamW_update_norm':norm(awu),'AdamW_decay':.1},'bare_CPU_jit_error':bare_jit_error,'named_CPU_mesh_axes':['cpu'],'original_AdamH_module_executed':True,'source_module_altered':False,'actual_Hero_zero_target_step':None,'actual_Hero_optimizer_group_binding':None,'actual_full_train_step':None,'actual_checkpoint_restore':None,'actual_GPU_TPU_collectives':None,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [F,H,M]},'library_function_source_sha256':{'optax.scale_by_adam':hashlib.sha256(__import__('inspect').getsource(optax.scale_by_adam).encode()).hexdigest()}}
(R/'analysis/zero_gradient_state_cpu.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print('Real original-module CPU optimizer checks:',len(checks),'warm update',norm(u_zero),'discard/freezing next delta',norm(u_after_discard-u_after_freeze))
