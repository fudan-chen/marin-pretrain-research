"""Real Optax grouped clipping controls and original AdamH on artificial CPU data.
Not the complete Hero optimizer build, real parameter-group assignment or training replay.
"""
import json,pathlib,hashlib,importlib.util,sys,inspect
import jax,jax.numpy as jnp,optax,numpy as np
R=pathlib.Path(__file__).resolve().parents[1];checks=[]
def ck(n,b):assert b,n;checks.append(n)
ck('Fixed genuine CPU runtime',jax.__version__=='0.7.2' and optax.__version__=='0.2.5' and jax.default_backend()=='cpu')
g={'u':jnp.array([3.]),'h':jnp.array([4.]),'a':jnp.array([12.])};labels={'u':'muonh','h':'adamh','a':'adam'}
tx=optax.multi_transform({n:optax.clip_by_global_norm(1.) for n in ['muonh','adamh','adam']},labels);out,state=tx.update(g,tx.init(g),g)
whole=optax.clip_by_global_norm(1.);allout,_=whole.update(g,whole.init(g),g)
identity=optax.multi_transform({n:optax.identity() for n in ['muonh','adamh','adam']},labels);identityout,_=identity.update(g,identity.init(g),g)
norm=lambda t:float(optax.global_norm(t));serial=lambda t:{k:np.asarray(v).tolist() for k,v in t.items()}
ck('Artificial raw whole norm is thirteen',norm(g)==13.)
ck('Real multi_transform clips each active group to one',all(np.allclose(v,1.) for v in out.values()) and np.isclose(norm(out),np.sqrt(3),atol=1e-6))
ck('Whole-model control clips combined norm to one',np.isclose(norm(allout),1.,atol=1e-6) and all(np.allclose(allout[k],g[k]/13) for k in g))
ck('No-clip grouped identity preserves artificial input',all(np.array_equal(identityout[k],g[k]) for k in g))
sys.dont_write_bytecode=True;F=R/'sources/adamh_2026_10_05/adamh.py';sp=importlib.util.spec_from_file_location('original_adamh_clip_cpu',F);m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
p=jnp.array([[2.,0.],[0.,1.]]);g1=jnp.array([[1.,4.],[2.,3.]]);g2=jnp.array([[-2.,.2],[3.,-1.]]);a=m.scale_by_adamh(.9,.95,1e-8,.1);s=a.init(p);cl=optax.clip_by_global_norm(.01);cg,_=cl.update(g1,cl.init(g1),p)
u,us=a.update(g1,s,p);cu,cs=a.update(cg,s,p);first_diff=float(jnp.linalg.norm(u-cu));next_u,_=a.update(g2,us,optax.apply_updates(p,u));next_cu,_=a.update(g2,cs,optax.apply_updates(p,u));next_diff=float(jnp.linalg.norm(next_u-next_cu))
ck('Tiny clipped gradient does not cap original AdamH parameter-update norm',float(jnp.linalg.norm(cg))<=.010001 and float(jnp.linalg.norm(cu))>.1)
ck('First-step near-equal updates can hide different moment state',first_diff<1e-6 and float(jnp.linalg.norm(us.mu-cs.mu))>.1)
ck('Common next gradient exposes first-step clipping history',next_diff>1e-3)
M=R/'sources/live_2026_10_07/meta.json';v=json.loads(M.read_text())['config']['optimizer']['value']['max_grad_norm'];ck('Latest archived recipe declares clipping disabled',v is None)
O=R/'sources/optimizer_2026_10_05/optimizer.py';ck('Pinned builder places three optional clippers inside groups',O.read_text().count('components.append(optax.clip_by_global_norm(self.max_grad_norm))')==3)
record={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'runtime':{'jax':jax.__version__,'optax':optax.__version__,'backend':jax.default_backend()},'declared_latest_max_grad_norm':v,'artificial_group_labels':labels,'group_control':{'raw':serial(g),'group_clipped':serial(out),'whole_clipped':serial(allout),'raw_combined_norm':norm(g),'group_clipped_combined_norm':norm(out),'whole_clipped_combined_norm':norm(allout)},'original_AdamH_control':{'clip_threshold':.01,'clipped_gradient_norm':float(jnp.linalg.norm(cg)),'first_parameter_update_norm':float(jnp.linalg.norm(cu)),'first_update_difference_norm':first_diff,'first_moment_difference_norm':float(jnp.linalg.norm(us.mu-cs.mu)),'common_next_update_difference_norm':next_diff,'next_parameter_input_shared':True,'lr':.1,'epsilon':1e-8},'actual_Hero_clipping_event':None,'actual_Hero_group_assignment':None,'actual_complete_Hero_optimizer_build':None,'actual_GPU_TPU_execution':None,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [F,O,M]},'library_source_sha256':{n:hashlib.sha256(inspect.getsource(fn).encode()).hexdigest() for n,fn in [('clip_by_global_norm',optax.clip_by_global_norm),('multi_transform',optax.multi_transform)]}}
(R/'analysis/group_clipping_cpu.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print('CPU/source clipping checks:',len(checks),'first/next diffs:',first_diff,next_diff)
