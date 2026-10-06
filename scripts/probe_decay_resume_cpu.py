"""Original gate/router decay + real Adam state and original nested path traversal.
Synthetic CPU parameters; direct state transfer, no checkpoint I/O/full Hero build.
"""
import ast,dataclasses,hashlib,json,pathlib,types
import equinox as eqx,jax,jax.numpy as jnp,numpy as np,optax
R=pathlib.Path(__file__).resolve().parents[1];O=R/'sources/clipping_precision_2026_10_07/optimizer.py';U=R/'sources/decay_resume_2026_10_07/jax_utils.py';M=R/'sources/live_2026_10_07/meta.json'
# Only unexecuted NamedArray type dispatch is supplied: all fixture leaves are raw JAX arrays.
class UnusedNamedArrayType:pass
ns={'jax':jax,'jnp':jnp,'optax':optax,'eqx':eqx,'fields':dataclasses.fields,'hax':types.SimpleNamespace(NamedArray=UnusedNamedArrayType)}
def extract(p,names):
 nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names];assert len(nodes)==len(names)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])),str(p),'exec'),ns)
extract(U,['leaf_key_paths','_isnamedtupleinstance','join_key']);extract(O,['_is_gate_or_router_weight','_gate_router_decay_mask','_scale_by_adam_gate_router_decay'])
cls=next(n for n in ast.parse(O.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='GrugMoeMuonHConfig');method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='create_mask');exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(O),'exec'),ns)
class Mlp(eqx.Module):
 router:jax.Array
 router_bias:jax.Array
class Attn(eqx.Module):
 attn_gate:jax.Array
class Block(eqx.Module):
 mlp:Mlp
 attn:Attn
 weight:jax.Array
class Fixture(eqx.Module):
 blocks:tuple
 token_embed:jax.Array
 output_proj:jax.Array
 static_note:str=eqx.field(static=True,default='synthetic')
def flat(t):return [np.asarray(x).tolist() for x in jax.tree.leaves(t)]
def equal(a,b):return all(np.array_equal(x,y) for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b))) and jax.tree.structure(a)==jax.tree.structure(b)
def maxerr(t):return max(float(jnp.max(jnp.abs(x))) for x in jax.tree.leaves(t))
def main():
 checks=[]
 def ck(n,c):assert c,n;checks.append(n)
 ck('Fixed CPU libraries',jax.default_backend()=='cpu' and jax.__version__=='0.7.2' and optax.__version__=='0.2.5')
 p=Fixture((Block(Mlp(jnp.array([2.,4.]),jnp.array([3.,6.])),Attn(jnp.array([1.,2.])),jnp.array([5.,10.])),),jnp.ones((2,2)),jnp.ones((2,2)))
 paths=ns['leaf_key_paths'](p);mask=ns['_gate_router_decay_mask'](p);groups=ns['create_mask'](None,p)
 ck('Original nested traversal selects only gate/router weights',[x for x,k in zip(jax.tree.leaves(paths),jax.tree.leaves(mask)) if k]==['blocks.0.mlp.router','blocks.0.attn.attn_gate'])
 ck('Original group helper uses actual fixture paths',jax.tree.leaves(groups)==['adam','adam','adam','adam','adam','adamh'])
 ck('Static Equinox metadata excluded and raw paths shape matched',len(jax.tree.leaves(paths))==6 and paths.static_note=='synthetic')
 meta=json.loads(M.read_text())['config']['optimizer']['value'];b1,b2,eps=meta['beta1'],meta['beta2'],meta['epsilon'];wd=meta['gate_router_weight_decay'];lr=.01;N=100
 adam=optax.scale_by_adam(b1,b2,eps);g=jax.tree.map(lambda x:jnp.full_like(x,.25),p);s=adam.init(p)
 for _ in range(50):_,s=adam.update(g,s,p)
 ck('Real artificial warm state count is 50',int(s.count)==50)
 d=ns['_scale_by_adam_gate_router_decay'](b1,b2,eps,wd,N);a,sa=adam.update(g,s,p);w,sw=d.update(g,s,p)
 expected=jax.tree.map(lambda u,x,k:u+(wd*.5)*x if k else u,a,p,mask)
 ck('Original wrapper accepts ordinary Adam state directly',type(sw)==type(s) and int(sw.count)==51)
 ck('Decay excludes moment accumulation and preserves exact next state',equal(sa,sw))
 ck('Decay addition is input-count based on exact nested mask',maxerr(jax.tree.map(lambda x,y:x-y,w,expected))<1e-7)
 ck('Original wrapper JIT and eager agree',equal(w,jax.jit(d.update)(g,s,p)[0]))
 results=[]
 for label,state,steps in [('preserved',s,100),('new_total_steps',s,200),('reset_all_state',adam.init(p),100),('reset_count_only',s._replace(count=jnp.array(0,jnp.int32)),100),('end_count',s._replace(count=jnp.array(100,jnp.int32)),100)]:
  tx=ns['_scale_by_adam_gate_router_decay'](b1,b2,eps,wd,steps);base,bs=adam.update(g,state,p);out,st=tx.update(g,state,p);addition=jax.tree.map(lambda x,y:x-y,out,base)
  final=optax.scale(-lr).update(out,optax.EmptyState(),p)[0]
  results.append({'case':label,'input_count':int(state.count),'output_count':int(st.count),'total_steps':steps,'decay_coefficient':wd*max(1-int(state.count)/steps,0),'router_adaptive_direction':flat(base.blocks[0].mlp.router),'router_decay_addition':flat(addition.blocks[0].mlp.router),'router_final_update':flat(final.blocks[0].mlp.router),'next_moments_equal_to_plain_Adam':equal(bs,st),'full_update':flat(final)})
 ck('Changing horizon alone changes decay at preserved count',np.allclose(results[1]['router_decay_addition'],1.5*np.array(results[0]['router_decay_addition']),rtol=1e-5))
 ck('Reinitialization doubles this artificial decay addition',np.allclose(results[2]['router_decay_addition'],2*np.array(results[0]['router_decay_addition']),rtol=1e-5))
 ck('Count reset alone also changes bias corrected adaptive direction',not np.allclose(results[3]['router_adaptive_direction'],results[0]['router_adaptive_direction']))
 ck('Count at horizon eliminates decay without freezing Adam',results[4]['router_decay_addition']==[[0.,0.]] and np.linalg.norm(results[4]['router_final_update'])>0)
 ck('All five controls preserve plain Adam next moments',all(c['next_moments_equal_to_plain_Adam'] for c in results))
 try:d.update(g,s,None);ok=False
 except ValueError:ok=True
 ck('Original wrapper rejects missing parameter tree',ok)
 out={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'runtime':{'jax':jax.__version__,'optax':optax.__version__,'equinox':eqx.__version__,'backend':jax.default_backend()},'source_sha256':{str(x.relative_to(R)):hashlib.sha256(x.read_bytes()).hexdigest() for x in [O,U,M]},'paths':jax.tree.leaves(paths),'groups':jax.tree.leaves(groups),'decay_mask':jax.tree.leaves(mask),'controls':results,'fixture':{'parameters':flat(p),'constant_gradient':.25,'warm_steps':50,'warm_params_held_fixed':True,'artificial_total_steps':N,'artificial_learning_rate':lr,'declared_decay':wd,'declared_b1':b1,'declared_b2':b2,'declared_epsilon':eps},'explicit_substitutions':['AST extracted original helper methods; full module/build not imported','NamedArray dispatch type placeholder never exercised; all leaves raw arrays','synthetic Equinox tree, not Hero parameter tree','direct in-memory ScaleByAdamState transfer; no checkpoint loader/I/O','fixed parameters during moment warm-up; five independent state/horizon controls, not training trajectories'],'actual_Hero_parameter_groups':None,'actual_Hero_state_restore':None,'actual_Hero_clock_mismatch':None,'actual_GPU_execution':None,'actual_training_benefit':None}
 (R/'analysis/decay_resume_cpu.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'checks_passed':len(checks),'controls':results},indent=2))
if __name__=='__main__':main()
