"""Synthetic integration: original router block, portable PR row-dot and division,
original decay wrapper with path stub, real Optax CPU. Expert outputs are constants;
not a complete ragged expert, GPU, Hero group build or checkpoint replay.
"""
import ast,hashlib,json,pathlib,types,inspect
import jax,jax.numpy as jnp,optax
from probe_router_weight_path_cpu import route,recipe,P as ROUTER
from probe_routing_gradient_proposal_cpu import code,P as PATCH,statements
R=pathlib.Path(__file__).resolve().parents[1];OPT=R/'sources/optimizer_2026_10_05/optimizer.py';META=R/'sources/live_2026_10_07/meta.json'
files=json.loads(PATCH.read_text());patch=next(f['patch'] for f in files if f['filename'].endswith('ep_ragged_all_to_all.py'));rowdot=next(l[1:].strip() for l in patch.splitlines() if l.startswith('+') and 'output_dot_cotangent = jnp.sum(out.astype' in l);rdcode=compile(ast.parse(rowdot),str(PATCH),'exec')
functions=['_is_gate_or_router_weight','_gate_router_decay_mask','_scale_by_adam_gate_router_decay'];tree=ast.parse(OPT.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in functions];ns={'jax':jax,'jnp':jnp,'optax':optax,'leaf_key_paths':lambda p:{k:k for k in p}};exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(OPT),'exec'),ns)
arr=lambda x:jnp.asarray(x).astype(jnp.float32).tolist()
def main():
 checks=[]
 def check(n,v):assert v,n;checks.append(n)
 p=jnp.asarray([[-4.,0.,-10.]],jnp.float32);x=jnp.ones((1,1),jnp.bfloat16);bias=jnp.zeros(3,jnp.float32)
 def weights(rr):return route(recipe,x,rr,bias)[2]
 w,pullback=jax.vjp(weights,p);selected=route(recipe,x,p,bias)[1]
 d=jnp.asarray(.001,jnp.bfloat16);out=jnp.full((2,1),3.,jnp.bfloat16);dy=(w*d).astype(jnp.bfloat16)
 rd={'jnp':jnp,'out':out,'cotangent':dy.reshape(2,1)};exec(rdcode,rd)
 div={'jnp':jnp,'routing':types.SimpleNamespace(accepted=jnp.ones_like(w,dtype=bool)),'weights_f32':w.astype(jnp.float32),'assignment_output_dot':rd['output_dot_cotangent'].reshape(w.shape)};exec(code,div)
 exact=jnp.full(w.shape,d.astype(jnp.float32)*3,dtype=jnp.bfloat16);candidate=div['weights_cotangent'].astype(jnp.bfloat16)
 ge=pullback(exact)[0];gc=pullback(candidate)[0]
 c=json.loads(META.read_text())['config']['optimizer']['value'];lr=c['adam_lr'];eps=c['epsilon'];b1=c['beta1'];b2=c['beta2'];wd=c['gate_router_weight_decay']
 tx=optax.chain(ns['_scale_by_adam_gate_router_decay'](b1,b2,eps,wd,1000),optax.scale(-lr));params={'layer.router':p};s0=tx.init(params)
 ue,se=tx.update({'layer.router':ge},s0,params);uc,sc=tx.update({'layer.router':gc},s0,params)
 _,sw=tx.update({'layer.router':jnp.asarray([[.1,-.2,.3]],jnp.float32)},s0,params)
 we,_=tx.update({'layer.router':ge},sw,params);wc,_=tx.update({'layer.router':gc},sw,params)
 norm=lambda z:float(jnp.linalg.norm(z));fresh=norm(ue['layer.router']-uc['layer.router']);warm=norm(we['layer.router']-wc['layer.router'])
 check('Executed CPU runtime',jax.default_backend()=='cpu' and optax.__version__=='0.2.5')
 check('Selected expert order is recorded',selected.tolist()==[[1,0]])
 check('All weighted cotangents stay positive; this is rounding not underflow',bool(jnp.all(dy>0)))
 check('Reference dS equal but candidate differs at exactly one assignment',float(exact[0,0])==float(exact[0,1]) and int(jnp.sum(exact!=candidate))==1)
 check('Reference router parameter gradient is zero in this control',norm(ge)==0)
 check('One assignment error reaches both selected parameters',float(gc[0,0])!=0 and float(gc[0,1])!=0 and float(gc[0,2])==0)
 check('Original source mask selects artificial router path',ns['_gate_router_decay_mask'](params)['layer.router'] is True)
 check('Identical fresh state gives differing updates beyond common decay',fresh>1e-4 and float(ue['layer.router'][0,2])==float(uc['layer.router'][0,2]))
 check('Both fresh optimizer counts advance equally',int(se[0].count)==int(sc[0].count)==1)
 check('Common warm state reduces this example update difference',warm<fresh and warm<1e-6)
 o={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'runtime':{'jax':jax.__version__,'optax':optax.__version__,'backend':jax.default_backend()},'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROUTER,PATCH,OPT,META]},'original_portable_rowdot_statement':rowdot,'original_division_statements':statements,'fixture':{'params':arr(p),'activation_dtype':'bfloat16','parameter_dtype':'float32','selected_experts':selected.tolist(),'combine_weights':arr(w),'cotangent':float(d),'constant_expert_outputs':arr(out),'rounded_weighted_cotangents':arr(dy),'dS_reference':arr(exact),'dS_candidate':arr(candidate),'router_gradient_reference':arr(ge),'router_gradient_candidate':arr(gc)},'optimizer_control':{'b1':b1,'b2':b2,'epsilon':eps,'diagnostic_constant_lr_from_declared_adam_lr':lr,'weight_decay':wd,'artificial_total_steps':1000,'actual_live_lr_schedule':None,'artificial_flat_path':'layer.router','leaf_key_paths_stub':True,'fresh_reference_update':arr(ue['layer.router']),'fresh_candidate_update':arr(uc['layer.router']),'fresh_update_difference_norm':fresh,'warm_reference_update':arr(we['layer.router']),'warm_candidate_update':arr(wc['layer.router']),'warm_update_difference_norm':warm,'warm_common_gradient':[[.1,-.2,.3]],'warm_parameter_input_held_common':True},'actual_Hero_parameter_group':None,'actual_GPU_execution':None,'actual_checkpoint_replay':None,'actual_Hero_loss_effect':None}
 o['library_source_sha256']={n:hashlib.sha256(inspect.getsource(fn).encode()).hexdigest() for n,fn in [('scale_by_adam',optax.scale_by_adam),('apply_updates',optax.apply_updates)]}
 o['optimizer_control']['fresh_parameter_distance_after_apply']=norm(optax.apply_updates(params,ue)['layer.router']-optax.apply_updates(params,uc)['layer.router'])
 (R/'analysis/router_coupling_update_cpu.json').write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'checks_passed':len(checks),'fixture':o['fixture'],'optimizer_control':o['optimizer_control']},indent=2))
if __name__=='__main__':main()
