"""Original portable expert class and original ragged_dot CPU/XLA wrapper.
Synthetic local rows/weights, independent dense grouped reference; no EP collective,
QuACK, production input, GPU or runtime reconstruction.
"""
import ast,dataclasses,hashlib,json,os,pathlib,types,warnings
from typing import Callable
import jax,jax.numpy as jnp,numpy as np
R=pathlib.Path(__file__).resolve().parents[1];P=R/'sources/engineering_v77_2026_10_07/current_ep_ragged_all_to_all.py';W=R/'sources/portable_ep_2026_10_07/ragged_dot.py'
assert os.environ.get('RAGGED_DOT_IMPL') is None,'Explicit unset RAGGED_DOT_IMPL required for CPU auto contract'
ns={'jax':jax,'jnp':jnp,'np':np,'dataclasses':dataclasses,'os':os,'warnings':warnings,'_has_pallas_triton':False,'_AUTO_FALLBACK_EXCEPTIONS':(NotImplementedError,RuntimeError),'_HAS_WARNED_AUTO_FALLBACK':False}
def execute(nodes,p):
 exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])),str(p),'exec'),ns)
wt=ast.parse(W.read_text());execute([n for n in wt.body if isinstance(n,ast.FunctionDef) and n.name in ['_ragged_dot_xla_impl','_preferred_implementations','_run_impl','ragged_dot']],W)
pt=ast.parse(P.read_text());cl=next(n for n in pt.body if isinstance(n,ast.ClassDef) and n.name=='_RaggedDotExpertMlp');execute([cl],P)
model=ns['_RaggedDotExpertMlp'](jax.nn.silu)
# Exact caller division statements; row/assignment mapping is identity in this local fixture.
assigns=[n for n in ast.walk(pt) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='divisible' for t in n.targets)]
div=assigns[0];parent=next(n for n in ast.walk(pt) if hasattr(n,'body') and isinstance(n.body,list) and div in n.body);idx=parent.body.index(div);tail=compile(ast.fix_missing_locations(ast.Module(body=parent.body[idx:idx+2],type_ignores=[])),str(P),'exec')
def restore_weight_gradient(rowdot,w,accepted):
 env={'jnp':jnp,'routing':types.SimpleNamespace(accepted=accepted),'weights_f32':w,'assignment_output_dot':rowdot};exec(tail,env);return env['weights_cotangent']
def dense_reference(x,w13,w2,physical,active):
 count=int(np.sum(active));m=(jnp.arange(x.shape[0])<count)[:,None];xx=jnp.where(m,x,0);rows=[];start=0
 for e,size in enumerate(physical):
  size=int(size);gu=xx[start:start+size]@w13[e];g,u=jnp.split(gu,2,axis=-1);rows.append((jax.nn.silu(g)*u)@w2[e]);start+=size
 return jnp.where(m,jnp.concatenate(rows,axis=0),0)
def main():
 checks=[];cases=[]
 def check(n,c):assert c,n;checks.append(n)
 rng=np.random.default_rng(913);x=jnp.array(rng.normal(size=(10,4)),dtype=jnp.float32);w13=jnp.array(rng.normal(size=(2,4,12))*.3,dtype=jnp.float32);w2=jnp.array(rng.normal(size=(2,6,4))*.3,dtype=jnp.float32);dout=jnp.array(rng.normal(size=(10,4)),dtype=jnp.float32)
 physical=jnp.array([3,7],dtype=jnp.int32);active=jnp.array([3,4],dtype=jnp.int32);mask=jnp.arange(10)<7
 weights=jnp.array([.25,.5,1,2,.75,1.5,.125,0,0,0]);cotangent=dout*weights[:,None]
 clean_x=x.at[7:].set(0);clean_dy=cotangent.at[7:].set(0);poison_x=x.at[7:].set(jnp.nan);poison_dy=cotangent.at[7:].set(jnp.nan)
 out,res=model.forward(clean_x,w13,w2,physical,active);clean_grad=model.backward(res,clean_dy)
 p_out,p_res=model.forward(poison_x,w13,w2,physical,active);p_grad=model.backward(p_res,poison_dy)
 check('CPU wrapper auto chooses original XLA branch',ns['_preferred_implementations']('auto')==('xla',))
 ref=lambda a,b,c:dense_reference(a,b,c,physical,active)
 ref_out,vjp=jax.vjp(ref,clean_x,w13,w2);ref_grad=vjp(clean_dy)
 check('Original forward matches independent dense grouped reference',np.allclose(out,ref_out,atol=1e-6,rtol=1e-5))
 check('All original ordinary gradients match dense reference',all(np.allclose(a,b,atol=1e-6,rtol=1e-5) for a,b in zip(clean_grad[:3],ref_grad)))
 check('Inactive poison leaves active outputs and ordinary gradients exactly equal',np.array_equal(out[:7],p_out[:7]) and all(np.array_equal(a,b) for a,b in zip(clean_grad[:3],p_grad[:3])))
 check('Ordinary gradients remain finite despite inactive NaN',all(np.isfinite(np.asarray(g)).all() for g in p_grad[:3]))
 scale_grad=jax.grad(lambda scale:jnp.sum(ref_out*scale[:,None]*clean_dy))(jnp.ones(10))
 check('Original row dot equals independently differentiated output scale',np.allclose(clean_grad[3][:7],scale_grad[:7],atol=1e-6,rtol=1e-5))
 check('Inactive row-dot NaN is confined to unspecified rows',np.isfinite(np.asarray(p_grad[3][:7])).all() and np.isnan(np.asarray(p_grad[3][7:])).all())
 recovered=restore_weight_gradient(p_grad[3],weights,mask);expected=jnp.sum(ref_out*dout,axis=-1)
 check('Original caller selection masks inactive row-dot and recovers positive-weight gradient',np.isfinite(np.asarray(recovered)).all() and np.allclose(recovered[:7],expected[:7],atol=1e-6,rtol=1e-5) and np.array_equal(np.asarray(recovered[7:]),np.zeros(3)))
 check('Portable residual actually retains entire output array',len(res)==6 and res[-1].shape==(10,4) and np.array_equal(res[-1],out))
 changed_res=res[:-1]+(res[-1].at[0].add(3),)
 changed_grad=model.backward(changed_res,clean_dy)
 check('Mutating saved active output changes only row-dot branch in this control',all(np.array_equal(a,b) for a,b in zip(changed_grad[:3],clean_grad[:3])) and not np.allclose(changed_grad[3][0],clean_grad[3][0]))
 for label,a,p in [('empty_first_expert',[0,7],[0,10]),('all_inactive',[0,0],[0,10])]:
  aa=jnp.array(a,dtype=jnp.int32);pp=jnp.array(p,dtype=jnp.int32);n=sum(a);xx=x.at[n:].set(jnp.nan);dy=cotangent.at[n:].set(jnp.nan)
  yy,rr=model.forward(xx,w13,w2,pp,aa);gg=model.backward(rr,dy)
  check(label+' ordinary outputs/gradients remain finite',np.isfinite(np.asarray(yy)).all() and all(np.isfinite(np.asarray(g)).all() for g in gg[:3]))
  check(label+' empty expert weight gradients are zero',np.array_equal(np.asarray(gg[1][0]),np.zeros((4,12))) and np.array_equal(np.asarray(gg[2][0]),np.zeros((6,4))))
  cases.append({'case':label,'active_group_sizes':a,'physical_group_sizes':p,'ordinary_gradient_norms':[float(jnp.linalg.norm(g)) for g in gg[:3]],'inactive_row_dot_nonfinite':int(np.sum(~np.isfinite(np.asarray(gg[3][n:]))))})
 o={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'runtime':{'jax':jax.__version__,'jaxlib':jax.lib.__version__,'numpy':np.__version__},'observations':{'capacity':10,'hidden':4,'intermediate':6,'experts':2,'active_rows':7,'physical_group_sizes':[3,7],'active_group_sizes':[3,4],'row_dot_active':np.asarray(p_grad[3][:7]).tolist(),'inactive_row_dot_nonfinite':3,'recovered_weight_gradient':np.asarray(recovered).tolist(),'residual_output_shape':list(res[-1].shape),'mutated_saved_output_control':{'ordinary_gradients_equal':True,'row_dot_before':float(clean_grad[3][0]),'row_dot_after':float(changed_grad[3][0]),'actual_corruption_event':None},'ordinary_gradient_norms':[float(jnp.linalg.norm(g)) for g in p_grad[:3]],'edge_cases':cases},'executed_original':['portable _apply/forward/backward class','CPU auto selector/run_impl/XLA ragged_dot_general','512-row padding and output slice','caller accepted/nonzero weight division and selection'], 'explicit_substitutions':['AST load skips unrelated imports and GPU/TPU definitions; no Haliax installation','fixed synthetic FP32 local fixtures and independent dense reference','caller receiver-to-assignment mapping identity; no all-to-all or routing planner','_has_pallas_triton=False on CPU; no environment override'],'actual_GPU_execution':None,'actual_Hero_input':None,'actual_transport_poison_test':None,'actual_memory_saved':None,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [P,W]}}
 arrays=dict(dx=p_grad[0],dw13=p_grad[1],dw2=p_grad[2],row_dot=p_grad[3],selected_weight_gradient=recovered)
 o['numeric_coverage']={k:{'elements':int(v.size),'finite':int(np.isfinite(np.asarray(v)).sum()),'nonfinite':int((~np.isfinite(np.asarray(v))).sum())} for k,v in arrays.items()}
 (R/'analysis/portable_expert_mlp_cpu.json').write_text(json.dumps(o,indent=2,allow_nan=False)+'\n');print('Original portable CPU checks:',len(checks));print(json.dumps(o['observations'],indent=2))
if __name__=='__main__':main()
