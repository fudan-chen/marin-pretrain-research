"""Execute the original moe_route block from frozen QBRoutedMoE on CPU.
Identity reshard and placeholder partition spec; no statistics, QB update, dispatch,
ragged expert, actual mesh, checkpoint, or optimizer execution.
"""
import ast,hashlib,json,pathlib,types
import jax,jax.numpy as jnp
R=pathlib.Path(__file__).resolve().parents[1];P=R/'sources/routing_2026_10_05/grug_moe.py'
tree=ast.parse(P.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='QBRoutedMoE');method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__call__');block=next(n for n in method.body if isinstance(n,ast.With) and isinstance(n.items[0].context_expr,ast.Call) and n.items[0].context_expr.args[0].value=='moe_route')
fn=ast.parse('def route(self,x,router,router_bias):\n pass').body[0];fn.body=[block,ast.parse('return router_logits,selected_experts,combine_weights,qb_alpha').body[0]]
ns={'jax':jax,'jnp':jnp,'reshard':lambda x,spec:x,'P':lambda *args:args};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(P),'exec'),ns)
route=ns['route'];recipe=types.SimpleNamespace(num_experts_per_token=2,routing_renorm_sum=2.5)
def run(name,logits,bias,dtype=jnp.bfloat16):
 x=jnp.ones((1,1),dtype);r=jnp.asarray([logits],dtype);b=jnp.asarray(bias,jnp.float32)
 vals=route(recipe,x,r,b);compiled=jax.jit(lambda rr,bb:route(recipe,x,rr,bb))(r,b)
 return {'name':name,'input_logits':logits,'bias':bias,'dtype':str(dtype),'router_logits':vals[0].tolist(),'selected_experts':vals[1].tolist(),'combine_weights':vals[2].astype(jnp.float32).tolist(),'weight_sum':float(jnp.sum(vals[2].astype(jnp.float32))),'qb_alpha':vals[3].tolist(),'eager_jit_equal':all(bool(jnp.array_equal(a,b)) for a,b in zip(vals,compiled))}
def main():
 checks=[]
 def check(n,v):assert v,n;checks.append(n)
 cases=[run('ordinary',[2.,1.,0.],[0.,0.,0.]),run('bias_selects_small_unbiased_sigmoid',[-50.,0.,-1.],[100.,0.,-100.]),run('all_selected_tiny_sigmoids',[-50.,-51.,-52.],[0.,0.,0.]),run('finite_extreme_sigmoid_zero',[-100.,-101.,-102.],[0.,0.,0.])]
 check('Real CPU execution',jax.default_backend()=='cpu')
 check('Original route block retains rounding barrier and epsilon',any(isinstance(n,ast.Assign) and 'optimization_barrier' in ast.unparse(n) for n in block.body) and '1e-09' in ast.unparse(block))
 check('Eager and JIT agree for all four controls',all(r['eager_jit_equal'] for r in cases))
 check('Bias chooses IDs while selected weight uses unbiased logits',cases[1]['selected_experts']==[[0,1]] and cases[1]['combine_weights'][0][0]>0 and cases[1]['combine_weights'][0][0]<1e-20)
 check('This ordinary bf16 control retains the renorm target',cases[0]['weight_sum']==2.5)
 check('Epsilon-dominated positive sigmoid sum is much below target',0<cases[2]['weight_sum']<1e-10)
 check('Finite extreme inputs can yield all-zero sigmoid weights on this CPU',cases[3]['weight_sum']==0)
 x=jnp.ones((1,1),jnp.bfloat16);r=jnp.asarray([[2.,1.,0.]],jnp.bfloat16);b=jnp.zeros((3,),jnp.float32)
 def surrogate(rr,bb):
  _,_,w,_=route(recipe,x,rr,bb);return jnp.sum(w.astype(jnp.float32)*jnp.asarray([[3.,1.]],jnp.float32))
 gr,gb=jax.grad(surrogate,argnums=(0,1))(r,b)
 check('Bias gradient of this surrogate is stopped',bool(jnp.all(gb==0)))
 check('Unselected router parameter gradient is zero for this surrogate',float(gr[0,2])==0 and bool(jnp.any(gr[0,:2]!=0)))
 out={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'runtime':{'jax':jax.__version__,'backend':jax.default_backend()},'extracted_source_block':ast.unparse(block),'source_sha256':{str(P.relative_to(R)):hashlib.sha256(P.read_bytes()).hexdigest()},'cases':cases,'surrogate':{'expert_outputs':[3.,1.],'router_parameter_gradient':gr.astype(jnp.float32).tolist(),'bias_gradient':gb.tolist()},'actual_Hero_extreme_logits':None,'actual_capacity_acceptance':None,'actual_GPU_execution':None,'actual_parameter_update':None,'actual_loss_effect':None}
 (R/'analysis/router_weight_path_cpu.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'checks_passed':len(checks),'cases':cases,'surrogate':out['surrogate']},indent=2))
if __name__=='__main__':main()
