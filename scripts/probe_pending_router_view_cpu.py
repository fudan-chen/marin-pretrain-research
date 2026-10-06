"""Original apply_qb_betas function with an explicit tree_at stub, then original
router block on CPU. Synthetic beta/parameters; no checkpoint IO or full model.
"""
import ast,hashlib,json,pathlib,types
import jax,jax.numpy as jnp
from probe_router_weight_path_cpu import route,recipe,P as ROUTER
R=pathlib.Path(__file__).resolve().parents[1];MODEL=R/'sources/contracts_2026_10_05/model.py'
def model(bias):return types.SimpleNamespace(stacked_blocks=types.SimpleNamespace(stacked=types.SimpleNamespace(mlp=types.SimpleNamespace(router_bias=bias))))
def tree_at_stub(select,m,new):
 assert select(m) is m.stacked_blocks.stacked.mlp.router_bias
 return model(new)
node=next(n for n in ast.parse(MODEL.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='apply_qb_betas')
ns={'jax':jax,'jnp':jnp,'Transformer':object,'eqx':types.SimpleNamespace(tree_at=tree_at_stub)};exec(compile(ast.Module(body=[node],type_ignores=[]),str(MODEL),'exec'),ns);apply=ns['apply_qb_betas']
def run(bias):
 logits=jnp.asarray([[.1,.2,0.]],jnp.float32);x=jnp.ones((1,1),jnp.bfloat16);out=route(recipe,x,logits,bias[0]);return {'selected_experts':out[1].tolist(),'combine_weights':out[2].astype(jnp.float32).tolist(),'qb_alpha':out[3].tolist()}
def main():
 checks=[]
 def check(n,v):assert v,n;checks.append(n)
 stored=model(jnp.zeros((1,3),jnp.float32));pending=jnp.asarray([[0.,3.,0.]],jnp.float32);next_view=apply(stored,pending);again=apply(next_view,pending);shifted=apply(stored,pending+7.)
 a=run(stored.stacked_blocks.stacked.mlp.router_bias);b=run(next_view.stacked_blocks.stacked.mlp.router_bias)
 check('Real CPU backend',jax.default_backend()=='cpu')
 check('Original helper centers and replaces bias',next_view.stacked_blocks.stacked.mlp.router_bias.tolist()==[[1.,-2.,1.]])
 check('Input stored model remains unchanged under stub replacement',stored.stacked_blocks.stacked.mlp.router_bias.tolist()==[[0.,0.,0.]])
 check('Applying same pending twice is idempotent',bool(jnp.array_equal(again.stacked_blocks.stacked.mlp.router_bias,next_view.stacked_blocks.stacked.mlp.router_bias)))
 check('Constant beta shift cancels in this exact-input control',bool(jnp.array_equal(shifted.stacked_blocks.stacked.mlp.router_bias,next_view.stacked_blocks.stacked.mlp.router_bias)))
 check('Same logits but different state view changes selected set',a['selected_experts']==[[1,0]] and b['selected_experts']==[[0,2]])
 # Shape remains fixed, but both weight pairing and K+1 threshold follow selected IDs.
 check('Same weight numbers need not mean same expert pairing',len(a['selected_experts'][0])==len(b['selected_experts'][0])==2 and a['combine_weights']==b['combine_weights'] and a['qb_alpha']!=b['qb_alpha'])
 nonfinite=apply(stored,jnp.asarray([[0.,jnp.inf,0.]],jnp.float32)).stacked_blocks.stacked.mlp.router_bias
 check('One nonfinite beta contaminates centered layer bias in this control',bool(jnp.all(~jnp.isfinite(nonfinite))))
 train=R/'sources/scale_2026_10_05/train_hero_ep.py';weights=R/'sources/state_2026_10_05/weights.py';txt=train.read_text();w=weights.read_text()
 check('Frozen training and inference consumers have different pending handling','model_getter=lambda s: s.params' in txt and 'qb_params = apply_qb_betas(state.params, state.pending_qb_betas)' in txt and 'return apply_qb_betas(cast(Transformer, state[weights_key]), pending)' in w)
 o={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'runtime':{'jax':jax.__version__,'backend':jax.default_backend()},'fixture':{'router_logits':[[.1,.2,0.]],'stored_bias':[[0.,0.,0.]],'pending_beta':pending.tolist(),'pending_applied_bias':next_view.stacked_blocks.stacked.mlp.router_bias.tolist(),'stored_view':a,'pending_applied_view':b,'nonfinite_beta_input':['0','Infinity','0'],'nonfinite_bias_output':[str(float(x)) for x in nonfinite[0]]},'actual_eqx_tree_at':False,'actual_checkpoint_IO':None,'actual_Hero_view_error_magnitude':None,'actual_Hero_nonfinite_beta_event':None,'actual_GPU_execution':None,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [MODEL,ROUTER,train,weights]}}
 (R/'analysis/pending_router_view_cpu.json').write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(json.dumps({'checks_passed':len(checks),'fixture':o['fixture']},indent=2))
if __name__=='__main__':main()
