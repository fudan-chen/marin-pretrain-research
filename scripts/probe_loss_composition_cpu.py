"""Original model loss method, Grug reduction and reference CE on fixed predictions.
Synthetic hidden/router providers and CPU reference delegate; no actual model or measured Hero loss.
"""
import ast,hashlib,json,pathlib,types
import jax,jax.numpy as jnp,numpy as np
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/contracts_2026_10_05';G=R/'sources/scale_2026_10_05/grug_loss.py';M=D/'model.py';ns={'jax':jax,'jnp':jnp,'named_call':lambda f:f,'_current_mesh':lambda:None}
def extract(p,names):
 nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names];assert len(nodes)==len(names);exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])),str(p),'exec'),ns)
extract(D/'api.py',['_apply_reduction']);extract(D/'reference.py',['_default_logsumexp','_apply_logit_soft_cap','linear_softmax_cross_entropy_loss_reference']);ns['_cross_entropy_logsumexp']=ns['_default_logsumexp']
api=next(n for n in ast.parse((D/'api.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='fused_cross_entropy_loss_and_logsumexp_penalty' and len(n.body)>1);loop=next(n for n in api.body if isinstance(n,ast.For));penalty=next(n for n in loop.body if isinstance(n,ast.If) and 'logsumexp_weight' in ast.unparse(n.test));idx=loop.body.index(penalty);tail=compile(ast.fix_missing_locations(ast.Module(body=loop.body[idx:idx+2],type_ignores=[])),str(D/'api.py'),'exec')
def delegate(h,l,w,**kw):
 loss,lse=ns['linear_softmax_cross_entropy_loss_reference'](h,l,w,dtype=kw['dtype']);env=dict(ns,loss=loss,lse=lse,logsumexp_weight=kw['logsumexp_weight'],reduction=kw['reduction'],weight=kw['weight']);exec(tail,env);return env['reduced_loss']
ns['fused_cross_entropy_loss_and_logsumexp_penalty']=delegate;extract(G,['_psum_over_axes','fused_linear_softmax_cross_entropy_loss']);ns['_CE_BLOCK_SIZES']=None;ns['_summarize_router_metrics']=lambda metrics:dict(metrics)
cl=next(n for n in ast.parse(M.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Transformer');method=next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name=='next_token_loss');exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),method],type_ignores=[])),str(M),'exec'),ns)
class Provider:
 def __init__(self,head):self.output_proj=head;self.config=types.SimpleNamespace(report_capacity_overflow=False)
 def __call__(self,tokens,mask=None):return jnp.ones((2,4,1)),{'router_z_loss_per_layer':jnp.array([0.])}
def main():
 checks=[];head=jnp.array([[2.,-1.]]);tokens=jnp.array([[0,0,0,0],[1,1,1,1]]);a=jnp.array([[1.,0,0,0],[1.,1,1,0]]);b=jnp.array([[1.,1,1,0],[1.,0,0,0]]);fraction=b*jnp.array([[.5],[1.]])
 def loss(h,weight,alpha=None,metrics=False):return ns['next_token_loss'](Provider(h),tokens,weight,logsumexp_weight=alpha,return_router_metrics=metrics)
 def check(n,c):assert c,n;checks.append(n)
 values={name:float(loss(head,wt)) for name,wt in [('A1_B3',a),('A3_B1',b),('fractional_A',fraction)]};expected_ce=np.logaddexp(2.,-1.)-np.array([2.,-1.]);check('Fixed predictions produce different loss under changed valid-target composition',abs(values['A1_B3']-values['A3_B1']-1.5)<1e-6)
 check('Original fractional mean uses weight sum not active count',abs(values['fractional_A']-float(np.dot([1.5,1],expected_ce)/2.5))<1e-6)
 val,grad=jax.value_and_grad(lambda h:loss(h,b))(head);v7,g7=jax.value_and_grad(lambda h:loss(h,b*7))(head);check('Uniform positive weight scaling leaves mean and gradient unchanged',np.allclose(val,v7,atol=1e-6) and np.allclose(grad,g7,atol=1e-6))
 plain_shift=float(loss(head+10,b));z=float(loss(head,b,1e-4));zshift=float(loss(head+10,b,1e-4));check('Common logit shift preserves pure CE in CPU tolerance',abs(plain_shift-float(val))<2e-6);check('Same probabilities change total objective when output z-loss enabled',zshift>z and zshift>plain_shift)
 _,zg=jax.value_and_grad(lambda h:loss(h,b,1e-4))(head);check('Output z-loss changes head gradient',not np.allclose(grad,zg,atol=1e-6))
 output,metrics=loss(head,b,1e-4,True);check('Original cross_entropy_loss metric includes enabled output penalty',float(metrics['train/cross_entropy_loss'])==float(output)==z and z>float(val))
 o={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'runtime':{'jax':jax.__version__,'numpy':np.__version__},'observations':{'per_domain_pure_ce':expected_ce.tolist(),'fixed_prediction_losses':values,'pure_ce':float(val),'shifted_pure_ce':plain_shift,'with_z_loss':z,'shifted_with_z_loss':zshift,'head_gradient_pure':np.asarray(grad).tolist(),'head_gradient_with_z':np.asarray(zg).tolist(),'weighted_target_mass':{'A1_B3':[1,3],'A3_B1':[3,1],'fractional_A':[1.5,1]}},'explicit_substitutions':['fixed hidden/router provider instead of Transformer forward','CPU original reference delegate replaces xla_fast_bwd dispatch','mesh=None; no collective','identity named_call; router summarizer identity'],'actual_Hero_loss_effect':None,'actual_Hero_target_composition':None,'actual_GPU_kernel':None,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [D/'api.py',D/'reference.py',G,M]}}
 (R/'analysis/loss_composition_cpu.json').write_text(json.dumps(o,indent=2)+'\n');print('Original loss composition checks:',len(checks))
if __name__=='__main__':main()
