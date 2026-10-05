"""Archived NS functions with float32 replacing BF16, global NumPy arrays, fake layout ops.
Control-flow and polynomial evidence only; no actual distributed slice, collective or QuACK kernel.
"""
import ast,collections,hashlib,json,math,pathlib,types
import numpy as np
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/muon_direction_2026_10_05';P=D/'grugmuon_hero.py';checks=[];layouts=[];mesh=None
class Spec(tuple):
 def __new__(cls,*x):return tuple.__new__(cls,x)
class Named:
 def __init__(self,spec):self.spec=spec
class Mesh:
 def __init__(self,shape):self.shape=shape;self.empty=not shape
mesh=Mesh({})
def reshard(x,spec):layouts.append({'shape':list(x.shape),'spec':list(spec.spec if isinstance(spec,Named) else spec)});return x
def vm(f):return lambda x:np.stack([f(y) for y in x])
def reshape(x,shape,**kw):return np.reshape(x,shape)
def einsum(eq,*x,**kw):return np.einsum(eq,*x)
jnp=types.SimpleNamespace(**{k:getattr(np,k) for k in dir(np) if not k.startswith('_')});jnp.bfloat16=np.float32;jnp.einsum=einsum
ut=ast.parse((D/'util.py').read_text());coeff=ast.literal_eval(next(x.value for x in ut.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='NEWTON_SCHULZ_COEFFICIENTS' for t in x.targets)))
ns={'jnp':jnp,'jax':types.SimpleNamespace(vmap=vm,sharding=types.SimpleNamespace(get_abstract_mesh=lambda:mesh),lax=types.SimpleNamespace(reshape=reshape),tree_util=types.SimpleNamespace(keystr=str)),'PartitionSpec':Spec,'NamedSharding':Named,'reshard':reshard,'math':math,'NEWTON_SCHULZ_COEFFICIENTS':coeff,'_axis_names':lambda x:() if x is None else x if isinstance(x,tuple) else (x,),'shard_map':lambda f,**kw:f}
names={'_intra_rack_axes','_zeropower_via_newtonschulz_local','_zeropower_via_newtonschulz_replicated','_newtonschulz_padded_stack_sharded','_newtonschulz_4d_distributed'};tree=ast.parse(P.read_text());body=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names];future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0);exec(compile(ast.fix_missing_locations(ast.Module(body=[future]+body,type_ignores=[])),str(P),'exec'),ns)
def ck(n,v):assert v,n;checks.append(n)
local=ns['_zeropower_via_newtonschulz_local'];x=np.diag([1.,.2,.01,1e-5]).astype(np.float32);trajectory=[]
for n in range(11):
 q=local(x,n,1e-8,'quintic');trajectory.append({'steps':n,'singular_values':np.linalg.svd(q,compute_uv=False).tolist(),'orthogonality_residual':float(np.linalg.norm(q@q.T-np.eye(4)))})
# Independent scalar spectrum recurrence, not a matrix multiply mirror.
s=np.diag(x)/(np.linalg.norm(x)+1e-8)
for a,b,c in coeff['quintic']:s=a*s+b*s**3+c*s**5
ck('Five-step matrix output agrees with scalar singular polynomial',np.allclose(np.sort(np.abs(s))[::-1],trajectory[5]['singular_values'],atol=1e-5))
ck('Five steps do not produce exact unit singular values',not np.allclose(trajectory[5]['singular_values'],np.ones(4),atol=1e-3))
ck('Six-step repeated sequence worsens artificial orthogonality residual',trajectory[6]['orthogonality_residual']>trajectory[5]['orthogonality_residual'])
ck('Exact rank deficiency remains zero',local(np.diag([1.,0.]).astype(np.float32),5,1e-8,'quintic')[1,1]==0)
ck('Zero matrix remains finite zero',np.array_equal(local(np.zeros((2,3),np.float32),5,1e-8,'quintic'),np.zeros((2,3))))
wide=np.array([[1.,2.,3.,-.5],[2.,-1.,.5,1.]],np.float32);ck('Tall transpose branch agrees with wide transpose',np.allclose(local(wide.T,5,1e-8,'quintic'),local(wide,5,1e-8,'quintic').T,atol=1e-5))
ck('No-mesh replicated and local outputs agree under float32 substitute',np.array_equal(local(wide,5,1e-8,'quintic'),ns['_zeropower_via_newtonschulz_replicated'](wide,5,1e-8,'quintic')))
ck('Rack selector excludes DCN when intra-rack axes exist',ns['_intra_rack_axes'](Mesh({'replica_dcn':8,'data':2,'model':4}))==[('data',2),('model',4)])
ck('Rack selector falls back to DCN if it is sole active axis',ns['_intra_rack_axes'](Mesh({'replica_dcn':8,'data':1}))==[('replica_dcn',8)])
stack=np.stack([wide,wide*2,wide*3]);mesh=Mesh({'data':2,'model':2});layouts.clear();q=ns['_newtonschulz_padded_stack_sharded'](stack,5,1e-8,'quintic',target_sharding=Named(Spec(None,None,None)));layout3=list(layouts)
ck('Three-layer stack pads to four and crops back',layout3[0]['shape'][0]==4 and q.shape==stack.shape)
ck('Zero padded layer does not change valid independent matrices',np.allclose(q,np.stack([local(y,5,1e-8,'quintic') for y in stack])))
mesh=Mesh({'data':2,'model':4});guard_stack=np.concatenate([stack,stack]);
try:ns['_newtonschulz_padded_stack_sharded'](guard_stack,5,1e-8,'quintic',target_sharding=Named(Spec('data',None,None)));raised=False
except ValueError:raised=True
ck('Padded leading sharded target is rejected by original guard',raised)
mesh=Mesh({'expert':1,'context':4,'data':2,'model':2});layouts.clear();x6=np.stack([stack,stack]);x4=np.stack([np.concatenate([stack,stack[:1]])]*2);target=Named(Spec(None,'context','data','model'));q4=ns['_newtonschulz_4d_distributed']([],x4,5,1e-8,'quintic',False,target);layout4=list(layouts)
ck('Composite bank context split uses bank-preserving 4D branch',layout4[0]['spec']==[None,'context',None,None] and len(layout4[0]['shape'])==4)
ck('Context-bank synthetic target dimensions divide declared mesh factors',x4.shape[1]%4==0 and x4.shape[2]%2==0 and x4.shape[3]%2==0)
ck('Bank branch restores exact declared target spec',layout4[-1]['spec']==list(target.spec))
ck('Each expert matrix receives independent NS before later hyperball',np.allclose(q4,np.stack([np.stack([local(z,5,1e-8,'quintic') for z in y]) for y in x4])))
mesh=Mesh({'data':2,'model':2});layouts.clear();target=Named(Spec(None,None,'data','model'));ns['_newtonschulz_4d_distributed']([],np.stack([stack[:2],stack[:2]]),5,1e-8,'quintic',False,target);merged_layout=list(layouts)
ck('Unsplit bank uses maximal divisible merged-axis shard product',merged_layout[0]['shape']==[4,2,4] and merged_layout[0]['spec']==[('data','model'),None,None])
try:ns['_newtonschulz_4d_distributed']([],x6,5,1e-8,'quintic',False,target);raised=False
except ValueError:raised=True
ck('Unsplit merged six does find a divisible two-axis subset',not raised)
mesh=Mesh({'data':4});
try:ns['_newtonschulz_4d_distributed']([],x6,5,1e-8,'quintic',False,Named(Spec(None,None,'data',None)));raised=False
except ValueError:raised=True
ck('Unsplit merged six with only shard four rejects instead of padding',raised)
# Positive common fan scale cancels from hyperball direction normalization outside epsilon floor.
g=np.array([[0.,1.],[1.,0.]],np.float32);p=np.array([[2.,0.],[0.,1.]],np.float32)
def projection(u):v=p-.1*u*np.linalg.norm(p)/np.linalg.norm(u);return v*np.linalg.norm(p)/np.linalg.norm(v)
ck('Common positive fan factor cancels from normalized hyperball direction',np.allclose(projection(g),projection(g*np.sqrt(3))))
out={'checks_passed':len(checks),'checks':checks,'scope':__doc__,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [P,D/'util.py']},'trajectory':trajectory,'artificial_input_diagonal':np.diag(x).tolist(),'three_dimensional_layout_calls':layout3,'context_bank_layout_calls':layout4,'merged_bank_layout_calls':merged_layout,'coefficient_sequence':coeff['quintic'],'actual_BF16_result':None,'actual_QuACK_result':None,'actual_distributed_result':None,'actual_training_effect':None,'historical_execution_sha':None};(R/'analysis/muon_direction_probe.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('Muon direction checks:',len(checks));print([(v['steps'],round(v['orthogonality_residual'],5)) for v in trajectory[:7]])
