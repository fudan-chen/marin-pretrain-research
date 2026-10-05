"""Original optimizer masks/decay wrapper with identity-Adam and flat-tree substitutes.
This does not reproduce Adam moments, Optax multi_transform or JAX sharding.
"""
import ast,collections,hashlib,json,pathlib,types
import numpy as np
R=pathlib.Path(__file__).resolve().parents[1];P=R/'sources/optimizer_2026_10_05/optimizer.py';OLD=R/'sources/code_optimizer.py';checks=[]
State=collections.namedtuple('SyntheticState','count moment_marker');GT=collections.namedtuple('GT','init update')
def tree_map(f,*trees):return {k:f(*(t[k] for t in trees)) for k in trees[0]}
def fake_adam(*args):return GT(lambda p:State(0,'synthetic-not-Adam'),lambda u,s,p:(u,State(s.count+1,s.moment_marker)))
jnp=types.SimpleNamespace(**{k:getattr(np,k) for k in dir(np) if not k.startswith('_')})
def clip(x,a_min,a_max):return np.maximum(x,a_min) if a_max is None else np.clip(x,a_min,a_max)
jnp.clip=clip
env={'jnp':jnp,'jax':types.SimpleNamespace(tree=types.SimpleNamespace(map=tree_map)),'optax':types.SimpleNamespace(scale_by_adam=fake_adam,GradientTransformation=GT),'leaf_key_paths':lambda p:{k:k for k in p}}
tree=ast.parse(P.read_text());body=[]
for f in tree.body:
 if isinstance(f,ast.FunctionDef) and f.name in {'_is_gate_or_router_weight','_gate_router_decay_mask','_scale_by_adam_gate_router_decay'}:body.append(f)
 if isinstance(f,ast.ClassDef) and f.name=='GrugMoeMuonHConfig':
  body.append(next(x for x in f.body if isinstance(x,ast.FunctionDef) and x.name=='create_mask'))
for f in body:
 f.decorator_list=[];f.returns=None
 for a in f.args.args+f.args.kwonlyargs:a.annotation=None
exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(P),'exec'),env)
def check(n,c):assert c,n;checks.append(n)
params={'blocks.attn.attn_gate':np.array([2.,4.]),'blocks.mlp.router':np.array([3.,6.]),'blocks.mlp.router_bias':np.array([5.,10.]),'blocks.mlp.router_extra':np.ones((2,2)),'token_embed':np.ones((2,2)),'output_proj':np.ones((2,2)),'blocks.norm.weight':np.ones(2),'blocks.gated_norm.weight':np.ones(2),'blocks.mlp.w1':np.ones((2,2))}
mask=env['_gate_router_decay_mask'](params);groups=env['create_mask'](None,params)
check('Only exact gate/router leaf suffixes receive specialized decay',[k for k,v in mask.items() if v]==['blocks.attn.attn_gate','blocks.mlp.router'])
check('Router bias remains Adam group but has no specialized decay',groups['blocks.mlp.router_bias']=='adam' and not mask['blocks.mlp.router_bias'])
check('Output projection routes to AdamH',groups['output_proj']=='adamh')
check('Gated norm precedes ordinary weight-suffix routing',groups['blocks.gated_norm.weight']=='muonh' and groups['blocks.norm.weight']=='adam')
check('Matrix and embedding groups follow their distinct rules',groups['blocks.mlp.w1']=='muonh' and groups['token_embed']=='adam')
grad={k:np.full_like(p,.5) for k,p in params.items()};transform=env['_scale_by_adam_gate_router_decay'](.9,.95,1e-8,.2,100);cases=[]
for count in [0,50,100,150]:
 u,s=transform.update(grad,State(count,'preserved-marker'),params);wd=.2*max(1-count/100,0);final={k:(p-.01*u[k]).tolist() for k,p in params.items()};cases.append({'input_adam_count':count,'output_adam_count':s.count,'decay_coefficient':wd,'router_final_with_artificial_lr':final['blocks.mlp.router'],'router_bias_final_with_artificial_lr':final['blocks.mlp.router_bias']})
 check('Original wrapper decay arithmetic at count '+str(count),np.allclose(u['blocks.mlp.router'],grad['blocks.mlp.router']+wd*params['blocks.mlp.router']))
check('Wrapper forwards underlying state without adding decay state',s.moment_marker=='preserved-marker' and s.count==151)
check('Decay uses input count before synthetic Adam increment',cases[0]['decay_coefficient']==.2 and cases[1]['decay_coefficient']==.1)
check('Bias update excludes parameter decay contribution',all(c['router_bias_final_with_artificial_lr']==[4.995,9.995] for c in cases))
try:transform.update(grad,State(0,'x'),None);bad=False
except ValueError:bad=True
check('Decay wrapper rejects missing params',bad)
cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='GrugMoeMuonHConfig');default=next(x.value.value for x in cls.body if isinstance(x,ast.AnnAssign) and x.target.id=='gate_router_weight_decay')
check('Current default is zero and historical optimizer has no dedicated branch',default==0 and '_scale_by_adam_gate_router_decay' not in OLD.read_text())
configs=[]
for p in sorted((R/'sources/scale_2026_10_05').glob('config_*.json')):
 c=json.loads(json.loads(p.read_text())['data']['project']['run']['config'])['optimizer']['value'];configs.append({'run':p.stem[7:],'dedicated_decay_present':'gate_router_weight_decay' in c,'dedicated_decay_value':c.get('gate_router_weight_decay'),'max_grad_norm':c.get('max_grad_norm'),'generic_weight_decay':c.get('weight_decay')})
check('Six ladder records retain three absent and three explicit zero settings',sum(c['dedicated_decay_present'] for c in configs)==3 and all(c['dedicated_decay_value'] in (None,0) for c in configs))
build=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='build');clip_calls=[x for x in ast.walk(build) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='clip_by_global_norm'];check('Clipping call sites lie in three separate group transforms',len(clip_calls)==3)
out={'checks_passed':len(checks),'checks':checks,'parameter_groups':groups,'dedicated_decay_mask':mask,'cases':cases,'configurations':configs,'default_dedicated_decay':default,'scope':'original flat-tree masks and decay wrapper with identity Adam dependency substitute; artificial learning rate; not actual Adam or multi_transform','source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [P,OLD]},'actual_optimizer_state':None,'actual_clipping_result':None,'actual_training_effect':None,'historical_execution_sha':None};(R/'analysis/optimizer_probe.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('Optimizer helper checks:',len(checks))
