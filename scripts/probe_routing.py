"""Original routing count/clip/combine helpers with NumPy substitutes.
No collective, kernel, model, gradient or historical batch is executed.
"""
import ast,contextlib,hashlib,json,pathlib,types
import numpy as np
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/routing_2026_10_05';checks=[]
jnp=types.SimpleNamespace(**{k:getattr(np,k) for k in dir(np) if not k.startswith('_')});jnp.einsum=lambda *a,**k:np.einsum(*a,**{n:v for n,v in k.items() if n!='preferred_element_type'})
env=dict(np=np,jnp=jnp,jax=types.SimpleNamespace(enable_x64=contextlib.nullcontext,Array=np.ndarray),sonic_gather_sum_available=lambda:False,_sort_activations=lambda a,p:a[p])
for name in ['DROPPED_ASSIGNMENTS','SENDER_DROPPED_ASSIGNMENTS','RECEIVER_DROPPED_ASSIGNMENTS','SKIPPED_PADDING_ASSIGNMENTS','VALID_ASSIGNMENTS']:env['MOE_'+name+'_METRIC']='moe/'+name.lower()
def extract(p,names,namespace):
 t=ast.parse(p.read_text());body=[]
 for f in t.body:
  if isinstance(f,ast.FunctionDef) and f.name in names:
   f.decorator_list=[];f.returns=None
   for a in f.args.args+f.args.kwonlyargs:a.annotation=None
   body.append(f)
 exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(p),'exec'),namespace)
def check(name,c):assert c,name;checks.append(name)
extract(D/'common.py',{'padding_skipped_assignments','_assignment_validity','_scaled_capacity'},env);extract(D/'ep_common.py',{'_prefix_cap_counts','_clip_receiver_group_sizes'},env);extract(D/'ragged.py',{'_unpermute_from_global_expert'},env)
T=R/'sources/scale_2026_10_05/train_hero_ep.py';OLD=R/'sources/code_train.py';oldenv=dict(env);extract(T,{'_drop_metrics'},env);extract(OLD,{'_drop_metrics'},oldenv)
valid=np.array([True,True,False,False]);mask=env['_assignment_validity'](valid,tokens=4,topk=8);skipped=env['padding_skipped_assignments'](valid,topk=8);check('Padding expands to skipped assignments independently of loss',skipped==16 and mask.sum()==16)
args=[np.array([4],np.int32),np.array([4],np.int32),np.array([0],np.int32)];kwargs=dict(batch_size=1,sequence_length=4,top_k=8,num_layers=1);new=env['_drop_metrics'](*args,np.array([16]),np.array([16]),**kwargs);old=oldenv['_drop_metrics'](*args,**kwargs);check('Denominator version changes padded example from 12.5 to 25 percent',old['moe/drop_fraction']==.125 and new['moe/drop_fraction']==.25)
try:env['_drop_metrics'](*args,np.array([15]),np.array([16]),**kwargs);bad=False
except ValueError:bad=True
check('Current metrics reject invalid-plus-padding accounting mismatch',bad)
check('Dynamic logical capacity clips to declared physical capacity',int(env['_scaled_capacity'](np.array(16),capacity_factor=1.15,divisor=2,minimum=2,maximum=8))==8)
groups=np.array([[4,0,0,4],[0,4,4,0]],np.int32);accepted=env['_clip_receiver_group_sizes'](groups,local_expert_size=2,receiver_capacity=4);check('Receiver clipping prioritizes expert then sender prefixes',accepted.tolist()==[[4,0,0,0],[0,0,4,0]]);check('Receiver clipping respects each receiver capacity',accepted.reshape(2,2,2).sum(axis=(0,2)).tolist()==[4,4])
n,k=8,8;weights=np.full((n,k),2.5/k,np.float32);cases=[]
for name in ['concentrated','distributed']:
 dropped=np.zeros((n,k),bool)
 if name=='concentrated':dropped[0,:]=True
 else:dropped[:,0]=True
 outputs=np.ones((n*k,1),np.float32);outputs[dropped.reshape(-1)]=0
 y=env['_unpermute_from_global_expert'](outputs,np.arange(n*k),weights,tokens_per_shard=n,topk=k);ce=np.logaddexp(0,-y[:,0]);case=dict(name=name,dropped_assignments=int(dropped.sum()),assignment_drop_fraction=float(dropped.mean()),affected_tokens=int(dropped.any(axis=1).sum()),fully_dropped_tokens=int(dropped.all(axis=1).sum()),routed_output=y[:,0].tolist(),synthetic_binary_nll=ce.tolist(),mean_synthetic_binary_nll=float(ce.mean()),lost_weight_mass=(weights*dropped).sum(axis=1).tolist(),drop_pattern=dropped.astype(int).tolist());cases.append(case)
check('Same assignment rate need not touch same number of tokens',cases[0]['assignment_drop_fraction']==cases[1]['assignment_drop_fraction']==.125 and [c['affected_tokens'] for c in cases]==[1,8]);check('Routed combine retains original weights without survivor renormalization',cases[1]['routed_output']==[2.1875]*8);check('Same lost count can yield different artificial NLL',cases[0]['mean_synthetic_binary_nll']!=cases[1]['mean_synthetic_binary_nll'])
# Exact combinatorial bounds, checked against every 3x2 pattern.
for bits in range(64):
 a=np.array([(bits>>i)&1 for i in range(6)]).reshape(3,2);d=int(a.sum());affected=int(a.any(axis=1).sum());full=int(a.all(axis=1).sum());assert (d+1)//2<=affected<=min(d,3);assert max(0,d-3)<=full<=d//2
check('Affected/full-drop bounds hold across all 64 small patterns',True)
configs=[]
for p in sorted((R/'sources/scale_2026_10_05').glob('config_*.json')):
 run=json.loads(p.read_text())['data']['project']['run'];c=json.loads(run['config'])['model']['value'];configs.append(dict(run=p.stem[7:],topk=c['num_experts_per_token'],shared_experts=c['num_shared_experts'],report_capacity_overflow=c['report_capacity_overflow']))
check('All six ladder configurations report top8/shared2/capacity counters',all(c['topk']==8 and c['shared_experts']==2 and c['report_capacity_overflow'] for c in configs))
paths=[D/'common.py',D/'ep_common.py',D/'ragged.py',D/'pooled.py',D/'grug_moe.py',T,OLD,R/'sources/contracts_2026_10_05/model.py'];out=dict(checks_passed=len(checks),checks=checks,metric_denominator_example=dict(old=old,new=new,scope='artificial padded input, no historical metric rebinding'),receiver_clip_example=dict(input=groups.tolist(),accepted=accepted.tolist(),receiver_capacity=4),cases=cases,ladder_configurations=configs,source_sha256={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},actual_collective_result=None,actual_gradient_result=None,actual_input_drop_pattern=None,historical_execution_sha=None);(R/'analysis/routing_probe.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
# Two transparent assignment grids; scalar output/NLL is a synthetic illustration.
svg=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 380" role="img" aria-labelledby="title desc"><title id="title">相同分配丢弃率，不同token损伤分布</title><desc id="desc">人工8 token乘8分配算例。左侧集中丢弃一个token的8分配，右侧每个token丢弃一个分配。</desc><rect width="800" height="380" fill="white"/>']
for ci,c in enumerate(cases):
 x0=35+ci*400;svg.append(f'<text x="{x0}" y="26" font-size="19" font-family="sans-serif">'+('集中：1个token失去8项分配' if ci==0 else '分散：8个token各失去1项分配')+'</text>')
 for t,row in enumerate(c['drop_pattern']):
  for e,lost in enumerate(row):svg.append(f'<rect x="{x0+e*31}" y="{42+t*31}" width="28" height="28" rx="3" fill="'+('#c75243' if lost else '#d9e8ef')+'"/>')
 svg.append(f'<text x="{x0}" y="318" font-size="16" font-family="sans-serif">分配丢弃率12.5%；受影响token：{c["affected_tokens"]}/8</text>')
 svg.append(f'<text x="{x0}" y="344" font-size="16" font-family="sans-serif">人工平均NLL：{c["mean_synthetic_binary_nll"]:.6f}</text>')
svg.append('<text x="35" y="372" font-size="14" font-family="sans-serif">红色为丢弃分配。仅原combine辅助函数＋人工输出/二分类目标，非真实模型结果。</text></svg>');(R/'assets/drop_patterns.svg').write_text('\n'.join(svg));print('Routing checks:',len(checks))
