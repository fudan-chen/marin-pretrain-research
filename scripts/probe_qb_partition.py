"""Original QB helpers with NumPy top_k substitute; no JAX collective or training."""
import ast,hashlib,json,pathlib,types
import numpy as np
R=pathlib.Path(__file__).resolve().parents[1];P=R/'sources/routing_2026_10_05/grug_moe.py';checks=[]
def top_k(x,k):
 order=np.argsort(-x,axis=-1,kind='stable')[...,:k]
 return np.take_along_axis(x,order,axis=-1),order
jnp=types.SimpleNamespace(**{k:getattr(np,k) for k in dir(np) if not k.startswith('_')})
env={'jnp':jnp,'jax':types.SimpleNamespace(lax=types.SimpleNamespace(top_k=top_k))}
names={'qb_topk_physical_count','qb_beta_topk_shard','reduce_moe_routing_stats'}
body=[]
for f in ast.parse(P.read_text()).body:
 if isinstance(f,ast.FunctionDef) and f.name in names:
  f.decorator_list=[];f.returns=None
  for a in f.args.args+f.args.kwonlyargs:a.annotation=None
  body.append(f)
exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(P),'exec'),env)
def check(n,c):assert c,n;checks.append(n)
def local(m,valid):
 k=env['qb_topk_physical_count'](len(m),num_experts_per_token=1,num_experts=2)
 return env['qb_beta_topk_shard'](m,np.array(valid,bool),physical_count=k,num_experts_per_token=1,num_experts=2)
def reduce(betas,weights):
 n=len(weights);stack={'routing_counts_local':np.ones((1,n,2),np.float32),'router_prob_sum_local':np.ones((1,n,2),np.float32),'router_z_sq_sum_local':np.zeros((1,n),np.float32),'valid_tokens_local':np.array([weights],np.float32),'qb_beta_local':np.array([betas],np.float32),'qb_beta_weight_local':np.array([weights],np.float32)}
 return env['reduce_moe_routing_stats'](stack,num_experts=2,num_experts_per_token=1)['qb_beta'][0]
logits=np.array([[100,0],[99,0],[0,1],[0,2]],np.float32)
alpha=np.min(logits,axis=-1,keepdims=True);margins=logits-alpha
cases=[]
for name,groups in [('contiguous',[[0,1],[2,3]]),('interleaved',[[0,2],[1,3]])]:
 parts=[local(margins[g],[True]*len(g)) for g in groups];beta=reduce([p[0] for p in parts],[p[1] for p in parts]);bias=-beta+beta.mean();next_logits=np.array([60,0],np.float32)+bias
 cases.append({'name':name,'groups':groups,'local_betas':[p[0].tolist() for p in parts],'valid_counts':[int(p[1]) for p in parts],'beta':beta.tolist(),'centered_bias':bias.tolist(),'next_query_raw_logits':[60,0],'next_query_biased_logits':next_logits.tolist(),'next_selected_expert':int(np.argmax(next_logits))})
global_beta=local(margins,[True]*4)[0]
check('Margins follow zero-bias K+1 threshold for a two-expert toy',np.array_equal(alpha,np.zeros((4,1))))
check('Identical global inputs retain identical exact global threshold',global_beta.tolist()==[99,1])
check('Original shard estimator differs under a new partition',[c['beta'] for c in cases]==[[50,1],[99.5,1.5]])
check('Weighted local threshold need not equal pooled threshold',all(c['beta']!=global_beta.tolist() for c in cases))
check('Centered bias can change next expert selection under partition',[c['next_selected_expert'] for c in cases]==[0,1])
empty,n=local(np.array([[1e6,1e6],[1e6,1e6]],np.float32),[False,False]);check('All-padding shard emits finite zero and zero weight',n==0 and empty.tolist()==[0,0])
check('Zero-weight empty shard cannot change reduced beta',np.array_equal(reduce([[50,1],empty],[4,0]),[50,1]))
b,n=local(np.array([[5,1],[1000,1000]],np.float32),[True,False]);check('Invalid extreme margin cannot enter the local threshold',b.tolist()==[5,1] and n==1)
check('Effective count weights local betas rather than equal shard mean',reduce([[10,0],[0,10]],[1,3]).tolist()==[2.5,7.5])
check('Local token permutation leaves order-statistic threshold unchanged',np.array_equal(local(margins[:2][::-1],[True,True])[0],local(margins[:2],[True,True])[0]))
check('Tiny local populations clamp target count to one',env['qb_topk_physical_count'](10,num_experts_per_token=8,num_experts=384)==1)
configs=[]
for p in sorted((R/'sources/scale_2026_10_05').glob('config_*.json')):
 c=json.loads(json.loads(p.read_text())['data']['project']['run']['config'])['model']['value'];configs.append({'run':p.stem[7:],'qb_estimator':c.get('qb_estimator'),'qb_hist_bins':c.get('qb_hist_bins')})
check('Six archived ladder configurations declare HIST with 10000 bins',len(configs)==6 and all(c['qb_estimator']=='HIST' and c['qb_hist_bins']==10000 for c in configs))
# Single-device pooled histogram helper; psum is identity, no device collective.
def bincount(x,length):
 x=np.asarray(x);return np.bincount(x[(x>=0)&(x<length)],minlength=length)[:length]
env['jnp'].bincount=bincount;env['jax'].lax.psum=lambda x,axis_name:x
model=R/'sources/contracts_2026_10_05/model.py'
f=next(f for f in ast.parse(model.read_text()).body if isinstance(f,ast.FunctionDef) and f.name=='_bincount_upper_quantile')
f.decorator_list=[];f.returns=None
for a in f.args.args+f.args.kwonlyargs:a.annotation=None
exec(compile(ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[])),str(model),'exec'),env)
def hist(m,bins=10000,valid=None):
 valid=np.ones(len(m),bool) if valid is None else np.array(valid,bool);active=m[valid];lo=float(active.min()) if len(active) else 0.;hi=float(active.max()) if len(active) else 0.;hi=max(hi,lo+1e-6)
 beta=env['_bincount_upper_quantile'](m,valid,num_experts=2,n_bins=bins,lo=lo,hi=hi,target_rank=float(valid.sum())/2,token_axes=('fake',))
 return {'beta':beta.tolist(),'lo':lo,'hi':hi,'bins':bins,'bin_width':(hi-lo)/bins}
base=hist(margins);outlier=margins.copy();outlier[3,1]=1e6;wide=hist(outlier)
check('HIST common range expands after another expert extreme margin',base['bin_width']==.01 and wide['bin_width']==100.)
check('HIST can change expert threshold without changing its own margins',np.array_equal(margins[:,0],outlier[:,0]) and base['beta'][0]!=wide['beta'][0])
check('HIST pooled histogram is invariant to toy row permutation',hist(margins[[0,2,1,3]])['beta']==base['beta'])
padded=np.concatenate([margins,[[1e6,1e6]]]).astype(np.float32)
check('Padding is excluded from artificial HIST range and counts',hist(padded,valid=[1,1,1,1,0])==base)
paths=[P,R/'sources/contracts_2026_10_05/model.py',R/'sources/scale_2026_10_05/train_hero_ep.py']
out={'checks_passed':len(checks),'checks':checks,'raw_logits':logits.tolist(),'margins':margins.tolist(),'cases':cases,'pooled_exact_topk_beta':global_beta.tolist(),'histogram_range_example':{'base':base,'other_expert_outlier':wide,'unchanged_expert':0,'valid_tokens':4},'unequal_count_example':{'betas':[[10,0],[0,10]],'weights':[1,3],'result':[2.5,7.5]},'configurations':configs,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'scope':'original local threshold, post-scan reduction and pooled histogram helper with NumPy dependency substitutes; centered bias and next query are transparent artificial arithmetic','actual_mesh_execution':None,'actual_training_effect':None,'historical_execution_sha':None}
(R/'analysis/qb_partition_probe.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('QB partition original helper checks:',len(checks))
