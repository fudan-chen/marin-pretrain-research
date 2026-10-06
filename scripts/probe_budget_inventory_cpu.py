"""Original train_sets method with original PRP/slice/mix bodies on CPU identity stores.
Replaces cache construction, AsyncDataset base/sync size bridge, and key_iterator.
Original __post_init__ constraints also checked; no full LmDataConfig init, real tokens, pilot model training or production repeat measurement.
"""
import ast,asyncio,hashlib,json,pathlib,platform,types
import jax
import probe_inner_shuffle_cpu as inner
import probe_mixture_identity_cpu as mixture
R=pathlib.Path(__file__).resolve().parents[1];P=R/'sources/deepening_2026_10_04/datasets_production.py';M=R/'sources/live_2026_10_07/meta.json'
class SizeBridge:
 def __init__(self,ds):self.ds=ds
 def __len__(self):return asyncio.run(self.ds.async_len())
inner.Base.as_sync_dataset=lambda self:SizeBridge(self)
# Explicit artificial key stream, not the production key_iterator implementation.
def artificial_keys(key):
 i=0
 while True:yield jax.random.fold_in(key,i);i+=1
ns=dict(inner.ns);ns['key_iterator']=artificial_keys
tr=ast.parse(P.read_text());cls=next(n for n in tr.body if isinstance(n,ast.ClassDef) and n.name=='LmDataConfig');method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='train_sets');post=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__post_init__');block=next(n for n in tr.body if isinstance(n,ast.ClassDef) and n.name=='BlockShuffleConfig')
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),block,method,post],type_ignores=[])),str(P),'exec'),ns)
class Harness:
 train_sets=ns['train_sets']
 __post_init__=ns['__post_init__']
 def __init__(self,lengths,exp=None,target=None,shuffle=False,validation=None,max_batches=None):
  self.components={};self.train_weights=None
  self.raw={n:inner.Identity(v) for n,v in lengths.items()};self.experiment_budget=exp;self.target_budget=target;self.shuffle=shuffle;self.permutation_type='feistel';self.num_validation_sequences=validation;self.shuffle_before_trainval_split=True;self.max_train_batches=max_batches
 def build_caches(self,split):return None
 def build_token_datasets(self,caches,Pos,split):return dict(self.raw)
def make(h,initial=4,check_initialization=True):
 if check_initialization:h.__post_init__()
 return h.train_sets(None,initial_batch_size=initial,key=jax.random.PRNGKey(7))
def identities(d):return {n:asyncio.run(inner.all_ids(ds)) for n,ds in d.items()}
class Labeled:
 def __init__(self,name,ds):self.name=name;self.ds=ds
 def is_finite(self):return True
 async def async_len(self):return await self.ds.async_len()
 async def get_batch(self,indices):return [f'{self.name}:{i}' for i in await self.ds.get_batch(indices)]
def mix_stream(d,slots=24):
 mx=mixture.Mix({n:Labeled(n,ds) for n,ds in d.items()},{n:.5 for n in d},8,key=7)
 return asyncio.run(mx.get_batch(list(range(slots))))
def main():
 checks=[]
 def check(n,v):assert v,n;checks.append(n)
 check('Genuine CPU backend',jax.default_backend()=='cpu')
 small=make(Harness({'A':7,'B':23},24,96));si=identities(small)
 check('Ratio truncates each inventory with integer floor',si=={'A':[0],'B':list(range(5))})
 stream=mix_stream(small)
 reference=mix_stream(make(Harness({'A':7,'B':23})),slots=96)
 check('Artificial reference and pilot preserve nominal per-domain exposure ratio',sum(x.startswith('A:') for x in reference)==48 and sum(x.startswith('B:') for x in reference)==48 and sum(x.startswith('A:') for x in stream)==12 and sum(x.startswith('B:') for x in stream)==12)
 check('Original restart mix exposes rounded small inventory',len(stream)==24 and len(set(stream))==6)
 distort={n:{'production_exposure':48,'production_inventory':L,'production_epochs':48/L,'pilot_exposure':12,'pilot_inventory':len(si[n]),'pilot_epochs':12/len(si[n]),'epoch_ratio':(12/len(si[n]))/(48/L)} for n,L in {'A':7,'B':23}.items()}
 check('Floor increases artificial repeat factors beyond continuous scaling',abs(distort['A']['epoch_ratio']-1.75)<1e-12 and abs(distort['B']['epoch_ratio']-1.15)<1e-12)
 zero=make(Harness({'A':3,'B':23},1,10));zi=identities(zero)
 err=None
 try:mix_stream(zero)
 except ValueError as e:err=str(e)
 check('Positive budget ratio can zero a small active child',zi['A']==[] and err is not None and 'empty finite dataset' in err)
 full=identities(make(Harness({'A':23},shuffle=True)))['A'];cut=identities(make(Harness({'A':23},1,4,shuffle=True)))['A']
 check('Budget cap takes shuffled logical prefix',cut==full[:5] and cut!=list(range(5)))
 withval=make(Harness({'A':23},1,4,validation={'A':4}),check_initialization=False);vi=identities(withval)['A']
 check('Budget cap follows train validation split',len(vi)==4)
 capped=identities(make(Harness({'A':23},1,2,max_batches={'A':2}),initial=4,check_initialization=False))['A']
 check('Max batches is a second cap after budget slicing',capped==list(range(8)))
 errors={}
 for label,h,initial in [('larger_budget',Harness({'A':23},5,4),4),('zero_target',Harness({'A':23},0,0),4),('missing_initial',Harness({'A':23},max_batches={'A':1}),None),('oversized_max_batches',Harness({'A':7},max_batches={'A':2}),4)]:
  try:make(h,initial)
  except (ValueError,ZeroDivisionError,AssertionError) as e:errors[label]=type(e).__name__
 check('Original method rejects experiment budget greater than target',errors.get('larger_budget')=='ValueError')
 check('Zero experiment and target budgets pass ordering guard then divide by zero',errors.get('zero_target')=='ZeroDivisionError')
 check('Batch cap requires initial batch and cannot exceed sliced inventory',errors.get('missing_initial')=='AssertionError' and errors.get('oversized_max_batches')=='AssertionError')
 initial_errors={}
 for label,h in [('split_and_budget',Harness({'A':23},1,4,validation={'A':4})),('cap_and_budget',Harness({'A':23},1,2,max_batches={'A':2}))]:
  try:make(h)
  except AssertionError as e:initial_errors[label]=str(e)
 check('Original post init forbids split together with simulation budgets','split_and_budget' in initial_errors)
 check('Original post init forbids max batches together with simulation budgets','cap_and_budget' in initial_errors)
 zero_config=Harness({'A':23},0,0);zero_config.__post_init__()
 check('Original post init has no positivity check for two zero budgets',zero_config.experiment_budget==zero_config.target_budget==0)
 decl=json.loads(M.read_text())['config']['data']['value']
 check('Latest declared recipe has both simulation budgets unset',decl['experiment_budget'] is None and decl['target_budget'] is None)
 paths=[P,M,inner.PP,inner.DP,mixture.P,R/'scripts/probe_inner_shuffle_cpu.py',R/'scripts/probe_mixture_identity_cpu.py']
 out={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'runtime':{'python':platform.python_version(),'jax':jax.__version__,'backend':jax.default_backend()},'ratio_example':{'ratio':.25,'artificial_reference_stream':reference,'inventories':si,'stream':stream,'repeat_factor_comparison':distort},'small_zero_example':{'inventories':zi,'original_mix_error':err},'shuffled_cap':{'full':full,'cap':cut},'bypassed_initialization_controls':True,'post_init_rejections':initial_errors,'split_then_cap':vi,'second_cap':capped,'errors':errors,'declared_experiment_budget':None,'declared_target_budget':None,'original_method_lines':[method.lineno,method.end_lineno],'original_post_init_lines':[post.lineno,post.end_lineno],'substitutions':['cache builder returning identity stores','AsyncDataset base and synchronous size bridge','artificial fold_in key stream replacing production key_iterator','null CPU mesh'],'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'actual_Hero_simulation_budget_event':None,'actual_pilot_model_loss':None,'actual_production_repeat_factors':None,'actual_token_store':None}
 (R/'analysis/budget_inventory_cpu.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('Original budget/inventory controls:',len(checks),'passed')
if __name__=='__main__':main()
