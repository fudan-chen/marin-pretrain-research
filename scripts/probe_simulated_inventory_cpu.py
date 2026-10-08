"""Historical original train_sets -> block shuffle -> simulated slice -> mixed restart on CPU.
Finite named identity children; no real token cache, packing, production checkpoint or model.
"""
import ast,asyncio,base64,collections,dataclasses,hashlib,json,math,pathlib,sys,types,typing,warnings
import jax,numpy as np
R=pathlib.Path(__file__).resolve().parents[1];H=R/'sources/run_code_2026_10_08';D=R/'sources/simulated_inventory_2026_10_08';T=R/'sources/historical_key_2026_10_08/train_ep.py';checks=[]
def check(n,v):assert v,n;checks.append({'name':n,'passed':True})
def adapter(script,name,replacements):
 p=R/'scripts'/script;s=p.read_text()
 for a,b in replacements.items():assert a in s;s=s.replace(a,b)
 m=types.ModuleType(name);m.__file__=str(p);sys.modules[name]=m;exec(compile(s,str(p),'exec'),m.__dict__);return m
inner=adapter('probe_inner_shuffle_cpu.py','sim_inner',{'sources/prp_2026_10_07/_prp.py':'sources/simulated_inventory_2026_10_08/_prp.py','sources/deepening_2026_10_04/dataset_production.py':'sources/simulated_inventory_2026_10_08/dataset.py','sources/deepening_2026_10_04/datasets_production.py':'sources/run_code_2026_10_08/datasets.py'})
mix=adapter('probe_mixture_identity_cpu.py','sim_mix',{'sources/deepening_2026_10_04/mixture_production.py':'sources/run_code_2026_10_08/mixture.py'})
class SyncLength:
 def __init__(self,ds):self.ds=ds
 def __len__(self):return asyncio.run(self.ds.async_len())
inner.Base.as_sync_dataset=lambda self:SyncLength(self)
env=dict(inner.ns,dataclass=dataclasses.dataclass,Sequence=typing.Sequence,BIG_INT=2**63-1,Axis=lambda name,size:types.SimpleNamespace(name=name,size=size),MixtureDataset=mix.Mix)
def extract(p,names):
 nodes=[n for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names];assert len(nodes)==len(names)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])),str(p),'exec'),env)
extract(H/'datasets.py',['BlockShuffleConfig','_has_nonzero_weight','build_token_datasets','train_sets']);extract(H/'schedule.py',['ScheduleStep','BatchSegment','BatchSchedule']);extract(H/'mixture.py',['rescale_mixture_schedule_for_batch_schedule']);extract(R/'sources/historical_key_2026_10_08/jax_utils.py',['key_iterator']);extract(T,['build_train_dataset'])
class Identity(inner.Identity):
 def __init__(self,name,length,trace):super().__init__(length);self.name=name;self.trace=trace
 async def get_batch(self,indices):return [self.name+':'+str(i) for i in indices]
 async def getitem_async(self,i):return self.name+':'+str(i)
 def block_shuffle(self,**kw):self.trace[self.name]=np.asarray(kw['key']).tolist();return inner.ns['BlockShufflingDataset'](self,**kw)
class Direct:
 def __init__(self,store):self.datasets={'train':store}
env['DirectDatasetComponent']=Direct
raw=json.loads((H/'continuation_inventory.json').read_text())['data']['project']['run'];cfg=json.loads(raw['config']);data=cfg['data']['value'];ratio=data['experiment_budget']/data['target_budget'];names=['c00q0','c00q1']
class Config:
 _has_nonzero_weight=env['_has_nonzero_weight'];build_token_datasets=env['build_token_datasets'];train_sets=env['train_sets']
 def __init__(self,length,ratio,weights=None,shuffle=None):
  self.trace={};self.components={n:Direct(Identity(n,length,self.trace)) for n in names};self.train_weights=weights or dict.fromkeys(names,.5);self.shuffle=env['BlockShuffleConfig'](**(shuffle or {'io_block_size':4,'window_blocks':3,'perm_type':'feistel'}));self.permutation_type='feistel';self.num_validation_sequences=None;self.max_train_batches=None;self.experiment_budget=None if ratio is None else ratio;self.target_budget=None if ratio is None else 1;self.stop_strategy='restart';self.mixture_block_size=8
 def build_caches(self,split):return {}
# Execute the exact uploaded root-key statements.
tree=ast.parse(T.read_text());assign=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Tuple) and [getattr(x,'id','') for x in n.targets[0].elts]==['data_key','model_key']);override=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and n.lineno==assign.end_lineno+1)
def key(ds):
 e={'jax':jax,'trainer':types.SimpleNamespace(seed=0),'config':types.SimpleNamespace(trainer=types.SimpleNamespace(data_seed=ds))};exec(compile(ast.fix_missing_locations(ast.Module(body=[assign,override],type_ignores=[])),str(T),'exec'),e);return e['data_key']
def build(length,simratio,ds=None,weights=None,shuffle=None):
 c=Config(length,simratio,weights,shuffle)
 with warnings.catch_warnings():warnings.simplefilter('ignore');m=env['build_train_dataset'](c,max_seq_len=4096,batch_schedule=env['BatchSchedule'](4),key=key(ds))
 return c,m
def read_pools(m):return {n:asyncio.run(inner.all_ids(child)) for n,child in m.datasets.items()}
jax.config.update('jax_threefry_partitionable',True);cases={};objects={}
for name,length,r,ds,sh in [('full_default',48,None,None,None),('full_zero',48,None,0,None),('half_default',48,.5,None,None),('half_zero',48,.5,0,None),('realratio_default',1000000,ratio,None,data['shuffle']),('realratio_zero',1000000,ratio,0,data['shuffle'])]:
 c,m=build(length,r,ds,shuffle=sh);pools=read_pools(m);objects[name]=(c,m);cases[name]={'full_identity_inventory_length':length,'simulation_ratio':r,'selected_lengths':{n:len(v) for n,v in pools.items()},'pools':pools,'pool_sha256':{n:hashlib.sha256('\n'.join(v).encode()).hexdigest() for n,v in pools.items()},'child_keys':c.trace}
check('Full shuffled inventory unchanged by None-zero key change despite order difference',all(set(cases['full_default']['pools'][n])==set(cases['full_zero']['pools'][n]) and cases['full_default']['pools'][n]!=cases['full_zero']['pools'][n] for n in names))
check('Original half slice keeps24unique identities per child and seed changes selected membership',all(len(set(cases[c]['pools'][n]))==24 for c in ['half_default','half_zero'] for n in names) and all(set(cases['half_default']['pools'][n])!=set(cases['half_zero']['pools'][n]) for n in names))
check('Public simulation budget ratio and real shuffle settings yield879identities from synthetic million',data['experiment_budget']==16483614720 and data['target_budget']==18750000000000 and int(1000000*ratio)==879 and all(cases[c]['selected_lengths']==dict.fromkeys(names,879) for c in ['realratio_default','realratio_zero']))
check('Real-setting synthetic selected pools remain unique and differ across data key variants',all(len(set(cases[c]['pools'][n]))==879 for c in ['realratio_default','realratio_zero'] for n in names) and all(set(cases['realratio_default']['pools'][n])!=set(cases['realratio_zero']['pools'][n]) for n in names))
# Change weights only while keeping the sampling pool seed fixed.
cnew,mnew=build(48,.5,None,{names[0]:.25,names[1]:.75});newp=read_pools(mnew);base=objects['half_default'][1];streams={};histograms={}
for name,m in [('equal',base),('redistributed',mnew)]:
 ids=asyncio.run(m.get_batch(list(range(512))));streams[name]=ids;histograms[name]={n:dict(collections.Counter(x for x in ids if x.startswith(n+':'))) for n in names}
check('Weights-only intervention preserves sampled child pools and keys',newp==cases['half_default']['pools'] and cnew.trace==cases['half_default']['child_keys'])
check('Original restart repeats selected24-member pools rather than excluded identities',all(set(histograms[g][n])==set(newp[n]) and min(histograms[g][n].values())>1 for g in histograms for n in names) and [sum(histograms['equal'][n].values()) for n in names]==[256,256] and [sum(histograms['redistributed'][n].values()) for n in names]==[128,384])
threshold=math.ceil(1/ratio);zero_c,zero_m=build(threshold-1,ratio,shuffle=data['shuffle']);one_c,one_m=build(threshold,ratio,shuffle=data['shuffle']);zero_len={n:asyncio.run(x.async_len()) for n,x in zero_m.datasets.items()};one_len={n:asyncio.run(x.async_len()) for n,x in one_m.datasets.items()};error=None
try:asyncio.run(zero_m.get_batch([0]))
except ValueError as exc:error=str(exc)
check('Floor threshold1138 permits zero-length positive-weight child below threshold and original read rejects it',threshold==1138 and zero_len==dict.fromkeys(names,0) and one_len==dict.fromkeys(names,1) and error is not None and 'empty finite dataset' in error)
# Source-only state/schema and execution order; no checkpoint contents inspected.
state=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GrugTrainState');fields=[n.target.id for n in state.body if isinstance(n,ast.AnnAssign)];text=T.read_text();check('Historical state schema stores six model-update fields but no explicit dataset recipe/key/cursor fields',fields==['step','params','master_params','opt_state','ema_params','pending_qb_betas'] and text.index('data_key, model_key =')<text.index('state = restore_grug_state_from_checkpoint(')<text.index('train_dataset = build_train_dataset(')<text.index('batch_source = train_loader.iter_from_step(int(state.step))'))
bindings=[]
for filename in ['dataset.py','_prp.py']:
 p=D/filename;k='lib/levanter/src/levanter/data/'+filename;md5=base64.b64encode(hashlib.md5(p.read_bytes()).digest()).decode();bindings.append({'file':str(p.relative_to(R)),'matches_both_run_manifests':all(json.loads((H/(r+'_code_manifest.json')).read_text())['contents'][k]['digest']==md5 for r in ['producer','continuation'])})
check('Two original shuffle/slice source files bound to both independent run manifests',all(x['matches_both_run_manifests'] for x in bindings))
contrasts={n:{cell:{'intersection':len(set(cases[n+'_default']['pools'][cell])&set(cases[n+'_zero']['pools'][cell])),'symmetric_difference':len(set(cases[n+'_default']['pools'][cell])^set(cases[n+'_zero']['pools'][cell]))} for cell in names} for n in ['full','half','realratio']}
files=[pathlib.Path(__file__),R/'scripts/probe_inner_shuffle_cpu.py',R/'scripts/probe_mixture_identity_cpu.py',T,H/'datasets.py',H/'mixture.py',H/'schedule.py',H/'continuation_inventory.json',H/'producer_code_manifest.json',H/'continuation_code_manifest.json',R/'sources/historical_key_2026_10_08/jax_utils.py']+list(D.iterdir())
out={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'runtime':{'jax':jax.__version__,'jaxlib':__import__('jaxlib').__version__,'numpy':np.__version__,'backend':jax.default_backend()},'declared_simulation_ratio':ratio,'declared_simulation_budgets':{'experiment_budget':data['experiment_budget'],'target_budget':data['target_budget']},'cases':cases,'pool_contrasts':contrasts,'weights_only_pools_preserved':True,'mixed_restart_histograms':histograms,'empty_inventory_control':{'first_nonempty_full_sequences':threshold,'below_threshold_lengths':zero_len,'at_threshold_lengths':one_len,'original_error':error},'state_schema_source_only':{'fields':fields,'file':str(T.relative_to(R)),'class_lines':[state.lineno,state.end_lineno],'actual_checkpoint_contents_verified':None},'bindings':bindings,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))},'substitutions':['Two synthetic named finite identity stores; not actual200cell inventory or cache','Actual recorded budget ratio and shuffle settings in million-identity control; small half/full controls use IO4/window3','Original train_sets, actual PRP/block shuffle, slicing, mixture builder and restart; AsyncDataset base, local CPU mesh, Axis and Direct component adapters','Sync length proxy calls original async_len outside event loop; no actual store sync wrapper','Original selected root key statements; actual entrypoint/imports and executed historical keys unknown'],'actual_training_pool_membership_verified':None,'actual_production_zero_length_cell':None,'actual_checkpoint_contents_verified':None,'actual_token_repetition_count':None,'actual_training_loss_effect':None}
(R/'analysis/simulated_inventory_cpu.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('Simulated inventory controls',len(checks));print(json.dumps(contrasts));print('ratio',ratio,'threshold',threshold,'error',error)
