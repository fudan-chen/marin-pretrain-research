"""Original historical seed split, mixture builder, child key iterator and mixed-block CPU controls.
Direct finite identity children record keys; no actual cache/shuffle/simulation/restore/GPU/import trace.
"""
import ast,asyncio,csv,dataclasses,hashlib,json,pathlib,sys,types,warnings,copy
import jax,numpy as np
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/historical_key_2026_10_08';H=R/'sources/run_code_2026_10_08';checks=[]
def check(n,v):assert v,n;checks.append({'name':n,'passed':True})
def extract(p,names,env):
 t=ast.parse(p.read_text());nodes=[n for n in ast.walk(t) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names];assert len(nodes)==len(names)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])),str(p),'exec'),env)
 return nodes
# Existing minimal MixtureDataset dependency adapters, with historical uploaded source.
p=R/'scripts/probe_mixture_identity_cpu.py';s=p.read_text().replace('sources/deepening_2026_10_04/mixture_production.py','sources/run_code_2026_10_08/mixture.py');mix=types.ModuleType('historical_key_mix');mix.__file__=str(p);sys.modules[mix.__name__]=mix;exec(compile(s,str(p),'exec'),mix.__dict__)
env={'dataclass':dataclasses.dataclass,'jax':jax,'BIG_INT':2**63-1,'Axis':lambda name,size:(name,size),'MixtureDataset':mix.Mix}
exec(compile('from __future__ import annotations\n'+(H/'schedule.py').read_text(),str(H/'schedule.py'),'exec'),env);extract(H/'mixture.py',['rescale_mixture_schedule_for_batch_schedule'],env);extract(H/'datasets.py',['BlockShuffleConfig','_has_nonzero_weight','build_token_datasets','train_sets'],env);extract(D/'jax_utils.py',['key_iterator'],env)
class Store(mix.Identity):
 def __init__(self,name,trace):super().__init__(name,10**12);self.trace=trace
 def block_shuffle(self,**kw):self.trace[self.name]=np.asarray(kw['key']).tolist();return self
class Direct:
 def __init__(self,name,trace):self.datasets={'train':Store(name,trace)}
env['DirectDatasetComponent']=Direct
class Config:
 _has_nonzero_weight=env['_has_nonzero_weight'];build_token_datasets=env['build_token_datasets'];train_sets=env['train_sets']
 def __init__(self,data):
  self.trace={};self.components={n:Direct(n,self.trace) for n in data['components']};self.train_weights=data['train_weights'];self.shuffle=env['BlockShuffleConfig'](**data['shuffle']);self.permutation_type=data['permutation_type'];self.num_validation_sequences=None;self.experiment_budget=None;self.target_budget=None;self.max_train_batches=None;self.stop_strategy=data['stop_strategy'];self.mixture_block_size=data['mixture_block_size']
 def build_caches(self,split):return {}
def root_nodes(p):
 t=ast.parse(p.read_text());a=next(n for n in ast.walk(t) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Tuple) and [getattr(x,'id','') for x in n.targets[0].elts]==['data_key','model_key']);b=next(n for n in ast.walk(t) if isinstance(n,ast.If) and n.lineno==a.end_lineno+1);return [a,b]
roots={n:root_nodes(D/n) for n in ['train_base.py','train_moe.py','train_ep.py']};builders={n:next(x for x in ast.walk(ast.parse((D/n).read_text())) if isinstance(x,ast.FunctionDef) and x.name=='build_train_dataset') for n in roots}
check('Three uploaded candidate entrypoints use identical selected root statements and mixture builder AST',len({ast.dump(ast.Module(body=v,type_ignores=[]),include_attributes=False) for v in roots.values()})==1 and len({ast.dump(n,include_attributes=False) for n in builders.values()})==1)
extract(D/'train_ep.py',['build_train_dataset'],env)
def root(seed,data_seed):
 ns={'jax':jax,'trainer':types.SimpleNamespace(seed=seed),'config':types.SimpleNamespace(trainer=types.SimpleNamespace(data_seed=data_seed))};exec(compile(ast.fix_missing_locations(ast.Module(body=roots['train_ep.py'],type_ignores=[])),str(D/'train_ep.py'),'exec'),ns);return ns['data_key'],ns['model_key']
def arr(k):return np.asarray(k).tolist()
def load(role):
 c=json.loads(json.loads((H/(role+'_inventory.json')).read_text())['data']['project']['run']['config']);return c,c['data']['value']
cs={r:load(r) for r in ['producer','continuation']};check('Both public declarations select seed0 default data seed MIXTURE and partitionable true',all(c['trainer']['value']['trainer']['seed']==0 and c['trainer']['value']['data_seed'] is None and c['trainer']['value']['training_data_mode']=='MIXTURE' and c['trainer']['value']['trainer']['jax_config']['jax_threefry_partitionable'] is True for c,d in cs.values()))
variants={};keyrows=[]
for flag in [True,False]:
 jax.config.update('jax_threefry_partitionable',flag)
 for seed,ds in [(0,None),(0,0),(1,None),(1,0)]:
  k,mk=root(seed,ds);cfg=Config(cs['continuation'][1]);schedule=env['BatchSchedule'](1024)
  with warnings.catch_warnings():warnings.simplefilter('ignore');m=env['build_train_dataset'](cfg,max_seq_len=4096,batch_schedule=schedule,key=k)
  a=np.asarray(m._get_block(8));prefix=a[:9216];mixkey,shufflekey=jax.random.split(k)
  name=f'partitionable={flag},seed={seed},data_seed={ds}'
  variants[name]={'partitionable':flag,'seed':seed,'data_seed':ds,'root_key':arr(jax.random.PRNGKey(seed)),'data_key':arr(k),'model_key':arr(mk),'mix_key':arr(mixkey),'shuffle_key':arr(shufflekey),'training_cells':len(cfg.trace),'child_keys':cfg.trace,'block8_sha256':hashlib.sha256(a.tobytes()).hexdigest(),'block8_prefix_sha256':hashlib.sha256(prefix.tobytes()).hexdigest(),'block8_prefix_counts':np.bincount(prefix.astype(np.int64)>>16,minlength=200).tolist()}
  keyrows += [{'variant':name,'cell':n,'key0':v[0],'key1':v[1]} for n,v in cfg.trace.items()]
base=variants['partitionable=True,seed=0,data_seed=None'];explicit=variants['partitionable=True,seed=0,data_seed=0'];other=variants['partitionable=True,seed=1,data_seed=0'];othernone=variants['partitionable=True,seed=1,data_seed=None'];off=variants['partitionable=False,seed=0,data_seed=None']
check('Default None and explicit zero preserve model key but alter data/mix/all200child keys',base['model_key']==explicit['model_key'] and base['data_key']!=explicit['data_key'] and base['mix_key']!=explicit['mix_key'] and len(base['child_keys'])==200 and all(base['child_keys'][n]!=explicit['child_keys'][n] for n in base['child_keys']))
check('Explicit same data seed decouples all data keys from trainer seed while model key changes',explicit['data_key']==other['data_key'] and explicit['mix_key']==other['mix_key'] and explicit['child_keys']==other['child_keys'] and explicit['model_key']!=other['model_key'])
check('Default data seed follows trainer seed',base['data_key']!=othernone['data_key'] and base['mix_key']!=othernone['mix_key'])
check('Partitionable flag changes split-derived keys and synthetic mixed-block prefix',base['data_key']!=off['data_key'] and base['mix_key']!=off['mix_key'] and base['block8_prefix_sha256']!=off['block8_prefix_sha256'])
check('Eight declared-key controls retain200support cells and9216prefixslots',len(variants)==8 and len(keyrows)==1600 and all(v['training_cells']==200 and sum(v['block8_prefix_counts'])==9216 for v in variants.values()))
# Replay boundary geometry using the source-derived key, no longer an integer seed passed straight to mixture.
jax.config.update('jax_threefry_partitionable',True);dk,_=root(0,None);built={}
for role,(c,data) in cs.items():
 cfg=Config(data)
 with warnings.catch_warnings():warnings.simplefilter('ignore');built[role]=env['build_train_dataset'](cfg,max_seq_len=4096,batch_schedule=env['BatchSchedule'](1024),key=dk)
pa=np.asarray(built['producer']._get_block(8));ca=np.asarray(built['continuation']._get_block(8));pp=pa[:9216];cp=ca[:9216];rep=np.intersect1d(pp,ca[9216:]);nextrep=np.intersect1d(pp,ca[9216:10240]);boundary={'producer_prefix_sha256':hashlib.sha256(pp.tobytes()).hexdigest(),'continuation_prefix_sha256':hashlib.sha256(cp.tobytes()).hexdigest(),'unmatched_prefix_logical_ids_each_side':9216-len(np.intersect1d(pp,cp)),'hybrid_block_repeated_logical_ids':len(rep),'next_batch_repeated_logical_ids':len(nextrep),'prefix_count_half_l1':int(abs(np.bincount(pp.astype(np.int64)>>16,minlength=200)-np.bincount(cp.astype(np.int64)>>16,minlength=200)).sum()//2)}
check('Two historical declarations with derived key retain200named child keys and nonzero synthetic boundary mismatch',len(built['producer'].dataset_index)==len(built['continuation'].dataset_index)==200 and boundary['unmatched_prefix_logical_ids_each_side']>0 and boundary['hybrid_block_repeated_logical_ids']>0)
# Falsy fallback is a deliberately incorrect counterfactual, not source behavior.
bad=copy.deepcopy(roots['train_ep.py']);bad[1].test=copy.deepcopy(bad[1].test.left);badns={'jax':jax,'trainer':types.SimpleNamespace(seed=1),'config':types.SimpleNamespace(trainer=types.SimpleNamespace(data_seed=0))};exec(compile(ast.fix_missing_locations(ast.Module(body=bad,type_ignores=[])),str(D/'train_ep.py')+'[counterfactual-if-data-seed]','exec'),badns);wrong=badns['data_key'];check('Counterfactual if-data-seed skips explicit zero while original is-not-None preserves it',arr(wrong)==othernone['data_key'] and arr(wrong)!=other['data_key'])
files=[pathlib.Path(__file__),p,H/'mixture.py',H/'datasets.py',H/'schedule.py',H/'producer_inventory.json',H/'continuation_inventory.json']+list(D.iterdir());runtime={'python':sys.version.split()[0],'jax':jax.__version__,'jaxlib':__import__('jaxlib').__version__,'numpy':np.__version__,'backend':jax.default_backend(),'default_prng_impl':str(jax.config.jax_default_prng_impl)}
output={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'runtime':runtime,'variants':variants,'boundary_with_declared_derived_key':boundary,'source_lines':{n:{'root':[roots[n][0].lineno,roots[n][-1].end_lineno],'builder':[builders[n].lineno,builders[n].end_lineno]} for n in roots},'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))},'substitutions':['Direct finite identity children length1e12; actual block shuffle only records assigned keys','simulation inventory slicing disabled after child key allocation','Selected AST statements and builder from three candidate entrypoints, not complete trainer','CPU threefry2x32 and explicitly selected partitionable flags; executed historical environment is not proven'],'actual_runtime_imports_verified':None,'actual_executed_historical_key':None,'actual_checkpoint_contents_verified':None,'actual_token_repetition_count':None,'actual_training_loss_effect':None}
version=jax.__version__.replace('.','_');dest=R/('analysis/historical_key_pipeline_jax_'+version+'.json');dest.write_text(json.dumps(output,indent=2)+'\n')
with (R/('analysis/historical_child_keys_jax_'+version+'.csv')).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(keyrows[0]),lineterminator='\n');w.writeheader();w.writerows(keyrows)
print('Historical key controls',len(checks),runtime);print(json.dumps({'declared_data_key':base['data_key'],'declared_mix_key':base['mix_key'],'boundary':boundary}))
