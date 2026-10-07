"""Original Hero seed statements, dataset construction/shuffle-key allocation and mixture on CPU.
Original Feistel/block shuffle on finite string identity stores. No tokens, checkpoint or model.
"""
import ast,asyncio,collections,dataclasses,hashlib,json,pathlib,types,typing
import jax,numpy as np
import probe_mixture_identity_cpu as mix
import probe_inner_shuffle_cpu as inner
R=mix.ROOT;T=R/'sources/main_incident_2026_10_07/train.py';D=R/'sources/boundaries_2026_10_05/datasets.py';U=R/'sources/tree_restore_2026_10_07/jax_utils.py';S=R/'sources/deepening_2026_10_04/schedule_production.py'
ns=dict(inner.ns,dataclass=dataclasses.dataclass,Sequence=typing.Sequence,BIG_INT=2**63-1)
def extract(p,names,container=None):
 tree=ast.parse(p.read_text());body=tree.body if container is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==container).body
 nodes=[n for n in body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names];assert len(nodes)==len(names)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])),str(p),'exec'),ns)
extract(U,{'key_iterator'});extract(D,{'BlockShuffleConfig'});extract(D,{'_has_nonzero_weight','build_token_datasets','train_sets'},'LmDataConfig')
extract(S,{'ScheduleStep','BatchSegment','BatchSchedule'});extract(mix.P,{'rescale_mixture_schedule_for_batch_schedule'})
ns.update(MixtureDataset=mix.Mix,Axis=lambda name,size:types.SimpleNamespace(name=name,size=size))
extract(T,{'build_train_dataset'})
main=next(n for n in ast.parse(T.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_run_grug_local')
position=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='(data_key, model_key)')
seed_nodes=main.body[position:position+2];assert isinstance(seed_nodes[1],ast.If) and ast.unparse(seed_nodes[1].test)=='config.trainer.data_seed is not None'
seed_code=compile(ast.Module(body=seed_nodes,type_ignores=[]),str(T),'exec')
def keys(seed,data_seed):
 env={'jax':jax,'trainer':types.SimpleNamespace(seed=seed),'config':types.SimpleNamespace(trainer=types.SimpleNamespace(data_seed=data_seed))};exec(seed_code,env);return env['data_key'],env['model_key']
def keyvalue(k):return np.array(k,copy=True).tolist()
class DirectAdapter:
 def __init__(self,datasets):self.datasets=datasets
ns['DirectDatasetComponent']=DirectAdapter
class LabelIdentity(inner.Identity):
 def __init__(self,name,trace):super().__init__(48);self.name=name;self.trace=trace
 async def get_batch(self,indices):return [self.name+':'+str(i) for i in indices]
 async def getitem_async(self,i):return self.name+':'+str(i)
 def block_shuffle(self,**kwargs):
  self.trace[self.name]=keyvalue(kwargs['key']);return inner.ns['BlockShufflingDataset'](self,kwargs['io_block_size'],window_blocks=kwargs['window_blocks'],key=kwargs['key'],perm_type=kwargs['perm_type'])
class Config:
 _has_nonzero_weight=ns['_has_nonzero_weight'];build_token_datasets=ns['build_token_datasets'];train_sets=ns['train_sets']
 def __init__(self,order,weights):
  self.trace={};self.components={n:DirectAdapter({'train':LabelIdentity(n,self.trace)}) for n in order};self.train_weights=weights
  self.shuffle=ns['BlockShuffleConfig'](io_block_size=4,window_blocks=3);self.permutation_type='feistel';self.num_validation_sequences=None;self.experiment_budget=None;self.target_budget=None;self.max_train_batches=None;self.stop_strategy='restart';self.mixture_block_size=8
 def build_caches(self,split):return {}
def build(seed=0,data_seed=None,order=('A','B'),weights=None):
 dk,mk=keys(seed,data_seed);cfg=Config(order,weights or {'A':.5,'B':.5});ds=ns['build_train_dataset'](cfg,max_seq_len=4,batch_schedule=ns['BatchSchedule'](4),key=dk)
 return cfg,ds,{'trainer_seed':seed,'data_seed':data_seed,'data_key':keyvalue(dk),'model_key':keyvalue(mk),'mix_key':keyvalue(ds.key),'shuffle_root_key':keyvalue(jax.random.split(dk)[1]),'child_keys':cfg.trace,'dataset_index':ds.dataset_index}
checks=[];cases={}
def check(n,c):assert c,n;checks.append({'name':n,'passed':True})
async def run():
 for name,args in {'default':{},'repeat':{},'trainer_seed1':{'seed':1},'explicit_data0':{'data_seed':0},'trainer1_data0':{'seed':1,'data_seed':0},'reordered':{'order':('B','A')},'zero_front':{'order':('X','A','B'),'weights':[(0,{'X':0,'A':.5,'B':.5}),(24,{'X':0,'A':.5,'B':.5})]},'future_front':{'order':('X','A','B'),'weights':[(0,{'X':0,'A':.5,'B':.5}),(24,{'X':.5,'A':.25,'B':.25})]},'future_tail':{'order':('A','B','X'),'weights':[(0,{'X':0,'A':.5,'B':.5}),(24,{'X':.5,'A':.25,'B':.25})]}}.items():
  cfg,ds,row=build(**args);row['declared_component_order']=list(cfg.components);row['first96']=await ds.get_batch(list(range(96)));row['resume_step6_batch']=await ds.get_batch(list(ns['BatchSchedule'](4).batch_indices_at_step(6)));row['child_first16']={n:await child.get_batch(list(range(16))) for n,child in ds.datasets.items()};row['sequence_stage_starts']=[a for a,_ in ds.weight_stages];cases[name]=row
 b=cases['default'];e=cases['explicit_data0'];f=cases['future_front'];z=cases['zero_front'];tail=cases['future_tail']
 check('Same source configuration and completed step reproduce identities',b['first96']==cases['repeat']['first96'] and b['resume_step6_batch']==cases['repeat']['resume_step6_batch'])
 check('Trainer seed changes both model key and default data stream',b['model_key']!=cases['trainer_seed1']['model_key'] and b['first96']!=cases['trainer_seed1']['first96'])
 check('Explicit data seed pins stream while model seed changes',e['first96']==cases['trainer1_data0']['first96'] and e['model_key']!=cases['trainer1_data0']['model_key'])
 check('Same integer zero is not same default versus explicit data key',b['data_key']!=e['data_key'] and b['first96']!=e['first96'] and b['resume_step6_batch']!=e['resume_step6_batch'])
 check('All seed variants preserve full synthetic A B inventories',all(collections.Counter(cases[n]['first96'])==collections.Counter(b['first96']) for n in ['trainer_seed1','explicit_data0','trainer1_data0','reordered','future_front','future_tail']))
 check('Child construction order reassigns keys by position',b['child_keys']['A']==cases['reordered']['child_keys']['B'] and b['child_keys']['B']==cases['reordered']['child_keys']['A'] and b['child_first16']['A']!=cases['reordered']['child_first16']['A'])
 check('Always zero front component filtered before key allocation',z['declared_component_order']==f['declared_component_order'] and z['sequence_stage_starts']==f['sequence_stage_starts'] and z['child_keys']==b['child_keys'] and z['dataset_index']==b['dataset_index'] and z['first96']==b['first96'])
 check('Future positive front component is constructed before activation',f['sequence_stage_starts']==[0,96] and f['dataset_index']==['X','A','B'] and all(not x.startswith('X:') for x in f['first96']) and 'X' in f['child_keys'])
 check('Future front component changes existing child mapping before activation',f['mix_key']==b['mix_key'] and f['child_keys']['A']!=b['child_keys']['A'] and f['first96']!=b['first96'] and f['resume_step6_batch']!=b['resume_step6_batch'])
 check('Current named-domain slot sequence stays fixed while future-front content changes',[x.split(':')[0] for x in f['first96']]==[x.split(':')[0] for x in b['first96']])
 check('Appending future component preserves existing key allocation and prefix',tail['child_keys']['A']==b['child_keys']['A'] and tail['child_keys']['B']==b['child_keys']['B'] and tail['first96']==b['first96'])
 meta=json.loads((R/'sources/live_2026_10_07/meta.json').read_text());cfg=meta['config'];tr=cfg['trainer']['value'];declared={'run':meta['name'],'trainer_seed':tr['trainer']['seed'],'data_seed':tr['data_seed'],'training_data_mode':tr['training_data_mode'],'shuffle':cfg['data']['value']['shuffle'],'snapshot':'sources/live_2026_10_07/meta.json','actual_executed_key':None}
 check('Archived run declares default data seed not explicit zero',declared['trainer_seed']==0 and declared['data_seed'] is None and declared['training_data_mode']=='MIXTURE')
 for name,row in cases.items():row['different_slots_vs_default']=sum(a!=c for a,c in zip(row['first96'],b['first96']));row['full_inventory_matches_default']=collections.Counter(row['first96'])==collections.Counter(b['first96'])
 paths=[pathlib.Path(__file__),T,D,U,S,mix.P,inner.PP,inner.DP,inner.TP,R/'scripts/probe_mixture_identity_cpu.py',R/'scripts/probe_inner_shuffle_cpu.py',R/'sources/live_2026_10_07/meta.json',R/'analysis/seed_pipeline_source_binding.json']
 out={'checks_passed':len(checks),'checks':checks,'cases':cases,'declared_run':declared,'scope':__doc__,'runtime':{'jax':jax.__version__,'jaxlib':__import__('jaxlib').__version__,'numpy':np.__version__,'backend':jax.default_backend(),'default_prng_impl':str(jax.config.jax_default_prng_impl)},'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'adapters':['AST extraction rather than full package import','Axis name/size adapter','DirectDatasetComponent type adapter; original support filter/build_token_datasets/train_sets executed','build_caches empty; finite named identity store, no real cache or tokenizer','single CPU and null CPU mesh','small block size 8, child IO block4/window3/inventory48, constant batch4','original seed statements executed; no original model or complete train loop'],'actual_Hero_data_discontinuity':None,'actual_Hero_next_batch_identity':None,'actual_token_store':None,'actual_causal_mixture_effect':None}
 (R/'analysis/seed_pipeline_cpu.json').write_text(json.dumps(out,indent=2)+'\n');print('Seed pipeline CPU:',len(checks),'passed; prefix differences:',{n:r['different_slots_vs_default'] for n,r in cases.items()})
asyncio.run(run())
