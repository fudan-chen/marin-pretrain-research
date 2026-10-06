"""Execute original mixture counting/index/remap methods with NumPy 2.0.2 and
synthetic datasets. Audit archived Hero declarations with Python integers, not
actual consumed tokens, runtime NumPy version or real loader/cursor replay.
"""
import ast,asyncio,hashlib,json,pathlib,types,warnings
import numpy as np
R=pathlib.Path(__file__).resolve().parents[1];P=R/'sources/deepening_2026_10_04/mixture_production.py';t=ast.parse(P.read_text());cl=next(x for x in t.body if isinstance(x,ast.ClassDef) and x.name=='MixtureDataset')
names=['_initialize_stage_counts','_compute_expected_counts_per_block','_compute_unpermuted_ids','_get_stage_for_block','_index_into_dataset_for_id','_remap_indices']
body=[x for x in cl.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name in names]
env={'np':np,'warnings':warnings,'StopStrategy':types.SimpleNamespace(RESTART_STRATEGY='restart',ALL_STOP_STRATEGY='all_exhausted',FIRST_STOP_STRATEGY='first_exhausted')}
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='Original',bases=[],keywords=[],body=body,decorator_list=[])],type_ignores=[])),str(P),'exec'),env)
C=env['Original'];K=49152;LIMIT=2**31-1;checks=[]
def ck(n,v):assert v,n;checks.append(n)
def make(starts):
 d=C();d.datasets={'A':None};d.dataset_index=['A'];d.block_size=K;d.weight_stages=[(x,{'A':1.}) for x in starts];d.stop_strategy='restart';d._counts_per_block_per_stage,d._counts_after_stage,d._unpermuted_ids_per_stage=d._initialize_stage_counts();d._stage_start_blocks=np.array([x//K for x in starts],np.int64);return d
first=LIMIT//K+1;start=first*K
with warnings.catch_warnings(record=True) as w:
 warnings.simplefilter('always');one=make([0,start]);base=int(one._counts_after_stage[0][0]);one_index=one._index_into_dataset_for_id(0,first)[1]
ck('First product-overflow boundary is aligned and just exceeds int32',start%K==0 and (first-1)*K<=LIMIT<start)
ck('Original per-stage array product wraps negative',base==start-2**32 and base<0)
ck('Original later-stage index inherits corrupt cumulative base',one_index==base)
ck('Python int cast after multiplication cannot recover original value',int(np.array([K],np.int32).__mul__(first)[0])!=start)
two=make([0,30000*K,60000*K]);ck('Each staged product fits int32 before their sum overflows',30000*K<LIMIT)
ck('Original cumulative sum wraps across individually safe stages',int(two._counts_after_stage[1][0])==60000*K-2**32)
with warnings.catch_warnings(record=True) as w:
 warnings.simplefilter('always');single=make([0]);within=single._index_into_dataset_for_id(0,first)[1];scalar_warnings=[str(x.message) for x in w]
ck('NumPy2 current-stage scalar product wraps before int conversion',within==start-2**32)
batch_index=single._index_into_dataset_for_id(np.int64(0),np.int64(first))[1]
ck('NumPy int64 block id preserves current-stage product in batched-style path',batch_index==start)
ck('Early conversion to Python integers preserves exact current-stage product',first*int(single._counts_per_block_per_stage[0][0])==start)
class Finite:
 def is_finite(self):return True
 async def async_len(self):return 1009
mapped=asyncio.run(one._remap_indices(Finite(),[one_index]))[0]
ck('Negative corrupted index is remapped into valid finite range',0<=mapped<1009)
ck('Restart modulo hides corruption while changing the underlying item',mapped!=start%1009)
# A local hypothetical patch only; do not edit archived source or claim upstream fix.
patched=ast.parse(ast.unparse(cl));pbody=[x for x in patched.body[0].body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name in names]
source=ast.unparse(ast.Module(body=pbody,type_ignores=[])).replace('cumulative_counts = np.zeros(len(self.datasets), dtype=np.int32)','cumulative_counts = np.zeros(len(self.datasets), dtype=np.int64)').replace('stage_total_counts = counts_this_stage * num_blocks_in_stage','stage_total_counts = counts_this_stage.astype(np.int64) * num_blocks_in_stage').replace('offset_in_stage * self._counts_per_block_per_stage[stage][dataset_id]','int(offset_in_stage) * int(self._counts_per_block_per_stage[stage][dataset_id])')
module=ast.parse('class Patched:\n'+''.join('    '+line+'\n' for line in source.splitlines()));exec(compile(module,'synthetic_int64_patch','exec'),env)
original=C;C=env['Patched'];fixed1=make([0,start]);fixed2=make([0,30000*K,60000*K]);fixed0=make([0]);C=original
ck('Hypothetical int64 cumulative patch preserves one-stage base',int(fixed1._counts_after_stage[0][0])==start)
ck('Hypothetical int64 cumulative patch preserves sum across stages',int(fixed2._counts_after_stage[1][0])==60000*K)
ck('Hypothetical early scalar conversion preserves long current-stage index',fixed0._index_into_dataset_for_id(0,first)[1]==start)
ck('Hypothetical patch leaves per-block allocation int32',fixed1._counts_per_block_per_stage[0].dtype==np.int32)
ck('Hypothetical patch promotes only cumulative storage to int64',fixed1._counts_after_stage[0].dtype==np.int64)
ck('Hypothetical patch preserves small-index behavior',all(fixed0._index_into_dataset_for_id(i,7)==single._index_into_dataset_for_id(i,7) for i in [0,1,K-1]))
META=R/'sources/wandb/hero-fa4sm100-nomask-step146k_meta.json';cfg=json.loads(META.read_text())['config'];data=cfg['data']['value'];tr=cfg['trainer']['value']['trainer'];batch=tr['train_batch_size'];stop=tr['num_train_steps'];block=data['mixture_block_size'];stages=data['train_weights'];names=list(dict.fromkeys(n for _,w in stages for n in w if any(q.get(n,0)>0 for _,q in stages)));counts={n:0 for n in names};rows=[]
for i,(step,w) in enumerate(stages):
 total=sum(w.values());weights={n:v/total for n,v in w.items() if v>0};obj=types.SimpleNamespace(datasets={n:None for n in names},dataset_index=names)
 with warnings.catch_warnings(record=True) as stage_warnings:
  warnings.simplefilter('always');per=single._compute_expected_counts_per_block.__func__(obj,weights,block)
 begin=step*batch;end=(stages[i+1][0] if i+1<len(stages) else stop)*batch;num=(end-begin+block-1)//block
 exact={n:int(per[k])*num for k,n in enumerate(names)}
 for n in names:counts[n]+=exact[n]
 row={'stage':i,'positive_weight_zero_count_components':[n for k,n in enumerate(names) if weights.get(n,0)>0 and int(per[k])==0],'warnings':[str(x.message) for x in stage_warnings],'step_start':step,'sequence_start':begin,'sequence_end':end,'start_aligned':begin%block==0,'whole_blocks_including_final_partial':num,'max_stage_component_count_upper_bound':max(exact.values()),'max_cumulative_component_count_upper_bound':max(counts.values()),'largest_cumulative_component':max(counts,key=counts.get)};rows.append(row)
ck('Archived Hero has fixed batch and three aligned starts',type(batch) is int and len(rows)==3 and all(x['start_aligned'] for x in rows))
ck('Archived Hero per-stage component products stay below int32 limit',all(x['max_stage_component_count_upper_bound']<=LIMIT for x in rows))
ck('Archived Hero cumulative component bounds stay below int32 limit',max(counts.values())<=LIMIT)
ck('Hero total exceeds int32 while every component stays below it',stop*batch>LIMIT and max(counts.values())<LIMIT)
ck('Final partial block is conservatively included',stop*batch%block!=0 and rows[-1]['whole_blocks_including_final_partial']*block>=rows[-1]['sequence_end']-rows[-1]['sequence_start'])
result={'scope':__doc__,'numpy_version':np.__version__,'checks_passed':len(checks),'checks':checks,'synthetic':{'block_size':K,'first_overflow_block':first,'expected_index':start,'original_index':one_index,'original_within_stage_index':within,'batched_int64_block_index':batch_index,'scalar_warnings':scalar_warnings,'finite_dataset_length':1009,'corrupt_remapped_index':mapped,'expected_remapped_index':start%1009,'expected_two_stage_sum':60000*K,'original_two_stage_sum':int(two._counts_after_stage[1][0])},'hero_declared_bounds':{'total_sequences':stop*batch,'int32_limit':LIMIT,'components':len(names),'rows':rows,'max_component_upper_bound':max(counts.values()),'component_counts_upper_bound':counts,'actual_dataset_key_order':None},'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [P,META]},'hypothetical_patch_status':'synthetic_only_not_applied_upstream','actual_Hero_runtime_numpy_version':None,'actual_historical_overflow':None,'actual_loader_replay':None}
import difflib
original_source=P.read_text();changed=original_source.replace('cumulative_counts = np.zeros(len(self.datasets), dtype=np.int32)','cumulative_counts = np.zeros(len(self.datasets), dtype=np.int64)').replace('stage_total_counts = counts_this_stage * num_blocks_in_stage','stage_total_counts = counts_this_stage.astype(np.int64) * num_blocks_in_stage').replace('offset_in_stage * self._counts_per_block_per_stage[stage][dataset_id]','int(offset_in_stage) * int(self._counts_per_block_per_stage[stage][dataset_id])')
(R/'analysis/mixture_range_candidate.patch').write_text(''.join(difflib.unified_diff(original_source.splitlines(True),changed.splitlines(True),fromfile='a/lib/levanter/src/levanter/data/mixture.py',tofile='b/lib/levanter/src/levanter/data/mixture.py')))
(R/'analysis/mixture_range_probe.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print('Mixture range checks:',len(checks));print(json.dumps({k:v for k,v in result['hero_declared_bounds'].items() if k not in ['component_counts_upper_bound','rows']},indent=2));print(json.dumps(result['synthetic'],indent=2))
