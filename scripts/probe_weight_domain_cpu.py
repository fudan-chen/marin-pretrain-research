"""Execute uploaded full MixtureDataset with finite identity stores and malformed numeric weights.
Original class dependency adapters from assertion probe; no tokens, model, actual incident or loss.
"""
import asyncio, hashlib, json, math, pathlib, platform, runpy, types, warnings
import jax, numpy as np
R=pathlib.Path(__file__).resolve().parents[1]
AD=R/'scripts/probe_assertion_contracts_cpu.py'
a=runpy.run_path(str(AD));M=a['M'];D=a['D'];checks=[]
def check(name,condition):
 if not condition:raise RuntimeError(name)
 checks.append({'name':name,'passed':True})
def make(guard=False):
 ns=a['environment'](0);Mix=ns['MixtureDataset'];original=Mix._normalize_weights
 if guard:
  def normalize(weights):
   if not all(math.isfinite(v) for v in weights.values()):raise ValueError('All weight values must be finite')
   total=sum(weights.values())
   if not math.isfinite(total) or total<=0:raise ValueError('Weight sum must be positive and finite')
   return original(weights)
  Mix._normalize_weights=staticmethod(normalize)
 return Mix
async def attempt(weights,order=('A','B'),guard=False):
 Mix=make(guard)
 try:
  with warnings.catch_warnings(record=True) as caught:
   m=Mix({n:a['Identity'](n) for n in order},weights,8,key=7,randomize_blocks=False)
   stream=await m.get_batch(list(range(16)))
  return {'accepted':True,'normalized_hex':[{n:float(v).hex() for n,v in w.items()} for _,w in m.weight_stages],'dataset_index':m.dataset_index,'quotas':[{n:int(v) for n,v in zip(m.dataset_index,q)} for q in m._counts_per_block_per_stage],'stream':stream,'warnings':[str(x.message) for x in caught]}
 except Exception as e:return {'accepted':False,'error_type':type(e).__name__,'error':str(e)}
async def main():
 case_weights={
  'valid_half':{'A':1.,'B':1.},
  'negative':{'A':1.,'B':-.1},
  'all_zero':{'A':0.,'B':0.},
  'nan_only_static':{'B':float('nan')},
  'nan_only_later':[(0,{'A':1.,'B':1.}),(8,{'B':float('nan')})],
  'inf_static':{'A':1.,'B':float('inf')},
  'inf_later':[(0,{'A':1.,'B':1.}),(8,{'A':1.,'B':float('inf')})],
  'finite_overflow_static':{'A':1e308,'B':1e308},
  'finite_overflow_later':[(0,{'A':1.,'B':1.}),(8,{'A':1e308,'B':1e308})],
  'tiny_positive':{'A':1e-320,'B':1.},
  'positive_ratio_underflow':{'A':1e-300,'B':1e300},
 }
 cases={n:await attempt(w) for n,w in case_weights.items()}
 config_acceptance={}
 for n,w in case_weights.items():
  cfg=types.SimpleNamespace(components={'A':object(),'B':object()},train_weights=w,max_train_batches=None,num_validation_sequences=None,experiment_budget=None,target_budget=None)
  try:a['environment'](0)['__post_init__'](cfg);config_acceptance[n]={'accepted':True}
  except Exception as e:config_acceptance[n]={'accepted':False,'error_type':type(e).__name__}
 reverse={n:await attempt(case_weights[n],order=('B','A')) for n in ['nan_only_later','finite_overflow_later']}
 fixed={n:await attempt(w,guard=True) for n,w in case_weights.items()}
 check('Actual CPU backend and normal compile level',jax.default_backend()=='cpu')
 check('Negative and zero-sum already rejected explicitly',all(not cases[n]['accepted'] and cases[n]['error_type']=='ValueError' for n in ['negative','all_zero']))
 check('NaN-only later stage silently normalizes to empty and assigns all eight to first A',cases['nan_only_later']['accepted'] and cases['nan_only_later']['normalized_hex'][1]=={} and cases['nan_only_later']['quotas'][1]=={'A':8,'B':0})
 check('Finite overflow later stage silently normalizes both weights to zero then assigns all to A',cases['finite_overflow_later']['accepted'] and cases['finite_overflow_later']['normalized_hex'][1]=={'A':0.0.hex(),'B':0.0.hex()} and cases['finite_overflow_later']['quotas'][1]=={'A':8,'B':0})
 check('First-child fallback follows reversed actual dataset order',all(reverse[n]['accepted'] and reverse[n]['quotas'][1]=={'B':8,'A':0} for n in reverse))
 check('Static forms fail later rather than share staged silent behavior',all(not cases[n]['accepted'] and cases[n]['error_type']=='ValueError' for n in ['nan_only_static','finite_overflow_static','inf_static','inf_later']))
 check('Tiny representable positive weight retained but receives zero integer quota',cases['tiny_positive']['dataset_index']==['A','B'] and cases['tiny_positive']['quotas'][0]=={'A':0,'B':8} and bool(cases['tiny_positive']['warnings']))
 check('Positive finite ratio underflows to zero and filters A from mixture dataset index',cases['positive_ratio_underflow']['accepted'] and cases['positive_ratio_underflow']['normalized_hex'][0]['A']==0.0.hex() and cases['positive_ratio_underflow']['dataset_index']==['B'])
 invalid=['negative','all_zero','nan_only_static','nan_only_later','inf_static','inf_later','finite_overflow_static','finite_overflow_later']
 check('Original config post_init does not preempt numeric malformed weight cases',all(config_acceptance[n]['accepted'] for n in case_weights))
 check('Counterfactual finite-total guard rejects all eight malformed inputs',all(not fixed[n]['accepted'] and fixed[n]['error_type']=='ValueError' for n in invalid))
 check('Minimal guard preserves selected valid and tiny-input trajectories without claiming underflow fix',all(fixed[n]==cases[n] for n in ['valid_half','tiny_positive','positive_ratio_underflow']))
 bindings=[M,D,AD]
 real=[]
 for role in ['producer','continuation']:
  path=R/('sources/run_code_2026_10_08/'+role+'_inventory.json');bindings.append(path)
  data=json.loads(json.loads(path.read_text())['data']['project']['run']['config'])['data']['value']
  stages=data['train_weights']; aligned=[(i*49152,w) for i,(_,w) in enumerate(stages)]
  names=list(data['components'])
  with warnings.catch_warnings():
   warnings.simplefilter('ignore')
   old=make(False)({n:a['Identity'](n) for n in names},aligned,49152,key=7,randomize_blocks=False)
   new=make(True)({n:a['Identity'](n) for n in names},aligned,49152,key=7,randomize_blocks=False)
  for i,(start,weights) in enumerate(stages):
   positives=[float(v) for v in weights.values() if v>0]
   real.append({'role':role,'declared_start':start,'diagnostic_sequence_begin':i*49152,'all_raw_finite_nonnegative':all(math.isfinite(v) and v>=0 for v in weights.values()),'raw_sum_hex':sum(weights.values()).hex(),'positive_min':min(positives),'positive_max':max(positives),'positive_weight_components_this_phase':len(positives),'normalized_equal':old.weight_stages[i]==new.weight_stages[i],'integer_quota_equal':bool(np.array_equal(old._counts_per_block_per_stage[i],new._counts_per_block_per_stage[i])),'retained_components_full_history':len(old.dataset_index)})
 check('Both archived recipes five stages finite nonnegative and no sum overflow',len(real)==5 and all(x['all_raw_finite_nonnegative'] and math.isfinite(float.fromhex(x['raw_sum_hex'])) for x in real))
 check('Minimal guard preserves all five archived normalized vectors and integer quotas',all(x['normalized_equal'] and x['integer_quota_equal'] and x['retained_components_full_history']==200 for x in real))
 out={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'runtime':{'python':platform.python_version(),'jax':jax.__version__,'numpy':np.__version__,'backend':jax.default_backend()},'cases':cases,'original_config_post_init':config_acceptance,'reversed_cases':reverse,'counterfactual_finite_guard_cases':fixed,'archived_recipe_geometry':'All declared phase weight dictionaries retained; phase starts remapped to synthetic aligned one-block boundaries for normalization/quota comparison, not historical exposure replay','archived_recipe_controls':real,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in bindings},'probe_script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'substitutions':['Original full historical class with assertion probe dependency adapters','Strict finite artificial identities length100, no actual200cell cache','Block size8 and randomize_blocks=False for diagnostic identities; real recipe integer vectors use49152','Counterfactual wrapper adds finite values and positive finite same-order sum guard, then calls unchanged original normalizer'], 'actual_Hero_nonfinite_weights':None,'actual_Hero_weight_sum_overflow':None,'actual_training_loss_effect':None,'actual_GPU_execution':None,'counterfactual_guard_deployed':False}
 (R/'analysis/weight_domain_cpu.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 print('Weight domain original source and candidate guard controls:',len(checks),'passed')
 print('Silent later-stage fallback quotas:',cases['nan_only_later']['quotas'],cases['finite_overflow_later']['quotas'])
if __name__=='__main__':asyncio.run(main())
