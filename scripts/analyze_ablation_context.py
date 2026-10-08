"""Observed baseline-conditional ablation responses and conditional original-source key allocation.
No new model training; supplied root key is not recovered historical launch key.
"""
import ast,csv,dataclasses,hashlib,json,pathlib,types,typing,collections,math,importlib.metadata,copy
import jax,numpy as np
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/deepening_2026_10_04/datasets.py';U=R/'sources/tree_restore_2026_10_07/jax_utils.py';checks=[]
def check(n,v):assert v,n;checks.append({'name':n,'passed':True})
env={'dataclass':dataclasses.dataclass,'jax':jax,'Sequence':typing.Sequence,'BIG_INT':2**63-1}
def extract(p,names,container=None):
 t=ast.parse(p.read_text());body=t.body if container is None else next(x for x in t.body if isinstance(x,ast.ClassDef) and x.name==container).body
 nodes=[x for x in body if isinstance(x,(ast.ClassDef,ast.FunctionDef)) and x.name in names];assert len(nodes)==len(names)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])),str(p),'exec'),env)
extract(D,['BlockShuffleConfig']);extract(D,['_has_nonzero_weight','build_token_datasets','train_sets','tagged_eval_sets'],'LmDataConfig');extract(U,['key_iterator'])
class Direct:
 def __init__(self,name,tags,trace):self.datasets={'train':Store(name,trace)};self.tags=tags
class Store:
 def __init__(self,name,trace):self.name=name;self.trace=trace
 def block_shuffle(self,**kw):self.trace[self.name]=np.array(kw['key']).tolist();return self
env['DirectDatasetComponent']=Direct
class Config:
 _has_nonzero_weight=env['_has_nonzero_weight'];build_token_datasets=env['build_token_datasets'];train_sets=env['train_sets'];tagged_eval_sets=env['tagged_eval_sets']
 def __init__(self,data):
  self.trace={};self.components={n:Direct(n,c['tags'],self.trace) for n,c in data['components'].items()};self.train_weights=data['train_weights'];self.shuffle=env['BlockShuffleConfig'](**data['shuffle']);self.permutation_type=data['permutation_type'];self.num_validation_sequences=None;self.experiment_budget=None;self.target_budget=None;self.max_train_batches=None
 def build_caches(self,split):return {}
 def validation_sets(self,Pos):return {n:object() for n in self.components if not n.startswith('c')}
files={D,U,pathlib.Path(__file__),R/'analysis/domain_ablation_audit.csv',R/'analysis/ablation_task_endpoints.csv'}
def load(name):
 p=R/'sources/practical_2026_10_04'/('config_'+name+'.json');files.add(p);c=json.loads(json.loads(p.read_text())['data']['project']['run']['config']);return c,c['data']['value']
audit=list(csv.DictReader((R/'analysis/domain_ablation_audit.csv').open()));groups=['proportional_domain_ablation','semantic_domain_ablation'];refs={g:next(r['control_run'] for r in audit if r['group']==g) for g in groups};baselines={};base_data={};root=jax.random.PRNGKey(0)
for g,name in refs.items():
 full,data=load(name);cfg=Config(data);ds=cfg.train_sets(None,key=root);tags=cfg.tagged_eval_sets(None);base_data[g]=data
 trim_data=copy.deepcopy(data);trim_data['train_weights']=trim_data['train_weights'][1:];bc=Config(trim_data);bd=bc.train_sets(None,key=root)
 baselines[g]={'trimmed_support':len(bd),'trimmed_removed_cells':[n for n in ds if n not in bd],'trimmed_changed_child_keys':sum(bc.trace[n]!=cfg.trace[n] for n in bd),'run':name,'train_support':list(ds),'child_keys':cfg.trace,'validation_component_tags':{n:t for n,t in zip([n for n in cfg.components if not n.startswith('c')],[t for _,t in tags])},'trainer_seed':full['trainer']['value']['trainer']['seed'],'data_seed':full['trainer']['value']['data_seed']}
key_rows=[];contexts=[];seed_ok=[];components_ok=[]
for row in audit:
 g=row['group'];domain=int(row['domain']);full,data=load(row['run_name']);cfg=Config(data);ds=cfg.train_sets(None,key=root);trim_data=copy.deepcopy(data);trim_data['train_weights']=trim_data['train_weights'][1:];trim=Config(trim_data);trim_ds=trim.train_sets(None,key=root);b=baselines[g];removed=[n for n in b['train_support'] if n not in ds];changed=[n for n in ds if cfg.trace[n]!=b['child_keys'][n]]
 contexts.append({'group':g,'domain':domain,'run':row['run_name'],'control_run':refs[g],'removed_cells':removed,'remaining_cells':len(ds),'remaining_changed_child_keys':len(changed),'changed_cells':changed,'deleted_cells_positive_in_first_historical_stage':[n for n in ds if n.startswith('c%02dq'%domain) and data['train_weights'][0][1].get(n,0)>0],'deleted_cells_zero_in_all_later_stages':all(w.get('c%02dq%d'%(domain,q),0)==0 for _,w in data['train_weights'][1:] for q in range(5)),'trimmed_support':len(trim_ds),'trimmed_changed_child_keys':sum(trim.trace[n]!=b['child_keys'][n] for n in trim_ds),'trimmed_index_shift_count':sum(list(ds).index(n)!=list(trim_ds).index(n) for n in trim_ds),'trimmed_removed_cells':[n for n in ds if n not in trim_ds],'recorded_stage_starts':[start for start,_ in data['train_weights']],'component_count':len(data['components']),'validation_components_zero_all_phases':all(w.get(n,0)==0 for _,w in data['train_weights'] for n in cfg.components if not n.startswith('c'))})
 seed_ok.append(full['trainer']['value']['trainer']['seed']==0 and full['trainer']['value']['data_seed'] is None);components_ok.append(data['components']==base_data[g]['components'])
 for n in ds:key_rows.append({'group':g,'deleted_domain':domain,'cell':n,'baseline_key':','.join(map(str,b['child_keys'][n])),'ablation_key':','.join(map(str,cfg.trace[n])),'key_changed':n in changed,'trimmed_key':','.join(map(str,trim.trace[n])) if n in trim.trace else '', 'trimmed_key_changed':trim.trace[n]!=b['child_keys'][n] if n in trim.trace else ''})
check('Two baseline declarations and eighty ablations available',len(audit)==80 and len(refs)==2)
check('Original source gives both baselines 200 training cells and 23 independent eval names',all(len(x['train_support'])==200 and len(x['validation_component_tags'])==23 and all(t==[n] for n,t in x['validation_component_tags'].items()) and not set(x['train_support'])&set(x['validation_component_tags']) for x in baselines.values()))
check('Recorded full-history ablations retain deleted-domain datasets through historical positive support',all(x['removed_cells']==[] and x['remaining_cells']==200 and len(x['deleted_cells_positive_in_first_historical_stage'])==5 and x['deleted_cells_zero_in_all_later_stages'] for x in contexts))
check('Recorded full-history ablations keep all named child keys unchanged in original source replay',all(x['remaining_changed_child_keys']==0 for x in contexts))
check('Counterfactual trimming removes declared domain plus other future-zero cell',all(x['trimmed_support']==(195 if x['domain']==22 else 194) and set(x['trimmed_removed_cells'])=={'c%02dq%d'%(x['domain'],q) for q in range(5)}|{'c22q0'} and x['trimmed_changed_child_keys']==x['trimmed_index_shift_count'] for x in contexts) and all(b['trimmed_support']==199 and b['trimmed_removed_cells']==['c22q0'] and b['trimmed_changed_child_keys']==89 for b in baselines.values()))
check('Validation components stay zero-weight and declarations retain identical components',all(x['component_count']==223 and x['validation_components_zero_all_phases'] for x in contexts) and all(components_ok))
check('All eighty records declare seed zero with default data seed',all(seed_ok))
repeat=Config(base_data[groups[0]]);repeat.train_sets(None,key=root)
check('Repeat original key allocation is deterministic for same supplied root key',repeat.trace==baselines[groups[0]]['child_keys'])
with (R/'analysis/ablation_child_key_replay.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(key_rows[0]),lineterminator='\n');w.writeheader();w.writerows(key_rows)
# Full 54-field response table, preserving overlapping mean fields as such.
endpoints=list(csv.DictReader((R/'analysis/ablation_task_endpoints.csv').open()));lookup={(r['run_name'],r['metric']):float(r['bpb']) for r in endpoints};metrics=sorted({r['metric'] for r in endpoints});responses=[];paired={}
for row in audit:
 for metric in metrics:
  value=lookup[row['run_name'],metric];base=lookup[row['control_run'],metric];delta=value-base;pct=100*delta/base
  responses.append({'group':row['group'],'domain':int(row['domain']),'domain_zh':row['domain_zh'],'metric':metric,'is_overlapping_mean_field':metric.endswith('_mean'),'control_run':row['control_run'],'run_name':row['run_name'],'baseline_bpb':base,'ablation_bpb':value,'delta_bpb':delta,'relative_change_pct':pct});paired[row['group'],int(row['domain']),metric]=pct
check('Full response export preserves 4320 run-field contrasts for 54 fields',len(metrics)==54 and len(responses)==4320)
for row in audit:
 for metric,short in [('logprob_gsm8k_5shot','gsm8k'),('logprob_humaneval_10shot','humaneval')]:
  assert abs(paired[row['group'],int(row['domain']),metric]-float(row[short+'_change_pct']))<1e-12
check('Independent endpoint recomputation matches prior GSM8K and HumanEval audit',True)
stats={}
for metric in metrics:
 pairs=[(paired[groups[0],i,metric],paired[groups[1],i,metric]) for i in range(40)];flips=[i for i,(a,b) in enumerate(pairs) if a*b<0]
 stats[metric]={'opposite_sign_domains':flips,'opposite_sign_count':len(flips),'is_overlapping_mean_field':metric.endswith('_mean'),'posthoc_display_thresholds_pct':{str(t):sum(a*b<0 and min(abs(a),abs(b))>=t for a,b in pairs) for t in [0,.1,.5,1.,2.]}}
# Add real training-eval Paloma macro without treating it as one of the 54 logprob fields.
pal={g:{int(r['domain']):float(r['paloma_change_pct']) for r in audit if r['group']==g} for g in groups};pairs=[(pal[groups[0]][i],pal[groups[1]][i]) for i in range(40)];stats['paloma_macro_bpb']={'opposite_sign_domains':[i for i,(a,b) in enumerate(pairs) if a*b<0],'opposite_sign_count':sum(a*b<0 for a,b in pairs),'posthoc_display_thresholds_pct':{str(t):sum(a*b<0 and min(abs(a),abs(b))>=t for a,b in pairs) for t in [0,.1,.5,1.,2.]}}
check('Observed strict-sign reversals counted without significance claim',stats['paloma_macro_bpb']['opposite_sign_count']==6 and stats['logprob_gsm8k_5shot']['opposite_sign_count']==9 and stats['logprob_humaneval_10shot']['opposite_sign_count']==31)
with (R/'analysis/ablation_context_responses.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(responses[0]),lineterminator='\n');w.writeheader();w.writerows(responses)
result={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'baselines':baselines,'conditional_key_replays':contexts,'response_fields':metrics,'response_rows':len(responses),'key_replay_rows':len(key_rows),'response_stats':stats,'exploratory_assumption_rejected':'Deleting a domain from continuation phases does not remove it from full schedule support when the historical phase is positive; recorded recipes do not show the hypothesized child-key reassignment in this source replay','input_root_key':np.asarray(root).tolist(),'runtime':{n:importlib.metadata.version(n) for n in ['jax','jaxlib','numpy']},'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)},'explicit_adapters':['DirectDatasetComponent replaces actual token stores; block_shuffle records its received key and does not permute/read samples','Supplied fixed root PRNGKey(0); historical launch seed derivation and code SHA not recovered','Original support/build/train_sets/tagged_eval_sets methods via AST; validation_sets adapter exposes recorded non-cell names only','Post-key allocation slicing and max-train-batch branches disabled; no simulated inventory or training stream replay'],'actual_historical_launch_code_binding':None,'actual_historical_child_keys':None,'actual_token_stream_difference':None,'actual_key_reassignment_loss_effect':None,'actual_independent_training_confirmation':None,'actual_535B_transfer_effect':None,'actual_training_to_eval_semantic_crosswalk':None,'posthoc_thresholds_are_significance_tests':False}
(R/'analysis/ablation_context.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print('Ablation context checks',len(checks),'response rows',len(responses),'key rows',len(key_rows))
