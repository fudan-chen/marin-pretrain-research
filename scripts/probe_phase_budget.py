"""Execute original Harrier phase builder/validation and budget helpers on host.
Recording config/component adapters; nominal full-sequence exposure, not actual token stream.
"""
import ast,dataclasses,hashlib,json,math,pathlib,platform,types
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/phase_budget_2026_10_07';H=D/'harrier.py';L=D/'launch_datakit_moe_mix.py';S=D/'harrier.json'
ns={'math':math,'dataclass':dataclasses.dataclass,'DatasetComponent':lambda **kw:types.SimpleNamespace(**kw),'TextLmDatasetFormat':lambda:None,'LmDataConfig':lambda **kw:types.SimpleNamespace(**kw),'prefix_join':lambda a,b:a.rstrip('/')+'/'+b,'marin_tokenizer':'marin-community/marin-tokenizer'}
def extract(p,names):
 nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names];assert len(nodes)==len(names)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])),str(p),'exec'),ns)
def constants(p,names):
 nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in names];assert len(nodes)==len(names);exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(p),'exec'),ns)
constants(H,{'PRETRAIN_TOKENS','COOLDOWN_TOKENS','TOTAL_TOKENS','SIMULATED_EPOCHING_MAX_FLOPS','_MIXTURE_SWITCH_FRACTION'});constants(L,{'_MIXTURE_BLOCK_SIZE','_PHASE_1_START_FRACTION'})
extract(H,['_HarrierMixSpec','_validate_spec','harrier_mix_2026_08_18_data_config']);extract(L,['_phase_1_start_step','_simulated_experiment_budget','_simulated_epoching_budgets'])
raw=json.loads(S.read_text());spec=ns['_HarrierMixSpec'](raw['tokenizer'],raw['candidate_store_uri'],tuple(raw['available_tokens'].items()),tuple(tuple(p['weights'].items()) for p in raw['phases']));ns['_SPEC']=spec;ns['HARRIER_MIX_2026_08_18_STORE']=types.SimpleNamespace(adopt_source=raw['candidate_store_uri']);ctx=types.SimpleNamespace(is_fingerprint=True,artifact_path=lambda _:raw['candidate_store_uri'])
def record(label,N,B=11264,seq=4096,flops=2.7e24):
 cfg=ns['harrier_mix_2026_08_18_data_config'](ctx=ctx,total_steps=N,batch_size=B,max_seq_len=seq,experiment_flops=flops,validation=[])
 stages=cfg.train_weights;phase=[];epochs={k:0. for k in raw['available_tokens']}
 for i,(start,w) in enumerate(stages):
  idx=next(j for j,p in enumerate(raw['phases']) if w==p['weights']);end=stages[i+1][0] if i+1<len(stages) else N;duration=max(0,min(end,N)-start)
  phase.append({'start':start,'end_clipped_to_run':min(end,N),'steps':duration,'phase_index':idx})
  for k in epochs:epochs[k]+=duration*B*seq*w[k]/raw['available_tokens'][k]
 peak=max(epochs,key=epochs.get)
 return {'case':label,'total_steps':N,'batch':B,'sequence_length':seq,'analytic_flops':flops,'step_multiple':ns['_MIXTURE_BLOCK_SIZE']//math.gcd(ns['_MIXTURE_BLOCK_SIZE'],B),'phase_stages':phase,'target_budget':cfg.target_budget,'experiment_budget':cfg.experiment_budget,'nominal_sequence_tokens':N*B*seq,'nominal_weighted_peak_cell':peak,'nominal_weighted_peak_epochs':epochs[peak],'phase_weight_sha256':[hashlib.sha256(json.dumps(w,sort_keys=True).encode()).hexdigest() for _,w in stages]}
def main():
 checks=[]
 def ck(n,c):assert c,n;checks.append(n)
 ns['_validate_spec'](spec);ck('Original fixed spec validation passes',len(spec.available_tokens)==200 and len(spec.phase_weights)==3)
 cases=[]
 for label,N,B,flops in [('hero_recipe',390251,11264,2.7e24),('one_extra_step',390252,11264,2.7e24),('double_horizon',780502,11264,5.4e24),('changed_batch_same_steps',390251,2048,2.7e24),('short_diagnostic',100,1024,1e20),('one_step',1,1024,1e20),('aligned_small_batch',100,49152,1e20),('threshold_equal',100,1024,1e23),('threshold_above',100,1024,math.nextafter(1e23,math.inf))]:cases.append(record(label,N,B,flops=flops))
 c={x['case']:x for x in cases};base=c['hero_recipe'];extended=c['double_horizon'];short=c['short_diagnostic'];tiny=c['one_step']
 ck('All recorded stage sequence offsets align',all((p['start']*x['batch'])%ns['_MIXTURE_BLOCK_SIZE']==0 for x in cases for p in x['phase_stages']))
 ck('Hero recipe boundaries match pinned source', [p['start'] for p in base['phase_stages']]==[0,108000,312192])
 ck('One extra planned step delays main phase by a block',[p['start'] for p in c['one_extra_step']['phase_stages']]==[0,108048,312192])
 ck('Double horizon moves both planned boundaries',[p['start'] for p in extended['phase_stages']]==[0,216000,624384])
 resume=121638
 def phase_at(x,t):return next(p['phase_index'] for p in reversed(x['phase_stages']) if p['start']<=t)
 ck('Same artificial resume step selects different phase under rebuilt horizon',phase_at(base,resume)==1 and phase_at(extended,resume)==0)
 ck('Short run dict collision removes main phase',[p['phase_index'] for p in short['phase_stages']]==[0,2] and short['phase_stages'][1]['start']==48)
 ck('One-step run never consumes recorded cooldown',tiny['phase_stages'][1]['start']==48 and tiny['phase_stages'][1]['steps']==0)
 ck('Batch alignment changes small run phase presence',[p['phase_index'] for p in c['aligned_small_batch']['phase_stages']]==[0,1,2])
 ck('Equal analytic threshold enables simulated budgets',c['threshold_equal']['target_budget']==ns['TOTAL_TOKENS'] and c['threshold_equal']['experiment_budget']==100*1024*4096)
 ck('Next float above threshold disables both budgets',c['threshold_above']['target_budget'] is None and c['threshold_above']['experiment_budget'] is None)
 ck('Raw builder permits nominal weighted exposure beyond fixed spec cap',base['nominal_weighted_peak_epochs']<=8 and extended['nominal_weighted_peak_epochs']>8 and extended['target_budget'] is None)
 ck('Changing batch alone changes nominal exposure scale',c['changed_batch_same_steps']['nominal_sequence_tokens']<base['nominal_sequence_tokens'])
 try:record('over_target_with_simulation',780502,11264,flops=1e20);budget_error=None
 except ValueError as e:budget_error=str(e)
 ck('Enabled simulation rejects nominal budget above target',budget_error is not None and 'exceeds target_budget' in budget_error)
 ck('Raw builder accepts same enlarged nominal token budget',extended['experiment_budget'] is None)
 out={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'runtime':{'python':platform.python_version()},'cases':cases,'artificial_resume_control':{'step':resume,'old_phase':phase_at(base,resume),'rebuilt_phase':phase_at(extended,resume),'actual_checkpoint_restore':None},'enabled_simulation_error':budget_error,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [H,L,S]},'explicit_substitutions':['AST-extracted original builder, fixed validation and exact phase/budget helpers','recording LmDataConfig and DatasetComponent adapters; no config post-init/cache/token reads','spec data passed into original dataclass without original filesystem loader','ctx fingerprint adapter and empty validation list','nominal exposure uses full sequence length and declared continuous weights, not actual integer-block counts or effective loss-token mass'],'actual_Hero_phase_regression':None,'actual_Hero_overexposure':None,'actual_Hero_loss_effect':None,'actual_token_identity_replay':None,'actual_GPU_execution':None}
 (R/'analysis/phase_budget_probe.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'checks_passed':len(checks),'cases':cases},indent=2))
if __name__=='__main__':main()
