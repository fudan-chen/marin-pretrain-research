"""Original runtime default helper on isolated host environment dictionaries.
No JAX/GPU initialization, launch, model constructor or production environment.
"""
import ast,hashlib,json,pathlib,types
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/runtime_defaults_2026_10_07';T=D/'train.py';M=D/'model.py';E=R/'sources/engineering_v77_2026_10_07/current_ep_ragged_all_to_all.py'
G=R/'sources/engineering_v77_2026_10_07/current_grug_moe.py'
ns={}
def execute(nodes,p):exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])),str(p),'exec'),ns)
et=ast.parse(E.read_text());flags=next(n for n in et.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='RAGGED_REQUIRED_XLA_FLAGS' for t in n.targets));execute([flags],E)
mt=ast.parse(M.read_text());offload=next(n for n in mt.body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.target.id=='OFFLOAD_CARRY_REMAT_MODE');execute([offload],M)
names=['HERO_EP_RUNTIME_ENV','OFFLOAD_CARRY_MEM_FRACTION','DEFAULT_SLOP_FACTOR','OFFLOAD_CARRY_SLOP_FACTOR','XLA_COLLECTIVE_OVERLAP_FLAG','XLA_HOST_MEMORY_OFFLOADING_FLAG','XLA_DISABLE_ASYNC_COLLECTIVES_FLAG','SYNC_COLLECTIVES','DEFAULT_COLLECTIVE_OVERLAP_LIMIT','INLINE_WATCH_COLLECTIVE_OVERLAP_LIMIT','SERIAL_COLLECTIVE_OVERLAP_LIMIT','RAGGED_MOE_IMPLEMENTATION','XLA_DISABLE_GPU_COMMAND_BUFFER_FLAG','_RAGGED_REQUIRED_XLA_FLAG_NAMES']
tt=ast.parse(T.read_text());nodes=[n for n in tt.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets)];assert len(nodes)==len(names)
helper=next(n for n in tt.body if isinstance(n,ast.FunctionDef) and n.name=='_apply_hero_ep_runtime_defaults');execute(nodes+[helper],T)
def run(env,remat='offload_carry',implementation='ragged_all_to_all'):
 copied=dict(env);ns['os']=types.SimpleNamespace(environ=copied)
 ns['_apply_hero_ep_runtime_defaults'](inline_watch_enabled=False,moe_implementation=implementation,remat_mode=remat,processes_per_task=1)
 return copied
def params(env):return {'memory_fraction':env['XLA_PYTHON_CLIENT_MEM_FRACTION'],'xla_flags':env['XLA_FLAGS'].split(),'parsed_flag_values':dict(f.partition('=')[::2] for f in env['XLA_FLAGS'].split())}
def main():
 checks=[]
 def check(n,c):assert c,n;checks.append(n)
 overlap=ns['XLA_COLLECTIVE_OVERLAP_FLAG'];slop='--xla_gpu_memory_limit_slop_factor';latency='--xla_gpu_enable_latency_hiding_scheduler'
 clean=run({});inherited=run({'XLA_PYTHON_CLIENT_MEM_FRACTION':'.75','XLA_FLAGS':f'{slop}=85 {overlap}=8 {latency}=false'});ordinary=run({},remat='remat');transition=run(ordinary);duplicate=run({'XLA_FLAGS':f'{slop}=85 {slop}=105 {overlap}=4 {overlap}=8'});fixed=run(clean)
 cases={n:params(e) for n,e in [('clean_offload',clean),('inherited_offload',inherited),('clean_ragged_no_offload',ordinary),('same_process_transition',transition),('duplicate_slop',duplicate),('repeat_same_mode',fixed)]}
 check('Clean carry defaults widen memory fraction and slop',cases['clean_offload']['memory_fraction']=='0.78' and cases['clean_offload']['parsed_flag_values'][slop]=='105')
 check('Explicit inherited fraction/slop/latency preserved',cases['inherited_offload']['memory_fraction']=='.75' and cases['inherited_offload']['parsed_flag_values'][slop]=='85' and cases['inherited_offload']['parsed_flag_values'][latency]=='false')
 check('Every ragged control forces single overlap despite inheritance',all(c['parsed_flag_values'][overlap]=='1' and sum(f.startswith(overlap+'=') for f in c['xla_flags'])==1 for c in cases.values()))
 check('All ragged required flags are exact and unique',all(all(c['xla_flags'].count(f)==1 for f in ns['RAGGED_REQUIRED_XLA_FLAGS']) for c in cases.values()))
 check('Ragged no-offload forces synchronous collectives',cases['clean_ragged_no_offload']['parsed_flag_values'][ns['XLA_DISABLE_ASYNC_COLLECTIVES_FLAG']]=='ALLCOLLECTIVES')
 check('Mode transition retains earlier defaults and sync setting',cases['same_process_transition']['memory_fraction']=='0.75' and cases['same_process_transition']['parsed_flag_values'][slop]=='85' and cases['same_process_transition']['parsed_flag_values'][latency]=='false' and cases['same_process_transition']['parsed_flag_values'][ns['XLA_DISABLE_ASYNC_COLLECTIVES_FLAG']]=='ALLCOLLECTIVES')
 check('Duplicate ordinary slop flags preserved without selecting backend precedence',sum(f.startswith(slop+'=') for f in cases['duplicate_slop']['xla_flags'])==2)
 check('Repeated same mode is idempotent in this dictionary control',clean==fixed)
 old=json.loads((R/'sources/engineering_v77_2026_10_07/comments_8506.json').read_text());new=json.loads((D/'comments_8506.json').read_text());issue=json.loads((D/'issue_8506.json').read_text())
 check('Latest public issue comments complete and bodies unchanged',len(new)==issue['comments']==59 and {x['id']:x['body'] for x in old}=={x['id']:x['body'] for x in new})
 pulls=[json.loads((D/f'pull_{n}.json').read_text()) for n in [9831,9832,9833]]
 check('All three current proposals remain open/unmerged',all(p['state']=='open' and p['merged'] is False and p['merged_at'] is None for p in pulls))
 check('Retrieved model runtime head matches audited portable head',pulls[-1]['head']['sha']=='b65be4c9550c5097f0a3add08933531a1c24d534')
 acq=json.loads((R/'analysis/runtime_defaults_acquisition.json').read_text());check('Seven exact acquisitions and prior 479 non-bookkeeping bytes preserved',len(acq['records'])==7 and acq['prior_bytes_preserved']==479 and all(hashlib.sha256((R/'sources'/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in acq['records']) and all(hashlib.sha256((R/'sources'/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in acq['prior_integrity_manifest']['files'] if x['file']!='source_manifest.json'))
 # Static source selection evidence, distinct from executing model initialization.
 selection=[n.value for n in ast.walk(mt) if isinstance(n,ast.keyword) and n.arg=='routing_weight_gradient'];assert len(selection)==1
 generic_cls=next(n for n in ast.parse(G.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='MoEExpertMlp')
 generic=next(n for n in generic_cls.body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.target.id=='routing_weight_gradient')
 check('Generic EXACT and hero ragged conditional EXPERT_SIDE are explicit static source selections','RoutingWeightGradient.EXACT' in ast.unparse(generic) and isinstance(selection[0],ast.IfExp) and ast.unparse(selection[0].body)=='RoutingWeightGradient.EXPERT_SIDE' and ast.unparse(selection[0].orelse)=='RoutingWeightGradient.EXACT')
 out={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'cases':cases,'hero_constructor_selection_ast':ast.unparse(selection[0]),'generic_default_ast':ast.unparse(generic),'current_pulls':[{'number':p['number'],'state':p['state'],'merged':p['merged'],'head':p['head']['sha'],'updated_at':p['updated_at']} for p in pulls],'issue_8506_comments':len(new),'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [T,M,E,G]},'explicit_substitutions':['os.environ replaced by independent dictionaries; no host environment mutation','AST extraction skips launcher imports and backend setup','hero constructor selection inspected statically, not instantiated'],'actual_production_deployment':None,'actual_effective_environment':None,'actual_XLA_parser_duplicate_precedence':None,'actual_HBM_measurement':None,'actual_GPU_execution':None}
 (R/'analysis/runtime_defaults.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'checks_passed':len(checks),'cases':cases,'selection':out['hero_constructor_selection_ast']},indent=2))
if __name__=='__main__':main()
