"""Bind two public code artifacts to selected source and rerun prior synthetic boundary replay.
Artifact declaration and host log are evidence; actual imports, checkpoint contents and tokens remain unverified.
"""
import ast,base64,hashlib,json,pathlib,types,sys,subprocess
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/run_code_2026_10_08';checks=[]
def check(n,v):assert v,n;checks.append({'name':n,'passed':True})
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
roles={r:json.loads((D/(r+'_inventory.json')).read_text())['data']['project']['run'] for r in ['producer','continuation']}
ms={r:json.loads((D/(r+'_code_manifest.json')).read_text())['contents'] for r in roles}
check('Anonymous inventories complete and explicitly bound to two different runs',roles['producer']['name']=='h100-ladder-d512-ep8-bs1024-791tpp-10pct-20260826-rno2a' and roles['continuation']['name']=='h100-d512-mix-proportional-zero-c00-seed0-from10pct-20260826-rno2a' and all(not x['files']['pageInfo']['hasNextPage'] and x['commit'] is None for x in roles.values()))
paths={'mixture.py':'data/mixture.py','loader.py':'data/loader.py','datasets.py':'data/text/datasets.py','schedule.py':'schedule.py','train_lm.py':'main/train_lm.py'};bindings=[]
for f,p in paths.items():
 k='lib/levanter/src/levanter/'+p;b=(D/f).read_bytes();md5=base64.b64encode(hashlib.md5(b).digest()).decode()
 bindings.append({'local_file':str((D/f).relative_to(R)),'artifact_path':k,'bytes':len(b),'sha256':digest(D/f),'manifest_md5_base64':md5,'matches_both_run_manifests':all(ms[r][k]['digest']==md5 and ms[r][k]['size']==len(b) for r in roles)})
check('Five downloaded files match size and W&B MD5 in both independent manifests',all(x['matches_both_run_manifests'] for x in bindings))
def nodes(p,name):return next(n for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and n.name==name)
def same(a,b,name):return ast.dump(nodes(a,name),include_attributes=False)==ast.dump(nodes(b,name),include_attributes=False)
oldmix=R/'sources/deepening_2026_10_04/mixture_production.py';olddata=R/'sources/deepening_2026_10_04/datasets_production.py';oldloader=R/'sources/eval_identity_2026_10_06/loader.py';oldschedule=R/'sources/deepening_2026_10_04/schedule_production.py'
check('Historical mixed-block source differs only in stage selection method AST',not same(oldmix,D/'mixture.py','_get_stage_for_block') and ast.dump(ast.parse(oldmix.read_text().replace(ast.get_source_segment(oldmix.read_text(),nodes(oldmix,'_get_stage_for_block')),ast.get_source_segment((D/'mixture.py').read_text(),nodes(D/'mixture.py','_get_stage_for_block')))),include_attributes=False)==ast.dump(ast.parse((D/'mixture.py').read_text()),include_attributes=False))
method_equivalence={n:same(olddata,D/'datasets.py',n) for n in ['_has_nonzero_weight','build_token_datasets','train_sets','tagged_eval_sets']}
loader_equivalence={n:same(oldloader,D/'loader.py',n) for n in ['_produce_batches','_dataset_get_available_batch_number','_do_retrieve_batch_of_batches']}
check('Selected support/key methods and original batch schedule identical to artifact sources',all(method_equivalence.values()) and oldschedule.read_bytes()==(D/'schedule.py').read_bytes())
check('Three previously replayed async loader methods identical despite whole-file differences',all(loader_equivalence.values()) and oldloader.read_bytes()!=(D/'loader.py').read_bytes())
logs={r:(D/(r+'_output.log.txt')).read_text().splitlines() for r in roles};evidence=[]
for role,lines in logs.items():
 for i,line in enumerate(lines,1):
  if ('Starting from scratch.' in line or 'Loading checkpoint from ' in line or 'Restore read ' in line or 'Progress on:train 394it/' in line or ('Saving checkpoint at step 393.' in line)):
   evidence.append({'role':role,'file':str((D/(role+'_output.log.txt')).relative_to(R)),'line':i,'text':line})
check('Producer log records scratch start and continuation records step393 read plus first progress394',any(x['role']=='producer' and 'Starting from scratch.' in x['text'] for x in evidence) and any(x['role']=='continuation' and 'Loading checkpoint from ' in x['text'] and '/step-393' in x['text'] for x in evidence) and any(x['role']=='continuation' and 'Restore read ' in x['text'] for x in evidence) and any(x['role']=='continuation' and 'train 394it/' in x['text'] for x in evidence))
# Load the existing dependency adapters, replacing their archived input file paths only.
def adapter(script,module,replacements):
 p=R/'scripts'/script;s=p.read_text()
 for a,b in replacements.items():assert a in s;s=s.replace(a,b)
 mod=types.ModuleType(module);mod.__file__=str(p);sys.modules[module]=mod;exec(compile(s,str(p),'exec'),mod.__dict__);return mod
mix=adapter('probe_mixture_identity_cpu.py','historical_mix_adapter',{"sources/deepening_2026_10_04/mixture_production.py":"sources/run_code_2026_10_08/mixture.py"})
loader=adapter('probe_loader_resume.py','historical_loader_adapter',{"sources/eval_identity_2026_10_06/loader.py":"sources/run_code_2026_10_08/loader.py","sources/deepening_2026_10_04/schedule_production.py":"sources/run_code_2026_10_08/schedule.py"})
p=R/'scripts/probe_ablation_resume_boundary.py';s=p.read_text().replace('import probe_mixture_identity_cpu as mix','').replace('import probe_loader_resume as loader','').replace('analysis/ablation_resume_boundary','analysis/historical_ablation_resume_boundary')
ns={'__file__':str(p),'__name__':'historical_boundary_replay','mix':mix,'loader':loader};exec(compile(s,str(p),'exec'),ns)
old=json.loads((R/'analysis/ablation_resume_boundary.json').read_text());new=json.loads((R/'analysis/historical_ablation_resume_boundary.json').read_text())
fields=['geometry','summary','cases','partial_block_population','identity_examples','unchanged_weights_control','unaligned_step393_error','alternate_supplied_key_controls']
check('All eighty synthetic boundary outputs and controls unchanged with uploaded historical sources',new['checks_passed']==12 and all(old[k]==new[k] for k in fields) and all((R/('analysis/'+f)).read_bytes()==(R/('analysis/historical_'+f)).read_bytes() for f in ['ablation_resume_boundary_cells.csv','ablation_resume_boundary_runs.csv']))
# Original stage methods agree on actual aligned declarations and supplied block probes.
probes=[]
for mname,m in [('producer',ns['build'](ns['pd'])),('continuation',ns['build'](ns['load'](roles['continuation']['name'])))]:
 for block in [0,7,8,9,64,65,66,81]:
  a=nodes(oldmix,'_get_stage_for_block');env={'np':ns['np']};exec(compile(ast.fix_missing_locations(ast.Module(body=[a],type_ignores=[])),str(oldmix),'exec'),env)
  probes.append({'role':mname,'block':block,'old_stage':env['_get_stage_for_block'](m,block),'historical_stage':m._get_stage_for_block(block)})
check('Stage representation difference preserves selected actual aligned-boundary probes',all(x['old_stage']==x['historical_stage'] for x in probes))
files=[pathlib.Path(__file__),R/'scripts/acquire_run_code_provenance.py',p,R/'scripts/probe_mixture_identity_cpu.py',R/'scripts/probe_loader_resume.py',oldmix,olddata,oldloader,oldschedule]+list(D.iterdir())+[R/'analysis/ablation_resume_boundary.json',R/'analysis/historical_ablation_resume_boundary.json']
out={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'run_names':{r:x['name'] for r,x in roles.items()},'git_commits':{r:x['commit'] for r,x in roles.items()},'manifest_entries':{r:len(m) for r,m in ms.items()},'file_bindings':bindings,'support_and_key_method_ast_equivalence':method_equivalence,'loader_method_ast_equivalence':loader_equivalence,'stage_probes':probes,'log_evidence':evidence,'historical_replay_checks':new['checks_passed'],'historical_replay_cases':len(new['cases']),'historical_replay_matches_prior':True,'source_sha256':{str(p.relative_to(R)):digest(p) for p in sorted(set(files))},'scope_limit':'Only two run artifacts; five selected source files. Uploaded bytes are not an import/module-origin trace. Synthetic IDs, supplied keys, finite identity stores remain substitutions.','actual_checkpoint_contents_verified':None,'actual_runtime_imports_verified':None,'actual_historical_mixture_key':None,'actual_token_repetition_count':None,'actual_training_loss_effect':None,'all_eighty_artifacts_verified':None}
(R/'analysis/run_code_provenance.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('Code provenance controls',len(checks))
