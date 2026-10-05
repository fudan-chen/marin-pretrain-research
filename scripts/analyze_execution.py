"""Bounded public execution-provenance audit. Runner files are inspected, never executed."""
import ast
import hashlib
import json
import pathlib
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/execution_2026_10_05'
meta=json.loads((D/'run_execution_files.json').read_text())['data']['project']
inventory=json.loads((D/'run_file_inventory.json').read_text())['data']['project']
rows=[]
for key,m in meta.items():
    r=inventory[key];names=[x['node']['name'] for x in r['files']['edges']]
    rows.append({'run':r['name'],'commit':m['commit'],'created_at':m['createdAt'],
                 'reported_file_count':r['fileCount'],'enumerated_file_count':len(names),
                 'has_next_page':r['files']['pageInfo']['hasNextPage'],
                 'exact_metadata_name_matches':[x['node']['name'] for x in m['files']['edges']],
                 'runner_present':'code/_callable_runner.py' in names,
                 'callable_pickle_listed':any(x.endswith('_callable.pkl') for x in names),
                 'evaluation_python_listed':any(x.endswith('/eval.py') for x in names),
                 'count_discrepancy':r['fileCount']-len(names)})
runner_rows=[]
for p in sorted(D.glob('runner_*.py')):
    source=p.read_text();tree=ast.parse(source)
    loads=any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='cloudpickle' and n.func.attr=='loads' for n in ast.walk(tree))
    runner_rows.append({'file':str(p.relative_to(R)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                        'loads_cloudpickle_callable':loads,'callable_pickle_name_present':'_callable.pkl' in source,
                        'source_executed':False,'deserialized_callable':False})
audit={'scope':'Six public run commit fields, exact-name metadata search, paginated-flag file inventory and two public runner entries; not a complete search of every job/artifact/log',
       'runs':rows,'runner_entries':runner_rows,'public_run_commit_binding_found':False,
       'historical_eval_code_binding':None,'historical_bpb_root_cause':None,
       'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(D.glob('*'))},
       'limitation':'Each fileCount exceeds the enumerated connection by one despite hasNextPage=false. No reason inferred; not a proof that metadata/source cannot exist elsewhere.'}
(R/'analysis/execution_provenance_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print('Run commits:',[r['commit'] for r in rows],'; public saved runner entries:',len(runner_rows))
