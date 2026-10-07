"""Archive complete live revision binding separately from the candidate's frozen base."""
import datetime,hashlib,json,pathlib
R=pathlib.Path(__file__).resolve().parents[1];S=R/'sources';D=S/'controller_candidate_2026_10_07';staged=pathlib.Path('/tmp/marin-v104');assert not D.exists();prior=json.loads((S/'archive_manifest.json').read_text());head=json.loads((staged/'head.json').read_text());rev=head['sha'];rows=[]
for name in ['head.json','checkpoint.py','main.py']:
 b=(staged/name).read_bytes()
 if name.endswith('.py'):compile(b,name,'exec')
 url='https://api.github.com/repos/marin-community/marin/commits/main' if name=='head.json' else f'https://raw.githubusercontent.com/marin-community/marin/{rev}/lib/iris/src/iris/cluster/controller/{name}'
 D.mkdir(exist_ok=True);(D/name).write_bytes(b)
 rows.append({'file':str((D/name).relative_to(S)),'url':url,'git_revision':rev,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'retrieved_utc':datetime.datetime.fromtimestamp((staged/name).stat().st_mtime,datetime.timezone.utc).isoformat(),'retrieval_time_basis':'Approximate staged complete download mtime'})
m=json.loads((S/'source_manifest.json').read_text());m.extend(rows);(S/'source_manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n');old=[x for x in prior['files'] if x['file']!='source_manifest.json'];assert all(hashlib.sha256((S/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in old)
bindings=[{'file':n,'same_bytes':(D/n).read_bytes()==(S/'controller_restore_2026_10_07/lib/iris/src/iris/cluster/controller'/n).read_bytes()} for n in ['checkpoint.py','main.py']]
(R/'analysis/controller_candidate_refresh.json').write_text(json.dumps({'records':rows,'prior_integrity_manifest':prior,'prior_bytes_preserved':len(old),'latest_revision':rev,'latest_subject':head['commit']['message'].splitlines()[0],'frozen_runtime_revision':'eee467718515b2383fc3a433014afce4ab075b05','bindings':bindings,'actual_latest_runtime_execution':None,'actual_production_deployed_revision':None},indent=2)+'\n');print('Archived',len(rows),'payloads; preserved',len(old),'prior files')
