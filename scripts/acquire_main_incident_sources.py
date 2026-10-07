"""Archive complete public refresh/code payloads; staged timestamps are approximate mtimes."""
import argparse,datetime,hashlib,json,pathlib,urllib.request
R=pathlib.Path(__file__).resolve().parents[1];S=R/'sources';D=S/'main_incident_2026_10_07';assert not D.exists()
a=argparse.ArgumentParser();a.add_argument('--staged-dir',type=pathlib.Path);args=a.parse_args();rev='eee467718515b2383fc3a433014afce4ab075b05';old='b65be4c9550c5097f0a3add08933531a1c24d534'
urls={f'issue_{n}.json':f'https://api.github.com/repos/marin-community/marin/issues/{n}' for n in [8435,8506]}
urls.update({f'comments_{n}.json':f'https://api.github.com/repos/marin-community/marin/issues/{n}/comments?per_page=100' for n in [8435,8506]})
urls.update({'head.json':'https://api.github.com/repos/marin-community/marin/commits/main','compare.json':f'https://api.github.com/repos/marin-community/marin/compare/{old}...{rev}?per_page=100','compare_reverse.json':f'https://api.github.com/repos/marin-community/marin/compare/{rev}...{old}?per_page=100','pull_9833.json':'https://api.github.com/repos/marin-community/marin/pulls/9833','k8s_taint.html':'https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration/'})
for n,p in {'train':'experiments/grug/moe_hero_ep/train.py','model':'experiments/grug/moe_hero_ep/model.py','ep_ragged_all_to_all':'lib/levanter/src/levanter/grug/_moe/ep_ragged_all_to_all.py','iris_k8s_tasks':'lib/iris/src/iris/cluster/backends/k8s/tasks.py'}.items():urls[n+'.py']=f'https://raw.githubusercontent.com/marin-community/marin/{rev}/{p}'
prior=json.loads((S/'archive_manifest.json').read_text());rows=[];payload=[]
for n,u in urls.items():
 if args.staged_dir:
  p=args.staged_dir/n;b=p.read_bytes();stamp=datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone.utc).isoformat();basis='Approximate local complete download mtime, not exact server timestamp'
 else:
  with urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'marin-source-research'}),timeout=30) as f:b=f.read()
  stamp=datetime.datetime.now(datetime.timezone.utc).isoformat();basis='HTTP body completion wall clock'
 if n.endswith('.json'):json.loads(b)
 if n.endswith('.py'):compile(b,n,'exec')
 payload.append((n,b));rows.append({'file':D.name+'/'+n,'url':u,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'retrieved_utc':stamp,'retrieval_time_basis':basis,'execution_binding':'Public source/snapshot; actual deployed revision unknown'})
D.mkdir()
for n,b in payload:(D/n).write_bytes(b)
p=S/'source_manifest.json';m=json.loads(p.read_text());m.extend(rows);p.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
preserved=[x for x in prior['files'] if x['file']!='source_manifest.json'];assert all(hashlib.sha256((S/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in preserved)
(R/'analysis/main_incident_acquisition.json').write_text(json.dumps({'records':rows,'prior_integrity_manifest':prior,'prior_bytes_preserved':len(preserved),'changed_bookkeeping':['source_manifest.json']},indent=2)+'\n');print('Archived',len(rows),'public payloads; preserved',len(preserved),'old non-bookkeeping files')
