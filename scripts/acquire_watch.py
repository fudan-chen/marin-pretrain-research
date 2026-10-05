"""Append fixed watch and tree-stat implementations; refuses existing archive."""
import pathlib,urllib.request,json,hashlib,datetime
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/watch_2026_10_06';SHA='84869ae8c91ffe64e9f761c5bd714542eb1876e0';assert not D.exists();payload=[];rows=[]
for name,path in [('watch.py','lib/levanter/src/levanter/callbacks/watch.py'),('tree_stats.py','lib/levanter/src/levanter/analysis/tree_stats.py')]:
 u=f'https://raw.githubusercontent.com/marin-community/marin/{SHA}/{path}'
 with urllib.request.urlopen(u,timeout=45) as f:b=f.read()
 compile(b,name,'exec');payload.append((name,b));rows.append({'file':'watch_2026_10_06/'+name,'url':u,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
D.mkdir()
for name,b in payload:(D/name).write_bytes(b)
p=R/'sources/source_manifest.json';j=json.loads(p.read_text());j.extend(rows);p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n');(R/'analysis/watch_acquisition.json').write_text(json.dumps({'revision':SHA,'files':rows},indent=2)+'\n')
