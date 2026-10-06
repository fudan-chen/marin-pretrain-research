"""Append fixed loader source; fetch completes before archive mutation."""
import pathlib,urllib.request,json,hashlib,datetime
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/eval_identity_2026_10_06';SHA='84869ae8c91ffe64e9f761c5bd714542eb1876e0';assert not D.exists()
payload=[];rows=[]
for name,path in [('loader.py','lib/levanter/src/levanter/data/loader.py'),('logging.py','lib/levanter/src/levanter/utils/logging.py')]:
 u=f'https://raw.githubusercontent.com/marin-community/marin/{SHA}/{path}'
 with urllib.request.urlopen(u,timeout=40) as f:b=f.read()
 compile(b,name,'exec');payload.append((name,b));rows.append({'file':'eval_identity_2026_10_06/'+name,'url':u,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
D.mkdir()
for name,b in payload:(D/name).write_bytes(b)
p=R/'sources/source_manifest.json';j=json.loads(p.read_text());j.extend(rows);p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n');(R/'analysis/eval_identity_acquisition.json').write_text(json.dumps({'revision':SHA,'files':rows},indent=2)+'\n')
