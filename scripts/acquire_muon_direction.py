"""Append fixed Muon direction implementation and coefficient source; refuse overwrite."""
import pathlib,urllib.request,json,hashlib,datetime
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/muon_direction_2026_10_05';SHA='84869ae8c91ffe64e9f761c5bd714542eb1876e0';assert not D.exists()
files=[('grugmuon_hero.py','experiments/grug/moe_hero_ep/grugmuon_hero.py'),('util.py','lib/levanter/src/levanter/optim/util.py')];rows=[];payload=[]
for name,path in files:
 u=f'https://raw.githubusercontent.com/marin-community/marin/{SHA}/{path}'
 with urllib.request.urlopen(u,timeout=45) as response:b=response.read()
 compile(b,name,'exec');payload.append((name,b));rows.append({'file':'muon_direction_2026_10_05/'+name,'url':u,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
D.mkdir()
for name,b in payload:(D/name).write_bytes(b)
p=R/'sources/source_manifest.json';j=json.loads(p.read_text());j.extend(rows);p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n');(R/'analysis/muon_direction_acquisition.json').write_text(json.dumps({'revision':SHA,'files':rows},indent=2)+'\n')
