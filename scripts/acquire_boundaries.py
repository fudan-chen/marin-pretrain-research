"""Archive pinned boundary implementation; run with Python 3.12+ for syntax checking."""
import concurrent.futures,datetime,hashlib,json,pathlib,sys,urllib.request
R=pathlib.Path(__file__).resolve().parents[1];S=R/'sources';D=S/'boundaries_2026_10_05'
PATHS={'examples.py':'lib/levanter/src/levanter/data/text/examples.py','datasets.py':'lib/levanter/src/levanter/data/text/datasets.py','text_init.py':'lib/levanter/src/levanter/data/text/__init__.py','attention_core.py':'lib/levanter/src/levanter/grug/attention/_core.py','fa4_cute.py':'lib/levanter/src/levanter/grug/attention/_fa4_cute.py','test_fa4.py':'lib/levanter/tests/grug/test_fa4_cute_attention.py','lm_model.py':'lib/levanter/src/levanter/models/lm_model.py'}
SHA='84869ae8c91ffe64e9f761c5bd714542eb1876e0'
def fetch(item):
 n,p=item;url='https://raw.githubusercontent.com/marin-community/marin/'+SHA+'/'+p
 with urllib.request.urlopen(url,timeout=45) as r:b=r.read()
 compile(b,n,'exec')
 return n,b,{'file':str((D/n).relative_to(S)),'url':url,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
if __name__=='__main__':
 if sys.version_info<(3,12):raise SystemExit('Use Python 3.12+')
 if D.exists():raise SystemExit('Refusing overwrite')
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(fetch,PATHS.items()))
 D.mkdir()
 for n,b,_ in rows:(D/n).write_bytes(b)
 p=S/'source_manifest.json';j=json.loads(p.read_text());j.extend(r[2] for r in rows);p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n')
 (R/'analysis/boundary_acquisition.json').write_text(json.dumps({'revision':SHA,'files':[r[2] for r in rows]},indent=2)+'\n')
