"""Archive pinned quality-scoring code; refuse overwrite."""
import concurrent.futures,datetime,hashlib,json,pathlib,sys,urllib.request
R=pathlib.Path(__file__).resolve().parents[1];S=R/'sources';D=S/'quality_2026_10_05';SHA='84869ae8c91ffe64e9f761c5bd714542eb1876e0'
NAMES=['artifact','calibrate','inference','rubric','score','scorer','data','metrics','README']
def fetch(n):
 u=f'https://raw.githubusercontent.com/marin-community/marin/{SHA}/experiments/datakit/cluster/quality/fast_transformer/{n}.{'md' if n=='README' else 'py'}';b=urllib.request.urlopen(u,timeout=45).read();
 if n!='README':compile(b,n,'exec')
 return n,b,dict(file=f'quality_2026_10_05/{n}.{"md" if n=="README" else "py"}',url=u,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
if __name__=='__main__':
 if sys.version_info<(3,12):raise SystemExit('Use Python3.12+')
 if D.exists():raise SystemExit('Refusing overwrite')
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(fetch,NAMES))
 D.mkdir()
 for n,b,_ in rows:(D/(n+('.md' if n=='README' else '.py'))).write_bytes(b)
 p=S/'source_manifest.json';j=json.loads(p.read_text());j.extend(r[2] for r in rows);p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n');(R/'analysis/quality_acquisition.json').write_text(json.dumps(dict(revision=SHA,files=[r[2] for r in rows]),ensure_ascii=False,indent=2)+'\n')
