"""Reacquire the V15 source snapshot in a fresh checkout (Python 3.12+).
Existing archives are never overwritten. Mutable Hub metadata must still point
at the observed revision, otherwise stop and start a separately dated snapshot.
"""
import concurrent.futures,datetime,hashlib,json,pathlib,sys,urllib.request
R=pathlib.Path(__file__).resolve().parents[1];S=R/'sources';D=S/'cache_2026_10_05'
def fetch(row):
 b=urllib.request.urlopen(row['url'],timeout=45).read()
 if row['file'].endswith('.py'):compile(b,row['file'],'exec')
 if row['file'].endswith('hf_model_metadata.json'):
  if json.loads(b)['sha']!='a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2':raise RuntimeError('Hub revision changed; create a new snapshot')
 elif hashlib.sha256(b).hexdigest()!=row['sha256']:raise RuntimeError('Pinned source bytes changed: '+row['file'])
 return b
if __name__=='__main__':
 if sys.version_info<(3,12):raise SystemExit('Use Python 3.12+')
 if D.exists():raise SystemExit('Refusing overwrite')
 p=R/'analysis/cache_acquisition.json';a=json.loads(p.read_text())
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:blobs=list(pool.map(fetch,a['files']))
 D.mkdir()
 manifest=json.loads((S/'source_manifest.json').read_text()); names={r['file'] for r in a['files']};manifest=[r for r in manifest if r['file'] not in names]
 for row,b in zip(a['files'],blobs):
  (S/row['file']).write_bytes(b);row.update(bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());manifest.append(row)
 p.write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n');(S/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
