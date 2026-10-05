"""Reacquire all V16 primary sources in a fresh checkout, Python 3.12+.
Refuse archive overwrite or changed GitHub snapshot bytes. A changed mutable
PR response belongs to a new dated snapshot, not a silent historical update.
"""
import concurrent.futures,datetime,hashlib,json,pathlib,sys,urllib.request
R=pathlib.Path(__file__).resolve().parents[1];S=R/'sources';D=S/'dedup_2026_10_05'
def fetch(row):
 b=urllib.request.urlopen(row['url'],timeout=45).read()
 if row['file'].endswith('.py'):compile(b,row['file'],'exec')
 else:json.loads(b)
 if hashlib.sha256(b).hexdigest()!=row['sha256']:raise RuntimeError('Source changed; create a separate snapshot: '+row['file'])
 return b
if __name__=='__main__':
 if sys.version_info<(3,12):raise SystemExit('Use Python 3.12+')
 if D.exists():raise SystemExit('Refusing overwrite')
 p=R/'analysis/dedup_acquisition.json';a=json.loads(p.read_text())
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:blobs=list(pool.map(fetch,a['files']))
 D.mkdir();manifest=json.loads((S/'source_manifest.json').read_text());names={r['file'] for r in a['files']};manifest=[r for r in manifest if r['file'] not in names]
 for row,b in zip(a['files'],blobs):
  (S/row['file']).write_bytes(b);row['retrieved_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();manifest.append(row)
 p.write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n');(S/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
