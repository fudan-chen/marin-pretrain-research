"""Verify archived full source bytes against public fixed-revision raw GitHub responses."""
import concurrent.futures,datetime,hashlib,json,pathlib,urllib.request
R=pathlib.Path(__file__).resolve().parents[1];revision='eee467718515b2383fc3a433014afce4ab075b05'
manifest={x['file']:x for x in json.loads((R/'sources/source_manifest.json').read_text())}
paths=['contracts_2026_10_05/model.py','contracts_2026_10_05/api.py','contracts_2026_10_05/reference.py','boundaries_2026_10_05/examples.py','scale_2026_10_05/grug_loss.py','deepening_2026_10_04/mixture_production.py']
def fetch(p):
 old=manifest[p]['url'];prefix='https://raw.githubusercontent.com/marin-community/marin/';assert old.startswith(prefix),old
 upstream=old[len(prefix):].split('/',1)[1];url=prefix+revision+'/'+upstream
 with urllib.request.urlopen(url,timeout=35) as f:b=f.read();status=f.status
 archive=(R/'sources'/p).read_bytes()
 return {'archive_file':'sources/'+p,'upstream_path':upstream,'verified_source_url':url,'frozen_revision':revision,'http_status':status,'bytes':len(b),'same_bytes':b==archive,'sha256':hashlib.sha256(b).hexdigest(),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:rows=list(pool.map(fetch,paths))
assert all(x['same_bytes'] and x['http_status']==200 for x in rows),rows
(R/'analysis/loss_mass_source_binding.json').write_text(json.dumps({'bindings':rows,'retrieval':'Complete HTTP response payload compared to existing complete archived source; no duplicate source files','actual_latest_head_execution':None},indent=2)+'\n');print('Verified',len(rows),'complete fixed-revision public source responses')
