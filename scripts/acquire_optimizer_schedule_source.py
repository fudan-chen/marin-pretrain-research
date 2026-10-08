"""Acquire pinned optimizer scheduler source; no historical runtime binding claimed."""
import datetime,hashlib,json,pathlib,urllib.request
R=pathlib.Path(__file__).resolve().parents[1]
u='https://raw.githubusercontent.com/marin-community/marin/84869ae8c91ffe64e9f761c5bd714542eb1876e0/lib/levanter/src/levanter/optim/config.py'
p=R/'sources/optimizer_schedule_2026_10_08/config.py';m=R/'sources/source_manifest.json';records=json.loads(m.read_text());rel=str(p.relative_to(R/'sources'))
if p.exists():
 prior=next(x for x in records if x['file']==rel)
 if hashlib.sha256(p.read_bytes()).hexdigest()!=prior['sha256']:raise RuntimeError('existing source mismatch')
else:
 with urllib.request.urlopen(u,timeout=45) as response:b=response.read()
 p.parent.mkdir(exist_ok=True);p.write_bytes(b)
 records.append({'file':rel,'url':u,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'git_revision':'84869ae8c91ffe64e9f761c5bd714542eb1876e0','execution_binding':'Pinned public source, not verified as historical deployed code'})
 m.write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
(R/'analysis/optimizer_schedule_acquisition_v133.json').write_text(json.dumps({'records':[x for x in records if x['file']==rel],'historical_runtime_binding':None},indent=2)+'\n')
print('Scheduler source bytes',p.stat().st_size)
