"""Hash the complete source archive; preserve original acquisition provenance."""
import pathlib,json,hashlib,datetime,re
ROOT=pathlib.Path(__file__).resolve().parents[1];S=ROOT/'sources'
known={}
for x in json.loads((S/'source_manifest.json').read_text()):
    if 'error' not in x:known[x['file']]=x
extra=ROOT/'analysis/prp_acquisition.json'
if extra.exists():
    x=json.loads(extra.read_text());known[x['file'][len('sources/'):] if x['file'].startswith('sources/') else x['file']]={k:x[k] for k in ['url','retrieved_utc','sha256']}
rows=[]
for p in sorted(S.rglob('*')):
    if not p.is_file() or p.name=='archive_manifest.json':continue
    rel=str(p.relative_to(S));b=p.read_bytes();r={'file':rel,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
    if rel in known:
        orig=known[rel];assert r['sha256']==orig['sha256'],rel
        r.update({k:orig[k] for k in ['url','retrieved_utc']})
    elif p.parent.name=='wandb':
        j=json.loads(p.read_text())
        if isinstance(j,dict) and 'retrieved_utc' in j:r['retrieved_utc']=j['retrieved_utc']
        run=j.get('run') if isinstance(j,dict) else None
        if not run and p.stem.endswith('_meta'):run=p.stem[:-5]
        if run:r['public_run_url']='https://wandb.ai/marin-community/marin_moe/runs/'+run
        r['origin']='Anonymous public GraphQL; request specs and collector code archived.'
    else:r['origin']='Derived readable copy, browser excerpt, or archive bookkeeping; see source_manifest and acquisition scripts.'
    rows.append(r)
(S/'archive_manifest.json').write_text(json.dumps({'integrity_manifest_created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':rows},ensure_ascii=False,indent=2))
print('Source archive hashed:',len(rows),'files')
