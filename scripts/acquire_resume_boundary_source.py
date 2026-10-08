"""Acquire public metadata for the shared d512 step-393 producer; verify snapshot on rerun."""
import pathlib,json,urllib.request,datetime,hashlib
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/resume_boundary_2026_10_08';D.mkdir(exist_ok=True);p=D/'producer_meta.json';url='https://api.wandb.ai/graphql'
query='query($e:String!,$p:String!,$r:String!){project(name:$p,entityName:$e){run(name:$r){name displayName state createdAt config summaryMetrics}}}'
variables={'e':'marin-community','p':'marin_moe','r':'h100-ladder-d512-ep8-bs1024-791tpp-10pct-20260826-rno2a'};payload={'query':query,'variables':variables}
manifest=R/'sources/source_manifest.json';records=json.loads(manifest.read_text());rel=str(p.relative_to(R/'sources'))
if p.exists():
 record=next(x for x in records if x['file']==rel);assert record['sha256']==hashlib.sha256(p.read_bytes()).hexdigest();print('Existing producer source verified')
else:
 request=urllib.request.Request(url,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','User-Agent':'Marin-research-resume-audit'})
 with urllib.request.urlopen(request,timeout=50) as response:body=response.read();final=response.url
 p.write_bytes(body);record={'file':rel,'url':url,'final_url':final,'request':payload,'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest(),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};manifest.write_text(json.dumps(records+[record],ensure_ascii=False,indent=2)+'\n')
 (R/'analysis/resume_boundary_acquisition_v121.json').write_text(json.dumps({'records':[record]},indent=2)+'\n');print('Producer payload acquired',len(body),'bytes')
