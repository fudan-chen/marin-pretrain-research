"""Acquire immutable V119 source payloads once; verify existing records on rerun."""
import pathlib,urllib.request,hashlib,json,datetime
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/eval_fix_2026_10_08';D.mkdir(exist_ok=True)
SHA='a00cb77a491f4777a2c66edce54f47ff7b255c40';raw=f'https://raw.githubusercontent.com/yonromai/marin/{SHA}/'
items=[('train.py',raw+'experiments/grug/moe_hero_ep/train.py'),('test_moe_hero_ep.py',raw+'tests/test_moe_hero_ep.py'),('commit.json',f'https://api.github.com/repos/yonromai/marin/commits/{SHA}'),('issue_9352.json','https://api.github.com/repos/marin-community/marin/issues/9352'),('state_adapter.py',raw+'lib/levanter/src/levanter/callbacks/state_adapter.py'),('callback_core.py',raw+'lib/levanter/src/levanter/callbacks/_core.py'),('eval.py',raw+'lib/levanter/src/levanter/eval.py')]
p=R/'sources/source_manifest.json';manifest=json.loads(p.read_text());known={x['file']:x for x in manifest};records=[];new=[]
for name,url in items:
 target=D/name;rel=str(target.relative_to(R/'sources'))
 if target.exists():
  record=known[rel];assert record['url']==url and hashlib.sha256(target.read_bytes()).hexdigest()==record['sha256']
 else:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Marin-research-source-audit'}),timeout=45) as response:body=response.read();final=response.url
  target.write_bytes(body);record={'file':rel,'url':url,'final_url':final,'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest(),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};new.append(record)
 records.append(record)
if new:p.write_text(json.dumps(manifest+new,ensure_ascii=False,indent=2)+'\n')
output=R/'analysis/eval_fix_acquisition_v119.json'
if not output.exists():output.write_text(json.dumps({'records':records},indent=2)+'\n')
else:assert json.loads(output.read_text())['records']==records
print('Verified',len(records),'payloads; new downloads:',len(new))
