"""Append-only issue acquisition and body refresh. Python 3.12 HTTPS. Refuses existing incident archive."""
import json,pathlib,urllib.request,datetime,hashlib,concurrent.futures
R=pathlib.Path(__file__).resolve().parents[1]; base='https://api.github.com/repos/marin-community/marin/issues/'
def get(url):
 req=urllib.request.Request(url,headers={'Accept':'application/vnd.github+json','User-Agent':'marin-research'})
 with urllib.request.urlopen(req,timeout=45) as f:return json.load(f)
def fetch(n):
 issue=get(base+str(n)); comments=[];page=1
 while True:
  batch=get(base+str(n)+f'/comments?per_page=100&page={page}');comments+=batch
  if len(batch)<100:break
  page+=1
 assert len(comments)==issue['comments']
 return n,issue,comments
rows=[];results=list(concurrent.futures.ThreadPoolExecutor(4).map(fetch,[8435,8506,8870,8073]))
for n,i,c in results:
 if n==8073:
  D=R/'sources/muon_geometry_2026_10_05';assert not D.exists();D.mkdir();manifest=json.loads((R/'sources/source_manifest.json').read_text());acq=[]
  for name,obj,url in [('issue_8073.json',i,base+'8073'),('comments_8073.json',c,base+'8073/comments?per_page=100&page=1')]:
   b=(json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode();(D/name).write_bytes(b);row={'file':str((D/name).relative_to(R/'sources')),'url':url,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};manifest.append(row);acq.append(row)
  (R/'sources/source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');(R/'analysis/muon_geometry_acquisition.json').write_text(json.dumps({'files':acq,'comment_count':len(c),'issue_updated_at':i['updated_at']},indent=2)+'\n')
  print('8073',len(c),i['body'][:1200]);print([(x['id'],x['body'][:180]) for x in c])
 else:
  D=R/'sources/engineering_2026_10_05';old=json.loads((D/f'issue_{n}_comments.json').read_text());oi=json.loads((D/f'issue_{n}.json').read_text());changed={x['id']:x.get('body') for x in old}!={x['id']:x.get('body') for x in c} or oi.get('body')!=i.get('body');rows.append({'issue':n,'comment_count':len(c),'body_or_comment_changed':changed,'updated_at':i['updated_at'],'url':i['html_url']});print(rows[-1])
(R/'analysis/issue_refresh_v30.json').write_text(json.dumps({'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'issue body and comment ID/body comparison; metadata changes excluded','issues':rows},ensure_ascii=False,indent=2)+'\n')
