"""Acquire pinned files or adopt complete raw downloads verified by frozen byte hashes.
The staged mode records file completion mtime as an approximate retrieval timestamp.
"""
import pathlib,urllib.request,json,hashlib,datetime,argparse
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/accumulation_2026_10_07';assert not D.exists();rev='84869ae8c91ffe64e9f761c5bd714542eb1876e0';payload=[];rows=[]
parser=argparse.ArgumentParser();parser.add_argument('--staged-dir',type=pathlib.Path);args=parser.parse_args()
expected={'grad_accum':'047a8bad9a9a9bdd837680c4fdb440fb19c12dc3001a56ab5c1b78521cba949b','trainer':'280284a5bf7461a9a8871e0773cc1f1ec9ef1c92a7a58d224e68ba06d2aa76d3'}
for name in ['grad_accum','trainer']:
 url=f'https://raw.githubusercontent.com/marin-community/marin/{rev}/lib/levanter/src/levanter/{name}.py'
 if args.staged_dir:
  p=args.staged_dir/('marin-'+name+'-v48.py');b=p.read_bytes();stamp=datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone.utc).isoformat();extra={'transport':'Complete raw download staged earlier in this research turn','retrieval_time_basis':'Approximate local download completion mtime; exact server timestamp unrecorded'}
 else:
  with urllib.request.urlopen(url,timeout=45) as f:b=f.read();final=f.url
  stamp=datetime.datetime.now(datetime.timezone.utc).isoformat();extra={'final_url':final,'transport':'Complete raw HTTP response body'}
 assert hashlib.sha256(b).hexdigest()==expected[name];compile(b,name+'.py','exec')
 payload.append((name+'.py',b));rows.append(dict(file='accumulation_2026_10_07/'+name+'.py',url=url,bytes=len(b),sha256=expected[name],retrieved_utc=stamp,git_revision=rev,execution_binding='Pinned public code, not verified historical Hero execution',**extra))
D.mkdir()
for n,b in payload:(D/n).write_bytes(b)
p=R/'sources/source_manifest.json';m=json.loads(p.read_text());m.extend(rows);p.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
(R/'analysis/accumulation_acquisition.json').write_text(json.dumps({'files':rows,'incomplete_downloads_not_archived':True},indent=2)+'\n');print('Archived two complete pinned source bodies')
