"""Validate split coverage, preserved text/assets, local targets and original hash."""
from pathlib import Path
from urllib.parse import urlsplit,unquote
import hashlib,json
from bs4 import BeautifulSoup
R=Path(__file__).resolve().parents[1]
manifest=json.loads((R/'reading/manifest.json').read_text())
original=R/'report_standalone.html'
assert hashlib.sha256(original.read_bytes()).hexdigest()==manifest['complete_report_sha256']
source=BeautifulSoup(original.read_text(),'html.parser')
tool=BeautifulSoup((R/'reading/tools.html').read_text(),'html.parser')
groups=manifest['groups']; ids=[i for g in groups for i in g['sections']]
assert len(ids)==len(set(ids))==manifest['sections']
for section in source.select('main > section[id]'):
 sid=section['id']
 page=tool if sid in groups[-1]['sections'] else BeautifulSoup((R/'reading'/f'{sid}.html').read_text(),'html.parser')
 new=page.find('section',id=sid)
 assert new is not None,sid
 assert section.get_text()==new.get_text(),sid
 if page is not tool:assert not page.find('script'),sid
count=0
for p in (R/'reading').glob('*.html'):
 page=BeautifulSoup(p.read_text(),'html.parser')
 for tag in page.select('[href],[src]'):
  for attr in ['href','src']:
   url=tag.get(attr,'');u=urlsplit(url)
   if not url or u.scheme or not u.path:continue
   assert (p.parent/unquote(u.path)).resolve().exists(),(p.name,url)
 count+=1
print(f'PASS: {len(ids)} sections preserved; {count} HTML pages; local assets/links present; original hash unchanged.')
