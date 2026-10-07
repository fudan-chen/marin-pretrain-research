"""V107 prior-table preservation, actual numeric figure, and scope checks."""
import hashlib,json,pathlib,subprocess,xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
R=pathlib.Path(__file__).resolve().parents[1];checks=[]
def ok(n,v):
 assert v,n
 checks.append({'name':n,'passed':True})
old=BeautifulSoup(subprocess.check_output(['git','show','report-v106-2026-10-07:index.html'],cwd=R),'html.parser');now=BeautifulSoup((R/'index.html').read_text(),'html.parser');stand=BeautifulSoup((R/'report_standalone.html').read_text(),'html.parser')
ok('All historical tables retained in order before appended comparison',all(now.select_one('#'+x['id']) and [t.get_text(' ',strip=True) for t in x.select('table')]==[t.get_text(' ',strip=True) for t in now.select_one('#'+x['id']).select('table')][:len(x.select('table'))] for x in old.select('section[id]') if x.select('table')))
j=json.loads((R/'analysis/resume_update_identity.json').read_text());b=json.loads((R/'analysis/resume_source_binding.json').read_text())
ok('Eight assembled CPU controls and five core source bindings valid',j['checks_passed']==8 and len(b['bindings'])==5 and all(x['same_bytes'] for x in b['bindings']) and all(hashlib.sha256((R/p).read_bytes()).hexdigest()==h for p,h in j['source_sha256'].items()))
svg=ET.parse(R/'assets/resume_update_identity.svg').getroot();rects=[x for x in svg.iter('{http://www.w3.org/2000/svg}rect') if x.get('fill') in ['#215f9a','#b26416']];o=j['observations'];values=[x for pair in zip(o['full_update'],o['weights_only_update']) for x in pair]
ok('Six figure bars encode actual updates with shared scale',len(rects)==6 and all(abs(float(x.get('width'))-abs(v)*28000)<.001 and abs(float(x.get('x'))-min(470,470+v*28000))<.001 for x,v in zip(rects,values)))
ok('Both HTML variants embed new accessible figure and comparison values',all(x.select_one('#resume-update-title') and '3.262185574' in x.select_one('#checkpoint-commit-guide').get_text() and '3.229082584' in x.select_one('#checkpoint-commit-guide').get_text() and not x.select('img[src="assets/resume_update_identity.svg"]') for x in [now,stand]))
ok('Original Hero update and causal mixture effects not inferred',j['actual_Hero_next_update_identity'] is None and j['actual_GPU_roundtrip'] is None and j['actual_causal_mixture_effect'] is None and o['pending_applied_in_quadratic'] is False)
(R/'analysis/static_validation_v107.json').write_text(json.dumps({'checks_passed':len(checks),'checks':checks,'actual_browser_rendering':None},indent=2)+'\n');print('Static checks',len(checks))
