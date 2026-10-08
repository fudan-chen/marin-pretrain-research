"""V116 archived prose/tables, four-device source binding and scope checks."""
import hashlib,json,pathlib,subprocess
from bs4 import BeautifulSoup
R=pathlib.Path(__file__).resolve().parents[1];checks=[]
def ok(n,v):
 assert v,n
 checks.append({'name':n,'passed':True})
old=BeautifulSoup(subprocess.check_output(['git','show','report-v115-2026-10-08:index.html'],cwd=R),'html.parser');now=BeautifulSoup((R/'index.html').read_text(),'html.parser');stand=BeautifulSoup((R/'report_standalone.html').read_text(),'html.parser')
ok('Historical prose retained before appendix',all((R/p).read_bytes().startswith(subprocess.check_output(['git','show','report-v115-2026-10-08:'+p],cwd=R)) for p in ['QB_ESTIMATION_ZH.md','RUBRICS_ZH.md','PIPELINE_ZH.md','SYNTHESIS_ZH.md','IMPROVEMENT_LOG_ZH.md','DELIVERY_AUDIT_ZH.md']))
ok('Historical tables retained before new projected-path table',all(now.select_one('#'+x['id']) and [t.get_text(' ',strip=True) for t in x.select('table')]==[t.get_text(' ',strip=True) for t in now.select_one('#'+x['id']).select('table')][:len(x.select('table'))] for x in old.select('section[id]') if x.select('table')))
j=json.loads((R/'analysis/qb_margin_path_cpu.json').read_text())
ok('Nine controls bind original projected path and four CPU devices',j['checks_passed']==9 and j['device_count']==4 and all(hashlib.sha256((R/p).read_bytes()).hexdigest()==h for p,h in j['source_sha256'].items()))
ok('Both HTML variants expose measured values and retain unknown reachability',all('V116' in s.select_one('#qb-guide').get_text() and '33.333290' in s.select_one('#qb-guide').get_text() for s in [now,stand]) and j['actual_Hero_margin_distribution'] is None and j['actual_multi_host_collectives'] is None and j['actual_real_data_domain_effect'] is None)
prior=json.loads(subprocess.check_output(['git','show','report-v115-2026-10-08:sources/archive_manifest.json'],cwd=R))
ok('All 566 original files unchanged',len(prior['files'])==566 and all(hashlib.sha256((R/'sources'/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in prior['files']))
(R/'analysis/static_validation_v116.json').write_text(json.dumps({'checks_passed':len(checks),'checks':checks,'actual_browser_rendering':None},indent=2)+'\n');print('Static checks',len(checks))
