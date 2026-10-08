"""Verify decision synthesis arithmetic, links, preserved evidence and HTML integration."""
import hashlib,json,pathlib,subprocess
from decimal import Decimal
from bs4 import BeautifulSoup
R=pathlib.Path(__file__).resolve().parents[1];checks=[]
def ok(name,condition):
    assert condition,name
    checks.append({'name':name,'passed':True})
j=json.loads((R/'analysis/training_decision_cases_v118.json').read_text())
source=(R/j['source']).read_text()
ok('Quoted author values and all computed differences match archived source',all(str(j[n][k]) in source for n in ['current','preferred'] for k in ['a1','a2']) and all(Decimal(str(j[n]['a2']))-Decimal(str(j[n]['a1']))==Decimal(str(j[n]['correction'])) for n in ['current','preferred']) and all(Decimal(str(j['preferred'][k]))-Decimal(str(j['current'][k]))==Decimal(str(j['preferred_minus_current'][k])) for k in ['a1','a2']) and Decimal(str(j['preferred_minus_current']['a2']))-Decimal(str(j['preferred_minus_current']['a1']))==Decimal(str(j['delta_change'])))
ok('No new training or production adoption claimed',j['actual_new_training_result'] is None and j['actual_production_fix_deployed'] is None and j['evidence_status']=='archived_author_report_not_local_training')
old=BeautifulSoup(subprocess.check_output(['git','show','report-v117-2026-10-08:index.html'],cwd=R),'html.parser')
now=BeautifulSoup((R/'index.html').read_text(),'html.parser');stand=BeautifulSoup((R/'report_standalone.html').read_text(),'html.parser')
ok('Both reports include decision guide with three tables and numerical example',all(s.select_one('#training-decisions-guide') and len(s.select_one('#training-decisions-guide').select('table'))==3 and '0.00001121' in s.select_one('#training-decisions-guide').get_text() for s in [now,stand]))
ok('Every historical section table retained',all(now.select_one('#'+x['id']) and [t.get_text(' ',strip=True) for t in x.select('table')]==[t.get_text(' ',strip=True) for t in now.select_one('#'+x['id']).select('table')][:len(x.select('table'))] for x in old.select('section[id]') if x.select('table')))
prior=json.loads(subprocess.check_output(['git','show','report-v117-2026-10-08:sources/archive_manifest.json'],cwd=R))
ok('All 566 historical source payloads byte preserved',len(prior['files'])==566 and all(hashlib.sha256((R/'sources'/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in prior['files']))
(R/'analysis/static_validation_v118.json').write_text(json.dumps({'checks_passed':len(checks),'checks':checks,'actual_browser_rendering':None},indent=2)+'\n')
print('Static checks',len(checks))
