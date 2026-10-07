"""Check V105 evidence refinement and preservation of previous tables."""
import hashlib,json,pathlib,subprocess,xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
R=pathlib.Path(__file__).resolve().parents[1];checks=[]
def ok(n,v):
 assert v,n
 checks.append({'name':n,'passed':True})
old=BeautifulSoup(subprocess.check_output(['git','show','report-v104-2026-10-07:index.html'],cwd=R),'html.parser');now=BeautifulSoup((R/'index.html').read_text(),'html.parser');stand=BeautifulSoup((R/'report_standalone.html').read_text(),'html.parser')
sections=[x for x in old.select('section[id]') if x.select('table')]
ok('Prior tables retained in order with new snapshot table appended',all(now.select_one('#'+x['id']) and [t.get_text(' ',strip=True) for t in x.select('table')]==[t.get_text(' ',strip=True) for t in now.select_one('#'+x['id']).select('table')][:len(x.select('table'))] for x in sections))
j=json.loads((R/'analysis/snapshot_semantics.json').read_text());a=json.loads((R/'analysis/snapshot_auth_tests.json').read_text())
ok('Eight snapshot controls source and candidate bound',j['checks_passed']==8 and all(hashlib.sha256((R/p).read_bytes()).hexdigest()==h for p,h in j['source_sha256'].items()))
ok('Two original auth cases match original JUnit',a['passed']==2 and len(list(ET.parse(R/'analysis/snapshot_auth_tests.xml').getroot().iter('testcase')))==2 and all(hashlib.sha256((R/p).read_bytes()).hexdigest()==h for p,h in a['artifact_sha256'].items()))
ok('Both HTML variants carry auth content and constructed-data distinction',all('当前 auth.sqlite3 已无业务表' in x.select_one('#controller-recovery-guide').get_text() and 'research_generation' in x.select_one('#controller-recovery-guide').get_text() and len(x.select_one('#controller-recovery-guide').select('table'))==4 for x in [now,stand]))
ok('No production invariant or training skew invented',j['fresh_auth_application_tables']==[] and j['actual_production_crossfile_invariant'] is None and j['actual_training_state_skew'] is None and j['actual_signing_key_loss'] is None)
(R/'analysis/static_validation_v105.json').write_text(json.dumps({'checks_passed':len(checks),'checks':checks,'actual_browser_rendering':None},ensure_ascii=False,indent=2)+'\n');print('Static checks',len(checks))
