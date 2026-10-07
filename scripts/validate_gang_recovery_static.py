"""Check the added report section against its upstream artifacts and V101 tables."""
import hashlib,json,pathlib,subprocess,xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
R=pathlib.Path(__file__).resolve().parents[1]
old=BeautifulSoup(subprocess.check_output(['git','show','report-v101-2026-10-07:index.html'],cwd=R),'html.parser')
now=BeautifulSoup((R/'index.html').read_text(),'html.parser');stand=BeautifulSoup((R/'report_standalone.html').read_text(),'html.parser');checks=[]
def ok(name,value):
 assert value,name
 checks.append({'name':name,'passed':True})
g=now.select_one('#gang-recovery-guide');sg=stand.select_one('#gang-recovery-guide')
ok('New chapter has two complete scene tables',len(g.select('table'))==2 and all(t.select('thead') and t.select('tbody') for t in g.select('table')))
ok('Both HTML variants inline the accessible recovery diagram',all(x.select_one('svg title') and x.select_one('svg desc') and not x.select('img[src="assets/gang_recovery_flow.svg"]') for x in [g,sg]))
sections=[x for x in old.select('section[id]') if x.select('table')]
ok('All V101 section tables retain ordered text',all(now.select_one('#'+x['id']) and [t.get_text(' ',strip=True) for t in x.select('table')]==[t.get_text(' ',strip=True) for t in now.select_one('#'+x['id']).select('table')] for x in sections))
j=json.loads((R/'analysis/gang_recovery_native_tests.json').read_text())
ok('33 native passes agree with complete JUnit XML',j['passed']==len(j['cases'])==33 and len(list(ET.parse(R/'analysis/gang_recovery_pytest.xml').getroot().iter('testcase')))==33 and j['returncode']==0)
ok('Native run outputs and archived source match recorded SHA256',all(hashlib.sha256((R/p).read_bytes()).hexdigest()==h for p,h in {**j['source_sha256'],**j['artifact_sha256']}.items()))
t=json.loads((R/'analysis/gang_recovery_trace_template.json').read_text())
ok('Template contains no invented recovery events',t['events']==[] and t['status']=='proposed_not_executed' and t['causal_mixture_effect'] is None)
ok('Production revision and recovery remain unknown',j['actual_Hero_deployed_revision'] is None and j['actual_Hero_recovery_trace'] is None and j['actual_controller_crash_restart'] is None)
record={'checks_passed':len(checks),'checks':checks,'preserved_V101_sections_with_tables':len(sections),'actual_browser_rendering':None,'quicklook_scope':'Only a cropped thumbnail inspected; not complete SVG or browser QA','source_sha256':{'assets/gang_recovery_flow.svg':hashlib.sha256((R/'assets/gang_recovery_flow.svg').read_bytes()).hexdigest()}}
(R/'analysis/static_validation_v102.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print('Static checks:',len(checks))
