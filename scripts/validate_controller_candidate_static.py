"""Check V104 candidate evidence and unchanged earlier table content."""
import hashlib,json,pathlib,subprocess
from bs4 import BeautifulSoup
R=pathlib.Path(__file__).resolve().parents[1];checks=[]
def ok(name,value):
 assert value,name
 checks.append({'name':name,'passed':True})
old=BeautifulSoup(subprocess.check_output(['git','show','report-v103-2026-10-07:index.html'],cwd=R),'html.parser');now=BeautifulSoup((R/'index.html').read_text(),'html.parser');stand=BeautifulSoup((R/'report_standalone.html').read_text(),'html.parser')
sections=[x for x in old.select('section[id]') if x.select('table')]
ok('All V103 section tables retain ordered text',all(now.select_one('#'+x['id']) and [t.get_text(' ',strip=True) for t in x.select('table')]==[t.get_text(' ',strip=True) for t in now.select_one('#'+x['id']).select('table')] for x in sections))
c=json.loads((R/'analysis/controller_recovery_candidate_controls.json').read_text());m=json.loads((R/'analysis/controller_recovery_candidate.json').read_text());b=json.loads((R/'analysis/controller_candidate_refresh.json').read_text())
ok('22 candidate checks and execution-bound code retain exact hashes',c['checks_passed']==len(c['checks'])==22 and all(x['passed'] for x in c['checks']) and all(hashlib.sha256((R/p).read_bytes()).hexdigest()==h for p,h in c['source_sha256'].items()))
ok('Candidate output captured and patch bound',hashlib.sha256((R/'analysis/controller_recovery_candidate_output.txt').read_bytes()).hexdigest()==m['output_sha256'] and all(hashlib.sha256((R/p).read_bytes()).hexdigest()==h for p,h in m['candidate_sha256'].items()))
ok('Both HTML variants explain candidate integration and legacy gaps',all('旧格式迁移尚未实现' in x.select_one('#controller-recovery-guide').get_text() and '22 项本地控制' in x.select_one('#controller-recovery-guide').get_text() for x in [now,stand]))
ok('Latest same core bytes do not claim latest runtime execution',all(x['same_bytes'] for x in b['bindings']) and b['actual_latest_runtime_execution'] is None and b['actual_production_deployed_revision'] is None)
ok('No candidate production or migration result invented',m['status']=='candidate_not_integrated' and c['actual_production_fix'] is None and c['actual_legacy_migration'] is None)
ok('Original upstream tracked checkout stays unchanged',subprocess.run(['git','diff','--quiet','HEAD'],cwd='/tmp/marin-iris-source-eee467').returncode==0)
(R/'analysis/static_validation_v104.json').write_text(json.dumps({'checks_passed':len(checks),'checks':checks,'preserved_V103_sections_with_tables':len(sections),'actual_browser_rendering':None},ensure_ascii=False,indent=2)+'\n');print('Static checks:',len(checks))
