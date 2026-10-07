"""V110 historical table preservation, public-source binding and bounded claims."""
import hashlib,json,pathlib,subprocess
from bs4 import BeautifulSoup
R=pathlib.Path(__file__).resolve().parents[1];checks=[]
def ok(n,v):
 assert v,n
 checks.append({'name':n,'passed':True})
old=BeautifulSoup(subprocess.check_output(['git','show','report-v109-2026-10-07:index.html'],cwd=R),'html.parser')
now=BeautifulSoup((R/'index.html').read_text(),'html.parser'); stand=BeautifulSoup((R/'report_standalone.html').read_text(),'html.parser')
ok('V109 historical table content retained',all(now.select_one('#'+x['id']) and [t.get_text(' ',strip=True) for t in x.select('table')]==[t.get_text(' ',strip=True) for t in now.select_one('#'+x['id']).select('table')][:len(x.select('table'))] for x in old.select('section[id]') if x.select('table')))
j=json.loads((R/'analysis/operational_progress.json').read_text()); a=json.loads((R/'analysis/operational_refresh_acquisition.json').read_text()); b=json.loads((R/'analysis/operational_plot_binding.json').read_text())
ok('Analysis and figure bind exact archived bytes',all(hashlib.sha256((R/p).read_bytes()).hexdigest()==h for p,h in j['source_sha256'].items()) and b['analysis_sha256']==hashlib.sha256((R/'analysis/operational_progress.json').read_bytes()).hexdigest() and b['figure_sha256']==hashlib.sha256((R/'assets/operational_progress.svg').read_bytes()).hexdigest() and len(a['records'])==10)
ok('Both HTML variants expose accessible figure and distinct endpoints',all(x.select_one('#operational-progress-guide #operational-progress-title') and '219263' in x.select_one('#operational-progress-guide').get_text() and '219518' in x.select_one('#operational-progress-guide').get_text() and not x.select('img[src="assets/operational_progress.svg"]') for x in [now,stand]))
ok('Public telemetry is not promoted into downtime or causal mixture effect',j['actual_fault_downtime'] is None and j['actual_duty_cycle'] is None and j['actual_loss_causal_effect'] is None and j['triage_alignment']['actual_independent_node_logs'] is None and all(x['actual_pause_seconds'] is None for x in j['intervals']))
prior=json.loads(subprocess.check_output(['git','show','report-v109-2026-10-07:sources/archive_manifest.json'],cwd=R))
ok('Every old nonbookkeeping source preserves bytes',all(hashlib.sha256((R/'sources'/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in prior['files'] if x['file']!='source_manifest.json'))
(R/'analysis/static_validation_v110.json').write_text(json.dumps({'checks_passed':len(checks),'checks':checks,'actual_browser_rendering':None},indent=2)+'\n');print('Static checks',len(checks))
