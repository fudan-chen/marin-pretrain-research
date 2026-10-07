"""V113 historical prose/table preservation, exact plotted losses and bounded reachability."""
import hashlib,json,pathlib,subprocess,xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
R=pathlib.Path(__file__).resolve().parents[1];checks=[]
def ok(n,v):
 assert v,n
 checks.append({'name':n,'passed':True})
old=BeautifulSoup(subprocess.check_output(['git','show','report-v112-2026-10-07:index.html'],cwd=R),'html.parser')
now=BeautifulSoup((R/'index.html').read_text(),'html.parser');stand=BeautifulSoup((R/'report_standalone.html').read_text(),'html.parser')
files=['FAILURE_BOUNDARIES_ZH.md','TRAIN_STATE_ZH.md','CHECKPOINT_COMMIT_ZH.md','RUBRICS_ZH.md','PIPELINE_ZH.md','SYNTHESIS_ZH.md','IMPROVEMENT_LOG_ZH.md','DELIVERY_AUDIT_ZH.md']
ok('Old prose and tables preserved before new appendices',all((R/p).read_bytes().startswith(subprocess.check_output(['git','show','report-v112-2026-10-07:'+p],cwd=R)) for p in files) and all(now.select_one('#'+x['id']) and [t.get_text(' ',strip=True) for t in x.select('table')]==[t.get_text(' ',strip=True) for t in now.select_one('#'+x['id']).select('table')][:len(x.select('table'))] for x in old.select('section[id]') if x.select('table')))
j=json.loads((R/'analysis/pending_finite_gap_cpu.json').read_text());b=json.loads((R/'analysis/pending_finite_plot_binding.json').read_text())
ok('Thirteen controls and sources match archived candidate and original bytes',j['checks_passed']==13 and all(x['passed'] for x in j['checks']) and all(hashlib.sha256((R/p).read_bytes()).hexdigest()==h for p,h in j['source_sha256'].items()))
texts=[x.text for x in ET.parse(R/'assets/pending_finite_gap.svg').getroot().iter('{http://www.w3.org/2000/svg}text')]
ok('Twelve plotted losses match original CPU records and figure binding',len(b['rows'])==4 and all('loss %.6f'%s['loss'] in texts for c in j['cases'].values() for s in c['stages']) and b['analysis_sha256']==hashlib.sha256((R/'analysis/pending_finite_gap_cpu.json').read_bytes()).hexdigest() and b['figure_sha256']==hashlib.sha256((R/'assets/pending_finite_gap.svg').read_bytes()).hexdigest())
ok('Both HTML variants preserve accessible figure and unknown estimator reachability',all(s.select_one('#failure-boundaries-guide #pending-finite-title') and not s.select('img[src="assets/pending_finite_gap.svg"]') for s in [now,stand]) and j['actual_original_QB_estimator_nonfinite_output'] is None and j['actual_production_poisoned_checkpoint'] is None and j['actual_Hero_nonfinite_beta_event'] is None and j['candidate_status']=='candidate_not_integrated')
prior=json.loads(subprocess.check_output(['git','show','report-v112-2026-10-07:sources/archive_manifest.json'],cwd=R))
ok('All 566 old sources unchanged',len(prior['files'])==566 and all(hashlib.sha256((R/'sources'/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in prior['files']))
(R/'analysis/static_validation_v113.json').write_text(json.dumps({'checks_passed':len(checks),'checks':checks,'actual_browser_rendering':None},indent=2)+'\n');print('Static checks',len(checks))
