"""V111 old tables, exact measured figure labels, source bindings and claim boundaries."""
import hashlib,json,pathlib,subprocess,xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
R=pathlib.Path(__file__).resolve().parents[1];checks=[]
def ok(n,v):
 assert v,n
 checks.append({'name':n,'passed':True})
old=BeautifulSoup(subprocess.check_output(['git','show','report-v110-2026-10-07:index.html'],cwd=R),'html.parser')
now=BeautifulSoup((R/'index.html').read_text(),'html.parser');stand=BeautifulSoup((R/'report_standalone.html').read_text(),'html.parser')
ok('Historical tables preserved',all(now.select_one('#'+x['id']) and [t.get_text(' ',strip=True) for t in x.select('table')]==[t.get_text(' ',strip=True) for t in now.select_one('#'+x['id']).select('table')][:len(x.select('table'))] for x in old.select('section[id]') if x.select('table')))
j=json.loads((R/'analysis/pending_resume_step_cpu.json').read_text());b=json.loads((R/'analysis/pending_resume_plot_binding.json').read_text())
ok('Original-step controls and adapters bind exact bytes',j['checks_passed']==19 and len(j['serialized_paths'])==11 and all(hashlib.sha256((R/p).read_bytes()).hexdigest()==h for p,h in j['source_sha256'].items()) and b['analysis_sha256']==hashlib.sha256((R/'analysis/pending_resume_step_cpu.json').read_bytes()).hexdigest() and b['figure_sha256']==hashlib.sha256((R/'assets/pending_resume_step.svg').read_bytes()).hexdigest())
texts=[x.text for x in ET.parse(R/'assets/pending_resume_step.svg').getroot().iter('{http://www.w3.org/2000/svg}text')]
ok('All five figure rows match measured bias route and loss',len(b['rows'])==5 and all(str(j['cases'][n]['next_forward_bias'][0]) in texts and str(j['cases'][n]['next_forward_ids'][0]) in texts and ('%.6f'%j['cases'][n]['train_loss']) in texts for n in b['rows']))
ok('Both HTML variants expose accessible figure and bounded EMA observation',all(s.select_one('#pending-resume-guide #pending-resume-title') and 'ema_beta=null' in s.select_one('#pending-resume-guide').get_text() and not s.select('img[src="assets/pending_resume_step.svg"]') for s in [now,stand]) and j['ema_alias_control']['actual_Hero_incident'] is None and j['actual_Hero_view_error_magnitude'] is None)
prior=json.loads(subprocess.check_output(['git','show','report-v110-2026-10-07:sources/archive_manifest.json'],cwd=R))
ok('All 566 source files unchanged including bookkeeping',len(prior['files'])==566 and all(hashlib.sha256((R/'sources'/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in prior['files']))
(R/'analysis/static_validation_v111.json').write_text(json.dumps({'checks_passed':len(checks),'checks':checks,'actual_browser_rendering':None},indent=2)+'\n');print('Static checks',len(checks))
