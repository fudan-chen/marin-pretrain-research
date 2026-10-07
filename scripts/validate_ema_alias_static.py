"""V112 historical prose/tables, measured output digests, figure and provenance."""
import hashlib,json,pathlib,subprocess,xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
R=pathlib.Path(__file__).resolve().parents[1];checks=[]
def ok(n,v):
 assert v,n
 checks.append({'name':n,'passed':True})
old=BeautifulSoup(subprocess.check_output(['git','show','report-v111-2026-10-07:index.html'],cwd=R),'html.parser')
now=BeautifulSoup((R/'index.html').read_text(),'html.parser');stand=BeautifulSoup((R/'report_standalone.html').read_text(),'html.parser')
files=['PENDING_RESUME_ZH.md','TRAIN_STATE_ZH.md','CHECKPOINT_COMMIT_ZH.md','RUBRICS_ZH.md','PIPELINE_ZH.md','SYNTHESIS_ZH.md','IMPROVEMENT_LOG_ZH.md','DELIVERY_AUDIT_ZH.md']
ok('All old prose in appended chapters and all historical tables retained',all((R/p).read_bytes().startswith(subprocess.check_output(['git','show','report-v111-2026-10-07:'+p],cwd=R)) for p in files) and all(now.select_one('#'+x['id']) and [t.get_text(' ',strip=True) for t in x.select('table')]==[t.get_text(' ',strip=True) for t in now.select_one('#'+x['id']).select('table')][:len(x.select('table'))] for x in old.select('section[id]') if x.select('table')))
j=json.loads((R/'analysis/ema_alias_matrix_cpu.json').read_text());b=json.loads((R/'analysis/ema_alias_plot_binding.json').read_text());c=j['cases']
ok('Fifteen controls bind all original sources adapters and local candidate',j['checks_passed']==15 and all(x['passed'] for x in j['checks']) and all(hashlib.sha256((R/p).read_bytes()).hexdigest()==h for p,h in j['source_sha256'].items()) and j['candidate_status']=='candidate_not_integrated' and j['actual_upstream_integration'] is None)
texts=[x.text for x in ET.parse(R/'assets/ema_alias_matrix.svg').getroot().iter('{http://www.w3.org/2000/svg}text')]
ok('Five measured rows and all eleven output digests match bindings',b['analysis_sha256']==hashlib.sha256((R/'analysis/ema_alias_matrix_cpu.json').read_bytes()).hexdigest() and b['figure_sha256']==hashlib.sha256((R/'assets/ema_alias_matrix.svg').read_bytes()).hexdigest() and len(b['rows'])==5 and all(str(c[n]['input_allocation']['logical_leaf_bytes']) in texts and str(c[n]['input_allocation']['unique_observed_buffer_bytes']) in texts for n in b['rows']) and all(len(r['full_output_leaf_digests'])==11 and r['full_output_leaf_digests']==c['ema_only_copy_donated']['full_output_leaf_digests'] for r in c.values() if r['error'] is None))
ok('Both HTML variants expose accessible matrix and bounded hardware claims',all(s.select_one('#pending-resume-guide #ema-alias-title') and not s.select('img[src="assets/ema_alias_matrix.svg"]') for s in [now,stand]) and j['actual_Hero_EMA_incident'] is None and j['actual_production_peak_memory'] is None and j['actual_GPU_execution'] is None)
prior=json.loads(subprocess.check_output(['git','show','report-v111-2026-10-07:sources/archive_manifest.json'],cwd=R))
ok('All 566 historical source files unchanged',len(prior['files'])==566 and all(hashlib.sha256((R/'sources'/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in prior['files']))
(R/'analysis/static_validation_v112.json').write_text(json.dumps({'checks_passed':len(checks),'checks':checks,'actual_browser_rendering':None},indent=2)+'\n');print('Static checks',len(checks))
