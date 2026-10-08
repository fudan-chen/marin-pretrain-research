"""V119 provenance, preserved historical content and HTML evidence-scope audit."""
import hashlib,json,pathlib,subprocess,ast
from bs4 import BeautifulSoup
R=pathlib.Path(__file__).resolve().parents[1];checks=[]
def ok(name,value):
 assert value,name
 checks.append({'name':name,'passed':True})
old=BeautifulSoup(subprocess.check_output(['git','show','report-v118-2026-10-08:index.html'],cwd=R),'html.parser');now=BeautifulSoup((R/'index.html').read_text(),'html.parser');stand=BeautifulSoup((R/'report_standalone.html').read_text(),'html.parser')
ok('Historical appended prose and all historical section tables retained',all((R/p).read_bytes().startswith(subprocess.check_output(['git','show','report-v118-2026-10-08:'+p],cwd=R)) for p in ['PENDING_RESUME_ZH.md','SYNTHESIS_ZH.md','RUBRICS_ZH.md','PIPELINE_ZH.md','IMPROVEMENT_LOG_ZH.md','DELIVERY_AUDIT_ZH.md']) and all(now.select_one('#'+x['id']) and [t.get_text(' ',strip=True) for t in x.select('table')]==[t.get_text(' ',strip=True) for t in now.select_one('#'+x['id']).select('table')][:len(x.select('table'))] for x in old.select('section[id]') if x.select('table')))
j=json.loads((R/'analysis/eval_pending_fix_cpu.json').read_text());a=json.loads((R/'analysis/eval_fix_acquisition_v119.json').read_text())
ok('Fifteen CPU controls and seven acquired payloads bind fixed source bytes',j['checks_passed']==15 and len(a['records'])==7 and all(hashlib.sha256((R/'sources'/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in a['records']) and all(hashlib.sha256((R/p).read_bytes()).hexdigest()==h for p,h in j['source_sha256'].items()))
prior=json.loads(subprocess.check_output(['git','show','report-v118-2026-10-08:sources/archive_manifest.json'],cwd=R))
ok('565 prior non-bookkeeping sources byte preserved and acquisition records append-only',len(prior['files'])==566 and all(hashlib.sha256((R/'sources'/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in prior['files'] if x['file']!='source_manifest.json') and json.loads((R/'sources/source_manifest.json').read_text())[:-7]==json.loads(subprocess.check_output(['git','show','report-v118-2026-10-08:sources/source_manifest.json'],cwd=R)))
ok('Both HTML variants explain force forwarding duplicate scoring and unknown deployment',all(s.select_one('#pending-resume-guide') and all(t in s.select_one('#pending-resume-guide').get_text() for t in ['V119','15项CPU','force','评分去重','生产采用均未知']) for s in [now,stand]) and j['actual_full_training_loop_execution'] is None and j['actual_GPU_execution'] is None)
train=ast.parse((R/'sources/eval_fix_2026_10_08/train.py').read_text());assigns=[n.lineno for n in ast.walk(train) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='pending_qb_betas_for_callbacks' for t in n.targets)]
commit=json.loads((R/'sources/eval_fix_2026_10_08/commit.json').read_text())
ok('Pinned final commit and three production closure-refresh sites inspected',commit['sha']=='a00cb77a491f4777a2c66edce54f47ff7b255c40' and sorted(assigns)==[1124,1257,1318])
(R/'analysis/static_validation_v119.json').write_text(json.dumps({'checks_passed':len(checks),'checks':checks,'production_refresh_assignment_lines':sorted(assigns),'actual_browser_rendering':None},indent=2)+'\n');print('Static checks',len(checks))
