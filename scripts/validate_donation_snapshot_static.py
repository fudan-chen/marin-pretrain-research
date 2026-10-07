"""V106 ownership documentation and historical table preservation."""
import hashlib,json,pathlib,subprocess
from bs4 import BeautifulSoup
R=pathlib.Path(__file__).resolve().parents[1];checks=[]
def ok(n,v):
 assert v,n
 checks.append({'name':n,'passed':True})
old=BeautifulSoup(subprocess.check_output(['git','show','report-v105-2026-10-07:index.html'],cwd=R),'html.parser');now=BeautifulSoup((R/'index.html').read_text(),'html.parser');stand=BeautifulSoup((R/'report_standalone.html').read_text(),'html.parser')
ok('All previous tables retained in order',all(now.select_one('#'+x['id']) and [t.get_text(' ',strip=True) for t in x.select('table')]==[t.get_text(' ',strip=True) for t in now.select_one('#'+x['id']).select('table')][:len(x.select('table'))] for x in old.select('section[id]') if x.select('table')))
j=json.loads((R/'analysis/donation_snapshot_cpu.json').read_text())
ok('Ten CPU checks source and script bound',j['checks_passed']==10 and all(hashlib.sha256((R/p).read_bytes()).hexdigest()==h for p,h in j['source_sha256'].items()))
ok('Actual donation distinguished from retained view control',all(x['input_deleted'] and x['saved_first']==0 and x['new_first']==200 for x in j['cases']) and not j['retained_host_view_control']['input_deleted'])
ok('Both HTML variants embed accessible ownership diagram',all(x.select_one('#checkpoint-memory-guide svg title') and not x.select('#checkpoint-memory-guide img[src="assets/donation_snapshot_flow.svg"]') for x in [now,stand]))
ok('GPU and full training restoration not claimed',j['actual_GPU_donation'] is None and j['actual_Hero_checkpoint_integrity'] is None and j['actual_training_resume'] is None)
(R/'analysis/static_validation_v106.json').write_text(json.dumps({'checks_passed':len(checks),'checks':checks,'actual_browser_rendering':None},indent=2)+'\n');print('Static checks',len(checks))
