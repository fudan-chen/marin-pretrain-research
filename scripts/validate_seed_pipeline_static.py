"""V108 preservation, recorded key diagram and real-run evidence boundaries."""
import hashlib,json,pathlib,subprocess
from bs4 import BeautifulSoup
R=pathlib.Path(__file__).resolve().parents[1];checks=[]
def ok(n,v):
 assert v,n
 checks.append({'name':n,'passed':True})
old=BeautifulSoup(subprocess.check_output(['git','show','report-v107-2026-10-07:index.html'],cwd=R),'html.parser');now=BeautifulSoup((R/'index.html').read_text(),'html.parser');stand=BeautifulSoup((R/'report_standalone.html').read_text(),'html.parser')
ok('Historical tables retained before appended seed comparison',all(now.select_one('#'+x['id']) and [t.get_text(' ',strip=True) for t in x.select('table')]==[t.get_text(' ',strip=True) for t in now.select_one('#'+x['id']).select('table')][:len(x.select('table'))] for x in old.select('section[id]') if x.select('table')))
j=json.loads((R/'analysis/seed_pipeline_cpu.json').read_text());b=json.loads((R/'analysis/seed_pipeline_source_binding.json').read_text());c=j['cases']
ok('Twelve original CPU controls and seven byte bindings valid',j['checks_passed']==12 and len(b['bindings'])==7 and all(x['same_bytes'] for x in b['bindings']) and all(hashlib.sha256((R/p).read_bytes()).hexdigest()==h for p,h in j['source_sha256'].items()))
svg=BeautifulSoup((R/'assets/seed_pipeline_keys.svg').read_text(),'html.parser').get_text()
ok('Key allocation figure uses actual baseline and future-front bindings',all('%s → [%s, %s]'%(name,key[0],key[1]) in svg for case in ['default','future_front'] for name,key in c[case]['child_keys'].items()) and c['future_front']['different_slots_vs_default']==92 and c['zero_front']['different_slots_vs_default']==c['future_tail']['different_slots_vs_default']==0)
ok('Both HTML variants embed accessible figure and current-slot distinction',all(x.select_one('#seed-pipeline-title') and '当前逐槽 A/B 域名序列' in x.select_one('#mixture-identity-guide').get_text() and not x.select('img[src="assets/seed_pipeline_keys.svg"]') for x in [now,stand]))
t=json.loads((R/'templates/data_seed_resume_review.json').read_text())
ok('Declared run and synthetic mapping not promoted to real token evidence',j['declared_run']['actual_executed_key'] is None and j['actual_Hero_next_batch_identity'] is None and j['actual_token_store'] is None and j['actual_causal_mixture_effect'] is None and t['status']=='proposed_not_executed' and t['before_and_after_domain_child_token_hashes'] is None)
(R/'analysis/static_validation_v108.json').write_text(json.dumps({'checks_passed':len(checks),'checks':checks,'actual_browser_rendering':None},indent=2)+'\n');print('Static checks',len(checks))
