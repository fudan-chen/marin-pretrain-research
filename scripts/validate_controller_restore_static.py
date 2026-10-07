"""Check V103 report structure, prior tables and recovery evidence bindings."""
import hashlib,json,pathlib,subprocess,xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
R=pathlib.Path(__file__).resolve().parents[1]
old=BeautifulSoup(subprocess.check_output(['git','show','report-v102-2026-10-07:index.html'],cwd=R),'html.parser');now=BeautifulSoup((R/'index.html').read_text(),'html.parser');stand=BeautifulSoup((R/'report_standalone.html').read_text(),'html.parser');checks=[]
def ok(name,value):
 assert value,name
 checks.append({'name':name,'passed':True})
g=now.select_one('#controller-recovery-guide');sg=stand.select_one('#controller-recovery-guide')
ok('Three recovery scene tables complete',len(g.select('table'))==3 and all(t.select('thead') and t.select('tbody') for t in g.select('table')))
ok('Both variants embed accessible diagram without external placeholder',all(x.select_one('svg title') and x.select_one('svg desc') and not x.select('img[src="assets/controller_restore_contract.svg"]') for x in [g,sg]))
sections=[x for x in old.select('section[id]') if x.select('table')]
ok('All V102 section tables retain ordered text',all(now.select_one('#'+x['id']) and [t.get_text(' ',strip=True) for t in x.select('table')]==[t.get_text(' ',strip=True) for t in now.select_one('#'+x['id']).select('table')] for x in sections))
j=json.loads((R/'analysis/controller_restore_native_tests.json').read_text());f=json.loads((R/'analysis/controller_restore_faults.json').read_text())
ok('51 original passes match two original JUnit records',j['passed']==51 and sum(len(list(ET.parse(R/'analysis'/('controller_restore_'+n+'.xml')).getroot().iter('testcase'))) for n in ['native','restart'])==51 and all(x['observed_tool_exitcode']==0 for x in j['runs']))
ok('Current source, fault script and native outputs match execution hashes',all(hashlib.sha256((R/p).read_bytes()).hexdigest()==h for p,h in {**j['source_sha256'],**j['artifact_sha256']}.items()) and hashlib.sha256((R/'scripts/probe_controller_restore_faults.py').read_bytes()).hexdigest()==f['executed_script_sha256'])
ok('Original file preservation and partial restore observations retained',f['checks_passed']==12 and all(f['cases']['ordinary_'+k]['original_file_bytes_preserved'] and not f['cases']['rollback_'+k]['local_main_exists'] for k in ['corrupt','missing']) and f['cases']['partial_download']['returned'] and not f['cases']['partial_download']['after_healthy'])
ok('Production, key impact and forced-kill recovery explicitly unknown',f['actual_production_incident'] is None and f['actual_auth_key_loss'] is None and j['actual_forced_process_kill_restore'] is None and j['actual_training_checkpoint_restore'] is None)
data={'checks_passed':len(checks),'checks':checks,'preserved_V102_sections_with_tables':len(sections),'actual_browser_rendering':None,'source_sha256':{'assets/controller_restore_contract.svg':hashlib.sha256((R/'assets/controller_restore_contract.svg').read_bytes()).hexdigest()}}
(R/'analysis/static_validation_v103.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');print('Static checks',len(checks))
