"""V109 unit correction, historical tables, plotted shares and backend scope."""
import hashlib,json,pathlib,subprocess,xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
R=pathlib.Path(__file__).resolve().parents[1];checks=[]
def ok(n,v):
 assert v,n
 checks.append({'name':n,'passed':True})
old=BeautifulSoup(subprocess.check_output(['git','show','report-v108-2026-10-07:index.html'],cwd=R),'html.parser');now=BeautifulSoup((R/'index.html').read_text(),'html.parser');stand=BeautifulSoup((R/'report_standalone.html').read_text(),'html.parser')
ok('All historical table content retained before appended accounting tables',all(now.select_one('#'+x['id']) and [t.get_text(' ',strip=True) for t in x.select('table')]==[t.get_text(' ',strip=True) for t in now.select_one('#'+x['id']).select('table')][:len(x.select('table'))] for x in old.select('section[id]') if x.select('table')))
j=json.loads((R/'analysis/loss_mass_cpu.json').read_text());b=json.loads((R/'analysis/loss_mass_source_binding.json').read_text())
ok('Thirteen controls and six complete public source bindings valid',j['checks_passed']==13 and len(b['bindings'])==6 and all(x['same_bytes'] and x['http_status']==200 and len((R/x['archive_file']).read_bytes())==x['bytes'] for x in b['bindings']) and all(hashlib.sha256((R/p).read_bytes()).hexdigest()==h for p,h in j['source_sha256'].items()))
c=j['cases'];q=j['integer_quotas'];fractions=lambda d:[d['A']/sum(d.values()),d['B']/sum(d.values())];values=fractions(q['equal'])+c['block8_equal_sequences']['weighted_target_share']+fractions(q['inverse_density'])+c['block8_inverse_density']['weighted_target_share'];svg=ET.parse(R/'assets/loss_mass_accounting.svg').getroot();bars=[x for x in svg.iter('{http://www.w3.org/2000/svg}rect') if x.get('fill') in ['#215f9a','#c77721']]
ok('Eight figure widths encode measured target shares and original quota',len(bars)==8 and all(abs(float(x.get('width'))-v*600)<.001 for x,v in zip(bars,values)))
ok('Both HTML variants preserve unit correction and accessible plot',all(x.select_one('#loss-mass-title') and 'V109修正了此前' in x.select_one('#data').get_text() and not x.select('img[src="assets/loss_mass_accounting.svg"]') for x in [now,stand]))
t=json.loads((R/'templates/domain_exposure_review.json').read_text())
ok('Actual Hero density kernel and learning benefit not promoted from fixture',j['actual_Hero_target_mass_by_domain'] is None and j['actual_Hero_gradient_share'] is None and j['actual_hardware_compute_cost'] is None and j['actual_xla_fast_bwd_execution'] is None and t['domain_rows']==[] and t['causal_mixture_effect'] is None)
(R/'analysis/static_validation_v109.json').write_text(json.dumps({'checks_passed':len(checks),'checks':checks,'actual_browser_rendering':None},indent=2)+'\n');print('Static checks',len(checks))
