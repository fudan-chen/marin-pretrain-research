# -*- coding: utf-8 -*-
"""Check source coverage, independently recompute key claims and inspect HTML links."""
import json,pathlib,re,statistics,hashlib,urllib.parse,csv,math,collections
import pyarrow.parquet as pq
from bs4 import BeautifulSoup
ROOT=pathlib.Path(__file__).resolve().parents[1];S=ROOT/'sources';A=ROOT/'analysis'
checks=[]
def ok(label,condition):
    if not condition:raise AssertionError(label)
    checks.append(label)
def read(p):return json.loads(p.read_text())
comments=read(S/'comments.json');issue=read(S/'issue.json')
ok('#8435 all 27 comments archived',len(comments)==issue['comments']==27)
translation=(ROOT/'ISSUE_8435_ZH.md').read_text()
ids=[int(x) for x in re.findall(r'^## C(\d+)\b',translation,re.M)]
ok('#8435 Chinese coverage 001 through 027',ids==list(range(1,28)))
for c in comments:ok('Translated source anchor '+str(c['id']),c['html_url'] in translation)
operations=(ROOT/'OPERATIONS_ZH.md').read_text();cs=read(S/'issue_8506_comments.json')
ok('#8506 all 58 production records indexed',len(cs)==read(S/'issue_8506.json')['comments']==58 and all(c['html_url'] in operations for c in cs))
spec=read(S/'harrier_spec_12d8b6f0.json');stock=spec['available_tokens']
ok('Stock has 200 buckets totaling 23.106103T',len(stock)==200 and sum(stock.values())==23106103007622)
meta=read(S/'wandb/hero-fa4sm100-nomask-step146k_meta.json');config=meta['config'];phases=config['data']['value']['train_weights']
trainer=config['trainer']['value']['trainer'];tpb=trainer['train_batch_size']*config['model']['value']['max_seq_len'];end=trainer['num_train_steps']
ok('4K snapshot and 46,137,344 tokens per update',config['model']['value']['max_seq_len']==4096 and tpb==46137344)
for s,w in phases:
    ws={k:v for k,v in w.items() if k in stock};ok('Phase weights normalized at step '+str(s),len(ws)==200 and abs(sum(ws.values())-1)<1e-9)
    ok('Validation weights zero at step '+str(s),all(v==0 for k,v in w.items() if k not in stock))
phase_ends=[phases[i+1][0] if i+1<len(phases) else end for i in range(len(phases))]
budgets=[(e-s)*tpb for (s,_),e in zip(phases,phase_ends)]
ok('18.005144633344T current planned budget',sum(budgets)==18005144633344)
cells=read(A/'cell_weights.json');ok('Cell table has exactly 200 unique buckets',len(cells)==200 and {c['cell'] for c in cells}==set(stock))
for c in cells:
    amount=sum(b*w[c['cell']] for b,(_,w) in zip(budgets,phases))
    ok('Exposure calculation '+c['cell'],abs(amount-c['planned_tokens'])<1e-4 and abs(amount/stock[c['cell']]-c['planned_epochs'])<1e-10)
ok('400 public examples preserved',sum(len(x['examples']) for x in read(A/'sample_texts.json'))==400)
summary=read(A/'summary.json')
for old,new,limit,label,delta in [('ep_control','ep_new',81916,'ep',-.0064983397722244264),('kernel_control','kernel_new',146339,'kernels',.00036009907722473145)]:
    def points(name):
        j=read(S/'wandb'/('window_'+name+'.json'));ix=next(i for i,s in enumerate(j['specs']) if s['keys'][-1]=='train/cross_entropy_loss')
        return {p['_step']:p['train/cross_entropy_loss'] for p in j['series'][ix] if p['_step']<limit}
    a,b=points(old),points(new);steps=sorted(a.keys()&b.keys())
    ok(label+' exact 200 contiguous paired steps',len(steps)==200 and steps==list(range(limit-200,limit)))
    ok(label+' independently verified CE mean delta',abs(statistics.mean(b[s]-a[s] for s in steps)-delta)<1e-12)
ok('No conflicting inherited measurements',not summary['duplicate_value_conflicts'])
known_ids=set()
for p in S.glob('*comments*.json'):
    j=read(p)
    if isinstance(j,list):known_ids.update(str(c['id']) for c in j if isinstance(c,dict) and 'id' in c)
for p in ROOT.glob('*_ZH.md'):
    used=set(re.findall(r'issuecomment-(\d+)',p.read_text()))
    ok('Cited comment IDs archived: '+p.name,used<=known_ids)
soup=BeautifulSoup((ROOT/'index.html').read_text(),'html.parser');all_ids=[x['id'] for x in soup.select('[id]')]
ok('HTML ids are unique',len(all_ids)==len(set(all_ids)))
missing=[]
for el,attr in [(x,'href') for x in soup.select('[href]')]+[(x,'src') for x in soup.select('[src]')]:
    target=el[attr];url=urllib.parse.urlsplit(target)
    if url.scheme or url.netloc:continue
    if not url.path and url.fragment:
        if urllib.parse.unquote(url.fragment) not in all_ids:missing.append(target)
    elif url.path and not (ROOT/urllib.parse.unquote(url.path)).exists():missing.append(target)
ok('HTML relative links and fragments resolve',not missing)
manifest=read(S/'archive_manifest.json')
ok('All archived file checksums match',all(hashlib.sha256((S/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in manifest['files']))
stand=(ROOT/'report_standalone.html').read_text()
figure_count=len(soup.select('img[src^="assets/"]'))
ok('Standalone embeds all twelve referenced figures',figure_count==12 and stand.count('src="data:image/png;base64,')==figure_count)
D=S/'deepening_2026_10_04';rows=pq.read_table(D/'hf_observations.parquet').to_pylist();swarm=read(A/'swarm_audit.json')
ok('Full pinned swarm has 934 distinct run names',len(rows)==len({r['run_name'] for r in rows})==934)
hashes={hashlib.sha256(json.dumps([r['phase0_weights'],r['phase1_weights']],sort_keys=True,separators=(',',':')).encode()).hexdigest() for r in rows}
ok('915 exact distinct weight pairs independently counted',len(hashes)==swarm['exact_distinct_weight_pairs']==915)
selected=read(S/'mix_study_swarm-2026.09.14.1_selected_runs.json');all_rows={r['run_name']:r for r in rows}
ok('Six selected runs exactly match full registry',all(all_rows[r['run_name']]==r for group in selected.values() for r in group))
paired={label:{int(re.search(r'seed(\d+)',r['run_name'])[1]):r for r in group} for label,group in selected.items()}
gsm_old=statistics.mean(paired['old'][i]['grouped_bpb']['logprob_gsm8k_5shot'] for i in range(3));gsm_new=statistics.mean(paired['new'][i]['grouped_bpb']['logprob_gsm8k_5shot'] for i in range(3))
ok('GSM8K BPB +10.8437 percent in all three seeds',abs(100*(gsm_new/gsm_old-1)-10.843711307791427)<1e-8 and all(paired['new'][i]['grouped_bpb']['logprob_gsm8k_5shot']>paired['old'][i]['grouped_bpb']['logprob_gsm8k_5shot'] for i in range(3)))
ok('54 task rows preserved in generated appendix',len(read(A/'paired_seed_metrics.json')['grouped_tasks_bpb'])==54 and all(k in (ROOT/'DEEP_DIVE_ZH.md').read_text() for k in selected['old'][0]['grouped_bpb']))
native=read(A/'loader_probe_python312.json');legacy=read(A/'loader_probe_python39.json');quant=list(csv.DictReader((A/'loader_quantization.csv').open()))
ok('Loader count probe executes production SHA arithmetic',native['python_version'].startswith('3.12.') and native['source_sha256']==hashlib.sha256((D/'mixture_production.py').read_bytes()).hexdigest())
for mode,probe in [('accurate',native),('legacy_sequential',legacy)]:
    for row in probe['rows']:
        expected={r['cell']:int(r['count_per_block']) for r in quant if r['sum_mode']==mode and int(r['block_size'])==row['block_size'] and int(r['phase'])==row['phase']}
        ok('Original arithmetic equals 200 computed counts '+mode+str((row['block_size'],row['phase'])),expected==row['counts'] and sum(expected.values())==row['block_size'])
ok('Cooldown old-runtime sensitivity not mistaken for 3.12',next(r for r in native['rows'] if r['phase']==2 and r['block_size']==49152)['remainder']==0 and next(r for r in legacy['rows'] if r['phase']==2 and r['block_size']==49152)['remainder']==183)
cursors=list(csv.DictReader((A/'cursor_audit.csv').open()))
for K,expected in [(49152,2951348224),(12288,108907479040)]:
    rr=[r for r in cursors if int(r['step'])==195125 and int(r['new_block_size'])==K];ds=[int(r['new_minus_old_tokens']) for r in rr]
    ok('Cursor conservation and exact L1 at K='+str(K),len(rr)==200 and sum(ds)==0 and sum(abs(d) for d in ds)==expected)
pr=read(A/'practical_audit.json');ab=list(csv.DictReader((A/'domain_ablation_audit.csv').open()));cfgs=list(csv.DictReader((A/'experiment_config_audit.csv').open()))
ok('80 domain interventions and 125 original configs archived',len(ab)==80 and len(cfgs)==pr['archived_configs']==125)
ok('All 80 recorded non-mixture settings match controls',pr['domain_control_config_matches']==80 and all(not r['controlled_config_differences'] for r in ab))
ok('125 original endpoint and weight cross-checks pass',all(abs(float(r['paloma_registry_minus_wandb']))<1e-12 and float(r['max_phase_weight_error'])<1.1e-8 for r in cfgs))
P=S/'practical_2026_10_04'
def raw_config(name):return json.loads(read(P/('config_'+name+'.json'))['data']['project']['run']['config'])
for r in ab:
    c,b=raw_config(r['run_name']),raw_config(r['control_run'])
    # Independently check critical settings directly in the source, rather than trusting stored hashes.
    ok('Raw control settings '+r['run_name'],c['model']==b['model'] and c['optimizer']==b['optimizer'] and c['resources']==b['resources'] and c['eval']==b['eval'] and c['trainer']['value']['trainer']['load_checkpoint_path'][-1]==b['trainer']['value']['trainer']['load_checkpoint_path'][-1])
q4=list(csv.DictReader((A/'math_q4_response.csv').open()))
ok('Only-Q4 Paloma regression and nonmonotone math response',any(r['run_name'].find('only-q4-')>=0 and abs(r['paloma_change_pct']-9.244000082388503)<1e-10 for r in pr['quality_ablations']) and float(q4[2]['gsm8k_bpb'])<float(q4[3]['gsm8k_bpb']))
orders=list(csv.DictReader((A/'order_budget_audit.csv').open()))
ok('Eight order comparisons preserve all three budget conventions',len(orders)==24 and len({r['pair'] for r in orders})==8)
ok('Order compensation and observed one-update mismatch are separated',all(float(r['l1_exposure_difference_tokens'])<1e-6 for r in orders if r['budget_convention']=='weight_compensation_2726_810_physical') and all(float(r['l1_exposure_difference_tokens'])>20000 for r in orders if r['budget_convention']=='observed_2727_810_physical'))
for short in ['r-stem-pre-high','r-stem-cool-high']:
    h=read(P/(short+'_step_windows.json'))['data']['project']['run']['sampledHistory'];pts=[json.loads(x) if isinstance(x,str) else x for x in h[0]]
    ok('Observed first resumed updates '+short,[x['_step'] for x in pts]==list(range(393,431)))
ok('80 full appendix rows and offline planner embedded',len(soup.select('#practical table')[-1].select('tbody tr'))==80 and 'MixPlanner.analyze' in stand and soup.select_one('#planner-example') is not None)
ok('Planner meaningful numeric checks passed',read(A/'planner_validation.json')['test_groups_passed']==12 and read(A/'planner_validation.json')['random_schedule_cases']==100)
# Fourth round: independently check endpoint, decomposition and frontier claims.
f=read(A/'findings_audit.json');PM='eval_dropless/paloma/macro_bpb';PC='eval_dropless/paloma/dolma_100_programing_languages-llama3/bpb';HE='logprob_humaneval_10shot';GM='logprob_gsm8k_5shot'
proportional={int(re.search(r'seed(\d+)',r['run_name'])[1]):r for r in rows if r['group']=='proportional_baseline'}
def metric_value(row,key):return row['training_eval_metrics' if key.startswith('eval') else 'grouped_bpb'][key]
def average(group,key):return statistics.mean(metric_value(group[i],key) for i in range(3))
ok('Strong baseline uses the same continuation seed labels 0 through 2',set(paired['new'])==set(paired['old'])=={0,1,2} and {0,1,2}<=set(proportional))
computed_gap=(average(paired['old'],PM)-average(proportional,PM))/(average(paired['old'],PM)-average(paired['new'],PM))
ok('92.052898 percent is the independently computed endpoint gap ratio',abs(computed_gap-f['proportional_fraction_of_observed_old_to_selected_macro_gap'])<1e-12 and abs(computed_gap-.920528980155)<1e-10)
ds=[metric_value(paired['new'][i],PM)-metric_value(proportional[i],PM) for i in range(3)]
half=4.302652729749462*statistics.stdev(ds)/math.sqrt(3)
ok('Three macro differences favor selected but conditional t interval crosses zero',all(d<0 for d in ds) and statistics.mean(ds)-half<0<statistics.mean(ds)+half and abs(f['three_seed_metrics'][PM]['conditional_t95_low']-(statistics.mean(ds)-half))<1e-12)
ok('Code and HumanEval improve in all three strong-baseline comparisons',all(metric_value(paired['new'][i],k)<metric_value(proportional[i],k) for k in [PC,HE] for i in range(3)))
ok('GSM8K has two regressions and one improvement against proportional',sum(metric_value(paired['new'][i],GM)>metric_value(proportional[i],GM) for i in range(3))==2)
comp=list(csv.DictReader((A/'strong_baseline_comparison.csv').open()));per_seed=list(csv.DictReader((A/'strong_baseline_paired.csv').open()))
for row in comp:
    key=row['metric'];p=average(proportional,key);n=average(paired['new'],key)
    ok('Strong baseline endpoint and every seed '+key,abs(float(row['proportional_mean'])-p)<1e-12 and abs(float(row['selected_mean'])-n)<1e-12 and abs(float(row['selected_vs_proportional_pct'])-100*(n/p-1))<1e-10 and all(abs(float(x['selected_minus_proportional'])-(metric_value(paired['new'][int(x['seed'])],key)-metric_value(proportional[int(x['seed'])],key)))<1e-12 for x in per_seed if x['metric']==key))
constituents=[k for k in paired['new'][0]['training_eval_metrics'] if k.startswith('eval_dropless/paloma/') and k.endswith('-llama3/bpb')]
ok('Macro uses exactly sixteen constituents and excludes token aggregate',len(constituents)==16 and 'eval_dropless/paloma/bpb' not in constituents and set(constituents)==set(f['sixteen_macro_constituents']))
ok('Every raw macro is recovered from actual sixteen-subset average',all(abs(statistics.mean(metric_value(g[i],k) for k in constituents)-metric_value(g[i],PM))<2e-7 for g in [paired['old'],paired['new'],proportional] for i in range(3)))
contributions=list(csv.DictReader((A/'macro_contributions.csv').open()))
ok('All sixteen contributions sum to raw constituent net change',len(contributions)==16 and all(abs(float(r['contribution_to_macro_bpb'])-(average(paired['new'],r['metric'])-average(proportional,r['metric']))/16)<1e-12 for r in contributions))
ok('Ten macro components improve and six regress',sum(average(paired['new'],k)<average(proportional,k) for k in constituents)==10)
ok('Noncode fifteen-subset mean slightly regresses',abs(100*(statistics.mean(average(paired['new'],k) for k in constituents if k!=PC)/statistics.mean(average(proportional,k) for k in constituents if k!=PC)-1)-f['noncode_selected_vs_proportional_pct'])<1e-10 and f['noncode_selected_vs_proportional_pct']>0)
candidates=[r for r in rows if r['group']=='mixprior_candidate' and re.search(r'seed(\d+)',r['run_name'])[1]=='0'];candidate_counts=collections.Counter(re.search(r'seed(\d+)',r['run_name'])[1] for r in rows if r['group']=='mixprior_candidate')
ok('Search is concentrated in 597 seed0, three seed1 and one seed2',dict(candidate_counts)=={'0':597,'1':3,'2':1} and len(candidates)==597)
frontier_rows=list(csv.DictReader((A/'observed_pareto_frontiers.csv').open()))
for objective,keys,expected,dominators in [('paloma_humaneval',[PM,HE],8,2),('paloma_gsm8k',[PM,GM],11,84),('paloma_humaneval_gsm8k',[PM,HE,GM],38,1)]:
    # Different implementation: explicit pair comparisons, not the producer's broadcast matrix.
    values={r['run_name']:tuple(metric_value(r,k) for k in keys) for r in candidates}
    counts={name:sum(all(x<=y for x,y in zip(other,v)) and any(x<y for x,y in zip(other,v)) for other in values.values()) for name,v in values.items()}
    ok('Independent observed frontier '+objective,sum(v==0 for v in counts.values())==expected and counts[paired['new'][0]['run_name']]==dominators and all(int(r['dominator_count'])==counts[r['run_name']] for r in frontier_rows if r['objective']==objective))
alt=all_rows[f['counterexample_run']]
ok('197c dominates selected seed0 on three axes but not all tasks',all(metric_value(alt,k)<metric_value(paired['new'][0],k) for k in [PM,HE,GM]) and sum(metric_value(alt,k)>metric_value(paired['new'][0],k) for k in alt['grouped_bpb'])==31 and sum(metric_value(alt,k)<metric_value(paired['new'][0],k) for k in alt['grouped_bpb'])==23)
ok('197c has no independent seed1 or seed2 counterpart in this registry',sum('197c9f5ceff6b9ee-' in r['run_name'] for r in rows)==1)
F=S/'findings_2026_10_04'
ok('Four new configurations and six resume windows archived',len(list(F.glob('config_*.json')))==4 and len(list(F.glob('window_*.json')))==6 and len(manifest['files'])==313)
for i,row in enumerate(f['configs']):
    def source_run(name):return read((F if 'mixprior-' in name else P)/('config_'+name+'.json'))['data']['project']['run']
    ra,rb=source_run(row['run_a']),source_run(row['run_b']);ca,cb=json.loads(ra['config']),json.loads(rb['config'])
    ok('New source critical settings and endpoints '+str(i),ca['model']==cb['model'] and ca['optimizer']==cb['optimizer'] and ca['resources']==cb['resources'] and ca['eval']==cb['eval'] and ca['trainer']['value']['trainer']['load_checkpoint_path'][-1]==cb['trainer']['value']['trainer']['load_checkpoint_path'][-1] and json.loads(ra['summaryMetrics'])[PM]==metric_value(all_rows[row['run_a']],PM) and json.loads(rb['summaryMetrics'])[PM]==metric_value(all_rows[row['run_b']],PM) and not row['differences_after_declared_path_and_weight_exclusions'])
for p in F.glob('window_*.json'):
    raw=read(p)['data']['project']['run']['sampledHistory'];h=[[json.loads(x) if isinstance(x,str) else x for x in series] for series in raw]
    ok('New source resume window '+p.stem,[x['_step'] for x in h[0]]==list(range(393,431)) and h[1][0]['throughput/total_tokens']==394*4194304)
effects=f['scale_effects'];ok('Proxy code gain does not keep its sign in every larger ladder',next(r['change_pct'] for r in effects if r['size']=='d512_proxy' and r['metric']==PC)<0 and next(r['change_pct'] for r in effects if r['size']=='d768')>0 and next(r['change_pct'] for r in effects if r['size']=='d1024')>0 and next(r['change_pct'] for r in effects if r['size']=='d1536')<0)
ok('Latest conclusions and all 54 task rows embedded',soup.select_one('main section.chapter')['id']=='workbench' and soup.select_one('#conclusions') is not None and len(soup.select('#conclusions table')[-1].select('tbody tr'))==54)
framework=read(A/'rubrics.json');workbench=read(A/'workbench_data.json');figure_notes=read(A/'figure_readings.json')
rule_ids={r['id'] for r in framework['rules']}
ok('All eighteen rubrics have explicit evidence anchors and provenance',rule_ids=={'R%02d'%i for i in range(1,19)} and len(framework['rules'])==18 and all(set(r['anchors'])=={'pass','partial','fail'} and r['question'] and r['why'] and r['evidence_links'] for r in framework['rules']))
ok('Seven pipeline stages refer only to declared required rules',[s['id'] for s in framework['pipeline']]==['P%d'%i for i in range(7)] and all(set(s['required'])<=rule_ids and s['outputs'] and s['next'] for s in framework['pipeline']))
for rule in framework['rules']:
    ok('Every rubric source exists '+rule['id'],all(urllib.parse.urlsplit(x['href']).scheme in ['http','https'] or (ROOT/x['href']).exists() for x in rule['evidence_links']))
ok('Rubric documentation contains every generated rule exactly once',re.findall(r'^### (R\d\d) ',(ROOT/'RUBRICS_ZH.md').read_text(),re.M)==['R%02d'%i for i in range(1,19)])
ok('Review example preserves unconfirmed capability and execution states',framework['example_audit']['judgments']['R07']['status']=='partial' and framework['example_audit']['judgments']['R12']['status']=='unassessed' and framework['example_audit']['judgments']['R14']['status']=='partial' and framework['example_audit']['judgments']['R15']['status']=='partial')
ok('Nine review input and arithmetic boundary groups passed',read(A/'review_validation.json')['test_groups_passed']==9)
ok('Five curve interpretation cases retain allowed and unsupported statements',len(workbench['cases'])==5 and all(c['allowed'] and c['forbidden'] and c['next'] and set(c['rules'])<=rule_ids for c in workbench['cases']))
ep_case=next(c for c in workbench['cases'] if c['id']=='ep');kernel_case=next(c for c in workbench['cases'] if c['id']=='kernel')
ok('Interpretation cases use actual paired-window CE values',format(summary['paired_windows']['ep']['train/cross_entropy_loss']['mean_new_minus_old'],'.7f') in ep_case['nodes'][-1]['text'] and format(summary['paired_windows']['kernels']['train/cross_entropy_loss']['mean_new_minus_old'],'+.7f') in kernel_case['nodes'][-1]['text'])
ok('Interpretation types do not present all arrows as observed causality',any(n['kind']=='mechanism' for c in workbench['cases'] for n in c['nodes']) and any(n['kind']=='hypothesis' for c in workbench['cases'] for n in c['nodes']) and '因果干预' in soup.select_one('#workbench').text)
for metric in workbench['metrics']:
    key=metric['key'];raw=[r for r in per_seed if r['metric']==key]
    ok('Interactive metric uses all original seed BPB values '+key,len(raw)==3 and [r['seed'] for r in metric['seeds']]==[0,1,2] and all(abs(metric['seeds'][int(r['seed'])][name]-float(r[name+'_bpb']))<1e-12 for r in raw for name in ['old','proportional','selected']) and all(abs(metric['means'][name]-statistics.mean(float(r[name+'_bpb']) for r in raw))<1e-12 for name in ['old','proportional','selected']))
ok('All twelve scientific figures have reading and original-value entries',len(figure_notes)==12 and {x['file'] for x in figure_notes}=={i['src'] for i in soup.select('img[src^="assets/"]')} and len(soup.select('.figure-reading'))==12 and all((ROOT/x['values']).exists() and (ROOT/x['chapter']).exists() and x['reading'] and x['boundary'] for x in figure_notes))
long_tables=[t for t in soup.select('table') if len(t.select('tbody tr'))>30]
ok('All five long tables are retained but initially collapsed',len(long_tables)==5 and all(t.find_parent('details',class_='large-table-details') is not None and not t.find_parent('details',class_='large-table-details').has_attr('open') for t in long_tables))
ok('Learning guide includes eighteen terms and five expandable exercises',len(soup.select('#learning table')[1].select('tbody tr'))==18 and len(soup.select('#learning details'))==5)
plan=read(ROOT/'templates/experiment_plan.json')
ok('Experiment template preserves planned status and unknown execution inputs',plan['status']=='planned_not_executed' and plan['scope']['checkpoint_content_digest'] is None and plan['budget']['token_budget'] is None and not plan['evidence'] and len(plan['comparisons'])==4)
browser=read(A/'browser_validation_v5.json')
ok('Browser review covers controls and preserves existing local data',all(browser['functional'][k] is True for k in ['fiveCases','blankEvidenceBlocked','conflictPreserved','invalidImportRetainsAudit','jsonRoundtrip','persistedAfterReload','originalLocalAuditRestored']) and not browser['errors'])
ok('Print reveals full tables then restores collapse state',browser['printTables']['count']==5 and browser['printTables']['allOpened'] and browser['printTables']['restored'])
ok('Offline workbench retains all images, chapters and no external requests',browser['standalone']['embeddedFigures']==browser['standalone']['loadedFigures']==12 and browser['standalone']['figureReadings']==12 and browser['standalone']['rubricCount']==18 and browser['standalone']['externalHttpRequests']==[])
ok('Narrow screen has no page-wide overflow',browser['mobile']['documentWidth']<=browser['mobile']['viewport'] and browser['mobile']['metricCards']==4 and browser['mobile']['metricSvgMaxWidth']<=browser['mobile']['viewport'])
report={'checks_passed':len(checks),'highlights':['27 comments translated and anchored','58 operations entries','200 buckets x 3 normalized phases','400 public examples','200-step EP and kernel paired windows','934 swarm observations / 915 exact weight pairs','54 task BPB comparisons','1200 counts match native 3.12 arithmetic and 1200 match legacy sensitivity','complete-block cursor differences independently totaled','80 domain interventions / 129 source configs','8 order comparisons / 3 token budget conventions','12 planner numeric test groups / 100 random schedules','strong proportional baseline / 16 macro contributions','597 seed0 endpoints / three independently checked observed frontiers','four new config contrasts / six step393 resume windows','twelve embedded figures','all relative HTML links valid','313 source archive checksums valid'],'not_verified':['GPU training/kernel reproduction','causality of individual mixture buckets','independent confirmation after candidate selection','535B task accuracy or long-context outcomes','full token ID / inner shuffle replay','all original experiment launch code SHAs and actual restored checkpoint contents','full selector objective, fitted model and hidden constraints']}
report['highlights']+=['18 authored rubrics / seven claim-stage decisions','five interpretation cases / four metrics with switchable baselines','12 figure-reading entries / five preserved collapsed tables','18 terms / five reading exercises','nine review logic and arithmetic checks','browser import/persistence/print and offline workbench checks']
report['not_verified']+=['external rubric scoring validity or actual reader comprehension study','truth of user-entered self-assessment evidence']
(A/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
