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
for p in S.rglob('*comments*.json'):
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
ok('Standalone embeds all seventeen referenced figures',figure_count==17 and stand.count('src="data:image/png;base64,')==figure_count)
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
ok('Four source configurations and six resume windows remain archived',len(list(F.glob('config_*.json')))==4 and len(list(F.glob('window_*.json')))==6 and sum(not x['file'].startswith(('decision_2026_10_04/','engineering_2026_10_05/','scale_2026_10_05/','execution_2026_10_05/','contracts_2026_10_05/','state_2026_10_05/','boundaries_2026_10_05/','cache_2026_10_05/','dedup_2026_10_05/','quality_2026_10_05/','routing_2026_10_05/','optimizer_2026_10_05/','short_conv_2026_10_05/','router_precision_2026_10_05/','eval_metrics_2026_10_05/','checkpoint_commit_2026_10_05/','checkpoint_memory_2026_10_05/','muon_geometry_2026_10_05/','adamh_2026_10_05/','muon_direction_2026_10_05/','watch_2026_10_06/','live_2026_10_06/','eval_identity_2026_10_06/','paloma_protocol_2026_10_06/','packing_2026_10_07/','live_2026_10_07/','accumulation_2026_10_07/')) for x in manifest['files'])==313)
for i,row in enumerate(f['configs']):
    def source_run(name):return read((F if 'mixprior-' in name else P)/('config_'+name+'.json'))['data']['project']['run']
    ra,rb=source_run(row['run_a']),source_run(row['run_b']);ca,cb=json.loads(ra['config']),json.loads(rb['config'])
    ok('New source critical settings and endpoints '+str(i),ca['model']==cb['model'] and ca['optimizer']==cb['optimizer'] and ca['resources']==cb['resources'] and ca['eval']==cb['eval'] and ca['trainer']['value']['trainer']['load_checkpoint_path'][-1]==cb['trainer']['value']['trainer']['load_checkpoint_path'][-1] and json.loads(ra['summaryMetrics'])[PM]==metric_value(all_rows[row['run_a']],PM) and json.loads(rb['summaryMetrics'])[PM]==metric_value(all_rows[row['run_b']],PM) and not row['differences_after_declared_path_and_weight_exclusions'])
for p in F.glob('window_*.json'):
    raw=read(p)['data']['project']['run']['sampledHistory'];h=[[json.loads(x) if isinstance(x,str) else x for x in series] for series in raw]
    ok('New source resume window '+p.stem,[x['_step'] for x in h[0]]==list(range(393,431)) and h[1][0]['throughput/total_tokens']==394*4194304)
effects=f['scale_effects'];ok('Proxy code gain does not keep its sign in every larger ladder',next(r['change_pct'] for r in effects if r['size']=='d512_proxy' and r['metric']==PC)<0 and next(r['change_pct'] for r in effects if r['size']=='d768')>0 and next(r['change_pct'] for r in effects if r['size']=='d1024')>0 and next(r['change_pct'] for r in effects if r['size']=='d1536')<0)
ok('After synthesis, engineering entry retains five earlier workbenches, conclusions and all 54 task rows',[s['id'] for s in soup.select('main > section.chapter')[1:7]]==['engineering-lab','assessment-lab','workbench','decision-lab','transfer-lab','order-lab'] and soup.select_one('#conclusions') is not None and len(soup.select('#conclusions table')[-1].select('tbody tr'))==54)
framework=read(A/'rubrics.json');workbench=read(A/'workbench_data.json');figure_notes=read(A/'figure_readings.json')
rule_ids={r['id'] for r in framework['rules']}
ok('All eighteen rubrics have explicit evidence anchors and provenance',rule_ids=={'R%02d'%i for i in range(1,19)} and len(framework['rules'])==18 and all(set(r['anchors'])=={'pass','partial','fail'} and r['question'] and r['why'] and r['evidence_links'] for r in framework['rules']))
ok('Seven pipeline stages refer only to declared required rules',[s['id'] for s in framework['pipeline']]==['P%d'%i for i in range(7)] and all(set(s['required'])<=rule_ids and s['outputs'] and s['next'] for s in framework['pipeline']))
for rule in framework['rules']:
    ok('Every rubric source exists '+rule['id'],all(urllib.parse.urlsplit(x['href']).scheme in ['http','https'] or (ROOT/x['href']).exists() for x in rule['evidence_links']))
ok('Rubric documentation contains every generated rule exactly once',re.findall(r'^### (R\d\d) ',(ROOT/'RUBRICS_ZH.md').read_text(),re.M)==['R%02d'%i for i in range(1,19)])
ok('Review example preserves unconfirmed capability and execution states',framework['example_audit']['judgments']['R07']['status']=='partial' and framework['example_audit']['judgments']['R12']['status']=='unassessed' and framework['example_audit']['judgments']['R14']['status']=='partial' and framework['example_audit']['judgments']['R15']['status']=='partial')
ok('Eleven review input, version migration and arithmetic boundary groups passed',read(A/'review_validation.json')['test_groups_passed']==11)
ok('Five curve interpretation cases retain allowed and unsupported statements',len(workbench['cases'])==5 and all(c['allowed'] and c['forbidden'] and c['next'] and set(c['rules'])<=rule_ids for c in workbench['cases']))
ep_case=next(c for c in workbench['cases'] if c['id']=='ep');kernel_case=next(c for c in workbench['cases'] if c['id']=='kernel')
ok('Interpretation cases use actual paired-window CE values',format(summary['paired_windows']['ep']['train/cross_entropy_loss']['mean_new_minus_old'],'.7f') in ep_case['nodes'][-1]['text'] and format(summary['paired_windows']['kernels']['train/cross_entropy_loss']['mean_new_minus_old'],'+.7f') in kernel_case['nodes'][-1]['text'])
ok('Interpretation types do not present all arrows as observed causality',any(n['kind']=='mechanism' for c in workbench['cases'] for n in c['nodes']) and any(n['kind']=='hypothesis' for c in workbench['cases'] for n in c['nodes']) and '因果干预' in soup.select_one('#workbench').text)
for metric in workbench['metrics']:
    key=metric['key'];raw=[r for r in per_seed if r['metric']==key]
    ok('Interactive metric uses all original seed BPB values '+key,len(raw)==3 and [r['seed'] for r in metric['seeds']]==[0,1,2] and all(abs(metric['seeds'][int(r['seed'])][name]-float(r[name+'_bpb']))<1e-12 for r in raw for name in ['old','proportional','selected']) and all(abs(metric['means'][name]-statistics.mean(float(r[name+'_bpb']) for r in raw))<1e-12 for name in ['old','proportional','selected']))
ok('All seventeen scientific figures have reading and original-value entries',len(figure_notes)==17 and {x['file'] for x in figure_notes}=={i['src'] for i in soup.select('img[src^="assets/"]')} and len(soup.select('.figure-reading'))==17 and all((ROOT/x['values']).exists() and (ROOT/x['chapter']).exists() and x['reading'] and x['boundary'] for x in figure_notes))
long_tables=[t for t in soup.select('table') if len(t.select('tbody tr'))>30]
ok('All five long tables are retained but initially collapsed',len(long_tables)==5 and all(t.find_parent('details',class_='large-table-details') is not None and not t.find_parent('details',class_='large-table-details').has_attr('open') for t in long_tables))
ok('Learning guide includes eighteen terms and five expandable exercises',len(soup.select('#learning table')[1].select('tbody tr'))==18 and len(soup.select('#learning details'))==5)
plan=read(ROOT/'templates/experiment_plan.json')
ok('Experiment template preserves planned status and unknown execution inputs',plan['status']=='planned_not_executed' and plan['scope']['checkpoint_content_digest'] is None and plan['budget']['token_budget'] is None and not plan['evidence'] and len(plan['comparisons'])==4)
decision=read(A/'decision_data.json');decision_checks=read(A/'decision_validation.json');trace=read(A/'selector_provenance_audit.json')
ok('597 seed0 candidates retain exact seven raw BPB metrics',len(decision['candidates'])==597 and len(decision['metrics'])==7 and all(all(metric_value(all_rows[r['run']],m['source_key'])==r['values'][m['id']] for m in decision['metrics']) for r in decision['candidates']))
ok('Only actual mixprior seed0 runs enter decision candidate set',{r['run'] for r in decision['candidates']}=={r['run_name'] for r in candidates})
for name,b in decision['baselines'].items():
    ok('Decision baseline retains original seed0 endpoints '+name,'seed0-' in b['run'] and all(metric_value(all_rows[b['run']],m['source_key'])==b['values'][m['id']] for m in decision['metrics']))
canonical={k:v for k,v in decision.items() if k!='dataset_sha256'}
ok('Decision source and canonical data digests match',hashlib.sha256((ROOT/decision['source']).read_bytes()).hexdigest()==decision['source_sha256'] and hashlib.sha256(json.dumps(canonical,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()==decision['dataset_sha256'])
ok('Nine decision boundary test groups pass without claiming training',decision_checks['test_groups_passed']==9 and decision_checks['not_verified'])
for scenario in decision_checks['scenarios']:
    b=decision['baselines'][scenario['baseline']]['values'];caps=scenario['caps'];primary=scenario['primary']
    eligible=[]
    for raw in candidates:
        v={m['id']:metric_value(raw,m['source_key']) for m in decision['metrics']}
        if all(100*(v[k]/b[k]-1)<=cap+1e-10 for k,cap in caps.items()):eligible.append((v[primary],re.search(r'mixprior-([0-9a-f]+)-',raw['run_name'])[1]))
    best=min(eligible)[1] if eligible else None
    ok('Independent hard-constraint intersection '+scenario['baseline']+' '+scenario['name'],len(eligible)==scenario['eligible'] and best==scenario['best'])
doc_tables=soup.select('#decision-guide table')
ok('Decision documentation preserves nine scenarios and six rule counterexamples',len(doc_tables)==3 and len(doc_tables[1].select('tbody tr'))==9 and len(doc_tables[2].select('tbody tr'))==6 and all(int(t.select('td')[2].text)==s['eligible'] and (s['best'] is None or s['best']==t.select('td')[3].text) for s,t in zip(decision_checks['scenarios'],doc_tables[1].select('tbody tr'))))
ok('Selector trace reports bounded search rather than universal absence',trace['status']=='not_recovered_in_checked_public_entries' and trace['tree_truncated'] is False and len(trace['limitations'])==4 and all(q['total_count']==q['items_returned']==1 and q['issue_numbers']==[9126] and q['incomplete_results'] is False for q in trace['issue_queries']) and trace['issue_9126_body_unchanged_from_prior_archive'])
ok('Five selector provenance files retain fixed revision in expanded archive',len(list((S/'decision_2026_10_04').glob('*.json')))==5 and len(manifest['files'])==443 and read(S/'decision_2026_10_04/marin_head.json')['sha']==read(S/'decision_2026_10_04/marin_tree.json')['sha']==trace['pinned_marin_revision'])
contract=read(ROOT/'templates/selection_contract.json')
ok('Confirmation contract cannot retroactively assert prior registration',contract['status']=='planned_not_executed' and contract['prior_search_results_already_seen'] is True and contract['contract_frozen_utc'] is None and contract['results'] is None and contract['independent_confirmation']['used_during_search'] is None)
embedded=json.loads(soup.select_one('#decision-data').text)
ok('Offline decision input embeds exact frozen candidate values',embedded==decision and soup.select_one('#decision-lab') is not None and 'exploratory_reanalysis_not_preregistered' in soup.text)
browser=read(A/'browser_validation_v5.json')
ok('Browser review covers controls and preserves existing local data',all(browser['functional'][k] is True for k in ['fiveCases','blankEvidenceBlocked','conflictPreserved','invalidImportRetainsAudit','jsonRoundtrip','persistedAfterReload','originalLocalAuditRestored']) and not browser['errors'])
ok('Print reveals full tables then restores collapse state',browser['printTables']['count']==5 and browser['printTables']['allOpened'] and browser['printTables']['restored'])
ok('Offline workbench retains all images, chapters and no external requests',browser['standalone']['embeddedFigures']==browser['standalone']['loadedFigures']==12 and browser['standalone']['figureReadings']==12 and browser['standalone']['rubricCount']==18 and browser['standalone']['externalHttpRequests']==[])
ok('Narrow screen has no page-wide overflow',browser['mobile']['documentWidth']<=browser['mobile']['viewport'] and browser['mobile']['metricCards']==4 and browser['mobile']['metricSvgMaxWidth']<=browser['mobile']['viewport'])
report={'checks_passed':len(checks),'highlights':['27 comments translated and anchored','58 operations entries','200 buckets x 3 normalized phases','400 public examples','200-step EP and kernel paired windows','934 swarm observations / 915 exact weight pairs','54 task BPB comparisons','1200 counts match native 3.12 arithmetic and 1200 match legacy sensitivity','complete-block cursor differences independently totaled','80 domain interventions / 129 source configs','8 order comparisons / 3 token budget conventions','12 planner numeric test groups / 100 random schedules','strong proportional baseline / 16 macro contributions','597 seed0 endpoints / three independently checked observed frontiers','four new config contrasts / six step393 resume windows','twelve embedded figures','all relative HTML links valid','313 source archive checksums valid'],'not_verified':['GPU training/kernel reproduction','causality of individual mixture buckets','independent confirmation after candidate selection','535B task accuracy or long-context outcomes','full token ID / inner shuffle replay','all original experiment launch code SHAs and actual restored checkpoint contents','full selector objective, fitted model and hidden constraints']}
report['highlights']+=['18 authored rubrics / seven claim-stage decisions','five interpretation cases / four metrics with switchable baselines','12 figure-reading entries / five preserved collapsed tables','18 terms / five reading exercises','nine review logic and arithmetic checks','browser import/persistence/print and offline workbench checks']
report['not_verified']+=['external rubric scoring validity or actual reader comprehension study','truth of user-entered self-assessment evidence']
browser6=read(A/'browser_validation_v6.json')
ok('Decision browser covers reference, constraints, infeasibility and invalid inputs',all(browser6['functional'][k] for k in ['nineScenariosMatch','infeasibleNotRelaxed','invalidInputBlocksOutput','jsonRetainsExploratoryState','completeCSVExport','existingLocalNotesPreserved']) and not browser6['errors'])
ok('Decision browser mobile and standalone retain full data without remote requests',browser6['mobile']['documentWidth']<=browser6['mobile']['viewport'] and browser6['standalone']['candidateCount']==597 and browser6['standalone']['externalHttpRequests']==[] and browser6['standalone']['embeddedFigures']==browser6['standalone']['loadedFigures']==12)
ok('Local zoom keeps all candidates in ranking and declares outside count',browser6['functional']['zoomPreservesRanking'] and browser6['zoom']['visible']+browser6['zoom']['outside']==597 and browser6['zoom']['markers']==browser6['zoom']['visible'] and browser6['mobile']['chartScrollWidth']>browser6['mobile']['chartContainerWidth'])
transfer=read(A/'transfer_data.json');tc=read(A/'transfer_validation.json')
ok('Transfer uses pinned source and embeds complete offline contracts',transfer['source_sha256']==decision['source_sha256'] and json.loads(soup.select_one('#transfer-data').text)==transfer and soup.select_one('#transfer-lab') is not None)
ok('Transfer contrast boundary checks preserve planned scope',tc['test_groups_passed']==13 and 'not model training' in tc['scope'])
tb=all_rows[decision['baselines']['selected']['run']];keys=sorted(tb['phase0_weights'])
ok('Transfer budgets distinguish nominal and physical demonstrations',transfer['budgets']['nominal']==[13125000000000,3750000000000] and transfer['budgets']['recorded_physical']==[11437867008,3397386240] and '未逐一证明四个候选实际读取量' in soup.text)
for profile in transfer['profiles']:
    raw=all_rows[profile['run']]
    ok('Transfer candidate retains both raw phases '+profile['id'],profile['weights']==[raw['phase0_weights'],raw['phase1_weights']] and profile['changed_phase_cells']==sum(any(raw['phase%d_weights'%i][k]!=tb['phase%d_weights'%i][k] for i in [0,1]) for k in keys))
    for convention,T in transfer['budgets'].items():
        delta={k:math.fsum(t*(raw['phase%d_weights'%i][k]-tb['phase%d_weights'%i][k]) for i,t in enumerate(T)) for k in keys}
        domains={c:math.fsum(delta[k] for k in keys if k.startswith('c%02dq'%c)) for c in range(40)}
        moved=math.fsum(map(abs,delta.values()))/2;cross=math.fsum(map(abs,domains.values()))/2
        phased=math.fsum(t*math.fsum(abs(raw['phase%d_weights'%i][k]-tb['phase%d_weights'%i][k]) for k in keys)/2 for i,t in enumerate(T));v=profile['comparisons'][convention]
        ok('Independent 200-bucket transfer arithmetic '+profile['id']+' '+convention,max(abs(r['delta_tokens']-delta[r['cell']]) for r in v['cells'])<.005 and max(abs(r['delta_tokens']-domains[r['id']]) for r in v['domains'])<.005 and abs(v['moved_tokens']-moved)<.005 and abs(v['cross_domain_net_tokens']-cross)<.005 and abs(v['within_domain_cancellation_tokens']-(moved-cross))<.005 and abs(v['temporal_cancellation_tokens']-(phased-moved))<.005)
ok('197c broad changes and decomposition remain explicit',transfer['profiles'][0]['changed_phase_cells']==195 and all(r['delta_tokens']!=0 for r in transfer['profiles'][0]['comparisons']['nominal']['domains']) and abs(transfer['profiles'][0]['comparisons']['nominal']['moved_tokens']/1e12-1.4728927612304688)<1e-12)
tm=transfer['math_timing'];r197=all_rows[transfer['profiles'][0]['run']];T=transfer['budgets']['nominal'];B=sum(T)
timed=[t*math.fsum(r197['phase%d_weights'%i][k]-tb['phase%d_weights'%i][k] for k in keys if k.startswith('c39q')) for i,t in enumerate(T)]
ok('Math timing residual is independently anchored to raw two-phase weights',max(abs(a-b) for a,b in zip(tm['stage_delta_tokens'],timed))<.005 and abs(tm['cumulative_delta_tokens']-sum(timed))<.005 and max(abs(tm['timing_residual_tokens'][i]-(timed[i]-T[i]/B*sum(timed))) for i in [0,1])<.005 and abs(tm['timing_residual_tokens'][0]/1e9-8.07020399305555)<1e-10)
csv_cells=list(csv.DictReader((A/'transfer_cell_differences.csv').open()));csv_domains=list(csv.DictReader((A/'transfer_domain_differences.csv').open()));original_rounding=list(csv.DictReader((A/'transfer_plan_quantization.csv').open()));fixed_rounding=list(csv.DictReader((A/'transfer_plan_integer_controlled.csv').open()))
expected_cells={(p['id'],convention,c['cell']):c for p in transfer['profiles'] for convention,v in p['comparisons'].items() for c in v['cells']}
ok('Transfer raw CSVs retain every profile and budget without filtering',len(csv_cells)==len({(r['candidate'],r['budget_convention'],r['cell']) for r in csv_cells})==1600 and len(csv_domains)==320 and len(original_rounding)==len(fixed_rounding)==2400 and all(float(r['delta_tokens'])==expected_cells[(r['candidate'],r['budget_convention'],r['cell'])]['delta_tokens'] for r in csv_cells))
probe=read(A/'transfer_loader_probe_python312.json');probe_rows={(r['donor'],r['arm'],r['phase']):r for r in probe['rows']};original_probe_rows={(r['donor'],r['arm'],r['phase']):r for r in probe['original_rows']}
probe_inputs=[{'donor':p['donor_domain'],'arms':[{'id':a['id'],'original':a['weights'],'integer_controlled':a['integer_controlled_weights']} for a in p['arms']]} for p in transfer['plans']]
ok('Transfer native count probe pins methods, inputs and Python3.12 source',probe['python_version'].startswith('3.12.') and probe['numpy_version']=='2.0.2' and probe['source_sha256']==hashlib.sha256((D/'mixture_production.py').read_bytes()).hexdigest() and probe['input_weights_sha256']==hashlib.sha256(json.dumps(probe_inputs,sort_keys=True,separators=(',',':')).encode()).hexdigest() and probe['source_methods']==['MixtureDataset._normalize_weights','MixtureDataset._compute_expected_counts_per_block'] and len(probe_rows)==len(original_probe_rows)==24)
baseline_counts=transfer['plans'][0]['arms'][0]['integer_repair']['target_phase_counts'];recipes=set()
csv_quantized={(int(r['donor']),r['arm'],r['cell']):r for r in original_rounding};csv_fixed={(int(r['donor']),r['arm'],r['cell']):r for r in fixed_rounding};csv_quantization_matches=[]
for p in transfer['plans']:
    file=read(ROOT/p['download']);embedded_plan={k:v for k,v in p.items() if k!='download'}
    ok('Prospective plan retains empty runtime and results c%d'%p['donor_domain'],file==embedded_plan and p['status']=='planned_not_executed' and p['launchable'] is False and all(p[k] is None for k in ['physical_training_budget','generation_results','checkpoint_content_digest','effective_inventory_and_history','floors_and_caps','independent_eval_manifest']) and p['block_units']=='sequences')
    for arm in p['arms']:
        recipes.add(json.dumps(arm['integer_repair']['target_phase_counts'],sort_keys=True))
        e={k:math.fsum(t*w[k] for t,w in zip(T,arm['weights'])) for k in keys};mathkeys=[k for k in keys if k.startswith('c39q')];donorkeys=[k for k in keys if k.startswith('c%02dq'%p['donor_domain'])]
        ok('Independent continuous contrast c%d %s'%(p['donor_domain'],arm['id']),abs(math.fsum(e[k] for k in mathkeys)-arm['target_receiver_tokens'])<.005 and max(abs(e[k]/arm['target_receiver_tokens']-arm['target_conditional_quality'][k]) for k in mathkeys)<1e-12 and all(arm['weights'][i][k]==tb['phase%d_weights'%i][k] for i in [0,1] for k in keys if k not in mathkeys+donorkeys))
        native_rows=[probe_rows[(p['donor_domain'],arm['id'],i)] for i in [0,1]];counts=arm['integer_repair']['target_phase_counts']
        ok('Native integer support and donor conservation c%d %s'%(p['donor_domain'],arm['id']),all(r['matches_target'] and r['counts']==counts[i] and sum(counts[i].values())==49152 and all(counts[i][k]==baseline_counts[i][k] for k in keys if k not in mathkeys+donorkeys) and sum(counts[i][k]-baseline_counts[i][k] for k in mathkeys)==61*arm['amount_factor'] and sum(counts[i][k]-baseline_counts[i][k] for k in donorkeys)==-61*arm['amount_factor'] for i,r in enumerate(native_rows)) and arm['integer_repair']['actual_finite_stream_verified'] is False and not arm['integer_repair']['extra_changed_cells_vs_baseline'])
        effective={k:math.fsum(T[i]*counts[i][k]/49152 for i in [0,1]) for k in keys}
        ok('Residual integer error remains reported c%d %s'%(p['donor_domain'],arm['id']),abs(math.fsum(abs(effective[k]-e[k]) for k in keys)-arm['integer_repair']['l1_nominal_exposure_error_tokens'])<.005 and abs(math.fsum(effective[k] for k in mathkeys)-arm['integer_repair']['receiver_tokens'])<.005 and (arm['id']=='M0Q0' or set(arm['full_block_audit']['extra_changed_cells_vs_baseline'])=={'c28q4','c30q4'}))
        original_counts=[original_probe_rows[(p['donor_domain'],arm['id'],i)]['counts'] for i in [0,1]];original_effective={k:math.fsum(T[i]*original_counts[i][k]/49152 for i in [0,1]) for k in keys}
        extras={k for k in keys if arm['constant_phase_delta'][k]==0 and any(original_counts[i][k]!=baseline_counts[i][k] for i in [0,1])}
        ok('Original global rounding leakage independently reproduced c%d %s'%(p['donor_domain'],arm['id']),original_counts==arm['full_block_audit']['phase_counts'] and all(original_probe_rows[(p['donor_domain'],arm['id'],i)]['matches_target'] for i in [0,1]) and extras==set(arm['full_block_audit']['extra_changed_cells_vs_baseline']) and abs(math.fsum(abs(original_effective[k]-e[k]) for k in keys)-arm['full_block_audit']['l1_nominal_exposure_error_tokens'])<.005)
        csv_quantization_matches.append(all(abs(float(csv_quantized[(p['donor_domain'],arm['id'],k)]['full_block_nominal_tokens'])-original_effective[k])<.005 and abs(float(csv_fixed[(p['donor_domain'],arm['id'],k)]['integer_controlled_nominal_tokens'])-effective[k])<.005 and [int(csv_fixed[(p['donor_domain'],arm['id'],k)]['phase%d_count'%i]) for i in [0,1]]==[counts[i][k] for i in [0,1]] for k in keys))
ok('Both full 2400-row rounding CSVs match independent native arithmetic',len(csv_quantized)==len(csv_fixed)==2400 and all(csv_quantization_matches))
ok('Twelve arm records contain eight distinct integer recipes',sum(len(p['arms']) for p in transfer['plans'])==12 and len(recipes)==transfer['unique_integer_recipes']==8)
browser7=read(A/'browser_validation_v7.json')
ok('Transfer browser exercises budget, domain, donor and exports',all(browser7['functional'][k] for k in ['eightProfileBudgetCasesMatch','fiveQualityRowsMatch','threePlansMatch','exportRetainsPlannedNullState','existingLocalDataPreserved']) and not browser7['errors'])
ok('Transfer print and offline views retain complete values',browser7['printTables']['count']==6 and browser7['printTables']['allOpened'] and browser7['printTables']['restored'] and browser7['standalone']['embeddedFigures']==browser7['standalone']['loadedFigures']==13 and browser7['standalone']['figureReadings']==13 and browser7['standalone']['externalHttpRequests']==[] and browser7['standalone']['profileCount']==4 and browser7['standalone']['armRecords']==12)
ok('Transfer mobile width is bounded and tables scroll',browser7['mobile']['documentWidth']<=browser7['mobile']['viewport'] and browser7['mobile']['tableScrollWidth']>browser7['mobile']['tableContainerWidth'])
report['checks_passed']=len(checks)
report['highlights']=[x.replace('313 source archive checksums valid','318 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['597 seed0 candidates / seven exact BPB inputs / nine constrained-selection scenarios','bounded selector provenance audit / five new source files','six rubric counterexample replays / confirmation selection contract','nine decision boundary test groups / separate raw-source recomputation','decision browser baseline, constraint, invalid-input, CSV and offline checks']
report['highlights']=[x.replace('twelve embedded figures','thirteen embedded figures').replace('12 figure-reading entries','13 figure-reading entries') for x in report['highlights']]
report['highlights']+=['four candidate budget decompositions / 1600 bucket and 320 domain rows','math amount and timing separated / explicit donor-factorial plans','13 transfer boundary groups / independent raw-weight arithmetic','12 planned arm records / eight unique integer recipes / 24 repaired and 24 original native loader counts','original global-rounding leakage preserved / declared-domain integer repair','transfer browser budget, donor, export, six print tables and offline checks']
report['not_verified']+=['prospective factorial GPU outcomes and actual finite data stream','portable integer weight encoding across other runtimes or block sizes']
order=read(A/'order_workbench_data.json');orderprobe=read(A/'order_loader_probe_python312.json');order_checks=read(A/'order_validation.json');order_js=read(A/'order_js_validation.json')
ok('Order input embeds exact archived audit and planned recipes',json.loads(soup.select_one('#order-data').text)==order and len(order['historical_pairs'])==8 and len(order['historical_runs'])==16 and soup.select_one('#order-lab') is not None)
ok('Order tests cover partial windows, count lattice and immutable exports',order_checks['test_groups_passed']==12 and order_checks['amplitudes_checked']==20 and order_js['test_groups_passed']==7 and order_js['mode_amplitude_cases']==40)
ok('Order probe executes six archived methods in pinned runtime',orderprobe['source_sha256']==hashlib.sha256((D/'mixture_production.py').read_bytes()).hexdigest() and orderprobe['python_version'].startswith('3.12.') and orderprobe['numpy_version']=='2.0.2' and len(orderprobe['source_methods'])==6 and len(orderprobe['historical_rows'])==48 and len(orderprobe['compiled_rows'])==480 and len(orderprobe['edge_proofs'])==20)
raw_counts={(r['run'],r['phase']):r['counts'] for r in orderprobe['historical_rows']}
for run,r in order['historical_runs'].items():
    cfg=json.loads(read(ROOT/r['source'])['data']['project']['run']['config']);w=cfg['data']['value']['train_weights']
    ok('Historical order inputs retain source, boundaries and native counts '+run,r['source_sha256']==hashlib.sha256((ROOT/r['source']).read_bytes()).hexdigest() and r['phase_weights']==[{k:v for k,v in p.items() if k in stock} for _,p in w] and cfg['data']['value']['mixture_block_size']==49152 and cfg['trainer']['value']['trainer']['train_batch_size']==1024 and [s*1024 for s,_ in w]==r['sequence_boundaries']==[0,393216,3194880] and all(raw_counts[(run,i)]==r['expected_counts'][i] for i in range(3)))
order_csv=list(csv.DictReader((A/'order_integer_cell_audit.csv').open()));order_csv_rows={(r['pair'],r['cell']):r for r in order_csv}
for p in order['historical_pairs']:
    a=[raw_counts[(p['run_a'],i)] for i in range(3)];b=[raw_counts[(p['run_b'],i)] for i in range(3)]
    full={k:56*(b[1][k]-a[1][k])+16*(b[2][k]-a[2][k]) for k in stock}
    ok('Independent complete-block order residual '+p['pair'],full==p['complete_block_difference_sequences'] and sum(abs(v) for v in full.values())*4096==p['complete_block_l1_tokens'] and sum(v for k,v in full.items() if k.startswith('c39q'))*4096==p['complete_block_math_net_tokens'] and p['whole_window_exact_difference_tokens'] is None and [x['complete_blocks'] for x in p['window_ledger']]==[56,16])
    lo=full.copy();hi=full.copy()
    for phase,read_sequences in [(1,39936),(2,43008)]:
        for k in stock:
            lo[k]+=max(0,b[phase][k]-(49152-read_sequences))-min(a[phase][k],read_sequences)
            hi[k]+=min(b[phase][k],read_sequences)-max(0,a[phase][k]-(49152-read_sequences))
    ok('Partial-bound audit remains a loose interval '+p['pair'],all(p['whole_window_difference_bounds_sequences'][k]==[lo[k],hi[k]] and int(order_csv_rows[(p['pair'],k)]['complete_block_delta_sequences'])==full[k] and int(order_csv_rows[(p['pair'],k)]['whole_window_delta_lower_sequences'])==lo[k] and int(order_csv_rows[(p['pair'],k)]['whole_window_delta_upper_sequences'])==hi[k] for k in stock))
ok('Every order CSV pair retains all 200 buckets',len(order_csv)==len(order_csv_rows)==1600)
from order_core import compile_order
native_recipes={(r['k'],r['lock_edges'],r['arm'],r['stage']):r for r in orderprobe['compiled_rows']};recipe_inputs=[]
for k in range(20):
    for lock in [True,False]:
        p=compile_order(order['baseline_counts'],order['initial_counts'],49152,k,lock_edges=lock)
        recipe_inputs.append({'k':k,'lock_edges':lock,'arms':[{'id':a['id'],'stages':a['stages']} for a in p['arms']]})
        independent=[]
        for a in p['arms']:
            e0=order['baseline_counts'][0].copy();e1=order['baseline_counts'][1].copy();n=a['sign']*k
            e0['c39q4']+=2*n;e0['c26q4']-=2*n;e1['c39q4']-=7*n;e1['c26q4']+=7*n
            targets=[order['initial_counts'],order['baseline_counts'][0],e0,e1,order['baseline_counts'][1]] if lock else [order['initial_counts'],e0,e1]
            independent.append(a['interior_counts']==[e0,e1] and all(native_recipes[(k,lock,a['id'],i)]['matches_target'] and native_recipes[(k,lock,a['id'],i)]['count_sha256']==hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest() and c==a['stages'][i]['counts'] and sum(c.values())==49152 and min(c.values())>=0 for i,c in enumerate(targets)) and all(56*(e0[c]-order['baseline_counts'][0][c])+16*(e1[c]-order['baseline_counts'][1][c])==0 for c in stock))
        ok('Independent integer schedule and native counts k%d locked=%s'%(k,lock),all(independent))
ok('Native schedule input digest matches every current encoded recipe',orderprobe['input_recipes_sha256']==hashlib.sha256(json.dumps(recipe_inputs,sort_keys=True,separators=(',',':')).encode()).hexdigest())
ok('All 20 conditional proofs retain identical outer arrays and cursors',[p['k'] for p in orderprobe['edge_proofs']]==list(range(20)) and all(p['interior_counts_exact'] and [e['block'] for e in p['edge_blocks']]==[8,81] and all(e['arrays_and_cursors_identical'] for e in p['edge_blocks']) and 'actual token stream unverified' in p['scope'] for p in orderprobe['edge_proofs']))
ok('Unlocked k10 diagnosis preserves explicit counterexample scope',[r['l1_difference_sequences'] for r in orderprobe['unlocked_debug_k10']]==[0,140,140] and all('randomize_blocks=False' in r['scope'] for r in orderprobe['unlocked_debug_k10']))
for file,field in [('templates/order_edge_locked.json','locked_default'),('templates/order_edges_changed_counterexample.json','unlocked_default')]:
    p=read(ROOT/file)
    ok('Order contract preserves unexecuted state '+field,p==order[field] and p['physical_budget_tokens']==14835253248 and p['sequence_interval']==[402432,4024320] and p['status']=='planned_not_executed' and p['launchable'] is False and p['actual_token_stream_verified'] is False and all(p[n] is None for n in ['actual_checkpoint_digest','actual_data_cursor','actual_shuffle_keys','token_store_manifest','independent_eval_manifest','generation_results']))
ok('Order baseline stays tied to unconfirmed V7 integer recipe',order['baseline_counts']==transfer['plans'][0]['arms'][-1]['integer_repair']['target_phase_counts'] and all(order['initial_counts']==raw_counts[(run,0)] for run in order['historical_runs']) and order['locked_default']['capacity']=={'early_units_per_k':2,'late_units_per_k':7,'forward_max_k':19,'reverse_max_k':56,'symmetric_max_k':19})
browser8=read(A/'browser_validation_v8.json')
ok('Order browser covers every history, boundary and count export',all(browser8['functional'][k] for k in ['eightHistoricalPairsMatch','fortyModeAmplitudeCasesMatch','unlockedCounterexampleVisible','invalidAmplitudeBlocksOutput','exportIsPlannedCountsOnly','existingLocalDataPreserved']) and not browser8['errors'])
ok('Order standalone and mobile retain complete local evidence',browser8['standalone']['embeddedFigures']==browser8['standalone']['loadedFigures']==14 and browser8['standalone']['figureReadings']==14 and browser8['standalone']['externalHttpRequests']==[] and browser8['standalone']['historicalPairs']==8 and browser8['mobile']['documentWidth']<=browser8['mobile']['viewport'] and browser8['mobile']['tableScrollWidth']>browser8['mobile']['tableContainerWidth'])
report['highlights']=[x.replace('thirteen embedded figures','fourteen embedded figures').replace('13 figure-reading entries','14 figure-reading entries') for x in report['highlights']]
report['highlights']+=['eight order comparisons / complete-block residuals and partial-edge bounds','48 historical and 480 compiled archived stage-count evaluations','20 conditional edge/index proofs / all 40 JS mode-amplitude cases','same original window budget with locked edge blocks / planned count-only exports','unlocked-edge diagnostic counterexample / separate actual-token-stream unknowns']
# Reader protocol checks cover authored inputs and tool state, not comprehension validity.
bank=read(A/'assessment_data.json');prior_rules=read(A/'rubrics_1_0_reference.json')
ok('Only R16 changes anchors between author framework 1.0 and 1.1',framework['version']=='1.1' and prior_rules['version']=='1.0' and framework['pipeline']==prior_rules['pipeline'] and framework['changes_from_1_0']['changed_rules']==['R16'] and all(r==old for r,old in zip(framework['rules'],prior_rules['rules']) if r['id']!='R16') and framework['rules'][15]!=prior_rules['rules'][15])
scope_audit=read(A/'r16_scope_audit.json')
ok('R16 scope audit preserves old anchor and absence of measured reader outcomes',scope_audit['prior_R16']==prior_rules['rules'][15] and scope_audit['current_R16']==framework['rules'][15] and scope_audit['unchanged_rule_count']==17 and scope_audit['pipeline_unchanged'] and scope_audit['concrete_evidence']['external_reader_study'] is None)
ok('R16 example claims figure preparation only, not verified comprehension','图表准备' in framework['example_audit']['scope'] and '未验证' in framework['example_audit']['judgments']['R16']['evidence'])
ok('Assessment bank has ten distinct cases and three separate dimensions',[c['id'] for c in bank['cases']]==['Q%02d'%i for i in range(1,11)] and [d['id'] for d in bank['dimensions']]==['scope','mechanism','next'] and bank['protocol_version']=='0.1' and bank['rubric_framework_version']=='1.1')
unsigned={k:v for k,v in bank.items() if k!='bank_sha256'}
ok('Question and provenance digest matches exact authored bank',bank['bank_sha256']==hashlib.sha256(json.dumps(unsigned,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest())
for f,digest in bank['source_sha256'].items():
    ok('Assessment source is pinned '+f,(ROOT/f).is_file() and hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==digest)
for c in bank['cases']:
    ok('Assessment case retains allowed claim, contrast, source and anchors '+c['id'],c['allowed_claim'] and c['explanation'] and c['next_evidence'] and set(c['anchors'])=={'scope','mechanism','next'} and all(set(x)=={'0','1','2'} and all(x.values()) for x in c['anchors'].values()) and c['critical_errors'] and len({e['id'] for e in c['critical_errors']})==len(c['critical_errors']) and set(c['rules'])<=rule_ids and all(l['href'] in bank['source_sha256'] for l in c['links']) and (ROOT/c['figure']).is_file() and len(soup.select('#figure-'+pathlib.Path(c['figure']).stem))==1)
ok('Embedded question bank is exactly the standalone authored input',json.loads(soup.select_one('#assessment-data').text)==bank and json.loads(BeautifulSoup(stand,'html.parser').select_one('#assessment-data').text)==bank)
he_rows=[r for r in csv.DictReader((A/'strong_baseline_paired.csv').open()) if r['metric']=='logprob_humaneval_10shot']
he_change=100*(sum(float(r['selected_bpb']) for r in he_rows)/sum(float(r['proportional_bpb']) for r in he_rows)-1)
ok('HumanEval question uses ratio of three means and identifies proxy metric',f'{he_change:.2f}%' in bank['cases'][1]['prompt'] and 'd512' in bank['cases'][1]['prompt'] and 'BPB' in bank['cases'][1]['prompt'])
ok('Search and order prompts match independently audited input sizes',f"{len(decision['candidates'])}个" in bank['cases'][3]['prompt'] and '56/16' in bank['cases'][8]['prompt'] and '480' in bank['cases'][9]['prompt'] and len(orderprobe['compiled_rows'])==480 and len(orderprobe['edge_proofs'])==20)
replay=read(A/'assessment_authored_replays.json')
ok('Thirty calibration answers are explicitly authored rather than reader outcomes',replay['bank_sha256']==bank['bank_sha256'] and len(replay['examples'])==30 and all(sum(e['case_id']==c['id'] for e in replay['examples'])==3 for c in bank['cases']) and all('not a reader observation' in e['provenance'] for e in replay['examples']))
ok('Reader and scoring validity outcomes remain absent',bank['external_readers'] is None and bank['external_scoring_validity'] is None and read(A/'assessment_validation.json')['test_groups_passed']==13 and read(A/'assessment_validation.json')['authored_replay_count']==30)
blank=read(ROOT/'templates/reader_assessment_blank.json')
ok('Blank reader template has no observations or inherited ratings',blank['bank_sha256']==bank['bank_sha256'] and blank['record_kind']=='local_unverified_reader_record' and all(not r['draft'] and r['reference_first_opened_at'] is None and not r['revisions'] for r in blank['cases'].values()))
browser9=read(A/'browser_validation_v9.json');bank9=read(A/'assessment_casebank_v9_reference.json')
ok('Historical V9 browser pins its own frozen bank and twelve behaviors',browser9['bank_sha256']==bank9['bank_sha256']==read(A/'release_v9.json')['casebank_sha256'] and len(browser9['functional'])==12 and all(browser9['functional'].values()) and not browser9['errors'])
ok('Historical question bank is the exact V9 released file',hashlib.sha256((A/'assessment_casebank_v9_reference.json').read_bytes()).hexdigest()==read(A/'release_v9.json')['artifacts']['analysis/assessment_data.json'])
ok('Historical migration checks old R16 fields before resetting anchor',browser9['migration_format_check']['invalidOldR16CannotBeHiddenByMigration'] and browser9['migration_format_check']['localDataRestored'] and browser9['migration_format_check']['bank_sha256']==bank9['bank_sha256'])
ok('Reader button export preserves unknown comprehension and training decision',browser9['exportSummary']['overallScore'] is None and browser9['exportSummary']['externalReaderStudy'] is None and browser9['exportSummary']['trainingGateDecision'] is None and browser9['exportSummary']['reviewEvents']==3 and browser9['exportSummary']['answerRevisions']==2)
ok('Historical V9 standalone used file protocol, ten questions and no HTTP requests',browser9['standalone']['protocol']=='file:' and browser9['standalone']['questionCount']==10 and browser9['standalone']['embeddedFigures']==browser9['standalone']['loadedFigures']==14 and browser9['standalone']['figureReadings']==14 and browser9['standalone']['controlsUpdated'] is True and browser9['standalone']['externalHttpRequests']==[] and browser9['standalone']['errors']==[])
ok('Reader tables scroll inside mobile container and desktop stays within viewport',browser9['mobile']['documentWidth']<=browser9['mobile']['viewport'] and browser9['mobile']['tableScrollWidth']>browser9['mobile']['tableContainerWidth'] and browser9['desktop']['documentWidth']<=browser9['desktop']['viewport'])
report['highlights']=[x.replace('nine review logic and arithmetic checks','eleven review logic, migration and arithmetic checks') for x in report['highlights']]
report['highlights']+=['R16 1.1 scope correction / seventeen unchanged anchors / explicit old-version migration','ten source-bound comprehension cases / thirty authored calibration examples','thirteen answer-revision, critical-error and reviewer-disagreement boundary groups','file-protocol offline reader workbench / actual Blob export / local data restored']
report['not_verified']+=['external reviewer identity, scoring agreement and transfer-task comprehension']
# Current engineering source bindings and actual wording repairs.
engineering=read(A/'engineering_data.json');refresh=read(A/'engineering_refresh_audit.json')
ok('Engineering atlas has eight episodes and twenty-four authored claims',len(engineering['episodes'])==8 and sum(len(e['claims']) for e in engineering['episodes'])==24 and engineering['actual_cluster_test'] is None and engineering['external_reader_study'] is None)
unsigned={k:v for k,v in engineering.items() if k!='atlas_sha256'}
ok('Atlas digest binds all claims, tests and provenance',engineering['atlas_sha256']==hashlib.sha256(json.dumps(unsigned,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest())
for f,digest in engineering['source_sha256'].items():
    ok('Engineering source bytes pinned '+f,hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==digest)
for e in engineering['episodes']:
    ok('Engineering case retains observed/unknown/test scopes '+e['id'],all(e[k] for k in ['observation','mechanism','action','outcome','unknown']) and all(e['next_test'][k] for k in ['plan','prediction','falsifier','boundary']) and set(e['rules'])<=rule_ids and len(e['claims'])==3 and all(c['status'] in ['supported','unknown','refuted'] and c['reason'] for c in e['claims']) and all(0<=v['ref_index']<len(e['refs']) for v in e['events']))
    for ref in e['refs']:
        original=read(ROOT/ref['file'])
        if 'comment_id' in ref:
            comment=next(c for c in original if c['id']==ref['comment_id'])
            ok('Comment anchor/time/body exists '+e['id']+'/'+str(ref['comment_id']),ref['needle'] in comment['body'] and ref['url']==comment['html_url'] and ref['created_utc']==comment['created_at'])
        else:
            pr=read(ROOT/ref['metadata_file'])
            ok('PR patch and exact revision exists '+e['id']+'/'+str(ref['pull_number']),any(ref['needle'] in f.get('patch','') for f in original) and ref['head_sha']==pr['head']['sha'] and ref['merge_sha']==pr['merge_commit_sha'])
ok('New public issue snapshot is complete and unchanged in comment bodies',refresh['new_acquisition_files']==12 and refresh['pagination_complete'] and all(not c['added_ids'] and not c['removed_ids'] and not c['changed_bodies'] for c in refresh['comment_comparisons']) and all(not c['changed'] for c in refresh['issue_body_changes']) and [c['new_count'] for c in refresh['comment_comparisons']]==[27,58,30])
prior_source_rows=refresh['prior_integrity_manifest']['files']
ok('All 317 old non-acquisition-manifest files retain exact bytes',len(prior_source_rows)==318 and sum(r['file']!='source_manifest.json' for r in prior_source_rows)==317 and all(hashlib.sha256((S/r['file']).read_bytes()).hexdigest()==r['sha256'] for r in prior_source_rows if r['file']!='source_manifest.json'))
for number in [9062,9183,9333]:
    pr=read(S/'engineering_2026_10_05'/('pull_%d.json'%number));files=read(S/'engineering_2026_10_05'/('pull_%d_files.json'%number))
    ok('All changed PR files archived '+str(number),pr['changed_files']==len(files) and all(f.get('patch') for f in files))
scope_edits=read(A/'engineering_scope_audit.json');report_text=(ROOT/'REPORT_ZH.md').read_text()
ok('All eleven semantic edits are present and old claims absent',len(scope_edits['changes'])==11 and all(c['after'] in (ROOT/c['file']).read_text() and c['before'] not in (ROOT/c['file']).read_text() for c in scope_edits['changes']))
ok('R13 applied without modifying framework anchors',engineering['rubric_framework_version']==framework['version']=='1.1' and 'R13' in scope_edits['rules_applied'] and scope_edits['independent_input_token_replay'] is None and scope_edits['component_ablation'] is None)
ok('Question wording and scoring anchors unchanged despite corrected source prose',bank['cases']==bank9['cases'] and bank['dimensions']==bank9['dimensions'] and bank['protocol_version']==bank9['protocol_version'] and bank['bank_sha256']!=bank9['bank_sha256'] and bank['source_sha256']['REPORT_ZH.md']!=bank9['source_sha256']['REPORT_ZH.md'])
ok('Both HTML variants embed exact current engineering atlas',json.loads(soup.select_one('#engineering-data').text)==engineering and json.loads(BeautifulSoup(stand,'html.parser').select_one('#engineering-data').text)==engineering)
# Independent arithmetic: do not confuse duration reduction with reciprocal speedup.
step_drop=100*(1-14.67/16.29);iteration_drop=100*(1-15.53/18.62)
step_speed=100*(16.29/14.67-1);iteration_speed=100*(18.62/15.53-1)
guide=(ROOT/'ENGINEERING_GUIDE_ZH.md').read_text()
ok('Step/iteration arithmetic preserves both denominators',all('%.2f%%'%v in guide for v in [step_drop,iteration_drop,step_speed,iteration_speed]))
browser10=read(A/'browser_validation_v10.json');bank10=read(A/'assessment_casebank_v10_reference.json')
expected_engineering_behaviors={'allEightCasesAndTwentyFourClaims','correctSourceLinks','quotaAndLaterStallRemainSeparate','caseNotesIsolatedAndTextEscaped','actualExportBlobPreservesUnknowns','oldBankNotSilentlyImported','tenQuestionAnchorsUnchanged','existingLocalStorageUnchanged','transientNotesDoNotPretendPersisted','offlineSelectorsAndEmbeddedInputs','actualOldBankImportRejectsWithoutOverwrite'}
ok('Historical V10 browser checks V10 bank and all eleven engineering behaviors',browser10['bank_sha256']==bank10['bank_sha256'] and browser10['atlas_sha256']==engineering['atlas_sha256'] and set(browser10['functional'])==expected_engineering_behaviors and all(browser10['functional'].values()) and not browser10['errors'])
ok('Historical V10 offline engineering page remains self-contained',browser10['standalone']['protocol']=='file:' and browser10['standalone']['episodeCount']==8 and browser10['standalone']['questionCount']==10 and browser10['standalone']['loadedFigures']==14 and browser10['standalone']['httpRequests']==[] and browser10['standalone']['errors']==[])
ok('Historical V10 engineering mobile and desktop fit their viewports',browser10['desktop']['documentWidth']<=browser10['desktop']['viewport'] and browser10['mobile']['documentWidth']<=browser10['mobile']['viewport'])
ok('Actual engineering button blob has no executed result or gate decision',browser10['export']['schema']=='marin-engineering-draft/1' and browser10['export']['execution_status']=='not_executed' and browser10['export']['reader_understanding'] is None and browser10['export']['training_gate_decision'] is None)
report['highlights']=[x.replace('318 source archive checksums valid','330 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['eight source-bound engineering timelines / twenty-four authored claim contrasts','eleven actual wording repairs / quota and post-cleanup stall separated','twelve dated public GitHub payloads / all 317 prior content files unchanged','current and historical question-bank digests kept separate']
import runpy
runpy.run_path(str(ROOT/'scripts/validate_scale.py'),init_globals={'check_scale':ok})
ok('V11 retains all ten V10 question texts and anchors while rebinding changed source',bank['cases']==bank10['cases'] and bank['dimensions']==bank10['dimensions'] and bank['protocol_version']==bank10['protocol_version'] and bank['bank_sha256']!=bank10['bank_sha256'] and bank['source_sha256']['CONCLUSIONS_ZH.md']!=bank10['source_sha256']['CONCLUSIONS_ZH.md'])
report['highlights']=[x.replace('330 source archive checksums valid','348 source archive checksums valid').replace('fourteen embedded figures','seventeen embedded figures').replace('14 figure-reading entries','17 figure-reading entries') for x in report['highlights']]
report['highlights']+=['full 48-endpoint scale contrasts / four observed direction changes','six public run configs / 20.3919 percent initial mixture half-L1','synthetic batch-sensitive BPB ranking counterexample / joint floor-slope diagnostics','48 unlaunched confirmation slots / 8.481788657664T planned continuation tokens']
report['not_verified']+=['September ladder executed evaluation-code binding and common-checkpoint BPB re-evaluation','causal isolated scale or 25-percent mixture-switch effects','48 proposed GPU continuations, shared-checkpoint and generation-evaluation costs']
browser11=read(A/'browser_validation_v11.json');bank12=read(A/'assessment_casebank_v12_reference.json')
ok('Historical V11 browser binds pinned prior source bank and all nine declared behaviors',browser11['bank_sha256']==bank12['bank_sha256'] and len(browser11['functional'])==9 and all(browser11['functional'].values()) and browser11['errors']==[])
ok('Historical V11 standalone embeds seventeen figures and works without HTTP',browser11['standalone']['bank_sha256']==bank12['bank_sha256'] and browser11['standalone']['protocol']=='file:' and browser11['standalone']['embeddedFigures']==browser11['standalone']['loadedFigures']==browser11['standalone']['figureReadings']==17 and browser11['standalone']['questionCount']==10 and browser11['standalone']['httpRequests']==browser11['standalone']['errors']==[] and browser11['standalone']['engineeringSelectorUpdated'] is True)
ok('Current V11 narrow tables scroll within viewport while old storage is preserved',browser11['desktop']['documentWidth']<=browser11['desktop']['viewport'] and browser11['mobile']['documentWidth']<=browser11['mobile']['viewport'] and browser11['mobile']['tableScrollWidth']>browser11['mobile']['tableContainerWidth'] and browser11['functional']['existingLocalStorageUnchanged'] is True and '导入未完成' in browser11['oldImportMessage'])
ok('Three actual V11 browser screenshots retain viewed artifact bytes',len(browser11['screenshots'])==3 and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==browser11['screenshot_sha256'][p] for p in browser11['screenshots']))
ok('Archived V10 bank is byte-identical to the V10 release reference',hashlib.sha256((A/'assessment_casebank_v10_reference.json').read_bytes()).hexdigest()==read(A/'release_v10.json')['artifacts']['analysis/assessment_data.json'] and bank10['bank_sha256']==read(A/'release_v10.json')['current_bank_sha256'])
report['highlights']+=['current V11 desktop/mobile/file-protocol browser / nine source-and-state behaviors','V10 answer-bank preserved bytewise / old bank import explicitly rejected']
execution=read(A/'execution_provenance_audit.json')
ok('Six public run commits remain null with bounded metadata and file-list search',len(execution['runs'])==6 and all(r['commit'] is None and r['has_next_page'] is False and r['exact_metadata_name_matches']==[] and r['count_discrepancy']==1 and r['runner_present'] is True for r in execution['runs']) and execution['historical_eval_code_binding'] is None)
ok('Two saved public code entries load a callable and are not executed locally',len(execution['runner_entries'])==2 and all(r['loads_cloudpickle_callable'] is True and r['callable_pickle_name_present'] is True and r['source_executed'] is False and r['deserialized_callable'] is False and hashlib.sha256((ROOT/r['file']).read_bytes()).hexdigest()==r['sha256'] for r in execution['runner_entries']))
for p,sha in execution['source_sha256'].items():ok('Execution provenance payload checksum '+p,hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha)
report['highlights']+=['six null public commit fields / two callable runner entries / execution-code binding remains unknown']
contract=read(A/'implementation_contract_probe.json')
ok('Source helper probe passes 46 local checks without GPU claims',contract['checks_passed']==46 and len(contract['checks'])==46 and contract['status']=='local_source_helper_probe_only' and len(contract['substitutions'])==3 and len(contract['not_verified'])==5)
ok('Fractional mask example distinguishes single and repeated weighting',contract['mask_example']['weighted_unreduced']==[2,3,0] and abs(contract['mask_example']['single_weight_mean']-10/3)<1e-14 and abs(contract['mask_example']['double_weight_mean']-7/3)<1e-14)
ok('Original environment helpers cover 18 precedence cases',len(contract['backward_precedence'])==18 and sum(r['error']=='ValueError' for r in contract['backward_precedence'])==3 and all(r['resolved'] is False for r in contract['backward_precedence'] if r['environment']=='0'))
ok('Checkpoint fallback is explicitly recorded',next(r for r in contract['restore_cases'] if r['case']=='fallback_older')['result']=={'restored':'root/10'} and next(r for r in contract['restore_cases'] if r['case']=='optional_existing_unreadable')['error']=='FileNotFoundError')
ok('Counter example is an upper-bound illustration, not actual drop telemetry',contract['counter_example']['int64_total']==6442450944 and contract['counter_example']['int32_total']==-2147483648 and contract['counter_example']['actual_drop_count'] is None)
for p,sha in contract['source_sha256'].items():ok('Implementation source checksum '+p,hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha)
ok('New source chapter and inline mechanism diagram are rendered',soup.select_one('#contracts-guide') is not None and len(soup.select('#contracts-guide svg'))==1 and len(soup.select('#contracts-guide table'))==3 and 'C7' in soup.select_one('#contracts-guide').text)
report['highlights']=[x.replace('348 source archive checksums valid','354 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['six pinned implementation files / 46 original-helper and arithmetic checks / seven acceptance rules','V11 browser evidence remains historical; new chapter separately verified']
browser12=read(A/'browser_validation_v12.json')
ok('New chapter desktop and mobile fit their viewports',all(browser12[k]['documentWidth']<=browser12[k]['viewport'] for k in ['desktop','mobile']) and browser12['mobile']['tableScrollWidth']>browser12['mobile']['tableContainerWidth'])
ok('New standalone chapter loads without HTTP or browser errors',browser12['offline']['protocol']=='file:' and browser12['offline']['diagram']==1 and browser12['offline']['tables']==3 and browser12['offline']['figures']==17 and browser12['offline']['requests']==browser12['offline']['errors']==[])
ok('Historical V12 chapter preserves pinned prior answer-bank identity',browser12['desktop']['bank']==browser12['offline']['bank']==bank12['bank_sha256'])
ok('Two viewed V12 browser screenshots preserve exact bytes',len(browser12['screenshots'])==2 and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==browser12['screenshot_sha256'][p] for p in browser12['screenshots']))
stateprobe=read(A/'train_state_probe.json')
ok('V12 answer-bank reference pins exact previous release bytes',hashlib.sha256((A/'assessment_casebank_v12_reference.json').read_bytes()).hexdigest()==read(A/'release_v12.json')['artifacts']['analysis/assessment_data.json'])
ok('Original state-control-flow probe passes 28 scoped checks',stateprobe['checks_passed']==len(stateprobe['checks'])==28 and stateprobe['actual_hardware_result'] is None and stateprobe['actual_checkpoint_restore'] is None and len(stateprobe['substitutions'])==6)
ok('Original state fields retain pending beta and no independent RNG leaf',stateprobe['train_state_fields']==['step','params','master_params','opt_state','ema_params','pending_qb_betas'])
ok('Precision probe distinguishes authoritative accumulation and BF16 counterexample',len(stateprobe['precision_rows'])==30 and abs(stateprobe['precision_rows'][9]['master_weight']-.9899899959564209)<1e-14 and stateprobe['precision_rows'][9]['compute_weight']==.98828125 and stateprobe['precision_rows'][19]['compute_weight']==1)
ok('Pending beta example records two distinct model views',stateprobe['qb_example']['after_first_stored_bias']==[[0,0]] and stateprobe['qb_example']['next_training_bias']==[[1,-1]] and stateprobe['qb_example']['stored_bias_top1']==1 and stateprobe['qb_example']['pending_applied_top1']==0)
ok('Original StepInfo properties preserve zero-based log offset',stateprobe['step_labels']==[{'completed_updates':1,'log_step':0,'next_step':1},{'completed_updates':393,'log_step':392,'next_step':393},{'completed_updates':394,'log_step':393,'next_step':394}])
for row in stateprobe['ladder_configs']:
 raw=read(ROOT/row['source'])['data']['project']['run'];cfg=json.loads(raw['config'])['trainer']['value']
 ok('State policy independently matches archived config '+row['run'],hashlib.sha256((ROOT/row['source']).read_bytes()).hexdigest()==row['source_sha256'] and row['master_param_mode']==cfg['master_param_mode']=='FP32_PINNED_HOST' and row['policy']==cfg['trainer']['mp'] and row['offload_opt_state']==cfg['offload_opt_state'])
for p,sha in stateprobe['source_sha256'].items():ok('State source checksum '+p,hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha)
ok('State chapter and mechanism diagram are rendered with four original-value tables',soup.select_one('#state-guide') is not None and len(soup.select('#state-guide svg'))==1 and len(soup.select('#state-guide table'))==4)
ok('Main report corrects CE objective and first-update label boundaries','最终logit z-loss计入这个键' in soup.select_one('#report').text and '第一次更新之后' in soup.select_one('#report').text)
report['highlights']=[x.replace('354 source archive checksums valid','361 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['seven new state sources / 28 original control-flow checks / master, pending routing and step-label audit','V12 bank retained bytewise before correcting main-report source semantics']
browser13=read(A/'browser_validation_v13.json');statecontract=read(ROOT/'templates/state_switch_acceptance.json')
ok('V13 desktop and mobile keep state tables within viewport',all(browser13[k]['documentWidth']<=browser13[k]['viewport'] for k in ['desktop','mobile']) and browser13['mobile']['tableScrollWidth']>browser13['mobile']['tableContainerWidth'])
ok('V13 source-bank revision preserves prompts but rejects old source record',all(browser13['bankChecks'][k] for k in ['oldImportRejected','localStorageUnchanged','promptsAndAnchorsUnchanged']) and browser13['bankChecks']['oldBank']==bank12['bank_sha256'] and browser13['desktop']['bank']==bank['bank_sha256']!=bank12['bank_sha256'] and '导入未完成' in browser13['oldImportMessage'])
ok('V13 standalone state chapter uses current bank without HTTP',browser13['offline']['bank']==bank['bank_sha256'] and browser13['offline']['protocol']=='file:' and browser13['offline']['loadedFigures']==17 and browser13['offline']['diagrams']==1 and browser13['offline']['tables']==4 and browser13['offline']['requests']==browser13['offline']['errors']==[])
ok('Three viewed V13 browser screenshots retain exact archived bytes',len(browser13['screenshots'])==3 and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==browser13['screenshot_sha256'][p] for p in browser13['screenshots']))
ok('State-switch template keeps 3 controls and all actual inputs unexecuted',statecontract['status']=='planned_not_executed' and all(v is None for v in statecontract['common_start'].values()) and all(v is None for v in statecontract['evaluation_contract'].values()) and len(statecontract['controls'])==3 and all(c['result']=='not_run' and c['observations'] is None and c['predeclared_tolerances'] is None for c in statecontract['controls']) and set(statecontract['gates'].values())=={'unknown'} and statecontract['evidence']==[])
report['highlights']+=['V13 current-bank browser / old-record rejection / untouched local storage / file-protocol chapter','state-switch contract retains 3 unexecuted controls and 5 unknown gates']
boundary=read(A/'boundary_probe.json')
ok('Boundary probe records 17 scoped original-helper checks',boundary['checks_passed']==len(boundary['checks'])==17 and boundary['actual_gpu_result'] is None and boundary['actual_training_input_segments'] is None and len(boundary['cases'])==6)
ok('Boundary target and attention isolation remain distinct',boundary['cases'][0]['segments']==[0,0,1,1] and boundary['cases'][0]['loss_weights']==[1,1,1,0] and boundary['cases'][0]['dense_attention'][2]==[0,0,1,0])
ok('Padding bounds and routing validity preserve separate meanings',boundary['cases'][3]['lower_bounds']==[2,2,2,2,4,4,7] and boundary['cases'][3]['kernel_valid']==[False,False,True,True,True,True,False] and boundary['cases'][3]['loss_weights']==[0,1,1,1,1,0,0])
ok('Contiguous masks match while reused-ID counterexample differs',all(boundary['cases'][i]['real_token_mask_mismatches']==0 for i in [2,3,4]) and boundary['cases'][5]['real_token_mask_mismatches']==2)
for p,sha in boundary['source_sha256'].items():ok('Boundary source checksum '+p,hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha)
ok('Boundary chapter retains three tables and explicit production limits',soup.select_one('#boundary-guide') is not None and len(soup.select('#boundary-guide table'))==3 and '没有证明Hero生成了复用ID' in soup.select_one('#boundary-guide').text)
report['highlights']=[x.replace('361 source archive checksums valid','368 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['seven boundary sources / 17 original-helper checks / EOS, padding, segment and target-position audit']
browser14=read(A/'browser_validation_v14.json')
ok('V14 boundary tables fit desktop/mobile viewport',all(browser14[k]['width']<=browser14[k]['viewport'] for k in ['desktop','mobile']) and browser14['mobile']['tableScrollWidth']>browser14['mobile']['tableWidth'])
ok('V14 boundary chapter loads offline with 17 figures and unchanged source bank',browser14['offline']['protocol']=='file:' and browser14['offline']['figures']==17 and browser14['offline']['tables']==3 and browser14['offline']['requests']==browser14['offline']['errors']==[] and browser14['desktop']['bank']==browser14['offline']['bank']==bank['bank_sha256']==read(A/'release_v13.json')['current_bank_sha256'])
ok('Two viewed V14 screenshots retain exact bytes',len(browser14['screenshots'])==2 and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==browser14['screenshot_sha256'][p] for p in browser14['screenshots']))
report['highlights']+=['V14 desktop/mobile and offline boundary chapter / current source-bank preserved']
cacheaudit=read(A/'cache_audit.json')
ok('Cache census covers 1200 component instances and two roots',cacheaudit['component_instances']==1200 and cacheaudit['unique_components']==200 and len(cacheaudit['cache_roots'])==2 and cacheaudit['component_split_counts']=={'validation':1200})
ok('Cache admission probe separates mismatch warning and unfinished rejection',[x['status'] for x in cacheaudit['admission_branch_cases']]==['opened','opened','opened','rejected_unfinished'] and cacheaudit['actual_cache_ledger'] is None and cacheaudit['actual_token_arrays'] is None and cacheaudit['historical_tokenizer_revision'] is None)
ok('Cache chapter exposes six acceptance rules and historical limits',soup.select_one('#cache-guide') is not None and 'K6' in soup.select_one('#cache-guide').text and '不是已确认的历史训练版本' in soup.select_one('#cache-guide').text)
report['highlights']+=['V15 1200 component census / two cache roots / four original admission branches / actual cache identity remains unknown']
browser15=read(A/'browser_validation_v15.json')
ok('V15 desktop mobile offline chapter and source bank remain valid',browser15['desktop']['width']<=browser15['desktop']['viewport'] and browser15['mobile']['width']<=browser15['mobile']['viewport'] and browser15['offline']['figures']==17 and browser15['offline']['tables']==4 and browser15['offline']['requests']==browser15['offline']['errors']==[] and browser15['offline']['bank']==bank['bank_sha256'])
ok('V15 viewed desktop screenshot retains exact bytes',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in browser15['screenshot_sha256'].items()))
report['highlights']=[x.replace('368 source archive checksums valid','378 source archive checksums valid') for x in report['highlights']]
dedup=read(A/'dedup_probe.json')
ok('V16 original store helpers pass scoped local Parquet checks',dedup['checks_passed']==len(dedup['checks'])==11 and dedup['actual_corpus'] is None and dedup['historical_execution_sha'] is None)
ok('V16 sparse-filter and exemption counters retain causal order',dedup['cases'][1]['counters']['datakit_store/exact_duplicate_dropped']==0 and dedup['cases'][2]['counters']['datakit_store/exact_duplicate_dropped']==1 and len(dedup['exempt_sources'])==16)
ok('V16 raw feature is not presented as current verifier verdict',dedup['raw_feature_counterexample']['containment']==2/3 and dedup['raw_feature_counterexample']['decision_rule_executed'] is False)
ok('V16 old dated spec does not impersonate actual old run store',dedup['spec_identity']['old_spec_matches_old_run'] is False and dedup['spec_identity']['old_named_spec_store'].endswith('0381a974') and dedup['spec_identity']['old_run_store'].endswith('81e7e39a'))
for p,sha in dedup['source_sha256'].items():ok('Dedup helper source checksum '+p,hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha)
ok('V16 chapter distinguishes author policy from verified corpus',soup.select_one('#dedup-guide') is not None and '不是已审定的误删率' in soup.select_one('#dedup-guide').text and '不能相减' in soup.select_one('#dedup-guide').text)
report['highlights']=[x.replace('378 source archive checksums valid','390 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['V16 sixteen-source fuzzy exemption / real local Parquet original helpers / sparse missing-path ambiguity / spec identity mismatch']
browser16=read(A/'browser_validation_v16.json')
ok('V16 chapter fits viewport and loads offline with unchanged assessment bank',browser16['desktop']['width']<=browser16['desktop']['viewport'] and browser16['mobile']['width']<=browser16['mobile']['viewport'] and browser16['offline']['figures']==17 and browser16['offline']['tables']==3 and browser16['offline']['requests']==browser16['offline']['errors']==[] and browser16['offline']['bank']==bank['bank_sha256'])
ok('V16 viewed screenshot retains exact bytes',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in browser16['screenshot_sha256'].items()))
quality=read(A/'quality_probe.json')
ok('V17 quality helpers preserve eleven scoped checks',quality['checks_passed']==len(quality['checks'])==11 and quality['actual_model'] is None and quality['actual_calibration_file'] is None and quality['historical_execution_sha'] is None)
ok('V17 quality windows distinguish duplication and coverage',quality['bme_windows'][2]['spans'][0]==quality['bme_windows'][2]['spans'][1] and quality['bme_windows'][6]['coverage']==.6 and quality['bme_windows'][7]['coverage']==.3 and quality['unsampled_damage']['real_model_result'] is None)
ok('V17 invalid interpolation is not treated as valid calibration',quality['collapsed_calibration']['valid_interp_contract'] is False)
for p,sha in quality['source_sha256'].items():ok('Quality source checksum '+p,hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha)
qframe=soup.select_one('#quality-window-frame')
ok('V17 explorer is embedded with offline srcdoc and explicit quality limits',qframe is not None and qframe['srcdoc']==(ROOT/'QUALITY_WINDOWS.html').read_text() and 'Q5' in soup.select_one('#quality-guide').text and '不是生产词表覆盖率' in soup.select_one('#quality-guide').text)
report['highlights']=[x.replace('390 source archive checksums valid','400 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['V17 quality observation / vocabulary / calibration / five acceptance rules / embedded window explorer']
browser17=read(A/'browser_validation_v17.json')
ok('V17 embedded explorer fits desktop/mobile and works offline',browser17['desktop']['width']<=browser17['desktop']['viewport'] and browser17['mobile']['width']<=browser17['mobile']['viewport'] and browser17['mobile']['inner_width']<=browser17['mobile']['inner_viewport'] and browser17['offline']['protocol']=='file:' and browser17['offline']['coverage']=='6000 / 10000' and browser17['offline']['requests']==browser17['offline']['errors']==[] and browser17['offline']['bank']==bank['bank_sha256'])
ok('V17 viewed screenshot keeps exact bytes',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in browser17['screenshot_sha256'].items()))
for filename in ['scripts/acquire_quality.py','scripts/probe_quality.py','scripts/build_quality_windows.py']:
    compile((ROOT/filename).read_bytes(),filename,'exec')
ok('V17 acquisition probe and explorer builder compile',True)
contractaudit=read(A/'ladder_contract_audit.json')
ok('V18 paired config census retains declared and actual equivalence distinction',len(contractaudit['pairs'])==3 and contractaudit['configuration_equal_is_execution_equal'] is False and all(p['groups']['model']['declared_equal'] and p['actual_code_sha'] is None and p['actual_start_state_digest'] is None and p['actual_cache_content_equivalence'] is None and p['actual_evaluation_equivalence'] is None for p in contractaudit['pairs']))
ok('V18 group equality agrees with configuration content hashes',all(g['declared_equal']==(g['old_sha256']==g['new_sha256']) for p in contractaudit['pairs'] for g in p['groups'].values()))
ok('V18 declared differences retain exact pair counts',[[len(p['groups'][k]['differences']) for k in ['model','optimizer','eval','trainer','resources']] for p in contractaudit['pairs']]==[[0,1,3,21,6],[0,5,4,24,6],[0,5,4,22,6]])
for p,sha in contractaudit['source_sha256'].items():ok('Ladder contract source checksum '+p,hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha)
changereview=read(ROOT/'templates/training_change_review.json')
ok('V18 training change form retains unexecuted controls and actual identity unknowns',changereview['status']=='planned_not_executed' and changereview['claim']['decision']=='pending_evidence' and all(v is None for arm in changereview['identities'].values() for v in arm.values()) and all(c['status']=='not_run' and c['evidence']==[] for c in changereview['controls'].values()) and changereview['evidence']==[])
ok('V18 chapter joins source checks without inventing deployment authority',soup.select_one('#change-guide') is not None and '五个对象' in soup.select_one('#change-guide').text and '不自动允许部署' in soup.select_one('#change-guide').text)
report['highlights']+=['V18 paired config census / five frozen objects / three engineering claim layers / unexecuted change-review form']
browser18=read(A/'browser_validation_v18.json')
ok('V18 change guide fits viewport with offline ordered chapters and unchanged bank',browser18['desktop']['width']<=browser18['desktop']['viewport'] and browser18['mobile']['width']<=browser18['mobile']['viewport'] and browser18['desktop']['has_template'] and browser18['offline']['protocol']=='file:' and browser18['offline']['figures']==17 and browser18['offline']['tables']==3 and browser18['offline']['guide_order'][0]=='change-guide' and browser18['offline']['requests']==browser18['offline']['errors']==[] and browser18['offline']['bank']==bank['bank_sha256'])
ok('V18 viewed screenshot retains exact bytes',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in browser18['screenshot_sha256'].items()))
routing=read(A/'routing_probe.json')
ok('V19 routing helpers retain eleven scoped original checks',routing['checks_passed']==len(routing['checks'])==11 and routing['actual_collective_result'] is None and routing['actual_gradient_result'] is None and routing['actual_input_drop_pattern'] is None and routing['historical_execution_sha'] is None)
ok('V19 old/new denominator distinction stays explicit',routing['metric_denominator_example']['old']['moe/drop_fraction']==.125 and routing['metric_denominator_example']['new']['moe/drop_fraction']==.25)
ok('V19 same assignment loss retains distinct token damage',routing['cases'][0]['assignment_drop_fraction']==routing['cases'][1]['assignment_drop_fraction']==.125 and [c['affected_tokens'] for c in routing['cases']]==[1,8] and routing['cases'][1]['routed_output']==[2.1875]*8)
for p,sha in routing['source_sha256'].items():ok('Routing source checksum '+p,hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha)
ok('V19 inline assignment diagram retains artificial scope',soup.select_one('#routing-guide svg') is not None and len(soup.select('#routing-guide svg rect'))==129 and '不是Hero的loss' in soup.select_one('#routing-guide').text)
report['highlights']=[x.replace('400 source archive checksums valid','406 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['V19 routing denominator / count vs token damage / original combine / prefix clip / synthetic diagram']
browser19=read(A/'browser_validation_v19.json')
ok('V19 routing diagram fits desktop/mobile and loads offline',browser19['desktop']['width']<=browser19['desktop']['viewport'] and browser19['mobile']['width']<=browser19['mobile']['viewport'] and browser19['mobile']['wrapper_width']<browser19['mobile']['scroll_width'] and browser19['offline']['figures']==browser19['offline']['loaded_figures']==17 and browser19['offline']['rects']==129 and browser19['offline']['tables']==3 and browser19['offline']['requests']==browser19['offline']['errors']==[])
ok('V19 viewed screenshot retains exact bytes',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in browser19['screenshot_sha256'].items()))
qb=read(A/'qb_partition_probe.json')
ok('V20 original QB helpers pass sixteen scoped checks',qb['checks_passed']==len(qb['checks'])==16 and qb['actual_mesh_execution'] is None and qb['actual_training_effect'] is None and qb['historical_execution_sha'] is None)
ok('V20 estimator applicability follows archived HIST configurations',len(qb['configurations'])==6 and all(c['qb_estimator']=='HIST' and c['qb_hist_bins']==10000 for c in qb['configurations']))
ok('V20 local quantile partition counterexample retains exact pooled reference',[c['beta'] for c in qb['cases']]==[[50,1],[99.5,1.5]] and qb['pooled_exact_topk_beta']==[99,1] and [c['next_selected_expert'] for c in qb['cases']]==[0,1])
hist=qb['histogram_range_example']
ok('V20 histogram outlier changes shared-grid resolution',hist['base']['bin_width']==.01 and hist['other_expert_outlier']['bin_width']==100 and hist['base']['beta'][0]!=hist['other_expert_outlier']['beta'][0])
for p,sha in qb['source_sha256'].items():ok('QB source checksum '+p,hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha)
ok('V20 chapter limits TOPK applicability and historical attribution',soup.select_one('#qb-guide') is not None and '不能据此认定这些HIST运行' in soup.select_one('#qb-guide').text and '不是Hero观察' in soup.select_one('#qb-guide').text)
report['highlights']+=['V20 local-quantile partition / HIST shared-grid range / next-step router state / sixteen original helper checks']
browser20=read(A/'browser_validation_v20.json')
ok('V20 QB chapter fits viewport and loads offline',browser20['desktop']['width']<=browser20['desktop']['viewport'] and browser20['mobile']['width']<=browser20['mobile']['viewport'] and browser20['offline']['figures']==browser20['offline']['loaded']==17 and browser20['offline']['tables']==4 and browser20['offline']['hasHistogramCaveat'] and browser20['offline']['requests']==browser20['offline']['errors']==[])
ok('V20 viewed screenshot retains exact bytes',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in browser20['screenshot_sha256'].items()))
op=read(A/'optimizer_probe.json')
ok('V21 optimizer masks and decay wrapper retain sixteen scoped checks',op['checks_passed']==len(op['checks'])==16 and op['actual_optimizer_state'] is None and op['actual_clipping_result'] is None and op['actual_training_effect'] is None and op['historical_execution_sha'] is None)
ok('V21 bias belongs to Adam but is excluded from dedicated decay',op['parameter_groups']['blocks.mlp.router_bias']=='adam' and op['dedicated_decay_mask']['blocks.mlp.router_bias'] is False)
ok('V21 count-based decay retains clipping to zero',[c['decay_coefficient'] for c in op['cases']]==[.2,.1,0,0] and op['default_dedicated_decay']==0)
ok('V21 six config declarations remain distinct from effective execution',len(op['configurations'])==6 and sum(c['dedicated_decay_present'] for c in op['configurations'])==3 and all(c['dedicated_decay_value'] in (None,0) and c['max_grad_norm'] is None and c['generic_weight_decay']==.1 for c in op['configurations']))
for p,sha in op['source_sha256'].items():ok('Optimizer source checksum '+p,hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha)
ok('V21 chapter distinguishes missing values from Python None',soup.select_one('#optimizer-guide') is not None and '不是显式传入Python' in soup.select_one('#optimizer-guide').text and '不是Adam算法数值结果' in soup.select_one('#optimizer-guide').text)
report['highlights']=[x.replace('406 source archive checksums valid','407 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['V21 parameter groups / dedicated decay mask / Adam-count clock / group clipping scope']
browser21=read(A/'browser_validation_v21.json')
ok('V21 optimizer chapter fits viewport and loads offline',browser21['desktop']['width']<=browser21['desktop']['viewport'] and browser21['mobile']['width']<=browser21['mobile']['viewport'] and browser21['offline']['figures']==browser21['offline']['loaded']==17 and browser21['offline']['tables']==4 and browser21['offline']['hasScope'] and browser21['offline']['requests']==browser21['offline']['errors']==[])
ok('V21 viewed screenshot retains exact bytes',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in browser21['screenshot_sha256'].items()))
browser22=read(A/'browser_validation_v22.json')
ok('V22 diagnostic chapter fits viewport and loads offline',browser22['desktop']['width']<=browser22['desktop']['viewport'] and browser22['desktop']['template_link'] and browser22['mobile']['width']<=browser22['mobile']['viewport'] and browser22['offline']['figures']==browser22['offline']['loaded']==17 and browser22['offline']['tables']==4 and browser22['offline']['requests']==browser22['offline']['errors']==[])
ok('V22 viewed screenshot retains exact bytes',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in browser22['screenshot_sha256'].items()))
sc=read(A/'short_conv_probe.json')
ok('V23 ShortConv helpers pass sixteen scoped checks',sc['checks_passed']==len(sc['checks'])==16 and sc['actual_pallas_result'] is None and sc['actual_bf16_result'] is None and sc['actual_gradient_result'] is None and sc['actual_distributed_result'] is None and sc['historical_execution_sha'] is None)
ok('V23 boundary and reused-ID counterexamples retain original outputs',sc['cases'][1]['output']==[1,11,100,1100,10000,110000] and sc['cases'][2]['output'][2]==101)
ok('V23 four simulated halo cases match pooled reference',len(sc['halo_cases'])==4 and all(c['got']==c['reference'] for c in sc['halo_cases']))
for p,sha in sc['source_sha256'].items():ok('ShortConv source checksum '+p,hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha)
ok('V23 chapter keeps actual GPU gradients unverified',soup.select_one('#short-conv-guide') is not None and '没有执行' in soup.select_one('#short-conv-guide').text and '不是已观察到的Hero数据故障' in soup.select_one('#short-conv-guide').text)
report['highlights']=[x.replace('407 source archive checksums valid','413 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['V23 ShortConv segment identity / halo metadata / initialization blindspot / rounding and fallback scopes']
browser23=read(A/'browser_validation_v23.json')
ok('V23 ShortConv chapter fits viewport and loads offline',browser23['desktop']['width']<=browser23['desktop']['viewport'] and browser23['mobile']['width']<=browser23['mobile']['viewport'] and browser23['offline']['figures']==browser23['offline']['loaded']==17 and browser23['offline']['tables']==4 and browser23['offline']['requests']==browser23['offline']['errors']==[])
ok('V23 viewed screenshot retains exact bytes',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in browser23['screenshot_sha256'].items()))
rp=read(A/'router_precision_audit.json')
ok('V24 fork identity and static scope remain explicit',rp['fork_revision']=='a00cb77a491f4777a2c66edce54f47ff7b255c40' and rp['status']=='experimental_not_live_deployment' and len(rp['router_branches'])==3 and rp['actual_gpu_dot'] is None and rp['actual_hlo_verification'] is None and rp['actual_fixture_loaded'] is None and rp['actual_training_result'] is None)
for p,sha in rp['source_sha256'].items():ok('Router precision source checksum '+p,hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha)
refresh=read(A/'issue_8435_refresh_v24.json')
ok('V24 refreshed comments preserve exact archived body identities',refresh['comment_count']==27 and refresh['pagination_complete'] and refresh['added_ids']==refresh['removed_ids']==refresh['body_edited_ids']==[] and all(hashlib.sha256(c['body'].encode()).hexdigest()==refresh['comment_body_sha256'][str(c['id'])] for c in read(ROOT/refresh['reference_comments'])))
ok('V24 precision chapter labels proposed crossed-policy evaluation',soup.select_one('#router-precision-guide') is not None and '四格是待设计' in soup.select_one('#router-precision-guide').text and '不是原router dot' in soup.select_one('#router-precision-guide').text)
report['highlights']=[x.replace('413 source archive checksums valid','416 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['V24 experimental fork router precision / schedule controls / proposed policy matrix / original issue body refresh']
browser24=read(A/'browser_validation_v24.json')
ok('V24 router precision chapter fits viewport and loads offline',browser24['desktop']['width']<=browser24['desktop']['viewport'] and browser24['mobile']['width']<=browser24['mobile']['viewport'] and browser24['offline']['figures']==browser24['offline']['loaded']==17 and browser24['offline']['tables']==4 and browser24['offline']['requests']==browser24['offline']['errors']==[])
ok('V24 viewed screenshot retains exact bytes',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in browser24['screenshot_sha256'].items()))
mt=read(A/'mix_trajectory.json')
ok('V26 corrected trajectory preserves subset and all four parent metrics',len(mt['subsets'])==16 and len(mt['complete_steps'])==67 and mt['subset_points']==2144 and mt['macro_points']==134 and mt['parent_points']==268 and mt['selected_points']==2412)
for p,sha in mt['source_sha256'].items():ok('Mix trajectory input checksum '+p,hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha)
# Cross-check against the independently implemented prior production-lineage stitch.
with (A/'mix_trajectory_points.csv').open() as f:
    point_rows=list(csv.DictReader(f))
prior_series=read(A/'series.json')
for row in point_rows:
    key='eval_dropless/paloma/'+('' if row['subset']=='__parent__' else row['subset']+'/')+row['metric']
    matches=[p for p in prior_series[key] if p['step']==int(row['step']) and p['run']==row['run']]
    assert len(matches)==1 and matches[0]['value']==float(row['value'])
ok('V26 independent extraction matches all 2412 prior values',len(point_rows)==2412)
ok('V25 counterfactual and historical evaluator remain unknown',mt['actual_mix_counterfactual'] is None and mt['actual_cluster_to_paloma_mapping'] is None and mt['actual_historical_eval_identity'] is None)
browser25=read(A/'browser_validation_v25.json')
ok('V25 real trajectory SVG fits viewport and works offline',browser25['desktop']['width']<=browser25['desktop']['viewport'] and browser25['mobile']['width']<=browser25['mobile']['viewport'] and browser25['offline']['figures']==browser25['offline']['loaded']==17 and browser25['offline']['svg'] and browser25['offline']['requests']==browser25['offline']['errors']==[] and soup.select_one('#mix-trajectory-guide svg') is not None)
ok('V25 viewed screenshot retains exact bytes',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in browser25['screenshot_sha256'].items()))
report['highlights']+=['V25 exact production-lineage subset trajectories / inherited history exclusion / aggregate direction sensitivity / no causal attribution']
em=read(A/'eval_metric_audit.json')
ok('V26 original helper and static dataflow scope remains bounded',em['checks_passed']==len(em['checks'])==14 and em['actual_evaluator_execution'] is None and em['actual_distributed_reduce'] is None and em['actual_token_byte_counts'] is None and em['historical_execution_sha'] is None)
for p,sha in em['source_sha256'].items():ok('Eval metric source checksum '+p,hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha)
ok('V26 real logged macro reconstructs at all 67 steps',len(mt['macro_reconstruction'])==67 and all(abs(x['CE_residual'])<5e-7 and abs(x['BPB_residual'])<5e-7 for x in mt['macro_reconstruction']))
ok('V26 naming correction is explicit in both chapters',soup.select_one('#eval-metrics-guide') is not None and '命名前提错了' in soup.select_one('#eval-metrics-guide').text and 'V26更正' in soup.select_one('#mix-trajectory-guide').text)
report['highlights']=[x.replace('416 source archive checksums valid','417 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['V26 correction of parent micro versus macro BPB / all 67 macro reconstructions / original log helpers and RunningMean units']
browser26=read(A/'browser_validation_v26.json')
ok('V26 corrected metric chapter fits and loads offline',browser26['desktop']['width']<=browser26['desktop']['viewport'] and browser26['mobile']['width']<=browser26['mobile']['viewport'] and browser26['offline']['figures']==browser26['offline']['loaded']==17 and browser26['offline']['tables']==3 and browser26['offline']['hasCorrection'] and browser26['offline']['requests']==browser26['offline']['errors']==[])
ok('V26 viewed screenshot retains exact bytes',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in browser26['screenshot_sha256'].items()))
cp=read(A/'checkpoint_commit_probe.json')
ok('V27 checkpoint helper scope preserves seventeen bounded checks',cp['checks_passed']==len(cp['checks'])==17 and cp['actual_tensorstore_write'] is None and cp['actual_cloud_permissions'] is None and cp['actual_checkpoint_restore'] is None and cp['actual_distributed_commit'] is None and cp['historical_execution_sha'] is None)
for p,sha in cp['source_sha256'].items():ok('Checkpoint commit source checksum '+p,hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha)
ok('V27 fabricated async marker and failure boundaries stay explicit',cp['cases'][0]['destination_visible'] and not cp['cases'][0]['caller_success'] and cp['cases'][1]['metadata_only_discovered'] and not cp['cases'][1]['real_arrays_written'] and cp['cases'][3]['actual_array_corruption'] is None)
ok('V27 commit flow distinguishes source stages and proposed restore',soup.select_one('#checkpoint-commit-guide svg') is not None and '额外验收建议' in soup.select_one('#checkpoint-commit-guide svg').text and '没有写模型数组' in soup.select_one('#checkpoint-commit-guide').text)
report['highlights']=[x.replace('417 source archive checksums valid','422 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['V27 layout versus completion markers / async save handoff / copy-delete permission / reused marker / four recovery watermarks']
browser27=read(A/'browser_validation_v27.json')
ok('V27 checkpoint chapter fits both viewports and runs offline',browser27['desktop_http']['width']<=browser27['desktop_http']['viewport'] and browser27['mobile_file']['width']<=browser27['mobile_file']['viewport'] and browser27['offline']['figures']==browser27['offline']['loaded']==17 and browser27['offline']['tables']==3 and browser27['offline']['svg'] and browser27['offline']['proposal'] and browser27['offline']['requests']==browser27['offline']['errors']==[])
ok('V27 viewed screenshot retains exact bytes',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in browser27['screenshot_sha256'].items()))
cm=read(A/'checkpoint_memory_probe.json')
ok('V28 memory planning scope retains seventeen bounded checks',cm['checks_passed']==len(cm['checks'])==17 and cm['actual_RSS'] is None and cm['actual_model_write_plan'] is None and cm['actual_GPU_slice'] is None and cm['actual_TensorStore_IO'] is None and cm['historical_execution_sha'] is None)
for p,sha in cm['source_sha256'].items():ok('Checkpoint memory source checksum '+p,hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha)
ok('V28 original budget and chunk boundaries retain artificial scope',cm['cases'][0]['oversized_peak_bytes']==14 and cm['cases'][0]['target_bytes']==10 and cm['cases'][2]['odd_chunk_bytes']==4356 and cm['cases'][2]['actual_Hero_oversized_chunk'] is None)
ok('V28 synthetic writer coverage and declared defaults remain exact',cm['cases'][1]['written_axis_intervals']==[[0,2],[2,4],[4,6],[6,8]] and cm['declared_defaults']['max_staged_host_bytes']==16*1024**3)
ok('V28 chapter separates reporting estimate from measurement',soup.select_one('#checkpoint-memory-guide') is not None and '不是节点实际峰值' in soup.select_one('#checkpoint-memory-guide').text and '没有生成Hero真实计划' in soup.select_one('#checkpoint-memory-guide').text)
report['highlights']=[x.replace('422 source archive checksums valid','423 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['V28 original asyncio budget / replica writer planning / odd chunk target boundary / no actual RSS claim']
browser28=read(A/'browser_validation_v28.json')
ok('V28 memory chapter DOM fits both viewports and runs offline',browser28['desktop']['width']<=browser28['desktop']['viewport'] and browser28['mobile']['width']<=browser28['mobile']['viewport'] and browser28['offline']['figures']==browser28['offline']['loaded']==17 and browser28['offline']['tables']==3 and browser28['offline']['hasScope'] and browser28['offline']['requests']==browser28['offline']['errors']==[])
case_map=read(A/'engineering_case_map.json')
ok('V29 maps all fifteen report cases without claiming new execution',len(case_map['cases'])==15 and {x['report_section'] for x in case_map['cases']}=={'3.'+str(i) for i in range(1,16)} and all(x['actual_new_cluster_execution'] is None and x['original_evidence_urls'] for x in case_map['cases']))
known_rules={x['id'] for x in read(A/'rubrics.json')['rules']}
ok('V29 routes use established rules and preserve proposed-check status',len(case_map['entry_routes'])==6 and all(set(x['rubric_ids'])<=known_rules for x in case_map['cases']+case_map['entry_routes']) and case_map['status']=='research_navigation_and_proposed_checks')
for p,sha in case_map['source_sha256'].items():ok('Case map input checksum '+p,hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha)
ok('V29 navigation and current audit are present in standalone report',soup.select_one('#engineering-map-guide') is not None and soup.select_one('#delivery-audit-guide') is not None and 'V18及此前的历史审计' in soup.select_one('#delivery-audit-guide').text and '不自动给主张评分' in soup.select_one('#engineering-map-guide').text)
report['highlights']+=['V29 fifteen-case symptom / source / local-scope / next-check map and six entry routes; current delivery audit separates historical records']
browser29=read(A/'browser_validation_v29.json')
ok('V29 fifteen-case map and six routes fit and load offline',browser29['desktop']['width']<=browser29['desktop']['viewport'] and browser29['mobile']['width']<=browser29['mobile']['viewport'] and browser29['offline']['figures']==browser29['offline']['loaded']==17 and browser29['offline']['cases']==15 and browser29['offline']['entryRows']==6 and browser29['offline']['historicalAudit'] and browser29['offline']['requests']==browser29['offline']['errors']==[])
mg=read(A/'muon_geometry_probe.json');refresh=read(A/'issue_refresh_v30.json')
ok('V30 original geometry helper retains twenty-one bounded checks',mg['checks_passed']==len(mg['checks'])==20 and mg['actual_Muon_direction'] is None and mg['actual_JAX_SPMD'] is None and mg['actual_GPU_result'] is None and mg['historical_execution_sha'] is None)
ok('V30 fixed geometry source digest matches',mg['source_sha256']==hashlib.sha256((S/'optimizer_2026_10_05/optimizer.py').read_bytes()).hexdigest())
ok('V30 four-dimensional counterexample conserves joint norm only',abs(mg['four_dimensional_joint_norm_before']-mg['four_dimensional_joint_norm_after'])<1e-5 and mg['four_dimensional_inner_norm_before']!=mg['four_dimensional_inner_norm_after'])
ok('V30 postprojection bound counterexample exceeds eta',mg['tangent_update_norm_eta_0p2']>mg['artificial_lr'] and abs(mg['tangent_update_norm_eta_0p2']-mg['postprojection_tight_bound_eta_0p2'])<1e-6)
ok('V30 complete four-comment incident archive preserves correction order',len(read(S/'muon_geometry_2026_10_05/comments_8073.json'))==read(S/'muon_geometry_2026_10_05/issue_8073.json')['comments']==4)
ok('V30 refresh checks three body/comment comparisons without changing snapshots',[x['issue'] for x in refresh['issues']]==[8435,8506,8870] and all(not x['body_or_comment_changed'] for x in refresh['issues']))
ok('V30 future geometry execution fields stay null',all(v is None for k,v in read(ROOT/'templates/muon_geometry_check.json').items() if k!='status'))
ok('V30 geometry chapter is present with four tables',soup.select_one('#muon-geometry-guide') is not None and len(soup.select('#muon-geometry-guide table'))==4)
bv30=read(A/'browser_validation_v30.json')
ok('V30 geometry diagrams and four tables load desktop mobile and offline',bv30['desktop']['width']<=bv30['desktop']['viewport'] and bv30['mobile']['width']<=bv30['mobile']['viewport'] and bv30['offline']['figures']==bv30['offline']['loaded']==17 and bv30['offline']['diagram'] and bv30['offline']['tables']==4 and bv30['requests']==bv30['errors']==[])
report['highlights']=[x.replace('423 source archive checksums valid','425 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['V30 twenty original geometry and analytic checks; four-comment incident diagnosis corrections; no GPU / SPMD reproduction']
ap=read(A/'adamh_probe.json');af=read(ROOT/'templates/mix_optimizer_state_factorial.json')
ok('V31 original AdamH function body retains seventeen scoped checks',ap['checks_passed']==len(ap['checks'])==17 and ap['actual_Optax_moments'] is None and ap['actual_JAX_SPMD'] is None and ap['actual_checkpoint_restore'] is None and ap['actual_training_effect'] is None and ap['historical_execution_sha'] is None)
ok('V31 AdamH fixed source digest matches',ap['source_sha256']==hashlib.sha256((S/'adamh_2026_10_05/adamh.py').read_bytes()).hexdigest())
ok('V31 artificial reset has equal norm but distinct direction',abs(ap['retained_norm']-ap['reset_norm'])<1e-5 and ap['angle_between_retained_and_reset_degrees']>3 and ap['second_step_with_moments']!=ap['second_step_reset_moments'])
ok('V31 mixture state factorial remains unexecuted with four null outcomes',af['status']=='planned_not_executed' and af['actual_training_executed'] is False and len(af['arms'])==4 and all(x['result'] is None for x in af['arms']) and all(af[k] is None for k in ['common_checkpoint_digest','execution_sha','token_budget_per_arm','fixed_eval_identity','scheduler_state_policy','restored_or_reset_fields']))
ok('V31 state chapter and diagram are present',soup.select_one('#adamh-state-guide svg') is not None and len(soup.select('#adamh-state-guide table'))==2)
report['highlights']=[x.replace('425 source archive checksums valid','426 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['V31 original AdamH body and explicit moment substitutes; state reset direction counterexample; count-only cancellation boundary']
b31=read(A/'browser_validation_v31.json')
ok('V31 state diagram loads desktop mobile and offline without overflow',b31['desktop']['width']<=b31['desktop']['viewport'] and b31['mobile']['width']<=b31['mobile']['viewport'] and b31['offline']['figures']==b31['offline']['loaded']==17 and b31['offline']['diagram'] and b31['offline']['tables']==2 and b31['requests']==b31['errors']==[])
md=read(A/'muon_direction_probe.json')
ok('V32 direction helper and layout probes preserve twenty scoped checks',md['checks_passed']==len(md['checks'])==20 and md['actual_BF16_result'] is None and md['actual_QuACK_result'] is None and md['actual_distributed_result'] is None and md['actual_training_effect'] is None and md['historical_execution_sha'] is None)
ok('V32 artificial sixth iteration worsens rather than guarantees convergence',len(md['trajectory'])==11 and md['trajectory'][6]['orthogonality_residual']>md['trajectory'][5]['orthogonality_residual'])
ok('V32 fixed Muon coefficient source digests match',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==v for p,v in md['source_sha256'].items()))
ok('V32 context-bank layout is preserved while matrix dims replicate',md['context_bank_layout_calls'][0]['spec']==[None,'context',None,None] and md['context_bank_layout_calls'][-1]['spec']==[None,'context','data','model'])
ok('V32 padded stack retains original global layer output shape',md['three_dimensional_layout_calls'][0]['shape'][0]==4)
ok('V32 direction chapter and spectrum illustration are embedded',len(soup.select('#muon-direction-guide table'))==4 and soup.select_one('#muon-direction-guide svg') is not None)
ok('V32 future direction execution fields remain null',all(v is None for k,v in read(ROOT/'templates/muon_direction_check.json').items() if k!='status'))
report['highlights']=[x.replace('426 source archive checksums valid','428 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['V32 original Newton-Schulz functions / float32 substitutions; nonmonotonic finite-step spectrum; bank-preserving layout boundaries']
b32=read(A/'browser_validation_v32.json')
ok('V32 direction spectrum and four tables load desktop mobile and offline',b32['desktop']['width']<=b32['desktop']['viewport'] and b32['mobile']['width']<=b32['mobile']['viewport'] and b32['offline']['figures']==b32['offline']['loaded']==17 and b32['offline']['diagram'] and b32['offline']['tables']==4 and b32['requests']==b32['errors']==[])
wp=read(A/'watch_probe.json');ob=read(A/'optimizer_bundle_validation.json')
ok('V33 watch probes preserve twenty-two bounded checks',wp['checks_passed']==len(wp['checks'])==22 and wp['actual_tree_norm_result'] is None and wp['actual_JAX_execution'] is None and wp['actual_watch_overhead'] is None and wp['historical_execution_sha'] is None)
ok('V33 fixed watch and caller source digests match',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in wp['source_sha256'].items()))
ok('V33 static watch clock uses preupdate optimizer state',wp['inline_static_fields']['opt_state']=='opt_state_in' and wp['inline_static_fields']['params']=='qb_params')
ok('V33 synthetic incident checks remain separate from historical GPU replay',ob['checks_passed']==len(ob['checks'])==23 and ob['actual_historical_bundle'] is None and ob['actual_GPU_replay'] is None)
ok('V33 observability chapter has three evidence tables',len(soup.select('#observability-guide table'))==3)
ok('V33 future normalized bundle contract remains blank',all(v is None for k,v in read(ROOT/'templates/optimizer_bundle_contract.json').items() if k!='status'))
report['highlights']=[x.replace('428 source archive checksums valid','430 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['V33 watch computation versus logging clocks; exact preupdate views; executable normalized-bundle observations without automatic root cause']
b33=read(A/'browser_validation_v33.json')
ok('V33 observability chapter and tool links load desktop mobile offline',b33['desktop']['width']<=b33['desktop']['viewport'] and b33['mobile']['width']<=b33['mobile']['viewport'] and b33['desktop']['hasToolLink'] and b33['offline']['figures']==b33['offline']['loaded']==17 and b33['offline']['hasChapter'] and b33['offline']['tables']==3 and b33['requests']==b33['errors']==[])
fl=read(A/'failure_loop_probe.json')
ok('V34 original loop fragment retains seventeen scoped checks',fl['checks_passed']==len(fl['checks'])==17 and fl['actual_GPU_behavior'] is None and fl['actual_callback_guard_coverage'] is None and fl['actual_checkpoint_commit'] is None and fl['actual_checkpoint_restore'] is None and fl['actual_historical_failure'] is None)
ok('V34 fixed loop source digest matches',fl['source_sha256']==hashlib.sha256((S/'scale_2026_10_05/train_hero_ep.py').read_bytes()).hexdigest())
ok('V34 finite-loss synthetic corruption reaches save handoff without claiming commit',fl['cases']['finite_loss_bad_update']['error'] is None and all(not x['parameter_finite'] for x in fl['cases']['finite_loss_bad_update']['checkpoint_handoffs']))
ok('V34 failed loss skips final callbacks and handoffs',fl['cases']['bad_loss']['callback_calls']==fl['cases']['bad_loss']['checkpoint_handoffs']==[] and fl['cases']['bad_loss']['events'][-1]=='training_finished')
ok('V34 numerical acceptance record remains planned and null',all(v is None for k,v in read(ROOT/'templates/numerical_acceptance_record.json').items() if k!='status'))
ok('V34 failure-boundary chapter has three tables',len(soup.select('#failure-boundaries-guide table'))==3)
report['highlights']+=['V34 finite-loss versus next-state health / original failure-loop control flow / proposed numeric acceptance watermark; no actual contaminated checkpoint claim']
b34=read(A/'browser_validation_v34.json')
ok('V34 failure boundaries load after verified preview recovery',b34['desktop']['width']<=b34['desktop']['viewport'] and b34['mobile']['width']<=b34['mobile']['viewport'] and b34['offline']['figures']==b34['offline']['loaded']==17 and b34['offline']['hasChapter'] and b34['offline']['tables']==3 and b34['requests']==b34['errors']==[])
bc=read(A/'batch_clock_probe.json')
ok('V35 bounded batch-clock checks have explicit non-replay scope',bc['checks_passed']==len(bc['checks'])==27 and bc['actual_GPU_behavior'] is None and bc['actual_historical_cursor_replay'] is None)
ok('V35 schedule and allocation source digests match',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in bc['synthetic_review']['source_sha256'].items()) and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in bc['source_sha256'].items()))
ok('V35 alignment and cursor counterexample remain rejected',not bc['synthetic_review']['construction_alignment_ok'] and bc['synthetic_review']['resume']['sequence_offset_delta']==-12)
ok('V35 equal totals do not imply equal batch prefixes',bc['equal_endpoint_counterexample']['sequence_offset_delta']==0 and not bc['equal_endpoint_counterexample']['consumed_batch_prefix_equal'])
ok('V35 original callback does not write removed component zero','mixture/weight/B' not in bc['original_callback_logs'][-1]['data'])
ok('V35 new clock chapter has three evidence tables',len(soup.select('#batch-clock-guide table'))==3)
report['highlights']+=['V35 exact cumulative batch clock / stage alignment rejection / consumed prefix review / configured weights versus block counts; static HTML verification only for new chapter']
mr=read(A/'mixture_range_probe.json')
ok('V36 integer range probes retain twenty-two bounded checks',mr['checks_passed']==len(mr['checks'])==22 and mr['actual_historical_overflow'] is None and mr['actual_loader_replay'] is None and mr['actual_Hero_runtime_numpy_version'] is None)
ok('V36 original source and archived configuration digests match',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in mr['source_sha256'].items()))
ok('V36 corrupt modulo returns a valid but different synthetic item',mr['synthetic']['corrupt_remapped_index']==553 and mr['synthetic']['expected_remapped_index']==936)
ok('V36 scalar and batch-style types have distinct range behavior',mr['synthetic']['original_within_stage_index']<0 and mr['synthetic']['batched_int64_block_index']==mr['synthetic']['expected_index'])
ok('V36 archived Hero bucket count bounds stay below int32',mr['hero_declared_bounds']['max_component_upper_bound']==256996542<mr['hero_declared_bounds']['int32_limit']<mr['hero_declared_bounds']['total_sequences'])
ok('V36 new range chapter has three evidence tables',len(soup.select('#mixture-range-guide table'))==3)
report['highlights']+=['V36 int32 intermediate/cumulative overflow and finite restart remap; actual declared Hero per-component upper bounds below limit; no historical overflow claim; static HTML only']
lv=read(A/'live_analysis_2026_10_06.json')
ok('V37 new observations have twelve bounded data checks',lv['checks_passed']==len(lv['checks'])==12 and lv['actual_eval_sample_identity'] is None and lv['actual_execution_SHA'] is None and lv['actual_mixture_counterfactual'] is None and lv['statistical_significance'] is None)
ok('V37 fresh W&B source payload digests match',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in lv['source_sha256'].items()))
ok('V37 new exact checkpoints preserve seventy-two old values',lv['overlap_values_unchanged']==72 and lv['new_eval_steps']==[200999,203999,206999,209999])
ok('V37 summary and evaluation progress stay separate',lv['snapshot']['summary_step']==211367 and lv['snapshot']['latest_complete_eval_step']==209999 and lv['snapshot']['eval_lag_updates']==1368)
ok('V37 macro sensitivity is descriptive and preserves improvement without ptb',lv['sensitivity']['loss']['ptb_share_of_equal_subset_delta']>.65 and lv['sensitivity']['loss']['equal_subset_delta_without_ptb']<0)
ok('V37 new observed chapter has three tables and inline scientific figure',len(soup.select('#live-observation-guide table'))==3 and soup.select_one('#live-observation-guide svg') is not None)
report['highlights']=[x.replace('430 source archive checksums valid','433 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['V37 new independent 9.75T / 4K snapshot; four new exact eval steps, 72 unchanged overlapping values; endpoint and PTB sensitivity decomposition; static HTML only for new chapter']
ei=read(A/'eval_identity_probe.json')
ok('V38 evaluation identity checks retain twenty-one bounded checks',ei['checks_passed']==len(ei['checks'])==21 and ei['actual_historical_execution_SHA'] is None and ei['actual_cache_content_hash'] is None and ei['actual_sharded_loader'] is None and ei['actual_eval_metric_replay'] is None and ei['actual_GPU_behavior'] is None)
ok('V38 archived source and declared config digests match',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in ei['source_sha256'].items()))
ok('V38 progress estimate differs from actual synthetic stream count',ei['synthetic_exact_multiple']=={'actual_batches':2,'len_estimate':3})
ok('V38 synthetic default stream includes finite partial items once',ei['synthetic_default_batches']==[[0,1,2,3],[4,5,6,7],[8,9]])
ok('V38 declared current evaluation has batch 704 and no cap',ei['declared_eval']['batch']==704 and ei['declared_eval']['max_eval_batches'] is None)
ok('V38 new identity chapter has three evidence tables',len(soup.select('#eval-identity-guide table'))==3)
report['highlights']=[x.replace('433 source archive checksums valid','435 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['V38 domain mapping / fresh default logical eval iteration / finite partial batch / progress-length overestimate; no actual cache or repeated model-score verification']
er=read(A/'eval_replay_validation.json')
ok('V39 replay comparison retains twenty-seven synthetic checks',er['checks_passed']==len(er['checks'])==27 and er['actual_historical_transcripts'] is None and er['actual_GPU_replay'] is None)
ok('V39 matching synthetic declarations do not imply actual GPU replay',er['synthetic_clean']['status']=='within_tolerance_under_matching_declarations' and er['synthetic_clean']['root_cause'] is None and er['synthetic_clean']['actual_GPU_replay'] is None)
ok('V39 regrouped synthetic transcript preserves totals but changes logged BPB',er['synthetic_regrouped']['status']=='identity_or_input_review_required' and er['synthetic_regrouped']['first']['micro_CE']==er['synthetic_regrouped']['second']['micro_CE'] and er['synthetic_regrouped']['first']['micro_logged_BPB']!=er['synthetic_regrouped']['second']['micro_logged_BPB'])
ok('V39 new replay chapter has three evidence tables',len(soup.select('#eval-replay-guide table'))==3)
report['highlights']+=['V39 executable declared global leaf-domain transcript comparison; token-weighted BPB regrouping counterexample; no raw-array identity verification or actual Hero replay']
ok('V40 synthesis is the first reading chapter before interactive labs',soup.select_one('.chapter').get('id')=='synthesis-guide')
ok('V40 synthesis chapter retains two decision tables',len(soup.select('#synthesis-guide table'))==2)
ok('V40 synthesis keeps historical evidence and proposed work distinct','本章没有新增训练结果' in (ROOT/'SYNTHESIS_ZH.md').read_text() and '不改动18条规则的版本' in (ROOT/'SYNTHESIS_ZH.md').read_text())
report['highlights']+=['V40 evidence-bounded synthesis and practical reading / mixture-experiment routes; no new experiments or reader comprehension claim']
ea=read(A/'eval_array_export_validation.json')
ok('V41 supplied-array export retains twenty-three synthetic checks',ea['checks_passed']==len(ea['checks'])==23 and ea['actual_Hero_arrays'] is None and ea['actual_GPU_forward'] is None and ea['actual_rank_gather'] is None)
ok('V41 synthetic array-export reference leaf sums match',[(r['weighted_loss_sum'],r['loss_weight_sum'],r['weighted_byte_sum']) for r in ea['synthetic_export']['records']]==[(2.,1.,3.),(3.,1.,1.)])
ok('V41 typed v2 hashes bind supplied arrays only',ea['synthetic_export']['input_digest_scheme']=='typed_global_eval_arrays_v2' and ea['synthetic_export']['array_export']['input_hash_verified_against_supplied_arrays'] is True)
ok('V41 new array export chapter has three evidence tables',len(soup.select('#eval-array-export-guide table'))==3)
report['highlights']+=['V41 offline supplied-global-array export / canonical typed v2 input digest / computed byte table / CPU synthetic tests; no actual forward, rank gathering or Hero data']
ef=read(A/'eval_format_probe.json');access=read(A/'cache_access_2026_10_06.json');pa=read(A/'paloma_protocol_acquisition.json')
ok('V42 original stream checks retain bounded synthetic scope',ef['checks_passed']==len(ef['checks'])==13 and ef['actual_Hero_cache_contents'] is None and ef['actual_GPU_loss_difference'] is None and ef['actual_PTB_root_cause'] is None)
ok('V42 original stream source identities match',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in ef['source_sha256'].items()))
ok('V42 synthetic fixed windows retain precise document cuts',ef['synthetic']['forward_windows']==[[10,11,0,20],[21,22,23,24]] and ef['synthetic']['reverse_windows']==[[20,21,22,23],[24,0,10,11]])
ok('V42 anonymous access does not prove object existence',len(access['requests'])==2 and all(x['status']==403 and x['response_error_code']=='AccessDenied' for x in access['requests']) and access['object_existence_verified'] is False and access['actual_cache_ledger'] is None)
ok('V42 public source acquisitions remain pinned without data requests',len(pa['hf_revision'])==40 and pa['data_payload_requested'] is False and len(pa['files'])==3 and pa['hf_revision'] in pa['files'][1]['url'])
ok('V42 public source payload checksums match',all(hashlib.sha256((S/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in pa['files']))
ok('V42 format chapter keeps three distinct evidence tables',len(soup.select('#eval-format-guide table'))==3)
report['highlights']=[x.replace('435 source archive checksums valid','438 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['V42 original continuous stream / format / tail checks on synthetic CPU cache; public pinned Paloma card and final paper; anonymous ledger 403; no actual Hero tail or PTB cause claim']
ta=read(A/'eval_target_alignment_validation.json')
ok('V43 target alignment checks retain synthetic and static scope',ta['checks_passed']==len(ta['checks'])==18 and ta['actual_Hero_arrays'] is None and ta['actual_GPU_forward'] is None and ta['actual_loss_array_alignment'] is None)
ok('V43 archived target/mask/length source identities match',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in ta['source_sha256'].items()))
ok('V43 contract marks supplied active next-token check explicitly',ta['synthetic_export']['array_export']['target_contract']=='causal_next_token_v1' and ta['synthetic_export']['array_export']['active_next_token_alignment_verified'] is True)
ok('V43 original mask and length coverage retain exact assumptions',len(ta['synthetic_coverage'])==5 and ta['synthetic_coverage'][1]['default_unit_weight_targets']==6 and all(x['positions_not_scored_as_targets']==x['remainder']+x['full_windows'] for x in ta['synthetic_coverage']))
ok('V43 blank production manifest defaults to causal target contract',read(ROOT/'templates/eval_array_manifest.json')['target_contract']=='causal_next_token_v1')
ok('V43 new target alignment chapter has two evidence tables',len(soup.select('#eval-target-alignment-guide table'))==2)
report['highlights']+=['V43 optional supplied-active-next-token coordinate contract and original default-window coverage checks; generic mode explicitly unverified; no actual model loss alignment, GPU forward or Hero data']
pp=read(A/'parallel_packing_probe.json')
ok('V44 packing probes retain fourteen synthetic/static checks',pp['checks_passed']==len(pp['checks'])==14 and pp['actual_Hero_parallel_field_mismatch'] is None and pp['actual_TensorStore_reads'] is None and pp['actual_GPU_loss'] is None)
ok('V44 original packing and wrapper source identities match',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in pp['source_sha256'].items()))
ok('V44 equal shapes preserve distinct synthetic field boundaries',pp['synthetic_equal_shape_mismatch'][0]['segments']['input_ids']==[0,0,0,1] and pp['synthetic_equal_shape_mismatch'][0]['segments']['loss_weights']==[0,0,1,1])
ok('V44 artificial N T and actual declared applicability stay separate',pp['synthetic_weighted_numerics']=={'matched_T':1.5,'mismatch_T':2.5,'matched_N':2.,'mismatch_N':11.} and pp['declared_components_checked']==223)
ok('V44 packing chapter has three evidence tables',len(soup.select('#packing-fields-guide table'))==3)
report['highlights']=[x.replace('438 source archive checksums valid','439 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['V44 synthetic parallel packing boundaries, clipping/drop/cap and weight-mask checks; declared Hero 223 text non-packing components; no TensorStore/GPU/Hero mismatch claim']
rexp=read(A/'repeat_exposure_validation.json')
ok('V45 repeat exposure checks retain twenty-two bounded tests',rexp['checks_passed']==len(rexp['checks'])==22 and rexp['actual_Hero_inventory'] is None and rexp['actual_Hero_cursor'] is None and rexp['actual_block_shuffle_replay'] is None and rexp['actual_training_benefit'] is None)
ok('V45 repeat helper source identities match',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in rexp['source_sha256'].items()))
ok('V45 new within-window indices may all be historically repeated',rexp['synthetic_after_one_cycle']['window_repeat_draws']==0 and rexp['synthetic_after_one_cycle']['new_sequence_indices_since_lifetime_start']==0 and rexp['synthetic_after_one_cycle']['repeat_draws_against_lifetime_history']==2)
ok('V45 artificial full permutation does not stand for actual block shuffle',rexp['synthetic_order']==[2,0,1,2,0,1,2,0] and rexp['actual_JAX_permutation'] is None)
ok('V45 repeat chapter retains two decision tables',len(soup.select('#repeat-exposure-guide table'))==2)
report['highlights']+=['V45 finite fixed-order repeat exposure projection; window versus lifetime index coverage; original helper and synthetic permutation, no actual Hero inventory/cursor/block shuffle or training benefit']
l7=read(A/'live_analysis_2026_10_07.json')
ok('V46 live observation retains nine bounded checks',l7['checks_passed']==len(l7['checks'])==9 and l7['actual_eval_input_identity'] is None and l7['actual_execution_SHA'] is None and l7['actual_mixture_counterfactual'] is None and l7['statistical_significance'] is None)
ok('V46 fresh and prior W&B payload identities match',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in l7['source_sha256'].items()))
ok('V46 one new exact step preserves 216 old values',l7['new_complete_eval_steps']==[212999] and l7['overlap_values_unchanged']==216)
ok('V46 micro and macro CE directions differ',l7['parent_deltas']['micro_loss']['delta']<0<l7['parent_deltas']['macro_loss']['delta'] and l7['sensitivity']['loss']['equal_subset_delta_without_ptb']<0)
ok('V46 new chapter retains three tables and independent inline figure',len(soup.select('#live-oct7-guide table'))==3 and soup.select_one('#live-oct7-guide svg') is not None)
report['highlights']=[x.replace('439 source archive checksums valid','441 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['V46 independent 9.915T / 4K snapshot; new 212999 evaluation and 216 unchanged old values; micro decline versus macro rebound, descriptive PTB decomposition; no causal or GPU claim']
ew=read(A/'eval_weight_inference.json')
ok('V47 conditional inference retains eight scoped checks',ew['checks_passed']==len(ew['checks'])==8 and ew['actual_domain_denominators'] is None and ew['actual_fixed_weight_identity'] is None and ew['actual_mixture_causal_effect'] is None)
ok('V47 source identities are unchanged',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in ew['source_sha256'].items()))
ok('V47 chronological partition is explicit and post hoc',len(ew['fit_steps'])==18 and ew['posthoc_holdout_steps']==[200999,203999,206999,209999,212999] and max(ew['fit_steps'])<min(ew['posthoc_holdout_steps']))
ok('V47 numerical perturbation is not statistical confidence',ew['numerical_perturbation']['is_confidence_interval'] is False and ew['affine_condition_number']>10000)
ok('V47 latest aggregates do not uniquely identify weights',ew['latest_nonidentifiability']['max_weight_difference']>.02 and ew['latest_nonidentifiability']['max_predicted_parent_difference']<1e-13)
ok('V47 inference chapter has three tables and inline scientific figure',len(soup.select('#eval-weight-guide table'))==3 and soup.select_one('#eval-weight-guide svg') is not None)
report['highlights']+=['V47 conditional eval weight inference, 18 fit and five post-hoc points; eight bounded checks, perturbation sensitivity and nonidentifiability; no actual denominator or training mixture benefit']
ga=read(A/'gradient_accumulation_probe.json')
ok('V48 accumulation checks retain fourteen bounded observations',ga['checks_passed']==len(ga['checks'])==14 and ga['actual_JAX_execution'] is None and ga['actual_Hero_accumulation_bug'] is None and ga['actual_optimizer_trajectory'] is None)
ok('V48 pinned accumulation source identities match',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in ga['source_sha256'].items()))
ok('V48 artificial effective denominators reverse analytic gradient',ga['synthetic_variable_denominator']['full_loss_gradient']==[2.625,.75] and ga['synthetic_variable_denominator']['ordinary_accumulated_loss_gradient']==[3.25,-.5] and ga['synthetic_variable_denominator']['denominator_weighted_loss_gradient']==[2.625,.75])
ok('V48 fixed position objective is preserved by equal-size splitting',ga['synthetic_fixed_positions']['full_loss_gradient']==ga['synthetic_fixed_positions']['accumulated_loss_gradient'])
ok('V48 new chapter preserves three tables and inline synthetic figure',len(soup.select('#gradient-accumulation-guide table'))==3 and soup.select_one('#gradient-accumulation-guide svg') is not None)
report['highlights']=[x.replace('441 source archive checksums valid','443 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['V48 generic accumulation versus specialized Hero path; fourteen original-body/serial/artificial objective checks; unequal denominator gradient reversal, no actual Hero bug or JAX/model/distributed parity']
import numpy as np
mn=read(A/'masked_numerics_cpu.json');dw=read(A/'default_target_weights.json')
ok('V49 genuine CPU probe records twenty-one bounded checks',mn['checks_passed']==len(mn['checks'])==21 and mn['runtime']['jax']==mn['runtime']['jaxlib']=='0.7.2' and mn['runtime']['backend']=='cpu' and mn['actual_GPU_or_TPU_kernel_execution'] is None and mn['actual_optimizer_step'] is None)
ok('V49 CPU function source identities match',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in mn['source_sha256'].items()))
ok('V49 zero mean has finite forward and nonfinite loss cotangents',all(mn['observations']['zero_mean_'+mode]['value']==0 and mn['observations']['zero_mean_'+mode]['gradient']==['NaN','NaN'] for mode in ['eager','jit']))
ok('V49 safe denominator contrast is explicit and locally finite',mn['proposed_safe_denominator_is_upstream_patch'] is False and mn['observations']['safe_zero_eager']['gradient']==[0.,0.] and mn['observations']['safe_zero_jit']['gradient']==[0.,0.])
ok('V49 original finite tail padding matches recorded CPU reference',np.allclose(mn['observations']['scan_positive_tail_padding']['hidden_gradient'],mn['observations']['scan_positive_tail_padding']['reference_hidden_gradient'],atol=1e-6,rtol=1e-6) and np.allclose(mn['observations']['scan_positive_tail_padding']['head_gradient'],mn['observations']['scan_positive_tail_padding']['reference_head_gradient'],atol=1e-6,rtol=1e-6))
ok('V49 default recipe audit records nine conditional checks',dw['checks_passed']==len(dw['checks'])==9 and dw['conditional_default_global_T']==46126080 and dw['actual_global_T'] is None and dw['actual_Hero_zero_T_event'] is None)
ok('V49 default-mask source identities match',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in dw['source_sha256'].items()))
ok('V49 chapter retains three evidence tables and CPU figure',len(soup.select('#masked-numerics-guide table'))==3 and soup.select_one('#masked-numerics-guide svg') is not None)
report['highlights']+=['V49 twenty-one genuine single-device JAX CPU numerical checks and nine conditional default-text-mask checks; forward zero versus reverse NaN, finite padding control, no actual Hero trigger or historical/GPU/TPU/full optimizer binding']
report['not_verified']+=['Actual Hero global zero-denominator or inactive nonfinite operand event and historical runtime binding', 'V49 complete custom_vjp dispatcher, GPU/TPU/distributed or full optimizer-step execution']
report['checks_passed']=len(checks)
(A/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
