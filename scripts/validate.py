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
ok('Four source configurations and six resume windows remain archived',len(list(F.glob('config_*.json')))==4 and len(list(F.glob('window_*.json')))==6 and sum(not x['file'].startswith(('decision_2026_10_04/','engineering_2026_10_05/','scale_2026_10_05/','execution_2026_10_05/','contracts_2026_10_05/','state_2026_10_05/','boundaries_2026_10_05/','cache_2026_10_05/','dedup_2026_10_05/','quality_2026_10_05/','routing_2026_10_05/','optimizer_2026_10_05/','short_conv_2026_10_05/','router_precision_2026_10_05/','eval_metrics_2026_10_05/','checkpoint_commit_2026_10_05/','checkpoint_memory_2026_10_05/','muon_geometry_2026_10_05/','adamh_2026_10_05/','muon_direction_2026_10_05/','watch_2026_10_06/','live_2026_10_06/','eval_identity_2026_10_06/','paloma_protocol_2026_10_06/','packing_2026_10_07/','live_2026_10_07/','accumulation_2026_10_07/','prp_2026_10_07/','engineering_current_2026_10_07/','tree_restore_2026_10_07/','state_restore_2026_10_07/','engineering_v77_2026_10_07/','portable_ep_2026_10_07/','receiver_layout_2026_10_07/','runtime_defaults_2026_10_07/','launch_binding_2026_10_07/','clipping_precision_2026_10_07/','decay_resume_2026_10_07/','phase_budget_2026_10_07/','integer_exposure_2026_10_07/','normalization_runtime_2026_10_07/','loader_background_2026_10_07/','main_incident_2026_10_07/','gang_recovery_2026_10_07/','controller_restore_2026_10_07/','controller_candidate_2026_10_07/','snapshot_semantics_2026_10_07/','donation_snapshot_2026_10_07/','operational_2026_10_07/')) for x in manifest['files'])==313)
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
ok('Five selector provenance files retain fixed revision in expanded archive',len(list((S/'decision_2026_10_04').glob('*.json')))==5 and len(manifest['files'])==566 and read(S/'decision_2026_10_04/marin_head.json')['sha']==read(S/'decision_2026_10_04/marin_tree.json')['sha']==trace['pinned_marin_revision'])
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
ok('V19 inline assignment diagram retains artificial scope',soup.select_one('#routing-guide svg') is not None and len(soup.select_one('#routing-guide svg').select('rect'))==129 and '不是Hero的loss' in soup.select_one('#routing-guide').text)
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
ok('V71 clock chapter retains five historical tables plus boundary-log table',len(soup.select('#batch-clock-guide table'))==9)
report['highlights']+=['V35 exact cumulative batch clock / stage alignment rejection / consumed prefix review / configured weights versus block counts; static HTML verification only for new chapter']
mr=read(A/'mixture_range_probe.json')
ok('V36 integer range probes retain twenty-two bounded checks',mr['checks_passed']==len(mr['checks'])==22 and mr['actual_historical_overflow'] is None and mr['actual_loader_replay'] is None and mr['actual_Hero_runtime_numpy_version'] is None)
ok('V36 original source and archived configuration digests match',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in mr['source_sha256'].items()))
ok('V36 corrupt modulo returns a valid but different synthetic item',mr['synthetic']['corrupt_remapped_index']==553 and mr['synthetic']['expected_remapped_index']==936)
ok('V36 scalar and batch-style types have distinct range behavior',mr['synthetic']['original_within_stage_index']<0 and mr['synthetic']['batched_int64_block_index']==mr['synthetic']['expected_index'])
ok('V36 archived Hero bucket count bounds stay below int32',mr['hero_declared_bounds']['max_component_upper_bound']==256996542<mr['hero_declared_bounds']['int32_limit']<mr['hero_declared_bounds']['total_sequences'])
ok('Range chapter retains historical tables plus runtime and read controls',len(soup.select('#mixture-range-guide table'))==6)
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
ok('V45 repeat chapter retains two decision tables',len(soup.select('#repeat-exposure-guide table'))==3)
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
ok('V48/V76 chapter retains original tables, new control table and historical figure',len(soup.select('#gradient-accumulation-guide table'))==4 and soup.select_one('#gradient-accumulation-guide svg') is not None)
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
zs=read(A/'zero_gradient_state_cpu.json')
ok('V50 fourteen CPU/source optimizer checks preserve original module scope',zs['checks_passed']==len(zs['checks'])==14 and zs['original_AdamH_module_executed'] is True and zs['source_module_altered'] is False and zs['actual_full_train_step'] is None and zs['actual_Hero_optimizer_group_binding'] is None)
ok('V50 original AdamH and declaration source identities match',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in zs['source_sha256'].items()))
ok('V50 warm zero gradient updates despite preserved norm',zs['warm_zero']['update_norm']>.1 and zs['warm_zero']['count_before']==1 and zs['warm_zero']['count_after']==2 and abs(zs['warm_zero']['parameter_norm_after']/zs['warm_zero']['parameter_norm_before']-1)<1e-6)
ok('V50 discarded update does not freeze next-step state',zs['discard_vs_freeze']['next_update_difference_norm']>.001 and zs['discard_vs_freeze']['discard_count_after_next']==3 and zs['discard_vs_freeze']['freeze_count_after_next']==2)
ok('V50 bare CPU jit compatibility failure remains explicitly recorded','nonempty mesh' in zs['bare_CPU_jit_error'] and zs['named_CPU_mesh_axes']==['cpu'])
ok('V50 chapter has three evidence tables and original-module CPU figure',len(soup.select('#zero-gradient-state-guide table'))==3 and soup.select_one('#zero-gradient-state-guide svg') is not None)
report['highlights']+=['V50 full unmodified AdamH module on artificial CPU matrices, fourteen CPU/source checks, nonzero warm-zero update and discard-versus-freeze distinction; bare empty-mesh JIT failure retained, named one-device JIT verified, no Hero full-step binding']
gc=read(A/'group_clipping_cpu.json')
ok('V51 ten clipping CPU/source checks retain bounded labels and build scope',gc['checks_passed']==len(gc['checks'])==10 and gc['actual_Hero_group_assignment'] is None and gc['actual_complete_Hero_optimizer_build'] is None and gc['declared_latest_max_grad_norm'] is None)
ok('V51 clipping source identities match',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in gc['source_sha256'].items()))
ok('V51 real group clipping differs from whole model control',abs(gc['group_control']['group_clipped_combined_norm']-3**.5)<1e-6 and abs(gc['group_control']['whole_clipped_combined_norm']-1)<1e-6)
ok('V51 shared next inputs expose distinct clipping moment history',gc['original_AdamH_control']['first_update_difference_norm']<1e-6 and gc['original_AdamH_control']['common_next_update_difference_norm']>.1 and gc['original_AdamH_control']['next_parameter_input_shared'] is True)
ok('V51/V85 optimizer five prior tables and one precision table retained',len(soup.select('#optimizer-guide table'))==7 and soup.select_one('#optimizer-guide svg') is not None)
report['highlights']+=['V51 real Optax grouped clipping controls and original AdamH fixed-next-input state comparison; ten checks, latest declared clipping disabled, no real Hero group/build/clipping event']
lr=read(A/'loader_resume_probe.json')
ok('V52 eleven original async loader host controls retain scope',lr['checks_passed']==len(lr['checks'])==11 and lr['actual_Hero_next_tokens'] is None and lr['actual_checkpoint_restore'] is None and lr['actual_JAX_batchification'] is None and lr['actual_background_prefetch'] is None)
ok('V52 loader source identities match',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in lr['source_sha256'].items()))
ok('V52 prefetch and completed-step resume are distinct',max(lr['prefetch']['first_store_request'])==19 and lr['prefetch']['resume_completed_step_one']['identities']==[4,5,6,7])
ok('V52 historical rewrite affects original host retrieval',lr['history_change']['old']['offset']==20 and lr['history_change']['rewritten']['offset']==32 and lr['history_change']['future_only']['offset']==20)
ok('V71 batch clock chapter retains five historical tables and one new table',len(soup.select('#batch-clock-guide table'))==9)
report['highlights']+=['V52 eleven original async loader host controls, identity-store retrieval and finite endpoints; no actual Hero token stream, background queue or complete checkpoint restore']
mi=read(A/'mixture_identity_cpu.json')
ok('V53 fourteen original mixture class CPU controls retain scope',mi['checks_passed']==len(mi['checks'])==14 and mi['runtime']['backend']=='cpu' and mi['actual_Hero_mapping'] is None and mi['actual_inner_shuffle'] is None and mi['actual_token_store'] is None)
ok('V53 mixture source identity matches',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in mi['source_sha256'].items()))
ok('V53 dictionary order changes tie allocation',mi['tie_counts']['ABC']=={'A':4,'B':2,'C':2} and mi['tie_counts']['CBA']=={'C':4,'B':2,'A':2})
ok('V53 key preserves whole-block multiset but changes partial content',sorted(mi['whole_seed7'])==sorted(mi['whole_seed8']) and sorted(mi['partial_seed7'])!=sorted(mi['partial_seed8']))
ok('V53 finite restart separates exposure from unique identity',len(mi['finite_length3_stream'])==24 and len(set(mi['finite_length3_stream']))==6)
report['highlights']+=['V53 fourteen original mixture class and real JAX CPU identity controls; ordered dataset IDs, integer ties, finite modulo and partial windows; no actual Hero token/shuffle/restore']
ok('V53 mixture identity chapter retains original and V54 V55 control tables',len(soup.select('#mixture-identity-guide table'))==8)
ish=read(A/'inner_shuffle_cpu.json')
ok('V54 fourteen CPU/source controls with bounded Hero claims',ish['checks_passed']==len(ish['checks'])==14 and ish['runtime']['backend']=='cpu' and ish['actual_Hero_inner_shuffle'] is None and ish['actual_Hero_split_leakage'] is None and ish['declared_num_validation_sequences'] is None)
ok('V54 PRP and dataset source identities match',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in ish['source_sha256'].items()))
ok('V54 original PRPs bijective in 130 small-domain controls',len(ish['permutation_grid'])==130 and all(x['bijective'] for x in ish['permutation_grid']))
ok('V54 fixed snapshot splits are disjoint but cross-snapshot overlap exists',set(ish['split22']['train']).isdisjoint(ish['split22']['validation']) and ish['old_train_new_validation_overlap']==[1])
ok('V54 block tail retains exact partial inventory',set(ish['block22'][-2:])=={20,21} and sorted(ish['block22'])==list(range(22)))
report['highlights']=[x.replace('443 source archive checksums valid','444 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['V54 fourteen original inner-shuffle/split CPU controls including 130 small PRP domains; same-snapshot disjointness versus cross-snapshot overlap, Hero declared split disabled, no actual token contamination']
bi=read(A/'budget_inventory_cpu.json')
ok('V55 sixteen original CPU budget and init controls retain scope',bi['checks_passed']==len(bi['checks'])==16 and bi['runtime']['backend']=='cpu' and bi['actual_Hero_simulation_budget_event'] is None and bi['actual_pilot_model_loss'] is None and bi['declared_experiment_budget'] is None and bi['declared_target_budget'] is None)
ok('V55 original sources and reused harness identities match',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in bi['source_sha256'].items()))
ok('V55 floor distorts artificial repeat factors',abs(bi['ratio_example']['repeat_factor_comparison']['A']['epoch_ratio']-1.75)<1e-12 and abs(bi['ratio_example']['repeat_factor_comparison']['B']['epoch_ratio']-1.15)<1e-12)
ok('V55 empty active inventory is rejected by original restart path',bi['small_zero_example']['inventories']['A']==[] and 'empty finite dataset' in bi['small_zero_example']['original_mix_error'])
ok('V55 cap applies after shuffled prefix and split',bi['shuffled_cap']['cap']==bi['shuffled_cap']['full'][:5] and len(bi['split_then_cap'])==4 and bi['errors']['zero_target']=='ZeroDivisionError')
report['highlights']+=['V55 sixteen original train_sets and post_init CPU controls with explicit cache/key/sync adapters; floor distorts repeat factors and can empty active inventory, no actual Hero simulation or pilot model loss']
ok('V55 original init rejects combinations used only as bypassed function controls',bi['bypassed_initialization_controls'] is True and set(bi['post_init_rejections'])=={'split_and_budget','cap_and_budget'})
rp=read(A/'routing_gradient_proposal_cpu.json')
ok('V56 seven proposal CPU/source checks retain unexecuted GPU scope',rp['checks_passed']==len(rp['checks'])==7 and rp['actual_ragged_MoE_execution'] is None and rp['actual_GPU_reproduction'] is None and rp['actual_Hero_deployment'] is None)
ok('V56 proposal source hashes match',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in rp['source_sha256'].items()))
ok('V56 formula controls retain boundary gradients',rp['cases'][0]['proposal_division_dS']==6 and rp['cases'][1]['proposal_division_dS']==0 and rp['cases'][1]['exact_reference_dS']==6 and rp['cases'][2]['proposal_division_dS']==0 and rp['cases'][2]['exact_reference_dS']>0)
for n in [9832,9833]:
    pr=read(S/f'engineering_current_2026_10_07/pull_{n}.json');fs=read(S/f'engineering_current_2026_10_07/pull_{n}_files.json')
    ok('V56 unmerged PR and complete file connection '+str(n),not pr['merged'] and pr['state']=='open' and pr['changed_files']==len(fs))
ec=read(A/'engineering_current_v56.json')
ok('V56 primary issue bodies and comment IDs unchanged',len(ec['comparisons'])==3 and all(not x['body_changed'] and not x['added'] and not x['removed'] and not x['changed'] for x in ec['comparisons']))
ok('V56/V57 historical tables plus V77 state refresh retained',len(soup.select('#recent-moe-guide table'))==7)
report['highlights']=[x.replace('444 source archive checksums valid','455 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['V56 three full issue refreshes unchanged and two unmerged MoE PR audits; seven patch-statement CPU/source controls, no GPU MoE or production deployment verification']
ps=read(A/'moe_proposal_source_audit.json')
ok('V56 nineteen independent raw API source audits retain deployment unknown',ps['checks_passed']==len(ps['checks'])==19 and ps['actual_deployment'] is None and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in ps['source_sha256'].items()))
pa=read(A/'moe_performance_attribution.json')
ok('V57 source table arithmetic preserves author and unknown scope',pa['checks_passed']==len(pa['checks'])==6 and len(pa['rows'])==18 and len(pa['timed_rows'])==12 and pa['unidentified']['component_independent_causal_effects'] is None and pa['unidentified']['actual_GPU_reproduction'] is None)
ok('V57 frozen performance source hash matches',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in pa['source_sha256'].items()))
ok('V57 conditional reversal retained with raw pairs unknown',pa['shared_epilogue']['sequential_delta_seconds']>0 and pa['shared_epilogue']['author_final_tip_removal_slowdowns']==[.0035,.0022] and pa['shared_epilogue']['raw_pair_step_traces'] is None)
pt=read(ROOT/'templates/performance_interaction_review.json')
ok('V57 interaction template remains unexecuted',pt['status']=='not_executed' and all(pt['comparison'][x] is None for x in ['base','A_only','B_only','A_and_B']) and pt['outputs']['interaction_seconds'] is None)
ok('V57 attribution figure embedded in existing source chapter',soup.select_one('#recent-moe-guide svg') is not None)
report['highlights']+=['V57 six source/arithmetic checks, 12 author one-rack timings, conditional performance reversal figure and unexecuted interaction template; no local GPU or full causal attribution']
re=read(A/'routing_gradient_envelope_cpu.json')
ok('V58 CPU envelope source and unknown scope',re['checks_passed']==len(re['checks'])==10 and len(re['grid'])==48 and re['actual_GPU_reproduction'] is None and re['actual_Hero_input_distribution'] is None and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in re['source_sha256'].items()))
ok('V58 input casting separated from positive product loss',re['summary']['float16']['input_cast_zero']==12 and all(re['summary'][x]['represented_positive_product_loss']==1 for x in ['float16','bfloat16','float32']))
rc=re['router_chain_controls']
ok('V58 local error need not survive router VJP',all(v==0 for k in ['float32_router_vjp_reference','float32_router_vjp_proposal'] for v in rc[0][k]) and any(v!=0 for v in rc[1]['float32_router_vjp_reference']) and all(v==0 for v in rc[1]['float32_router_vjp_proposal']))
rt=read(ROOT/'templates/routing_gradient_acceptance.json')
ok('V58 template remains unexecuted',rt['status']=='not_executed' and all(v is None for v in rt['comparisons'].values()) and rt['production_failure_probability'] is None)
ok('V58 envelope figure embedded',len(soup.select('#recent-moe-guide svg'))>=2)
report['highlights']+=['V58 48 selected scalar CPU controls and two artificial router VJPs, ten checks; no production incidence or GPU/update/loss attribution']
rw=read(A/'router_weight_path_cpu.json')
ok('V59 source hash and unknown scope',rw['checks_passed']==len(rw['checks'])==9 and rw['actual_GPU_execution'] is None and rw['actual_Hero_extreme_logits'] is None and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in rw['source_sha256'].items()))
ok('V59 epsilon and selection controls',rw['cases'][1]['selected_experts']==[[0,1]] and 0<rw['cases'][2]['weight_sum']<1e-10 and rw['cases'][3]['weight_sum']==0 and all(x['eager_jit_equal'] for x in rw['cases']))
ok('V59 surrogate stopped bias and unselected parameter',all(x==0 for x in rw['surrogate']['bias_gradient']) and rw['surrogate']['router_parameter_gradient'][0][2]==0)
report['highlights']+=['V59 original moe_route CPU block with reshard/spec stubs; nine checks, four inputs and surrogate gradients; no complete expert or training attribution']
cu=read(A/'router_coupling_update_cpu.json')
ok('V60 source hashes and untested full production',cu['checks_passed']==len(cu['checks'])==10 and cu['actual_GPU_execution'] is None and cu['actual_Hero_parameter_group'] is None and cu['actual_Hero_loss_effect'] is None and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in cu['source_sha256'].items()))
cf=cu['fixture'];co=cu['optimizer_control']
ok('V60 rounding without underflow retained',all(x>0 for x in cf['rounded_weighted_cotangents'][0]) and sum(a!=b for a,b in zip(cf['dS_reference'][0],cf['dS_candidate'][0]))==1)
ok('V60 selected gradient coupling retained',all(x==0 for x in cf['router_gradient_reference'][0]) and all(x!=0 for x in cf['router_gradient_candidate'][0][:2]) and cf['router_gradient_candidate'][0][2]==0)
ok('V60 state-sensitive updates and diagnostic schedule retained',co['fresh_update_difference_norm']>1e-4 and co['warm_update_difference_norm']<1e-6 and co['actual_live_lr_schedule'] is None and co['leaf_key_paths_stub'])
ok('V60 coupling and V78 portable figures embedded',len(soup.select('#recent-moe-guide svg'))==4)
report['highlights']+=['V60 ten synthetic integration CPU controls, coupling and fresh/warm update figure; constant experts, path/reshard stubs, no full Hero execution or loss attribution']
mi=read(A/'mix_event_identifiability.json')
ok('V61 design audit source binding and causal unknowns',mi['checks_passed']==len(mi['checks'])==9 and mi['actual_independent_mix_effect'] is None and mi['actual_independent_execution_effect'] is None and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in mi['source_sha256'].items()))
ok('V61 both event designs remain deficient',len(mi['retained_steps'])==67 and mi['intermediate_retained_steps']==[] and mi['level_design']['rank']==2 and mi['segmented_design']['rank']==4)
ok('V61 synthetic timestamps are separate',mi['synthetic_design_controls']['one_intermediate']['segmented_rank']==5 and mi['synthetic_design_controls']['two_intermediate']['segmented_rank']==6)
mt=read(ROOT/'templates/mixture_execution_comparison.json')
ok('V61 experiment template unexecuted',mt['status']=='not_executed' and all(x['fixed_eval_results'] is None and x['feasible'] is None for x in mt['arms'].values()))
ok('V61 event figure embedded',soup.select_one('#mix-trajectory-guide svg') is not None)
report['highlights']+=['V61 nine archived design/algebra checks; aliased mix/execution events, no causal estimates; synthetic timestamp rank controls and unexecuted comparison template']
sp=read(A/'swarm_seed_pairs.json')
ok('V62 paired archive audit and unknown confirmation',sp['checks_passed']==len(sp['checks'])==9 and sp['actual_independent_confirmation'] is None and sp['post_selection_p_value'] is None and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in sp['source_sha256'].items()))
ok('V62 all endpoints and mixed direction retained',len(sp['endpoints'])==25 and sp['summary']=={'metrics':25,'all_three_improve':20,'all_three_regress':2,'mixed_direction':3})
ok('V62 three-pair resolution remains synthetic',sp['synthetic_sign_flip_resolution']['patterns']==8 and sp['synthetic_sign_flip_resolution']['two_sided_min_probability']==.25 and sp['synthetic_sign_flip_resolution']['not_a_selected_mixture_p_value'])
ok('V62 paired figure embedded beside event audit',soup.select_one('#mix-trajectory-guide #swarmpair-figure_1') is not None and soup.select_one('#mix-trajectory-guide #mixevent-figure_1') is not None)
report['highlights']+=['V62 six swarm observations, 25 endpoints and 75 descriptive seed-label pairs; nine checks, consistent gains and regressions; no independent confirmation or post-selection p-value']
pv=read(A/'pending_router_view_cpu.json')
ok('V63 original pending/router source hashes and unknown IO',pv['checks_passed']==len(pv['checks'])==9 and not pv['actual_eqx_tree_at'] and pv['actual_checkpoint_IO'] is None and pv['actual_GPU_execution'] is None and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in pv['source_sha256'].items()))
pf=pv['fixture']
ok('V63 same numeric weights differ in expert pairing',pf['stored_view']['combine_weights']==pf['pending_applied_view']['combine_weights'] and pf['stored_view']['selected_experts']==[[1,0]] and pf['pending_applied_view']['selected_experts']==[[0,2]])
ok('V63 nonfinite propagation not a production event',pf['nonfinite_bias_output']==['inf','nan','inf'] and pv['actual_Hero_nonfinite_beta_event'] is None)
ok('V63 state chapter explanation present','V63' in soup.select_one('#state-guide').get_text())
report['highlights']+=['V63 nine JAX CPU/source controls, ID/weight pairing and pending centering with explicit tree_at stub; no checkpoint IO or production view/NaN attribution']
wc=read(A/'weights_consumer_faults.json')
ok('V64 consumer checks retain actual array IO unknown',wc['checks_passed']==len(wc['checks'])==9 and len(wc['cases'])==10 and wc['actual_checkpoint_array_IO'] is None and wc['actual_OCDBT_execution'] is None and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in wc['source_sha256'].items()))
ok('V64 metadata guards precede layout calls',all(wc['cases'][i]['error']=='ValueError' and wc['cases'][i]['calls']==[] for i in [2,3,4,5]))
ok('V64 layout and failure propagation retained',wc['cases'][0]['calls'][1]['requested_keys']==['master_params','pending_qb_betas'] and wc['cases'][7]['calls'][1]['requested_keys']==['params','pending_qb_betas'] and wc['cases'][9]['calls']==['manifest'])
report['highlights']+=['V64 ten original-consumer fault controls, nine checks; real temporary metadata IO with explicit layout/array/digest/template/tree stubs, no actual checkpoint or OCDBT restore']
ti=read(A/'tensorstore_roundtrip.json')
ok('V65 real storage source helpers and production unknowns',ti['checks_passed']==len(ti['checks'])==8 and ti['actual_full_Marin_serializer'] is None and ti['actual_production_missing_chunk_event'] is None and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in ti['source_sha256'].items()))
to=ti['observations'];te=ti['fixture']['expected_values']
ok('V65 complete fresh process versus default fill controls',to['complete_child']['values']==te and to['partial_child']['values'][:2]==te[:2] and to['partial_child']['values'][2:]==[[0.0]*4]*4 and to['deleted_chunk_child']['values']==[[0.0]*4]*2+te[2:])
ok('V65 real storage figure embedded',soup.select_one('#checkpoint-commit-guide #tsio-figure_1') is not None)
report['highlights']+=['V65 eight actual local TensorStore Zarr3/OCDBT IO checks with original spec helpers and independent read processes; synthetic arrays, no full Marin serializer or production restore']
si=read(A/'serialize_arrays_real_io.json')
ok('V66 original host serializer real manager scope and source binding',si['checks_passed']==len(si['checks'])==9 and si['actual_GlobalAsyncCheckpointManager'] and si['actual_HostByteBudget'] and si['actual_full_tree_serializer'] is None and si['actual_multirank_commit'] is None and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in si['source_sha256'].items()))
so=si['observations']
ok('V66 failed future budget release does not trigger success',so['success']['peak_bytes']==so['failure']['peak_bytes']==96 and 'commit_callback' in so['success']['events'] and 'commit_callback' not in so['failure']['events'] and 'local_failed' in so['failure']['events'] and so['failure']['second_array_restored_values']==so['success']['restored_values'][1])
ok('V66 previous attempt error and reader explanation retained',so['failure']['next_save_events']==[] and so['failure']['next_save_error'] is not None and 'V66' in soup.select_one('#checkpoint-commit-guide').get_text())
report['highlights']+=['V66 nine real local IO controls in original host serializer with original budget and installed single-process JAX manager; explicit omissions, no full tree or distributed/production restore']
ri=read(A/'restore_candidate_real_io.json')
ok('V67 original discovery reader source and production boundary',ri['checks_passed']==len(ri['checks'])==9 and ri['original_CheckpointArray_schema'] and ri['original_leaf_read_pipeline'] and ri['actual_full_state_restore'] is None and ri['actual_production_stale_marker_event'] is None and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in ri['source_sha256'].items()))
ro=ri['observations']
ok('V67 marker qualification and partial contents separated',ro['discovered_steps']==[10,20,30] and ro['selected_after_removing_step30_marker']=='step20' and ro['partial']['values'][1][2:]==[[0.0]*4]*4 and 'NOT_FOUND' in ro['missing_array_error'])
ok('V67 stored shape versus manifest field kept explicit',len(ro['manifest_shape_mismatch']['values'][1])==2 and 'V67' in soup.select_one('#checkpoint-commit-guide').get_text())
cr=read(ROOT/'templates/checkpoint_commit_review.json')
ok('V67 four restore contract template results remain unexecuted',cr['status']=='planned_not_executed' and len(cr['restore_contracts'])==4 and all(v is None for row in cr['restore_contracts'].values() for v in row.values()))
report['highlights']+=['V67 nine actual original leaf restore/discovery controls, real manifest schema and fsspec with local StoragePath adapter; synthetic markers, no full state, production publisher or multirank restore']
tc=read(A/'tree_restore_contracts.json')
ok('V68 original wrapper and NamedArray contract source binding',tc['checks_passed']==len(tc['checks'])==7 and len(tc['cases'])==6 and tc['original_load_checkpoint'] and tc['actual_full_model_restore'] is None and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in tc['source_sha256'].items()))
tt=tc['cases']
ok('V68 raw and named shape checks separated',tt[2]['error'] is None and tt[2]['restored_shape']==[2,4] and 'different sizes' in tt[3]['error'])
ok('V68 dtype evidence and original static field merge retained',all(x['error'] is None and x['restored_dtype']=='int32' and x['non_array_label']=='diagnostic' for x in tt[4:]))
ta=read(A/'tree_restore_acquisition.json')
ok('V68 six fixed acquisitions and old bytes preserved except bookkeeping',len(ta['records'])==6 and ta['prior_bytes_preserved']==454 and ta['changed_bookkeeping']==['source_manifest.json'] and all(hashlib.sha256((S/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in ta['records']))
report['highlights']=[x.replace('455 source archive checksums valid','461 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['V68 six actual local wrapper controls and seven checks, original NamedArray class with explicit single-device sharding adapter; no full model, Hero dtype event or execution-package reconstruction']
gs=read(A/'grug_state_restore_real_io.json')
ok('V69 original state policy source binding and actual inventory unknown',gs['checks_passed']==len(gs['checks'])==9 and gs['actual_Hero_state_inventory'] is None and gs['actual_multirank_restore'] is None and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in gs['source_sha256'].items()))
go=gs['observations']
ok('V69 real missing-leaf fallback differs from absent actual array',go['missing_manifest_leaf']['step']==10 and len(go['missing_manifest_leaf']['calls'])==3 and len(go['manifest_listed_array_absent']['calls'])==1 and go['manifest_listed_array_absent']['error'].startswith('ValueError:'))
ok('V69 master migration selects differing content not only dtype',go['migration']['compute_values']!=go['migration']['migrated_values'] and go['migration']['migrated_values']==[1.375,-2.375,3.875] and 'V69' in soup.select_one('#checkpoint-commit-guide').get_text())
ga=read(A/'state_restore_acquisition.json')
ok('V69 single fixed acquisition old source bytes preserved except bookkeeping',len(ga['records'])==1 and ga['prior_bytes_preserved']==460 and ga['changed_bookkeeping']==['source_manifest.json'])
report['highlights']=[x.replace('461 source archive checksums valid','462 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['V69 nine original Grug-schema policy controls with six-leaf real IO, actual Optax fixture and explicit payload/barrier/publisher adapters; no full Hero state or production incident']
dc=read(A/'restore_data_clock.json')
ok('V70 joined restore/data source binding and actual tokens unknown',dc['checks_passed']==len(dc['checks'])==7 and dc['actual_Hero_next_tokens'] is None and dc['actual_full_training_step'] is None and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in dc['source_sha256'].items()))
do=dc['observations']
ok('V70 marker selection versus state data clock retained',do['checkpoint']['marker_step']==100 and do['checkpoint']['loaded_state_step']==20 and do['checkpoint']['selected_calls']==['candidate'] and do['restored_step_batch']['offset']==148 and do['marker_step_counterfactual_batch']['offset']==788)
ok('V70 history rewrite and original stage boundary retained',do['rewritten_history_batch']['offset']==160 and do['mixture_boundary_batch']['offset']==156 and do['restored_step_batch']['identities']!=do['rewritten_history_batch']['identities'])
ok('V70 measured identity order figure embedded',soup.select_one('#batch-clock-guide #restoreclock-figure_1') is not None)
report['highlights']+=['V70 seven joined original-policy/loader/mixture controls and measured sample-order figure; synthetic marker and identity data, no Hero token stream, training step or loss attribution']
bl=read(A/'mixture_boundary_logging.json')
ok('V71 original callback mixture sources and true loss unknown',bl['checks_passed']==len(bl['checks'])==6 and bl['actual_Hero_loss_change'] is None and bl['actual_Hero_boundary_crossing'] is None and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in bl['source_sha256'].items()))
bo=bl['observations']
ok('V71 normal builder-conversion alignment retained',bo['normal_before']['domain_counts']=={'A':8} and bo['normal_after']['domain_counts']=={'B':8} and bl['converted_sequence_boundaries']=={'old':156,'new':168})
ok('V71 completed callback clock matches returned batch',all(x['callback_step_info']['step']==x['batch']['step'] and x['callback_step_info']['next_step']==x['batch']['step']+1 for x in bo.values()))
ok('V71 artificial mixed batch differs from start-stage configured log',bo['frozen_stages_new_loader_crossing']['domain_counts']=={'A':4,'B':4} and bo['frozen_stages_new_loader_crossing']['stage_log']['values']['mixture/weight/A']==1)
bt=read(ROOT/'templates/mixture_boundary_review.json')
ok('V71 boundary review has no unexecuted loss results',bt['status']=='planned_not_executed' and all(v is None for v in bt['loss_evidence'].values()) and bt['decision']['loss_attribution_supported'] is None)
ok('V71 boundary-count figure embedded',soup.select_one('#batch-clock-guide #mixboundary-figure_1') is not None)
report['highlights']+=['V71 five original callback/loader/mixture controls, six checks and measured domain-count figure; normal alignment and deliberately mismatched schedule separated, no measured loss or Hero crossing event']
lc=read(A/'loss_composition_cpu.json')
ok('V72 source-bound original loss controls retain real Hero effect unknown',lc['checks_passed']==len(lc['checks'])==7 and lc['actual_Hero_loss_effect'] is None and lc['actual_GPU_kernel'] is None and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in lc['source_sha256'].items()))
lo=lc['observations']
ok('V72 fixed prediction target composition changes mean',abs(lo['fixed_prediction_losses']['A1_B3']-lo['fixed_prediction_losses']['A3_B1']-1.5)<1e-6 and lo['weighted_target_mass']['fractional_A']==[1.5,1])
ok('V72 logit shift pure CE versus z-loss distinguished',abs(lo['pure_ce']-lo['shifted_pure_ce'])<2e-6 and lo['shifted_with_z_loss']>lo['with_z_loss']>lo['pure_ce'] and 'V72' in soup.select_one('#loss-triage').get_text())
report['highlights']+=['V72 seven original-loss CPU controls, fixed-prediction composition and output-penalty metric checks; synthetic forward/reference delegate, no Hero loss attribution or GPU execution']
lr=read(A/'loss_cross_replay_cpu.json')
ok('V73 four-cell source binding retains unknown training attribution',lr['checks_passed']==len(lr['checks'])==18 and lr['actual_Hero_loss_attribution'] is None and lr['actual_mixture_training_experiment'] is None and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in lr['source_sha256'].items()))
la=lr['observations']['pure_ce']['allocation']
ok('V73 reference-distribution reversal and interaction retained',la['prediction_at_old_weights']<0<la['prediction_at_new_weights'] and abs(la['interaction']-2)<1e-6 and abs(la['total']-la['composition_at_old_prediction']-la['prediction_at_old_weights']-la['interaction'])<1e-12)
lt=read(ROOT/'templates/loss_cross_replay_review.json')
ok('V73 missing cell and unexecuted receipt never filled',lr['missing_cell_control']['allocation'] is None and lt['status']=='planned_not_executed' and all(v is None for cells in lt['cells'].values() for v in cells.values()))
ok('V73 four-cell figure embedded',soup.select_one('#loss-triage #losscross-figure_1') is not None)
report['highlights']+=['V73 eighteen original-loss CPU cross-replay checks and descriptive interaction figure; common synthetic targets, missing-cell rejection, no Hero attribution or training effect']
te=read(A/'tagged_eval_accumulator_cpu.json')
ok('V74 source-bound evaluator fixture retains production impact unknown',te['checks_passed']==len(te['checks'])==12 and te['actual_Hero_affected'] is None and te['actual_GPU_execution'] is None and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in te['source_sha256'].items()))
to=te['observations']
ok('V74 root versus parent empty-domain behavior preserved',to['separate']['root_macro_CE']==1 and to['separate']['parent_macro_CE']==2 and to['separate']['leaf_CE']['paloma/B']==0)
ok('V74 fractional root denominator is partition-sensitive',to['fractional']['micro_CE']==.5 and to['fractional']['leaf_CE']['paloma/A']==2 and to['root_fractional_repartition']['micro_CE']==.5 and to['root_fractional_joined']['micro_CE']==1)
ok('V74 masked nonfinite propagation and finite control separated',to['finite_zero_after']['micro_CE']==2 and to['nan_zero_after']['micro_CE']=={'nonfinite':'nan'} and to['nan_zero_after']['leaf_CE']['paloma/A']==2 and to['nan_zero_after']['leaf_BPB']['paloma/A']=={'nonfinite':'nan'} and 'V74' in soup.select_one('#eval-metrics-guide').get_text())
report['highlights']+=['V74 twelve original evaluator CPU controls, true Equinox state and original result construction; explicit adapters, root/parent empty policy, fractional denominator and NaN propagation; no Hero incident or GPU proof']
es=read(A/'eval_callback_shapes_cpu.json')
ok('V75 original callback controls retain actual production impact unknown',es['checks_passed']==len(es['checks'])==16 and len(es['cases'])==12 and es['actual_Hero_malformed_callback'] is None and es['actual_upstream_fix'] is None and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in es['source_sha256'].items()))
ec={x['case']:x for x in es['cases']}
ok('V75 silent numerator and byte broadcasting preserved',ec['weight_column_broadcast']['source_error'] is None and ec['weight_column_broadcast']['source_result']['micro_CE']==4 and ec['token_column_broadcast']['source_result']['micro_CE']==2 and abs(ec['token_column_broadcast']['source_result']['micro_BPB']-2*ec['valid']['source_result']['micro_BPB'])<1e-6 and all(ec[n]['offline_gate_error'] is not None for n in ['weight_column_broadcast','token_column_broadcast','tag_row_broadcast','loss_column_broadcast']))
ok('V75 original multitag support differs from exclusive gate',es['original_multitag_arrays']==[[1,1],[0,1]] and es['overlap_gate_receipt']['tag_mass']==[2,4] and ec['multi_membership']['offline_gate_error'] is not None and abs(ec['multi_membership']['source_result']['parent_micro_CE']-5/3)<1e-6)
ok('V75 finite wrong index and semantic identity limits retained',ec['negative_token_ids']['source_error'] is None and ec['oversized_token_ids']['source_error'] is None and ec['swapped_tag_columns']['offline_gate_error'] is None and ec['swapped_tag_columns']['source_result']['leaf_CE']['paloma/A']==3 and 'V75' in soup.select_one('#eval-metrics-guide').get_text())
report['highlights']+=['V75 sixteen original evaluator callback controls and report-side structural gate; explicit overlap policy and semantic-identity limits, no Hero malformed callback, GPU or production fix']
mc=read(A/'microbatch_loss_cpu.json')
ok('V76 source-bound true CPU microbatch controls retain Hero path unknown',mc['checks_passed']==len(mc['checks'])==10 and mc['actual_Hero_accumulation_bug'] is None and mc['actual_upstream_fix'] is None and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in mc['source_sha256'].items()))
mo=mc['observations']
ok('V76 ordinary gradient reversal versus weighted recovery',mo['positive_blocks']['full']['gradient'][0][0]>0>mo['positive_blocks']['ordinary']['gradient'][0][0] and mo['positive_blocks']['weighted_mean']==mo['positive_blocks']['full'])
ok('V76 empty local mean stays nonfinite after zero scaling',mo['one_empty_block']['mass']==[0,3] and mo['one_empty_block']['ordinary']['gradient']==[[{'nonfinite':'nan'},{'nonfinite':'nan'}]] and mo['one_empty_block']['weighted_mean']['gradient']==mo['one_empty_block']['ordinary']['gradient'])
ok('V76 original numerator control recovers finite whole-step gradient',mo['one_empty_block']['numerator_global']==mo['one_empty_block']['full'] and 'V76' in soup.select_one('#gradient-accumulation-guide').get_text())
report['highlights']+=['V76 ten original microbatch/CE CPU autodiff controls with explicit scan/physical/sharding adapters; local-zero gradient contamination and numerator-first control, no Hero path or GPU fix proof']
ev=read(A/'engineering_current_v77.json');ea=read(A/'engineering_v77_acquisition.json')
ok('V77 complete refresh and pinned acquisitions preserve prior bytes',ev['checks_passed']==len(ev['checks'])==12 and len(ea['records'])==16 and ea['prior_bytes_preserved']==461 and all(hashlib.sha256((S/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in ea['records']))
ok('V77 latest run entry translated without confirming node root cause',ev['comparisons'][1]['comments']==59 and ev['comparisons'][1]['added_ids']==[6025364160] and ev['new_comment']['underlying_taint_condition_confirmed'] is False and ev['new_comment']['url'] in operations)
ok('V77 closed umbrella remains unmerged and split proposals open',ev['pulls'][0]['state']=='closed' and ev['pulls'][0]['merged'] is False and all(x['state']=='open' and x['merged'] is False for x in ev['pulls'][1:]))
ok('V77 two computational AST comparisons and portable residual correction',len(ev['ast_comparisons'])==2 and all(x['computational_ast_equal'] for x in ev['ast_comparisons']) and 'V77' in soup.select_one('#recent-moe-guide').get_text() and ev['actual_GPU_test'] is None)
report['highlights']=[x.replace('462 source archive checksums valid','478 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['V77 sixteen new public acquisitions and twelve checks; new NoExecute retry report, unmerged closure, doc-only computational AST comparison and portable residual scope correction; no production logs, GPU or deployment proof']
pe=read(A/'portable_expert_mlp_cpu.json');pa=read(A/'portable_ep_acquisition.json')
ok('V78 original portable CPU controls bound to fixed sources',pe['checks_passed']==len(pe['checks'])==14 and pe['actual_GPU_execution'] is None and pe['actual_transport_poison_test'] is None and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in pe['source_sha256'].items()))
pc=pe['numeric_coverage'];po=pe['observations']
ok('V78 inactive row-dot versus finite ordinary and selected gradients',pc['row_dot']=={'elements':10,'finite':7,'nonfinite':3} and all(pc[k]['nonfinite']==0 for k in ['dx','dw13','dw2','selected_weight_gradient']) and po['edge_cases'][1]['ordinary_gradient_norms']==[0,0,0])
ok('V78 saved-output branch and figure retained',po['mutated_saved_output_control']['ordinary_gradients_equal'] and po['mutated_saved_output_control']['row_dot_before']!=po['mutated_saved_output_control']['row_dot_after'] and po['mutated_saved_output_control']['actual_corruption_event'] is None and soup.select_one('#recent-moe-guide #portableep-figure_1') is not None)
ok('V78 one acquisition and old bytes preserved except bookkeeping',len(pa['records'])==1 and pa['prior_bytes_preserved']==477 and all(hashlib.sha256((S/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in pa['prior_integrity_manifest']['files'] if x['file']!='source_manifest.json'))
report['highlights']=[x.replace('478 source archive checksums valid','479 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['V78 fourteen original portable expert/CPU ragged wrapper checks and measured finite-element figure; dense gradient reference, allowed inactive row-dot and saved-output control, no GPU/transport/production proof']
rl=read(A/'receiver_layout_cpu.json');ra=read(A/'receiver_layout_acquisition.json')
ok('V79 original receiver metadata controls source-bound with real domain bias unknown',rl['checks_passed']==len(rl['checks'])==10 and rl['actual_GPU_execution'] is None and rl['actual_domain_drop_bias'] is None and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in rl['source_sha256'].items()))
ro=rl['observations']
ok('V79 aggregate room still locally drops',ro['aggregate_room']['demand']==ro['aggregate_room']['total_logical_capacity']==32 and ro['aggregate_room']['accepted']==27 and ro['aggregate_room']['dropped']==5)
ok('V79 numbering changes identity not global acceptance count',ro['low_capacity']['accepted']==ro['expert_relabeling']['accepted']==12 and ro['low_capacity']['accepted_by_sender']==[8,4] and ro['expert_relabeling']['accepted_by_sender']==[6,6] and all(x['accepted_identity_roundtrip'] for x in ro.values()) and soup.select_one('#routing-guide #receiverlayout-figure_1') is not None)
ok('V79 one new helper and all old non-bookkeeping bytes preserved',len(ra['records'])==1 and ra['prior_bytes_preserved']==478 and all(hashlib.sha256((S/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in ra['prior_integrity_manifest']['files'] if x['file']!='source_manifest.json'))
report['highlights']=[x.replace('479 source archive checksums valid','480 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['V79 ten original JAX clip/offset/chunk-plan checks with host identity replay and measured capacity figure; local capacity and numbering policy, no collective/GPU/domain-bias proof']
pc=read(A/'post_clip_router_cpu.json');pcc=pc['cases']
ok('V80 original route mask combine source bound with production unknown',pc['checks_passed']==len(pc['checks'])==15 and pc['actual_Hero_domain_effect'] is None and pc['actual_GPU_execution'] is None and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in pc['source_sha256'].items()))
ok('V80 dropped selected logit retains normalization gradient',pcc['keep_first']['pre_clip_weight_gradient'][0][1]==0 and pcc['keep_first']['router_logit_gradient'][0][1]<-.49)
ok('V80 same count drops differ in weight mass and output',pcc['keep_first']['retained_weight_mass']>pcc['keep_second']['retained_weight_mass'] and pcc['keep_first']['output']>pcc['keep_second']['output'])
ok('V80 altered renormalization explicitly separate and figure present',pc['post_clip_renormalized_control']['output']==7.5 and pc['post_clip_renormalized_control']['upstream_fix'] is False and soup.select_one('#routing-guide #postclip-figure_1') is not None)
report['highlights']+=['V80 fifteen original route/mask/portable combine CPU checks; retained mass and normalization gradients, no production/GPU attribution']
pp=read(A/'portable_router_precision_cpu.json')
ok('V81 same-head real portable router controls source bound',pp['checks_passed']==len(pp['checks'])==17 and len(pp['cases'])==9 and pp['actual_GPU_execution'] is None and pp['actual_Hero_precision_incident'] is None and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in pp['source_sha256'].items()))
ph=[c for c in pp['cases'] if c['dtype']=='float16']
ok('V81 float16 positive weight weighted cotangent vanishes and reaches router',ph[-1]['weights'][0][1]>0 and ph[-1]['weighted_cotangent_sorted'][0][0]==0 and ph[-1]['dweight_expert_side'][0][1]==0 and ph[-1]['router_gradient_delta_norm']>0)
ok('V81 rounded-up subnormal amplifies local recovered gradient',ph[1]['dweight_expert_side'][0][1]>1.9*ph[1]['dweight_exact'][0][1] and len(pp['original_combine_transpose_statements'])==3)
ok('V81 finite ordinary expert gradients retained across all controls',all(c['ordinary_gradients_finite'] for c in pp['cases']) and 'V81' in soup.select_one('#routing-guide').get_text())
report['highlights']+=['V81 seventeen same-head router and actual portable expert CPU checks; nine positive-weight precision controls plus accepted-zero boundary, no optimizer/GPU/production claim']
rt=read(A/'runtime_defaults.json');rta=read(A/'runtime_defaults_acquisition.json')
ok('V82 original host helper source-bound without production/backend claims',rt['checks_passed']==len(rt['checks'])==13 and len(rt['cases'])==6 and rt['actual_production_deployment'] is None and rt['actual_GPU_execution'] is None and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in rt['source_sha256'].items()))
ok('V82 explicit and residual runtime values preserve old memory budget',rt['cases']['clean_offload']['memory_fraction']=='0.78' and rt['cases']['inherited_offload']['memory_fraction']=='.75' and rt['cases']['same_process_transition']['memory_fraction']=='0.75' and rt['cases']['same_process_transition']['parsed_flag_values']['--xla_gpu_memory_limit_slop_factor']=='85')
ok('V82 fresh public proposals unmerged and complete issue comments',all(p['state']=='open' and p['merged'] is False for p in rt['current_pulls']) and rt['issue_8506_comments']==59 and 'V82' in soup.select_one('#routing-guide').get_text())
ok('V82 seven new sources retain 479 prior non-bookkeeping bytes',len(rta['records'])==7 and rta['prior_bytes_preserved']==479 and all(hashlib.sha256((S/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in rta['records']) and all(hashlib.sha256((S/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in rta['prior_integrity_manifest']['files'] if x['file']!='source_manifest.json'))
report['highlights']=[x.replace('480 source archive checksums valid','487 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['V82 thirteen original host runtime helper/static configuration checks; seven fresh sources and exact prior preservation, production deployment/environment/GPU unknown']
lb=read(A/'launch_binding.json');lba=read(A/'launch_binding_acquisition.json')
ok('V83 launch host checks source bound with actual child/plugin unknown',lb['checks_passed']==len(lb['checks'])==13 and lb['actual_child_environment'] is None and lb['actual_plugin_loaded'] is None and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in lb['source_sha256'].items()))
ok('V83 forwarding exclusions and independent metadata restoration distinct',len(lb['omitted_at_forwarding_stage'])==4 and 'JAX_PLATFORMS' not in lb['dispatch_capture']['forwarded'] and 'GIT_COMMIT' not in lb['dispatch_capture']['forwarded'] and lb['independently_augmented_environment']['GIT_COMMIT']=='synthetic-parent-commit')
ok('V83 lexical PJRT guard controls not loaded plugin evidence',len(lb['guard_cases'])==6 and [c['passed'] for c in lb['guard_cases']]==[False,False,False,True,True,True] and all(c['actual_plugin_loaded'] is None for c in lb['guard_cases']) and 'V83' in soup.select_one('#routing-guide').get_text())
ok('V83 three full modules retain prior 486 non-bookkeeping source bytes',len(lba['records'])==3 and lba['prior_bytes_preserved']==486 and all(hashlib.sha256((S/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in lba['records']) and all(hashlib.sha256((S/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in lba['prior_integrity_manifest']['files'] if x['file']!='source_manifest.json'))
report['highlights']=[x.replace('487 source archive checksums valid','490 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['V83 thirteen original host launcher/forwarding/metadata guard checks plus independent downstream augmentation; no child/plugin/GPU execution']
dr=read(A/'diagnostic_routes.json')
ok('V84 seven curated routes retain actual reader/training unknown',len(dr['routes'])==7 and len({r['id'] for r in dr['routes']})==7 and dr['actual_reader_comprehension_measurement'] is None and dr['actual_user_training_replay'] is None)
de=[e for r in dr['routes'] for e in r['evidence']]
ok('V84 ten source-control result records exact and counts matched',len(de)==len({e['file'] for e in de})==10 and all(hashlib.sha256((ROOT/e['file']).read_bytes()).hexdigest()==e['sha256'] and read(ROOT/e['file'])['checks_passed']==e['checks_passed'] for e in de))
ok('V84 native routes have visible evidence scope and all decision fields',len(soup.select('#synthesis-guide #diagnostic-routes details'))==7 and all(len(d.select('summary'))==1 and len(d.select('.route-boundary'))==1 for d in soup.select('#diagnostic-routes details')))
ok('V84 synthesis historical decision tables intact and chapters linked',len(soup.select('#synthesis-guide table'))==2 and all((ROOT/r['chapter']).exists() for r in dr['routes']))
report['highlights']+=['V84 seven native symptom routes bind ten prior result records; curated navigation/control plans, no new training or comprehension results']
cp=read(A/'clipping_precision_cpu.json');cpa=read(A/'clipping_precision_acquisition.json')
ok('V85 clipping optional and source bound with actual Hero incident unknown',cp['checks_passed']==len(cp['checks'])==19 and cp['archived_Hero_max_grad_norm'] is None and cp['actual_Hero_clipping_incident'] is None and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in cp['source_sha256'].items()))
cf=[c for c in cp['cases'] if c['dtype']=='float16']
ok('V85 finite float16 gradient norm overflow and underflow controls retained',len(cp['cases'])==6 and cf[0]['source_norm']=='Infinity' and cf[0]['direct_clipped_norm_float64']==0 and cf[1]['source_norm']==0 and cf[1]['direct_clipped_norm_float64']>19*cf[1]['threshold'])
ok('V85 collapsed clip still updates original warm AdamH state',cp['warm_AdamH']['direct_warm_update_norm']>.18 and cp['warm_AdamH']['update_difference_norm']>.01 and cp['warm_AdamH']['count_after_direct']==2 and 'V85' in soup.select_one('#optimizer-guide').get_text())
ok('V85 two same-head modules preserve 489 old non-bookkeeping files',len(cpa['records'])==2 and cpa['prior_bytes_preserved']==489 and all(hashlib.sha256((S/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in cpa['records']) and all(hashlib.sha256((S/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in cpa['prior_integrity_manifest']['files'] if x['file']!='source_manifest.json'))
report['highlights']=[x.replace('490 source archive checksums valid','492 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['V85 nineteen optional clip precision and original same-head AdamH CPU checks; production clipping declared disabled, no actual Hero incident/dtype/GPU claim']
dc=read(A/'decay_resume_cpu.json');dca=read(A/'decay_resume_acquisition.json')
ok('V86 original nested path and real Adam controls source bound',dc['checks_passed']==len(dc['checks'])==15 and dc['actual_Hero_clock_mismatch'] is None and dc['actual_Hero_state_restore'] is None and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in dc['source_sha256'].items()))
ok('V86 real mask scope and five state controls retained',len(dc['paths'])==6 and dc['decay_mask']==[True,False,True,False,False,False] and len(dc['controls'])==5 and all(c['next_moments_equal_to_plain_Adam'] for c in dc['controls']))
ok('V86 horizon and count change distinct mechanisms',dc['controls'][1]['decay_coefficient']==.015 and dc['controls'][0]['decay_coefficient']==.01 and dc['controls'][3]['router_adaptive_direction'][0][0]>2.3 and dc['controls'][4]['decay_coefficient']==0 and 'V86' in soup.select_one('#optimizer-guide').get_text())
ok('V86 one same-head source preserves 491 prior non-bookkeeping bytes',len(dca['records'])==1 and dca['prior_bytes_preserved']==491 and all(hashlib.sha256((S/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in dca['prior_integrity_manifest']['files'] if x['file']!='source_manifest.json'))
report['highlights']=[x.replace('492 source archive checksums valid','493 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['V86 fifteen real Adam and original nested-path CPU checks; state/horizon/bias correction controls, no disk restore or Hero clock/GPU proof']
pb=read(A/'phase_budget_probe.json');pba=read(A/'phase_budget_acquisition.json');pbc={c['case']:c for c in pb['cases']}
ok('V87 original phase builder controls source-bound with actual effects unknown',pb['checks_passed']==len(pb['checks'])==15 and len(pb['cases'])==9 and pb['actual_Hero_phase_regression'] is None and pb['actual_Hero_overexposure'] is None and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in pb['source_sha256'].items()))
ok('V87 horizon rebound and short phase collisions explicit',pb['artificial_resume_control']['old_phase']==1 and pb['artificial_resume_control']['rebuilt_phase']==0 and [p['phase_index'] for p in pbc['short_diagnostic']['phase_stages']]==[0,2] and soup.select_one('#batch-clock-guide #phasebudget-figure_1') is not None)
ok('V87 fixed spec cap and analytic threshold scope retained',pbc['double_horizon']['nominal_weighted_peak_epochs']>13 and pbc['hero_recipe']['nominal_weighted_peak_epochs']<8 and pbc['threshold_equal']['target_budget'] is not None and pbc['threshold_above']['target_budget'] is None)
ok('V87 three same-head sources preserve 492 prior non-bookkeeping bytes',len(pba['records'])==3 and len({x['revision'] for x in pba['records']})==1 and pba['prior_bytes_preserved']==492 and all(hashlib.sha256((S/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in pba['prior_integrity_manifest']['files'] if x['file']!='source_manifest.json'))
report['highlights']=[x.replace('493 source archive checksums valid','496 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['V87 fifteen original Harrier builder/spec/helper host checks; nine stage/budget controls and nominal exposure diagram, no actual token stream or GPU claim']
ie=read(A/'integer_exposure_cpu.json');iea=read(A/'integer_exposure_acquisition.json');iec={c['cell']:c for c in ie['cells']}
ok('V88 original quota and permutation controls source bound',ie['checks_passed']==len(ie['checks'])==18 and ie['actual_Hero_token_exposure'] is None and ie['actual_Hero_logical_assignment_replay'] is None and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in ie['source_sha256'].items()))
ok('V88 200 cells preserve zero support and integer exposure distinction',len(iec)==200 and iec['c22q0']['continuous_sequence_budget']>0 and iec['c22q0']['exact_sequence_budget_seed7']==0 and iec['c27q0']['integer_full_token_epochs_seed7']<iec['c27q0']['continuous_full_token_epochs'])
ok('V88 partial tail keys and global budgets accounted',len(ie['tail_controls'])==2 and sum(c['exact_sequence_budget_seed7'] for c in ie['cells'])==sum(c['exact_sequence_budget_seed8'] for c in ie['cells'])==4395787264 and iec['c27q0']['exact_sequence_budget_seed7']!=iec['c27q0']['exact_sequence_budget_seed8'])
ok('V88 one source retains all prior 495 non-bookkeeping sources',len(iea['records'])==1 and iea['prior_bytes_preserved']==495 and all(hashlib.sha256((S/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in iea['prior_integrity_manifest']['files'] if x['file']!='source_manifest.json'))
report['highlights']=[x.replace('496 source archive checksums valid','497 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['V88 eighteen original quota/packed ID/CPU permutation checks and 200-cell logical exposure CSV; no production key/token identity claim']
nr=read(A/'normalization_runtime_compare.json');nra=read(A/'normalization_runtime_acquisition.json')
ok('V89 actual runtime records exact with production unknown',nr['checks_passed']==len(nr['checks'])==7 and nr['actual_Hero_quota_replay'] is None and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in nr['runtime_record_sha256'].items()))
ok('V89 both original runtime controls source-bound and covered',all(read(ROOT/p)['checks_passed']==54 and len(read(ROOT/p)['cases'])==27 and all(hashlib.sha256((ROOT/q).read_bytes()).hexdigest()==h for q,h in read(ROOT/p)['source_sha256'].items()) for p in nr['runtime_records']))
ok('V89 same logged cooldown changes 184 quotas with scope correction',next(x for x in nr['rows'] if x['source']=='historical_wandb' and x['phase']==2)['changed_cell_count']==184 and 'V89对本段补证' in soup.select_one('#mixture-range-guide').get_text())
ok('V89 one official source preserves 496 prior non-bookkeeping bytes',len(nra['records'])==1 and nra['prior_bytes_preserved']==496 and all(hashlib.sha256((S/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in nra['prior_integrity_manifest']['files'] if x['file']!='source_manifest.json'))
report['highlights']=[x.replace('497 source archive checksums valid','498 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['V89 original normalization in two actual Python runtimes, isolated sum controls and historical scope correction; production quota and token stream unknown']
ms=read(A/'mixture_support_cpu.json');msb=read(A/'mixture_support_source_binding.json')
ok('V90 current original-class support controls source-bound',ms['checks_passed']==len(ms['checks'])==25 and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in ms['source_sha256'].items()) and ms['actual_Hero_children'] is None)
ok('V90 low-level counterexample and normal entry guard distinguished',ms['cases']['missing_positive']=={'A':6,'B':4} and not ms['cases']['config_guard_dict']['accepted'] and ms['cases']['declared_typo_component_guard']['accepted'] and len(ms['archived_support_comparison'])==9 and all(not x['unknown_positive_inventory_keys'] for x in ms['archived_support_comparison']))
ok('V90 live fixed-head full module bound to unchanged archived bytes',msb['exact_bytes_equal'] and msb['revision']=='b65be4c9550c5097f0a3add08933531a1c24d534' and msb['download_sha256']==msb['archived_sha256']==hashlib.sha256((ROOT/msb['archived_file']).read_bytes()).hexdigest())
report['highlights']+=['V90 twenty-five original mixture/entry support controls; guard versus actual children distinguished, no Hero missing-child incident claim']
cc=read(A/'component_construction_cpu.json')
ok('V91 component construction source-bound with explicit IO limits',cc['checks_passed']==len(cc['checks'])==15 and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in cc['source_sha256'].items()) and cc['actual_Hero_empty_concat'] is None and cc['actual_token_stream'] is None)
ok('V91 normal missing train cache differs from empty concat chain',cc['cases']['ordinary_train_missing']['error']=='ValueError' and cc['cases']['empty_concat_actual_names']==['A'] and cc['cases']['empty_concat_integer_quota']==[10] and len(cc['cases']['empty_concat_identity'])==10)
ok('V91 partial validation content and archived scope distinguished',len(cc['cases']['partial_concat_identity'])==3 and len(cc['cases']['full_concat_identity'])==6 and cc['archived_components']=={'count':223,'concat_shapes':[]})
report['highlights']+=['V91 fifteen original component/config/build/concat/mixture checks; missing training cache protection, partial validation panel and empty concat boundary, no Hero incident claim']
cd=read(A/'cache_dispatch_probe.json')
ok('V92 real-thread original cache dispatch checks source-bound',cd['checks_passed']==len(cd['checks'])==10 and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in cd['source_sha256'].items()) and cd['actual_distributed_hang'] is None)
ok('V92 completion order does not become build order',cd['cases']['completion_cba']['observed_completion']==['C','B','A'] and cd['cases']['completion_bac']['observed_completion']==['B','A','C'] and cd['cases']['completion_cba']['build_dispatch']==cd['cases']['completion_bac']['build_dispatch']==['A','B','C'])
ok('V92 error cleanup and visibility boundaries explicit',cd['cases']['metadata_failure']['error_observed_before_caller_exit'] and cd['cases']['metadata_failure']['cleanup_exception_type']=='RuntimeError' and cd['cases']['metadata_failure']['build_dispatch']==[] and cd['cases']['hit_A']['build_dispatch']!=cd['cases']['hit_B']['build_dispatch'])
report['highlights']+=['V92 ten original build_caches real-thread controls; preserved dispatch order and waiting exception cleanup, no distributed IO or Hero hang claim']
dc=read(A/'data_record_checker_probe.json');dca=read(A/'data_execution_record_archived_check.json')
ok('V93 proposed checker controls explicitly not source execution',dc['checks_passed']==len(dc['checks'])==18 and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in dc['source_sha256'].items()) and dc['actual_production_execution'] is None)
ok('V93 archived missing evidence not silently promoted',dca['status']=='needs_evidence' and dca['production_execution_verified'] is False and dca['gates'][0]['status']=='consistent' and all(x['status']=='missing' for x in dca['gates'][1:]))
ok('V93 synthetic consistency never approves production',all(x['production_execution_verified'] is False and x['training_benefit_verified'] is False for x in dc['cases'].values()) and dc['cases']['Empty template cannot pass']['status']=='needs_evidence')
report['highlights']+=['V93 proposed record checker with eighteen tamper controls; archived declarations plus local quota retain missing execution evidence, no production approval']
sd=read(A/'strict_data_record_probe.json')
ok('V94 strict tool source-bound regression records',sd['checks_passed']==len(sd['checks'])==12 and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in sd['source_sha256'].items()) and sd['actual_production_execution'] is None)
ok('V94 old consistency failure and strict correction explicit',sd['cases']['Unsupported kind no longer self-consistent']['old_status']=='record_consistent_only' and sd['cases']['Unsupported kind no longer self-consistent']['new']['status']=='conflict' and sd['cases']['Oversized JSON integer reports conflict without crash']['old_status']=='OverflowError')
ok('V94 missing host evidence not promoted to execution',sd['cases']['Missing expected host is missing evidence']['new']['status']=='needs_evidence' and sd['cases']['Archive remains missing execution evidence']['new']['status']=='needs_evidence' and all(x['new']['production_execution_verified'] is False for x in sd['cases'].values()))
report['highlights']+=['V94 strict proposed checker layer and twelve old/new regressions; field shape and explicit host coverage, no production validation']
mr=read(A/'mixture_read_range_cpu.json')
ok('V95 current original constructor/read controls source-bound',mr['checks_passed']==len(mr['checks'])==13 and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in mr['source_sha256'].items()) and mr['actual_Hero_overflow'] is None)
ok('V95 distinct APIs and shared corrupt prefix retain wide reference',mr['cases']['one_stage']['single']=='A:319' and mr['cases']['one_stage']['batch']=='A:702' and mr['cases']['staged']['single']==mr['cases']['staged']['batch']=='A:319' and mr['cases']['staged']['reference']['expected_identity']=='A:702')
ok('V95 declared Hero bound distinguished from synthetic overflow',mr['cases']['hero_declaration']['all_child_max_full_block_count_bound']==253722855 and mr['cases']['hero_declaration']['single']==mr['cases']['hero_declaration']['batch'] and mr['cases']['too_large_multiplier']['type']=='OverflowError')
report['highlights']+=['V95 thirteen current original mixture constructor/read CPU controls; API type divergence and shared prefix corruption, Hero declaration bound below int32, no actual token replay']
me=read(A/'mixture_exhaustion_cpu.json')
ok('V96 finite original-class controls source-bound',me['checks_passed']==len(me['checks'])==8 and me['grid_cells']==32 and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in me['source_sha256'].items()) and me['actual_Hero_exhaustion_incident'] is None)
ok('V96 randomized finite prefix errors distinguished from ordered controls',me['failing_grid_cells']==8 and all(r['read_error'] is None for r in me['rows'] if not r['randomize']) and any(r['reported_length']==1 and r['safe_prefix_length']==0 and r['read_error'] is not None for r in me['rows']))
ok('V96 all-stop count versus identity coverage retained',me['all_stop']['reported_length']==5 and me['all_stop']['unique_count']==3 and me['all_stop']['unrandomized']==['A:0','A:1','A:2','A:3','A:4'] and me['example_single_error']['type']=='IndexError')
report['highlights']+=['V96 thirty-two original finite-length/read controls, eight checks; partial-block bounds and count versus coverage, Hero declares restart, no actual loader incident']
rc=read(A/'restart_prefix_coverage_cpu.json')
ok('V97 original restart and conditional formula controls source-bound',rc['checks_passed']==len(rc['checks'])==16 and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in rc['source_sha256'].items()) and rc['actual_Hero_coverage'] is None)
ok('V97 partial prefix counters cannot substitute contiguous index coverage',rc['cases']['one_A']['actual_prefix_unique']==3 and rc['cases']['one_A']['conditional_prefix_projection']['window_distinct_sequence_indices']==5 and rc['cases']['one_A']['actual_new_against_history']==['A:0','A:3'])
ok('V97 complete staged blocks preserve reference multisets',len(rc['staged_full_blocks'])==8 and all(x['actual_multiset']==x['reference_multiset'] for x in rc['staged_full_blocks']) and rc['cases']['one_A']['actual_window_internal_repeats']==1)
report['highlights']+=['V97 sixteen original restart/coverage controls; contiguous formula narrowed to actual index intervals and complete blocks, edge coverage unknown for Hero']
mh=read(A/'mixture_resume_history_cpu.json')
ok('V98 original history replay controls source-bound',mh['checks_passed']==len(mh['checks'])==16 and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in mh['source_sha256'].items()) and mh['actual_loader_resume'] is None and mh['actual_Hero_stage_rewrite'] is None)
ok('V98 same current quota and names do not imply same identities',mh['same_current_quota']==[4,4] and mh['old_prefix']==[8,8] and mh['changed_prefix']==[12,4] and all(x.split(':')[0]==y.split(':')[0] and x!=y for x,y in zip(mh['old_read'],mh['changed_past_read'])))
ok('V98 modulo alias retains distinct logical offsets',mh['modulo_masked_old']==mh['modulo_masked_changed'] and mh['old_logical']!=mh['changed_logical'] and mh['reset_cursor_read']!=mh['old_read'])
report['highlights']+=['V98 sixteen original-class historical-stage replay controls; unchanged current quota with shifted base and modulo-masked history, no actual loader restore']
ls=read(A/'loader_stall_cpu.json');lb=read(A/'loader_current_source_binding.json')
ok('V99 original loader waits source-bound and current bytes verified',ls['checks_passed']==len(ls['checks'])==13 and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in ls['source_sha256'].items()) and lb['same_bytes'] and hashlib.sha256((ROOT/lb['archived_file']).read_bytes()).hexdigest()==lb['sha256'] and lb['git_revision']=='b65be4c9550c5097f0a3add08933531a1c24d534' and ls['actual_Hero_stall'] is None)
ok('V99 watchdog only observes selected read waits',ls['cases']['read_wait']['automatic_timeout'] is False and any('10.0 seconds' in x for x in ls['cases']['read_wait']['warnings']) and ls['cases']['length_wait']['pending_before_release'] and ls['cases']['length_wait']['warnings_before_release']==[] and ls['cases']['explicit_cancellation']['read_cancelled'])
ok('V99 prefetch fault and post-return warning scopes explicit',ls['cases']['fault_fetch_4']['returned']==[] and len(ls['cases']['fault_fetch_1']['returned'])==1 and ls['cases']['sync_next_True']['warnings_after_release']==[] and soup.select_one('#loader-stall-title') is not None)
report['highlights']+=['V99 thirteen original loader host watchdog/next controls; no automatic timeout, selected waits unwrapped and prefetch fault boundaries, no actual Hero stall']
bqa=read(A/'loader_background_acquisition.json');bq=read(A/'background_queue_cpu.json')
ok('V100 two background modules preserve historical non-bookkeeping source bytes',len(bqa['records'])==2 and bqa['prior_bytes_preserved']==497 and all(hashlib.sha256((S/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in bqa['records']) and all(hashlib.sha256((S/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in bqa['prior_integrity_manifest']['files'] if x['file']!='source_manifest.json'))
ok('V100 original and candidate queue controls source-bound',bq['checks_passed']==len(bq['checks'])==24 and bq['original_checks_passed']==19 and bq['candidate_checks_passed']==5 and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in bq['source_sha256'].items()) and bq['actual_Hero_shutdown_incident'] is None)
ok('V100 original shutdown and repeated exhaustion gaps retained',bq['cases']['repeated_end']['pending_after_first_end'] and not bq['cases']['repeated_end']['producer_alive'] and bq['cases']['stop_blocked_consumer']['consumer_pending_after_producer_exit'] and bq['cases']['join_wait']['stop_pending_before_release'])
ok('V100 positive and cleanup controls bound interpretation',[x['kind'] for x in bq['cases']['unbuffered_repeated_end']]==['value','value','StopIteration','StopIteration'] and bq['cases']['unbuffered_fault_RuntimeError']['kind']=='StopIteration' and bq['cases']['buffered_runtime_fault']['kind']=='RuntimeError' and bq['cases']['stop_blocked_consumer']['cleanup_consumer_finished'] and [x['kind'] for x in bq['cases']['loader_fault_fetch_1']['events']]==['value','OSError'])
ok('V100 local consumer candidate is not production IO cancellation',not bq['candidate_integrated_into_Marin'] and not bq['candidate_remote_IO_cancellation'] and bq['cases']['candidate_stop_consumer']['consumer_finished_before_release'] and bq['cases']['candidate_stop_consumer']['producer_alive_before_release'] and bq['cases']['candidate_stop_consumer']['all_finished_after_release'])
report['highlights']=[x.replace('498 source archive checksums valid','500 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['V100 nineteen original background queue/thread controls and five local consumer-candidate controls; repeated exhaustion and stop wakeup gaps, no production integration or remote IO cancellation']
ma=read(A/'main_incident_acquisition.json');mi=read(A/'main_incident_revision.json');mb=read(A/'main_revision_binding.json')
ok('V101 thirteen live/main source payloads preserve old content',len(ma['records'])==13 and ma['prior_bytes_preserved']==499 and all(hashlib.sha256((S/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in ma['records']) and all(hashlib.sha256((S/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in ma['prior_integrity_manifest']['files'] if x['file']!='source_manifest.json'))
ok('V101 five same bytes and different train file separately bound',len(mb['records'])==6 and sum(x['same_bytes'] for x in mb['records'])==5 and all((hashlib.sha256((ROOT/x['archived_file']).read_bytes()).hexdigest()==x['sha256'])==x['same_bytes'] for x in mb['records']) and mb['actual_deployed_revision'] is None)
ok('V101 eighteen original host classification and refresh controls source-bound',mi['checks_passed']==len(mi['checks'])==18 and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in mi['source_sha256'].items()) and mi['source_relationship']['status']=='diverged' and mi['source_relationship']['pull_merged'] is False)
ok('V101 main and proposal runtime defaults and constraints differ',mi['runtime_cases']['clean_carry']['main']['memory_fraction']=='0.75' and mi['runtime_cases']['clean_carry']['proposal']['memory_fraction']=='0.78' and mi['runtime_cases']['inherited_no_carry']['main']['flag_values']['--xla_gpu_experimental_parallel_collective_overlap_limit']=='8' and mi['runtime_cases']['inherited_no_carry']['proposal']['flag_values']['--xla_gpu_experimental_parallel_collective_overlap_limit']=='1')
ok('V101 pod evidence classification not node-root-cause proof',mi['pod_classification']['taint_137']=='PREEMPTED' and mi['pod_classification']['plain_137']=='FAILED' and mi['pod_classification']['oom_137']=='FAILED' and mi['actual_noexecute_root_cause'] is None and mi['actual_kubernetes_pod_status'] is None)
ok('V101 public issue refresh does not create new events',mi['issue_refresh']['8435']['count']==27 and mi['issue_refresh']['8506']['count']==59 and all(x['ids_and_bodies_unchanged'] for x in mi['issue_refresh'].values()) and mi['actual_deployed_revision'] is None)
report['highlights']=[x.replace('500 source archive checksums valid','513 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['V101 eighteen host main/proposal runtime and pod classification/refresh controls; branch divergence, defaults versus forced environment, no new issue body or actual deployment attribution']
ga=read(A/'gang_recovery_acquisition.json'); gn=read(A/'gang_recovery_native_tests.json')
ok('V102 nineteen complete gang source blobs preserve previous archive',len(ga['records'])==19 and ga['prior_bytes_preserved']==512 and all(hashlib.sha256((S/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in ga['records']) and all(hashlib.sha256((S/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in ga['prior_integrity_manifest']['files'] if x['file']!='source_manifest.json'))
ok('V102 native local controller tests source and output bound',gn['returncode']==0 and gn['passed']==len(gn['cases']) and gn['passed']>20 and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in gn['source_sha256'].items()) and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in gn['artifact_sha256'].items()))
required_gang={'test_coscheduled_crash_loop_fails_on_cumulative_budget','test_coscheduled_retriable_failure_bounces_siblings_to_pending','test_coscheduled_cascade_holds_sibling_resources_until_heartbeat','test_heartbeat_finalizes_stranded_attempt_after_producer_terminal','test_stale_attempt_ignored','test_coscheduled_cascade_survives_same_batch_sibling_update','test_coscheduled_bounced_job_recoschedules_to_single_slice','test_timeout_charges_cumulative_failure_budget'}
ok('V102 upstream behavioral scenarios actually executed',required_gang <= {x['name'] for x in gn['cases'] if x['outcome']=='passed'})
ok('V102 safe marker exclusions and production unknowns retained',gn['safe_default_marker_expression']=='not slow and not docker and not requires_cluster and not manual' and '-m' not in gn['command'] and gn['actual_Hero_deployed_revision'] is None and gn['actual_Hero_retry_parameters'] is None and gn['actual_Hero_recovery_trace'] is None)
gs=soup.select_one('#gang-recovery-guide')
ok('V102 recovery chapter tables and accessible inline mechanism present',gs is not None and len(gs.select('table'))==2 and gs.select_one('svg title') is not None and not gs.select('img[src="assets/gang_recovery_flow.svg"]'))
report['highlights']=[x.replace('513 source archive checksums valid','532 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['V102 original upstream local SQLite transition tests; gang cumulative budget, sibling cascade, stale attempts and capacity finalization; no production training recovery claim']
gt=read(A/'gang_recovery_trace_template.json')
ok('V102 recovery trace template not misrepresented as production evidence',gt['status']=='proposed_not_executed' and gt['events']==[] and gt['actual_checkpoint_digest'] is None and gt['actual_loader_cursor'] is None and gt['causal_mixture_effect'] is None and gn['tracked_source_unchanged'])
ca=read(A/'controller_restore_acquisition.json'); cn=read(A/'controller_restore_native_tests.json'); cf=read(A/'controller_restore_faults.json')
ok('V103 complete restore source blobs preserve previous archive',len(ca['records'])==14 and ca['prior_bytes_preserved']==531 and all(hashlib.sha256((S/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in ca['records']) and all(hashlib.sha256((S/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in ca['prior_integrity_manifest']['files'] if x['file']!='source_manifest.json'))
ok('V103 original upstream 51 passes and native outputs source bound',cn['passed']==51 and sum(x['passed'] for x in cn['runs'])==51 and all(x['observed_tool_exitcode']==0 and all(c['outcome']=='passed' for c in x['cases']) for x in cn['runs']) and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in {**cn['source_sha256'],**cn['artifact_sha256']}.items()))
ok('V103 twelve original module fault controls source bound',cf['checks_passed']==len(cf['checks'])==12 and all(x['passed'] for x in cf['checks']) and hashlib.sha256((ROOT/'scripts/probe_controller_restore_faults.py').read_bytes()).hexdigest()==cf['executed_script_sha256'] and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in cf['source_sha256'].items()))
ok('V103 ordinary errors retain both original files while rollback clears them',all(cf['cases']['ordinary_'+k]['original_file_bytes_preserved'] and cf['cases']['ordinary_'+k]['local_marker']==7 and not cf['cases']['rollback_'+k]['local_main_exists'] and cf['cases']['rollback_'+k]['rollout_phase']=='rollback_requested' for k in ['corrupt','missing']))
ok('V103 partial publish discoverable despite unchanged local ancestry',cf['cases']['partial_upload']['local_ancestry'] is None and cf['cases']['partial_upload']['main_uploaded'] and not cf['cases']['partial_upload']['auth_uploaded'] and cf['cases']['partial_download']['returned'] and not cf['cases']['partial_download']['after_healthy'] and not cf['cases']['partial_startup']['before_db_open_healthy'] and cf['cases']['partial_startup']['after_db_open_healthy'])
ok('V103 controller reopen not forced kill or model checkpoint proof',cn['tracked_source_unchanged'] and cn['actual_forced_process_kill_restore'] is None and cn['actual_training_checkpoint_restore'] is None and cf['actual_production_incident'] is None and cf['actual_auth_key_loss'] is None)
cr=soup.select_one('#controller-recovery-guide')
ok('V103 accessible recovery diagram and three case tables inline',cr is not None and len(cr.select('table'))==4 and cr.select_one('svg title') is not None and not cr.select('img[src="assets/controller_restore_contract.svg"]'))
report['highlights']=[x.replace('532 source archive checksums valid','546 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['V103 51 upstream recovery/reopen cases and 12 fault controls; ordinary failure preservation, requested rollback preclear and partial-publish completeness gap; no production incident attribution']
required_restore={'test_controller_when_reopened_preserves_attempt_and_resumes_without_duplicate_launch','test_checkpoint_restore_preserves_snapshot_and_discards_later_writes','test_active_task_restored_from_checkpoint_retries_after_runtime_loss','test_backup_with_concurrent_commits_preserves_snapshot','test_reopening_a_migrated_db_changes_nothing','test_replace_from_resets_dict','test_coordinator_endpoint_survives_controller_restart_and_moves_to_the_replacement_attempt'}
ok('V103 native restore/reopen scenario identities executed',required_restore <= {c['name'] for r in cn['runs'] for c in r['cases'] if c['outcome']=='passed'})
ccr=read(A/'controller_candidate_refresh.json'); ccp=read(A/'controller_recovery_candidate.json'); cct=read(A/'controller_recovery_candidate_controls.json')
ok('V104 refreshed core source payloads preserve full prior archive',len(ccr['records'])==3 and ccr['prior_bytes_preserved']==545 and all(hashlib.sha256((S/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in ccr['records']) and all(hashlib.sha256((S/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in ccr['prior_integrity_manifest']['files'] if x['file']!='source_manifest.json'))
ok('V104 same core bytes do not assert current dependencies executed',len(ccr['bindings'])==2 and all(x['same_bytes'] for x in ccr['bindings']) and all((S/'controller_candidate_2026_10_07'/x['file']).read_bytes()==(S/'controller_restore_2026_10_07/lib/iris/src/iris/cluster/controller'/x['file']).read_bytes() for x in ccr['bindings']) and ccr['actual_latest_runtime_execution'] is None)
ok('V104 candidate and original source bound to 22 passing controls',cct['checks_passed']==len(cct['checks'])==22 and all(x['passed'] for x in cct['checks']) and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in cct['source_sha256'].items()) and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in ccp['candidate_sha256'].items()))
ok('V104 failed candidate rollback retains both original files and request',all(cct['cases']['True-'+k]['original_bytes_preserved'] and cct['cases']['True-'+k]['marker']==7 and cct['cases']['True-'+k]['rollout_phase']=='rollback_requested' for k in ['corrupt','missing']))
ok('V104 candidate publication failures do not select partial directory',all(cct['cases'][k]['error']=='OSError' and cct['cases'][k]['ancestry']==cct['cases'][k]['latest'] and cct['cases'][k]['latest'] is not None for k in ['second-file','completion-file']))
ok('V104 malformed candidate content rejected including valid digest bad SQLite',all(cct['cases'][k]['error'] is not None for k in ['missing-auth','wrong-digest','corrupt-auth-valid-digest','corrupt-main-valid-digest','wrong-identity','epoch-collision']))
ok('V104 legacy and production integration remain explicit unfinished work',ccp['status']==cct['status']=='candidate_not_integrated' and cct['cases']['legacy']['error']=='ValueError' and cct['actual_production_fix'] is None and cct['actual_legacy_migration'] is None and cct['actual_cloud_transport'] is None)
ok('V104 recorded candidate output matches execution artifact',hashlib.sha256((A/'controller_recovery_candidate_output.txt').read_bytes()).hexdigest()==ccp['output_sha256'])
report['highlights']=[x.replace('546 source archive checksums valid','549 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['V104 22 candidate recovery controls; completion records and original-state retention; legacy migration and production call-chain integration unfinished; two latest core sources byte-identical, current runtime not executed']
sa=read(A/'snapshot_semantics_acquisition.json'); sp=read(A/'snapshot_semantics.json'); st=read(A/'snapshot_auth_tests.json')
ok('V105 four complete snapshot/auth sources preserve prior content',len(sa['records'])==4 and sa['prior_bytes_preserved']==548 and all(hashlib.sha256((S/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in sa['records']) and all(hashlib.sha256((S/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in sa['prior_integrity_manifest']['files'] if x['file']!='source_manifest.json'))
ok('V105 eight snapshot controls bound to source and candidate bytes',sp['checks_passed']==len(sp['checks'])==8 and all(x['passed'] for x in sp['checks']) and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in sp['source_sha256'].items()))
ok('V105 actual fresh auth empty while constructed skew passes strict restore',sp['fresh_auth_application_tables']==[] and sp['cases']['original_quiet']['backup_pair']==[0,0] and sp['cases']['original_interleaved']['backup_pair']==[0,1] and sp['cases']['candidate_interleaved']['restored_pair']==[0,1] and sp['actual_production_crossfile_invariant'] is None and sp['actual_training_state_skew'] is None)
ok('V105 original auth tests executed with complete output bound',st['passed']==len(st['cases'])==2 and st['observed_tool_exitcode']==0 and all(x['outcome']=='passed' for x in st['cases']) and {x['name'] for x in st['cases']}=={'test_require_persistent_signing_key','test_worker_token_differs_after_restart'} and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in st['artifact_sha256'].items()) and st['actual_production_secret_identity'] is None)
report['highlights']=[x.replace('549 source archive checksums valid','553 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['V105 auth empty-schema/config-key risk refinement; eight constructed snapshot controls and two original auth tests; valid file pair not proof of shared state cut or production training skew']
da=read(A/'donation_snapshot_acquisition.json'); dp=read(A/'donation_snapshot_cpu.json')
ok('V106 three complete sources preserve previous archive',len(da['records'])==3 and da['prior_bytes_preserved']==552 and all(hashlib.sha256((S/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in da['records']) and all(hashlib.sha256((S/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in da['prior_integrity_manifest']['files'] if x['file']!='source_manifest.json'))
ok('V106 ten actual CPU controls source and script bound',dp['checks_passed']==len(dp['checks'])==10 and all(x['passed'] for x in dp['checks']) and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in dp['source_sha256'].items()))
ok('V106 actual deleted inputs and delayed full snapshot verified',len(dp['cases'])==2 and all(x['input_deleted'] and x['independent_host_memory'] and x['saved_first']==0 and x['new_first']==200 for x in dp['cases']) and dp['retained_host_view_control']['input_deleted'] is False)
ok('V106 GPU and production restore remain unknown',dp['actual_GPU_donation'] is None and dp['actual_Hero_checkpoint_integrity'] is None and dp['actual_training_resume'] is None)
dsection=soup.select_one('#checkpoint-memory-guide')
ok('V106 ownership diagram embedded with accessible title',dsection is not None and dsection.select_one('svg title') is not None and not dsection.select('img[src="assets/donation_snapshot_flow.svg"]'))
report['highlights']=[x.replace('553 source archive checksums valid','556 source archive checksums valid') for x in report['highlights']]
report['highlights']+=['V106 ten CPU donation and delayed-consumer controls; observation-view non-donation control; ownership diagram; GPU and full training restore unverified']
ru=read(A/'resume_update_identity.json'); rb=read(A/'resume_source_binding.json')
ok('V107 eight assembled CPU controls source and script bound',ru['checks_passed']==len(ru['checks'])==8 and all(x['passed'] for x in ru['checks']) and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in ru['source_sha256'].items()))
ok('V107 core archived sources bound to frozen main bytes',len(rb['bindings'])==5 and all(x['same_bytes'] and hashlib.sha256((ROOT/x['archive_file']).read_bytes()).hexdigest()==x['sha256'] for x in rb['bindings']) and rb['actual_latest_head_execution'] is None)
ro=ru['observations']
ok('V107 same preloss but first update and next loss differ',ro['restored_step']==ro['restored_adam_count']==3 and ro['pre_update_loss']==ro['weights_only_pre_update_loss'] and ro['full_update']!=ro['weights_only_update'] and ro['next_weights_only_loss']<ro['next_full_loss'] and ro['pending_applied_in_quadratic'] is False)
ok('V107 actual Hero and causal mixture effects remain unknown',ru['actual_Hero_next_update_identity'] is None and ru['actual_GPU_roundtrip'] is None and ru['actual_causal_mixture_effect'] is None)
rs=soup.select_one('#checkpoint-commit-guide')
ok('V107 update comparison diagram embedded accessibly',rs is not None and rs.select_one('#resume-update-title') is not None and not rs.select('img[src="assets/resume_update_identity.svg"]'))
report['highlights']+=['V107 eight assembled CPU state staging/donation/real IO/next-update controls; weights-only same-loss counterexample; original Hero train_step and loader unverified']
sk=read(A/'seed_pipeline_cpu.json'); sb=read(A/'seed_pipeline_source_binding.json'); sr=read(ROOT/'templates/data_seed_resume_review.json')
ok('V108 twelve original pipeline CPU controls source bound',sk['checks_passed']==len(sk['checks'])==12 and all(x['passed'] for x in sk['checks']) and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in sk['source_sha256'].items()))
ok('V108 seven core sources match frozen main complete bytes',len(sb['bindings'])==7 and all(x['same_bytes'] and hashlib.sha256((ROOT/x['archive_file']).read_bytes()).hexdigest()==x['sha256'] for x in sb['bindings']) and sb['actual_latest_head_execution'] is None)
sc=sk['cases']
ok('V108 None explicit zero and future support controls kept distinct',sc['default']['data_key']!=sc['explicit_data0']['data_key'] and sc['explicit_data0']['first96']==sc['trainer1_data0']['first96'] and sc['future_front']['different_slots_vs_default']==92 and sc['zero_front']['different_slots_vs_default']==sc['future_tail']['different_slots_vs_default']==0 and sc['zero_front']['declared_component_order']==sc['future_front']['declared_component_order'] and sc['zero_front']['sequence_stage_starts']==sc['future_front']['sequence_stage_starts'] and [x.split(':')[0] for x in sc['default']['first96']]==[x.split(':')[0] for x in sc['future_front']['first96']])
ok('V108 declared snapshot not actual executed keys or token evidence',sk['declared_run']['data_seed'] is None and sk['declared_run']['actual_executed_key'] is None and sk['actual_Hero_data_discontinuity'] is None and sk['actual_Hero_next_batch_identity'] is None and sk['actual_token_store'] is None and sk['actual_causal_mixture_effect'] is None)
ok('V108 prefix review remains an unexecuted real-run template',sr['status']=='proposed_not_executed' and sr['data_key'] is None and sr['prefix_identity_gate'] is None and sr['before_and_after_domain_child_token_hashes'] is None and sr['causal_mixture_effect'] is None)
ok('V108 child-key figure embedded accessibly',soup.select_one('#mixture-identity-guide #seed-pipeline-title') is not None and not soup.select('img[src="assets/seed_pipeline_keys.svg"]'))
report['highlights']+=['V108 twelve CPU seed/build/shuffle/mix controls; future positive support can change current child keys; None versus explicit zero; no production token or causal loss claim']
lm=read(A/'loss_mass_cpu.json'); lb=read(A/'loss_mass_source_binding.json'); le=read(ROOT/'templates/domain_exposure_review.json'); lc=lm['cases']
ok('V109 thirteen original-interface CPU controls source and script bound',lm['checks_passed']==len(lm['checks'])==13 and all(x['passed'] for x in lm['checks']) and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in lm['source_sha256'].items()))
ok('V109 six complete fixed-source HTTP payloads match archive',len(lb['bindings'])==6 and all(x['http_status']==200 and x['same_bytes'] and hashlib.sha256((ROOT/x['archive_file']).read_bytes()).hexdigest()==x['sha256'] for x in lb['bindings']) and lb['actual_latest_head_execution'] is None)
ok('V109 equal loss distinct domain coefficients and uniform scale invariance',lc['dense_both']['loss']==lc['equal_sequences_sparse_b']['loss']==lc['uniform_half_weight']['loss'] and lc['dense_both']['first_logit_gradient']!=lc['equal_sequences_sparse_b']['first_logit_gradient'] and lc['uniform_half_weight']['first_logit_gradient']==lc['equal_sequences_sparse_b']['first_logit_gradient'] and lc['equal_sequences_sparse_b']['weighted_target_share']==[.75,.25])
ok('V109 sampler quotas and nominal-input budget not conflated with target mass',lm['integer_quotas']=={'equal':{'A':4,'B':4},'inverse_density':{'A':2,'B':6}} and lc['block8_equal_sequences']['input_slots']==lc['block8_inverse_density']['input_slots']==32 and sum(lc['block8_equal_sequences']['weighted_target_mass'])==16 and sum(lc['block8_inverse_density']['weighted_target_mass'])==12)
ok('V109 actual Hero objective and requested fused backend remain unexecuted',lm['actual_Hero_target_mass_by_domain'] is None and lm['actual_Hero_gradient_share'] is None and lm['actual_hardware_compute_cost'] is None and lm['actual_causal_mixture_effect'] is None and lm['actual_xla_fast_bwd_execution'] is None and all(x['requested_implementation']=='xla_fast_bwd' and x['executed']=='original CPU reference' for x in lm['backend_calls']))
ok('V109 real exposure template remains unknown and figure embedded',le['status']=='proposed_not_executed' and le['domain_rows']==[] and le['actual_Hero_objective_shares'] is None and soup.select_one('#data #loss-mass-title') is not None and not soup.select('img[src="assets/loss_mass_accounting.svg"]'))
report['highlights']+=['V109 thirteen original loss-interface CPU controls; sequence/position/target/mass ledgers and unit correction; requested fused backend explicitly replaced; actual Hero density and causal mixture benefit unknown']

op=read(A/'operational_progress.json'); oa=read(A/'operational_refresh_acquisition.json'); ob=read(A/'operational_plot_binding.json')
ok('V110 eight operational analyses bind exact sources',op['checks_passed']==len(op['checks'])==8 and all(x['passed'] for x in op['checks']) and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in op['source_sha256'].items()))
ok('V110 ten public payloads retain prior archive and full comments',len(oa['records'])==10 and oa['prior_bytes_preserved']==555 and all(x['old_count']==x['new_count'] and not x['added_ids'] and not x['removed_ids'] and not x['changed_bodies'] for x in op['comment_comparisons']))
ok('V110 sampled endpoint and later summary remain separate',op['sampled_rows']==1000 and len(op['intervals'])==999 and op['sampled_endpoint_span']['end_step']==219263 and op['latest_summary']['step']==219518 and op['nominal_tokens_per_logged_step']==46137344)
ok('V110 outage cause actual MFU and causal mixture benefit remain unknown',all(op[k] is None for k in ['actual_fault_downtime','actual_duty_cycle','actual_downtime_cause_decomposition','actual_loss_causal_effect','actual_production_deployed_revision']) and op['original_formula_replay']['actual_hardware_peak_verified'] is None and all(x['actual_pause_seconds'] is None and x['initiating_cause'] is None for x in op['intervals']))
ok('V110 scientific figure bound and embedded accessibly',ob['rows']==999 and ob['analysis_sha256']==hashlib.sha256((A/'operational_progress.json').read_bytes()).hexdigest() and ob['figure_sha256']==hashlib.sha256((ROOT/'assets/operational_progress.svg').read_bytes()).hexdigest() and soup.select_one('#operational-progress-guide #operational-progress-title') is not None and not soup.select('img[src="assets/operational_progress.svg"]'))
ok('V110 timeline proposal records no actual execution',read(ROOT/'templates/operational_timeline_review.json')['status']=='proposed_not_executed' and read(ROOT/'templates/operational_timeline_review.json')['events']==[])
report['highlights']+=['V110 eight operational analyses; 1000 sampled observations; calendar progress distinct from step duration; source archive expanded 556 to 566; no outage attribution']

ps=read(A/'pending_resume_step_cpu.json'); pb=read(A/'pending_resume_plot_binding.json'); pc=ps['cases']; pa=ps['ema_alias_control']
ok('V111 nineteen original-step CPU controls bind source and adapter bytes',ps['checks_passed']==len(ps['checks'])==19 and all(x['passed'] for x in ps['checks']) and ps['backend']=='cpu' and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in ps['source_sha256'].items()))
ok('V111 complete original IO restore preserves next-update route loss and state',len(ps['serialized_paths'])==11 and 'pending_qb_betas' in ps['serialized_paths'] and pc['full_restore']['next_state']==pc['uninterrupted']['next_state'] and pc['full_restore']['train_loss']==pc['uninterrupted']['train_loss'])
ok('V111 pending zero is replacement not consumed-state no-op',pc['clear_pending']['next_forward_bias']==pc['materialize_clear_pending']['next_forward_bias']==[[0.,0.,0.]] and pc['materialize_keep_pending']['next_state']==pc['full_restore']['next_state'] and pc['clear_pending']['next_state']==pc['materialize_clear_pending']['next_state'] and pc['clear_pending']['next_state']!=pc['full_restore']['next_state'])
ok('V111 EMA alias failure isolated from disabled Hero branch',pa['parameter_and_ema_leaf_identity_shared'] and 'donate the same buffer twice' in pa['error'] and pa['independent_EMA_buffers_succeed'] and pa['EMA_disabled_original_step_succeeds'] and pa['archived_Hero_ema_beta'] is None and pa['actual_Hero_incident'] is None)
ok('V111 stored callback route differs from next-forward route without claiming Hero magnitude',ps['callback_views'][0]['ids']==pc['full_restore']['stored_ids'] and ps['callback_views'][0]['ids']!=pc['full_restore']['next_forward_ids'] and ps['actual_Hero_view_error_magnitude'] is None and ps['actual_GPU_execution'] is None and ps['actual_causal_mixture_effect'] is None)
ok('V111 state diagram binds measured fixture and embeds accessibly',pb['analysis_sha256']==hashlib.sha256((A/'pending_resume_step_cpu.json').read_bytes()).hexdigest() and pb['figure_sha256']==hashlib.sha256((ROOT/'assets/pending_resume_step.svg').read_bytes()).hexdigest() and soup.select_one('#pending-resume-guide #pending-resume-title') is not None and not soup.select('img[src="assets/pending_resume_step.svg"]'))
ok('V111 consumer template has no actual checkpoint or execution result',read(ROOT/'templates/pending_consumer_review.json')['status']=='proposed_not_executed' and read(ROOT/'templates/pending_consumer_review.json')['actual_original_next_step_equivalence'] is None and read(ROOT/'templates/pending_consumer_review.json')['comparisons']==[])
report['highlights']+=['V111 nineteen original-step CPU controls; real local OCDBT/Adam/Equinox; pending-zero replacement and EMA-alias donation boundary; Hero EMA disabled, no production incident attribution']
report['checks_passed']=len(checks)
(A/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
