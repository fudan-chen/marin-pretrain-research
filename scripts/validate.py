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
ok('Standalone embeds all ten referenced figures',figure_count==10 and stand.count('src="data:image/png;base64,')==figure_count)
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
report={'checks_passed':len(checks),'highlights':['27 comments translated and anchored','58 operations entries','200 buckets x 3 normalized phases','400 public examples','200-step EP and kernel paired windows','934 swarm observations / 915 exact weight pairs','54 task BPB comparisons','1200 counts match native 3.12 arithmetic and 1200 match legacy sensitivity','complete-block cursor differences independently totaled','80 domain interventions / 125 source configs','8 order comparisons / 3 token budget conventions','12 planner numeric test groups / 100 random schedules','ten embedded figures','all relative HTML links valid','complete source archive checksums valid'],'not_verified':['GPU training/kernel reproduction','causality of individual mixture buckets','independent confirmation after candidate selection','535B task accuracy or long-context outcomes','full token ID / inner shuffle replay','all original experiment launch code SHAs and actual restored checkpoint contents']}
(A/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
