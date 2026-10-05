"""Independent arithmetic and source checks for V11; no producer function imports."""
import ast
import csv
import hashlib
import json
import math
import pathlib
import statistics

R=pathlib.Path(__file__).resolve().parents[1];A=R/'analysis';S=R/'sources';D=S/'scale_2026_10_05'
checks=[]
def local_check(label,condition):
    if not condition:raise AssertionError(label)
    checks.append(label)
ok=globals().get('check_scale',local_check)
def read(path):return json.loads(path.read_text())
def rows(path):return list(csv.DictReader(path.open()))
raw=rows(S/'mix_study_final-2026.09.15.1_point_speedups.csv');new=rows(A/'scale_endpoints.csv');audit=read(A/'scale_audit.json')
sizes=['d768','d1024','d1536'];oldnames=['rav-ladder-d768-v2','rav-ladder-d1024','rav-ladder-d1536']
ok('Scale raw file has 48 BPB and 48 CE rows, with no macro',len(raw)==96 and sum(r['metric']=='bpb' for r in raw)==48 and all(r['subset']!='macro' for r in raw))
for r in new:
    b=next(x for x in raw if x['metric']=='bpb' and x['size']==r['size'] and x['subset']==r['subset'])
    c=next(x for x in raw if x['metric']=='loss' and x['size']==r['size'] and x['subset']==r['subset'])
    delta=(float(b['variant_loss'])-float(b['baseline_loss']))/float(b['baseline_loss'])*100
    ok('Independent raw endpoint BPB/CE '+r['size']+'/'+r['subset'],float(r['old_bpb'])==float(b['baseline_loss']) and float(r['new_bpb'])==float(b['variant_loss']) and abs(float(r['bpb_change_pct'])-delta)<1e-12 and float(r['old_ce'])==float(c['baseline_loss']) and float(r['new_ce'])==float(c['variant_loss']))
flip=[]
for subset in {r['subset'] for r in new}:
    directions=[float(r['new_bpb'])>float(r['old_bpb']) for r in new if r['subset']==subset]
    if any(directions) and not all(directions):flip.append(subset)
ok('Exactly four observed subset direction changes',sorted(flip)==audit['direction_changes']==['dolma_100_programing_languages','ptb','redpajama','wikitext_103'])
ok('Exactly two CE/BPB direction conflicts are code on d768/d1024',audit['ce_bpb_direction_disagreements']==[{'size':'d768','subset':'dolma_100_programing_languages'},{'size':'d1024','subset':'dolma_100_programing_languages'}])
weights=rows(A/'scale_initial_weights.csv');contrasts=rows(A/'scale_config_contrasts.csv');metadata=read(A/'scale_run_metadata.json')
for size,oldname,improved in zip(sizes,oldnames,[11,12,15]):
    rs=[r for r in raw if r['metric']=='bpb' and r['size']==size];m=next(r for r in audit['macro_rows'] if r['size']==size)
    ok('Reconstructed sixteen-subset macro '+size,abs(m['old_bpb']-sum(float(r['baseline_loss']) for r in rs)/16)<1e-14 and abs(m['new_bpb']-sum(float(r['variant_loss']) for r in rs)/16)<1e-14 and m['improved']==improved)
    configs=[]
    for name in [oldname,'h100-mix25-20260912-'+size]:
        p=D/('config_'+name+'.json');run=read(p)['data']['project']['run'];c=json.loads(run['config']);sm=json.loads(run['summaryMetrics']);configs.append(c)
        md=next(r for r in metadata if r['run']==name)
        ok('Public run config fingerprint and token arithmetic '+name,md['config_sha256']==hashlib.sha256(p.read_bytes()).hexdigest() and md['configured_total_tokens']==sm['throughput/total_tokens'] and md['final_step']+1==c['trainer']['value']['trainer']['num_train_steps'])
        for r in rs:
            side='baseline_loss' if name==oldname else 'variant_loss';metric='eval_dropless/paloma/'+r['subset']+'-llama3/bpb'
            ok('Frozen CSV cross-check against live archived summary '+name+'/'+r['subset'],abs(float(r[side])-sm[metric])<2e-7)
    oc,nc=configs;wa=oc['data']['value']['train_weights'][0][1];wb=nc['data']['value']['train_weights'][0][1]
    half_l1=sum(abs(wa.get(k,0)-wb.get(k,0)) for k in wa.keys()|wb.keys())/2
    ww=[r for r in weights if r['size']==size]
    ok('Two hundred initial weights differ before treatment switch '+size,len(ww)==200 and abs(sum(abs(float(r['new_minus_old'])) for r in ww)/2-half_l1)<1e-14 and abs(half_l1-.20391888499269878)<1e-14 and all(float(r['old_initial_weight'])==wa[r['cell']] and float(r['new_initial_weight'])==wb[r['cell']] for r in ww))
    val=[k for k in oc['data']['value']['components'] if k.startswith('paloma/')]
    ok('Sixteen validation component configs match despite batch difference '+size,len(val)==16 and all(oc['data']['value']['components'][k]==nc['data']['value']['components'][k] for k in val) and oc['eval']['value']['eval_batch_size']!=nc['eval']['value']['eval_batch_size'])
ok('d1536 physical totals retain one new-update difference',next(r['logged_total_tokens'] for r in metadata if r['run']=='rav-ladder-d1536')-next(r['logged_total_tokens'] for r in metadata if r['run']=='h100-mix25-20260912-d1536')==4194304)

# Closed-form covariance slope, not numpy.polyfit used by the producer.
for r in rows(A/'scale_fit_sensitivity.csv'):
    points=audit['macro_rows'] if r['subset']=='macro' else [{'old_bpb':float(x['baseline_loss']),'new_bpb':float(x['variant_loss']),'compute':float(x['baseline_compute'])} for x in raw if x['metric']=='bpb' and x['subset']==r['subset']]
    f=float(r['floor_bpb']);xs=[math.log(p['compute']/points[0]['compute']) for p in points];ys=[math.log(p['old_bpb']-f) for p in points]
    xm=statistics.mean(xs);ym=statistics.mean(ys);alpha=-sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/sum((x-xm)**2 for x in xs)
    ratio=(points[-1]['old_bpb']-f)/(points[-1]['new_bpb']-f);speed=math.exp(math.log(ratio)/alpha)
    ok('Independent joint floor/slope inversion '+r['subset']+'/'+r['floor_fraction'],abs(alpha-float(r['refit_alpha']))<1e-12 and abs(speed-float(r['refit_recentered_speedup']))<1e-11 and int(r['observations'])==3 and int(r['free_fit_parameters_at_fixed_floor'])==2)
for r in rows(A/'scale_fit_leave_one_rung.csv'):
    ps=audit['macro_rows'] if r['subset']=='macro' else [{'size':x['size'],'old_bpb':float(x['baseline_loss']),'compute':float(x['baseline_compute'])} for x in raw if x['metric']=='bpb' and x['subset']==r['subset']]
    kept=[x for x in ps if x['size']!=r['held_out_size']];held=next(x for x in ps if x['size']==r['held_out_size']);alpha=math.log(kept[0]['old_bpb']/kept[1]['old_bpb'])/math.log(kept[1]['compute']/kept[0]['compute']);pred=kept[0]['old_bpb']*(held['compute']/kept[0]['compute'])**(-alpha)
    ok('Independent two-point held-rung diagnostic '+r['subset']+'/'+r['held_out_size'],abs(pred-float(r['predicted_bpb']))<1e-12)

toy=read(A/'scale_batch_counterexample.json');a,b=toy['results'];ln2=math.log(2)
ok('Synthetic example keeps CE fixed but reverses logged BPB ranking',a['old_ce']==b['old_ce']==1 and a['new_ce']==b['new_ce']==.8 and abs(a['old_batch_token_weighted_bpb']-.55/ln2)<1e-12 and abs(a['new_batch_token_weighted_bpb']-.575/ln2)<1e-12 and a['logged_formula_change_pct']>0 and b['logged_formula_change_pct']<0 and all(abs(r['global_ratio_change_pct']+20)<1e-12 for r in [a,b]))
tree=ast.parse((D/'eval.py').read_text());calls=[x for x in ast.walk(tree) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='add' and isinstance(x.func.value,ast.Attribute) and x.func.value.attr=='bpb_per_tag']
ok('Pinned public TaggedEvaluator uses token rather than byte weight for batch BPB',any(len(c.args)==2 and isinstance(c.args[1],ast.Name) and c.args[1].id=='this_weights_per_tag' for c in calls))
ok('Mechanism counterexample is explicitly synthetic with no execution binding','Synthetic arithmetic replay' in toy['scope'] and audit['actual_gpu_confirmation'] is None and audit['actual_scale_causal_experiment'] is None)
plan=read(A/'scale_confirmation_plan.json');arms=plan['arms'];ok('Forty-eight proposed continuation slots have distinct factor combinations',len(arms)==plan['continuation_slots']==48 and len({(x['size'],x['target_start_fraction'],x['arm'],x['continuation_seed']) for x in arms})==48)
ok('Budget is independently summed using actual rounded update intervals',plan['total_continuation_tokens']==sum((x['total_updates']-x['start_update'])*4194304 for x in arms)==8481788657664 and all(x['start_update']%48==0 and x['start_update']>=x['total_updates']*x['target_start_fraction'] and x['start_update']-48<x['total_updates']*x['target_start_fraction'] for x in arms))
ok('Confirmation plan cannot be misread as launched or pure order effect',plan['status']=='planned_not_executed' and plan['launchable'] is False and plan['order_effect_claim_permitted'] is False and plan['initialization_tokens_in_budget'] is False and all(plan[x] is None for x in ['checkpoint_digests','actual_data_state','frozen_weight_digests','independent_eval_manifest','primary_metric_and_guardrails','actual_launch','actual_results','reader_efficacy_study']))
refresh=read(A/'scale_refresh_audit.json');prior=refresh['prior_archive_files']
ok('Fourteen new scale sources preserve all 329 prior content files',len(prior)==330 and len(refresh['new_files'])==14 and all(hashlib.sha256((S/r['file']).read_bytes()).hexdigest()==r['sha256'] for r in prior if r['file']!='source_manifest.json'))
for r in refresh['new_files']:ok('New scale snapshot source checksum '+r['file'],hashlib.sha256((S/r['file']).read_bytes()).hexdigest()==r['sha256'])
doc=(R/'SCALE_TRANSFER_ZH.md').read_text()
ok('New chapter keeps observed, synthetic and proposed evidence separate',all(x in doc for x in ['0.20391888499269878','8,481,788,657,664','人为构造','不把它写成历史指标冲突的已确认根因','未修改判断锚点','不能计算完整候选排序的Spearman相关']))
if __name__=='__main__' and 'check_scale' not in globals():print('Scale checks:',len(checks))
