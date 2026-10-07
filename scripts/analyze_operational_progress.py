"""Observed sampled W&B frontier rates, public issue delta and original metric formula replay.
No duty-cycle estimate, downtime attribution, unique/scored tokens or real-device MFU verification.
"""
import ast,collections,dataclasses,datetime,hashlib,json,math,pathlib,statistics,types
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/operational_2026_10_07';A=R/'analysis';checks=[]
def load(p):return json.loads(p.read_text())
def ck(n,v):assert v,n;checks.append({'name':n,'passed':True})
def utc(t):return datetime.datetime.fromtimestamp(t,datetime.timezone.utc).isoformat()
acq=load(A/'operational_refresh_acquisition.json');raw=load(D/'wandb_meta.json')['data']['project']['run'];summary=json.loads(raw['summaryMetrics']) if isinstance(raw['summaryMetrics'],str) else raw['summaryMetrics'];cfg=json.loads(raw['config']) if isinstance(raw['config'],str) else raw['config'];tr=cfg['trainer']['value'];model=cfg['model']['value'];batch=tr['trainer']['train_batch_size'];length=model['max_seq_len'];unit=batch*length
rows=load(D/'wandb_operational_history.json')['data']['project']['run']['sampledHistory'][0];rows=[json.loads(r) if isinstance(r,str) else r for r in rows]
ck('Complete recorded sampled rows expose all five requested fields',len(rows)==1000 and all(set(acq['history_spec']['keys'])<=set(r) for r in rows))
ck('Observed steps and timestamps strictly increase in returned sample',all(b['_step']>a['_step'] and b['_timestamp']>a['_timestamp'] for a,b in zip(rows,rows[1:])))
ck('All observed nominal token counters match configured constant batch and length',unit==46137344 and all(r['throughput/total_tokens']==(r['_step']+1)*unit for r in rows) and summary['throughput/total_tokens']==(summary['_step']+1)*unit)
intervals=[]
for a,b in zip(rows,rows[1:]):
 ds=b['_step']-a['_step'];dt=b['_timestamp']-a['_timestamp'];dn=b['throughput/total_tokens']-a['throughput/total_tokens']
 intervals.append({'start_step':a['_step'],'end_step':b['_step'],'start_utc':utc(a['_timestamp']),'end_utc':utc(b['_timestamp']),'delta_steps':ds,'elapsed_seconds':dt,'nominal_token_delta':dn,'calendar_seconds_per_logged_step':dt/ds,'nominal_frontier_tokens_per_second':dn/dt,'end_logged_duration':b['throughput/duration'],'end_logged_iteration_time':b['throughput/iteration_time'],'actual_pause_seconds':None,'initiating_cause':None})
def span(a,b):
 ds=b['_step']-a['_step'];dt=b['_timestamp']-a['_timestamp'];dn=b['throughput/total_tokens']-a['throughput/total_tokens']
 return {'start_step':a['_step'],'end_step':b['_step'],'start_utc':utc(a['_timestamp']),'end_utc':utc(b['_timestamp']),'elapsed_seconds':dt,'delta_steps':ds,'nominal_token_delta':dn,'calendar_seconds_per_logged_step':dt/ds,'nominal_frontier_tokens_per_second':dn/dt}
full=span(rows[0],rows[-1]);old=load(R/'sources/live_2026_10_07/meta.json')['summaryMetrics'];recent=span(old,summary)
ck('Sampled interval accounting telescopes to observed endpoints',math.isclose(sum(x['elapsed_seconds'] for x in intervals),full['elapsed_seconds']) and sum(x['delta_steps'] for x in intervals)==full['delta_steps'] and sum(x['nominal_token_delta'] for x in intervals)==full['nominal_token_delta'])
changes=[]
for c in acq['comment_comparisons']:
 n=c['issue'];prior=load(R/'sources/engineering_v77_2026_10_07'/f'issue_{n}.json');current=load(D/f'issue_{n}.json');changes.append(dict(c,issue_body_changed=prior['body']!=current['body'],issue_state_changed=prior['state']!=current['state']))
ck('Three issue comments complete and compared by identity and body',len(changes)==3 and all(load(D/f"issue_{c['issue']}.json")['comments']==c['new_count'] for c in changes))
# Replay original formula and full logging closure. Device kind/count/peak are explicit logged-input adapters.
P=D/'_metrics.py';S=R/'sources/deepening_2026_10_04/schedule_production.py';ns={'dataclass':dataclasses.dataclass,'collections':collections,'statistics':statistics}
future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
exec(compile(ast.fix_missing_locations(ast.Module(body=[future]+ast.parse(S.read_text()).body,type_ignores=[])),str(S),'exec'),ns)
names={'InstantThroughput','compute_instant_throughput','log_performance_stats'};nodes=[n for n in ast.parse(P.read_text()).body if getattr(n,'name',None) in names];assert len(nodes)==3
exec(compile(ast.fix_missing_locations(ast.Module(body=[future]+nodes,type_ignores=[])),str(P),'exec'),ns)
logs=[];summaries=[];theory=summary['throughput/theoretical_flops'];perdevice=summary['throughput/theoretical_flops_per_device'];device_count=int(theory/perdevice)
ns.update(jax=types.SimpleNamespace(device_count=lambda:device_count,devices=lambda:[types.SimpleNamespace(device_kind=summary['throughput/device_kind'])]),device_flops_for_jax_device=lambda _:perdevice,levanter=types.SimpleNamespace(tracker=types.SimpleNamespace(log=lambda metrics,step:logs.append({'step':step,'metrics':metrics}),log_summary=lambda metrics:summaries.append(metrics))))
callback=ns['log_performance_stats'](length,ns['BatchSchedule'](batch),summary['throughput/flops_per_example']);callback(types.SimpleNamespace(step=int(summary['_step']),step_duration=summary['throughput/duration']))
replay=logs[0]['metrics'];fields=['throughput/total_tokens','throughput/duration','throughput/tokens_per_second','throughput/examples_per_second','throughput/gflops_per_second','throughput/mfu']
ck('Original logging closure reproduces six latest logged metric values from declared inputs',all(math.isclose(replay[k],summary[k],rel_tol=1e-10,abs_tol=1e-8) for k in fields))
zero=ns['compute_instant_throughput'](batch,0.,length,summary['throughput/flops_per_example'],theory)
ck('Original formula zero duration is unavailable rather than zero throughput',zero.tokens_per_second is None and zero.mfu is None)
comment=next(c for c in load(D/'comments_8506.json') if c['id']==6025364160);ct=datetime.datetime.fromisoformat(comment['created_at'].replace('Z','+00:00')).timestamp();pairs=[(a,b,x) for a,b,x in zip(rows,rows[1:],intervals) if a['_timestamp']<=ct<=b['_timestamp']];assert len(pairs)==1
_,_,bracket=pairs[0];alignment={'comment_id':comment['id'],'comment_url':comment['html_url'],'created_at':comment['created_at'],'author_reported_step':215756,'sampled_bracket':bracket,'reported_step_inside_bracket':bracket['start_step']<=215756<=bracket['end_step'],'actual_independent_node_logs':None,'actual_taint_initiating_condition':None,'actual_downtime_seconds':None}
ck('Public triage timestamp and reported step lie within sampled bracket without causal attribution',alignment['reported_step_inside_bracket'])
paths=[pathlib.Path(__file__),P,D/'state_adapter.py',S,R/'sources/main_incident_2026_10_07/train.py',R/'sources/state_2026_10_05/callback_core.py',A/'operational_refresh_acquisition.json']+[D/x for x in ['wandb_meta.json','wandb_operational_history.json']]+[D/f'{kind}_{n}.json' for kind in ['issue','comments'] for n in [8435,8506,8870]]
out={'checks_passed':len(checks),'checks':checks,'scope':__doc__,'comment_comparisons':changes,'sampled_rows':len(rows),'spec':acq['history_spec'],'nominal_tokens_per_logged_step':unit,'configured_batch':batch,'configured_sequence_length':length,'configured_log_every':tr['log_every'],'sampled_endpoint_span':full,'previous_to_new_summary_span':recent,'sampled_logged_duration_median':statistics.median(r['throughput/duration'] for r in rows),'sampled_logged_iteration_median':statistics.median(r['throughput/iteration_time'] for r in rows),'latest_summary':{'step':summary['_step'],'state':raw['state'],'heartbeatAt':raw['heartbeatAt'],'tokens_per_second':summary['throughput/tokens_per_second'],'mfu':summary['throughput/mfu'],'mean_mfu':summary['throughput/mean_mfu'],'mfu_sample_count':summary['throughput/mfu_sample_count']},'largest_calendar_seconds_per_step_intervals':sorted(intervals,key=lambda x:x['calendar_seconds_per_logged_step'],reverse=True)[:5],'triage_alignment':alignment,'original_formula_replay':{'fields':fields,'reproduced':replay,'reported':{k:summary[k] for k in fields},'device_count_adapter':device_count,'theoretical_peak_adapter':theory,'actual_hardware_peak_verified':None},'intervals':intervals,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'actual_unique_or_scored_token_progress':None,'actual_duty_cycle':None,'actual_fault_downtime':None,'actual_downtime_cause_decomposition':None,'actual_loss_causal_effect':None,'actual_production_deployed_revision':None}
(A/'operational_progress.json').write_text(json.dumps(out,indent=2)+'\n');print('Operational checks',len(checks),'passed; sampled rate',full['nominal_frontier_tokens_per_second'],'recent rate',recent['nominal_frontier_tokens_per_second'])
