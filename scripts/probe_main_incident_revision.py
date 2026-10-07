"""Host execution of main/proposal runtime helpers and Iris pod classification.
No actual launcher, XLA parsing, Kubernetes control plane, training or production deployment.
"""
import ast,hashlib,json,pathlib,runpy,types
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/main_incident_2026_10_07'
a=runpy.run_path(str(R/'scripts/probe_runtime_defaults.py'));old=a['ns'];new={}
def execute(nodes,p,ns):exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])),str(p),'exec'),ns)
ep=D/'ep_ragged_all_to_all.py';mt=D/'model.py';tp=D/'train.py'
flags=next(n for n in ast.parse(ep.read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='RAGGED_REQUIRED_XLA_FLAGS' for t in n.targets));execute([flags],ep,new)
off=next(n for n in ast.parse(mt.read_text()).body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.target.id=='OFFLOAD_CARRY_REMAT_MODE');execute([off],mt,new)
names={'HERO_EP_RUNTIME_ENV','XLA_COLLECTIVE_OVERLAP_FLAG','DEFAULT_COLLECTIVE_OVERLAP_LIMIT','INLINE_WATCH_COLLECTIVE_OVERLAP_LIMIT','RAGGED_COLLECTIVE_OVERLAP_LIMIT','OFFLOAD_CARRY_COLLECTIVE_OVERLAP_LIMIT','RAGGED_MOE_IMPLEMENTATION','XLA_DISABLE_GPU_COMMAND_BUFFER_FLAG','_RAGGED_REQUIRED_XLA_FLAG_NAMES'}
tr=ast.parse(tp.read_text());nodes=[n for n in tr.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets)];assert len(nodes)==len(names)
helper=next(n for n in tr.body if isinstance(n,ast.FunctionDef) and n.name=='_apply_hero_ep_runtime_defaults');execute(nodes+[helper],tp,new)
def run(ns,env,mode):
 e=dict(env);ns['os']=types.SimpleNamespace(environ=e);ns['_apply_hero_ep_runtime_defaults'](inline_watch_enabled=False,moe_implementation='ragged_all_to_all',remat_mode=mode,processes_per_task=1)
 return {'memory_fraction':e['XLA_PYTHON_CLIENT_MEM_FRACTION'],'flags':e['XLA_FLAGS'].split(),'flag_values':dict(f.partition('=')[::2] for f in e['XLA_FLAGS'].split())}
checks=[]
def ck(n,c):assert c,n;checks.append(n)
O='--xla_gpu_experimental_parallel_collective_overlap_limit';S='--xla_gpu_memory_limit_slop_factor';H='--xla_gpu_enable_host_memory_offloading';Y='--xla_gpu_disable_async_collectives'
cases={}
for label,mode,env in [('clean_carry','offload_carry',{}),('clean_no_carry','remat',{}),('inherited_no_carry','remat',{'XLA_FLAGS':O+'=8'}),('inherited_carry','offload_carry',{'XLA_FLAGS':O+'=8 '+S+'=85','XLA_PYTHON_CLIENT_MEM_FRACTION':'.75'})]:cases[label]={'main':run(new,env,mode),'proposal':run(old,env,mode)}
ck('Main clean carry retains .75/85 while proposal defaults .78/105',cases['clean_carry']['main']['memory_fraction']=='0.75' and cases['clean_carry']['main']['flag_values'][S]=='85' and cases['clean_carry']['proposal']['memory_fraction']=='0.78' and cases['clean_carry']['proposal']['flag_values'][S]=='105')
ck('No-carry explicit overlap remains eight on main but proposal forces one',cases['inherited_no_carry']['main']['flag_values'][O]=='8' and cases['inherited_no_carry']['proposal']['flag_values'][O]=='1')
ck('Carry mode forces overlap one in both snapshots',all(cases[k][rev]['flag_values'][O]=='1' for k in ['clean_carry','inherited_carry'] for rev in ['main','proposal']))
ck('Synchronous no-carry flag present only in proposal',Y not in cases['clean_no_carry']['main']['flag_values'] and cases['clean_no_carry']['proposal']['flag_values'][Y]=='ALLCOLLECTIVES')
ck('Host memory accounting flag present only in proposal carry',H not in cases['clean_carry']['main']['flag_values'] and cases['clean_carry']['proposal']['flag_values'][H]=='true')
ck('Explicit inherited fraction/slop still override both defaults',all(cases['inherited_carry'][rev]['memory_fraction']=='.75' and cases['inherited_carry'][rev]['flag_values'][S]=='85' for rev in ['main','proposal']))
ck('Both use exact required ragged flags in selected controls',all(all(f in cases[k][rev]['flags'] for f in ns['RAGGED_REQUIRED_XLA_FLAGS']) for k in cases for rev,ns in [('main',new),('proposal',old)]))
# Exact pure classification helpers; enum identities adapted to readable labels.
ip=D/'iris_k8s_tasks.py';it=ast.parse(ip.read_text());ins={'IRIS_TASK_CONTAINER_NAME':'task','job_pb2':types.SimpleNamespace(TASK_STATE_PREEMPTED='PREEMPTED',TASK_STATE_WORKER_FAILED='WORKER_FAILED',TASK_STATE_FAILED='FAILED')}
const={'_INFRASTRUCTURE_FAILURE_REASONS','_DISRUPTION_TARGET_CONDITION','_KUEUE_TERMINATION_TARGET_CONDITION','_KUEUE_WORKLOAD_EVICTION_REASON_PREFIX','_TERMINAL_REASON_MAX_CHARS'}
func={'_task_container_status','_disruption_condition','_format_disruption_reason','_pod_failure_state'}
nodes=[n for n in it.body if (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in const for t in n.targets)) or (isinstance(n,ast.FunctionDef) and n.name in func)];assert len(nodes)==len(const)+len(func);execute(nodes,ip,ins)
def pod(reason='Error',condition=None,status='True',ctype='DisruptionTarget'):
 x={'status':{'phase':'Failed','containerStatuses':[{'name':'task','state':{'terminated':{'reason':reason,'exitCode':137}}}]}}
 if condition is not None:x['status']['conditions']=[{'type':ctype,'status':status,'reason':condition,'message':'artificial condition'}]
 return x
inputs={'taint_137':pod(condition='DeletionByTaintManager'),'plain_137':pod(),'oom_137':pod('OOMKilled'),'false_condition':pod(condition='DeletionByTaintManager',status='False'),'kueue_evict':pod(condition='WorkloadEvictedByPreemption',ctype='TerminationTarget'),'unrelated_termination':pod(condition='UnrelatedShutdown',ctype='TerminationTarget'),'pod_evicted_no_container':{'status':{'phase':'Failed','reason':'Evicted'}}}
classified={k:ins['_pod_failure_state'](p) for k,p in inputs.items()}
ck('Exit 137 plus true taint disruption is preempted',classified['taint_137']=='PREEMPTED')
ck('Exit 137 alone is failed rather than evidence of taint',classified['plain_137']=='FAILED')
ck('Own-limit OOM follows application failure policy',classified['oom_137']=='FAILED')
ck('False condition does not classify as disruption',classified['false_condition']=='FAILED')
ck('Only specific Kueue termination reason marks preemption',classified['kueue_evict']=='PREEMPTED' and classified['unrelated_termination']=='FAILED')
ck('Pod-level eviction without task container is worker failure',classified['pod_evicted_no_container']=='WORKER_FAILED')
sidecar=pod();sidecar['status']['containerStatuses'].insert(0,{'name':'sidecar','state':{'terminated':{'reason':'Evicted','exitCode':137}}})
ck('Named task status takes priority over first sidecar',ins['_pod_failure_state'](sidecar)=='FAILED')
ck('Disruption description is bounded at five hundred chars',len(ins['_format_disruption_reason']({'reason':'DeletionByTaintManager','message':'x'*1000}))==500)
head=json.loads((D/'head.json').read_text());cmp=json.loads((D/'compare.json').read_text());pr=json.loads((D/'pull_9833.json').read_text())
ck('Observed main and proposal are diverged with fourteen and ten commits',head['sha']=='eee467718515b2383fc3a433014afce4ab075b05' and cmp['status']=='diverged' and cmp['ahead_by']==14 and cmp['behind_by']==10 and pr['head']['sha']=='b65be4c9550c5097f0a3add08933531a1c24d534' and pr['merged'] is False)
def canonical(xs):return [(x['id'],x['body']) for x in xs]
refresh={}
for n,oldp in [(8435,R/'sources/comments.json'),(8506,R/'sources/runtime_defaults_2026_10_07/comments_8506.json')]:
 current=json.loads((D/f'comments_{n}.json').read_text());prior=json.loads(oldp.read_text());issue=json.loads((D/f'issue_{n}.json').read_text());refresh[str(n)]={'count':len(current),'prior_count':len(prior),'ids_and_bodies_unchanged':canonical(current)==canonical(prior),'updated_at':issue['updated_at'],'last_id':current[-1]['id']}
 ck('Live issue '+str(n)+' matches latest archived identities and bodies',len(current)==issue['comments'] and canonical(current)==canonical(prior))
files=[tp,mt,ep,ip,D/'head.json',D/'compare.json',D/'pull_9833.json',D/'comments_8435.json',D/'comments_8506.json',R/'scripts/probe_runtime_defaults.py',R/'sources/runtime_defaults_2026_10_07/train.py',a['M'],a['E'],pathlib.Path(__file__)]
out={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'main_revision':head['sha'],'proposal_revision':pr['head']['sha'],'runtime_cases':cases,'pod_inputs':inputs,'pod_classification':classified,'source_relationship':{'status':cmp['status'],'ahead_by':cmp['ahead_by'],'behind_by':cmp['behind_by'],'merge_base':cmp['merge_base_commit']['sha'],'pull_state':pr['state'],'pull_merged':pr['merged']},'issue_refresh':refresh,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'actual_deployed_revision':None,'actual_noexecute_root_cause':None,'actual_kubernetes_pod_status':None,'actual_GPU_effect':None,'substitutions':['os.environ isolated dictionaries','enum labels and task-container name adapter for pure pod helpers','no launcher imports, XLA parser or Kubernetes control loop']}
(R/'analysis/main_incident_revision.json').write_text(json.dumps(out,indent=2)+'\n');print('Main/proposal runtime, pod classification and source refresh controls:',len(checks),'passed')
