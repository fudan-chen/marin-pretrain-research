"""Original schedule and callback bodies on declared synthetic inputs, not historical loader replay."""
import ast,json,pathlib,types,hashlib,subprocess,sys,tempfile
from typing import Sequence,Tuple,List
import audit_batch_clock as a
R=a.R;checks=[]
def ck(n,v):assert v,n;checks.append(n)
rows=[{'start':0,'value':4},{'start':3,'value':8}];bs=a.schedule(rows)
ck('Piecewise offsets equal independently summed prior batches',all(bs.global_data_offset_by_step(s)==sum(4 if i<3 else 8 for i in range(s)) for s in range(20)))
ck('New batch applies at boundary, old batch owns preceding step',bs.batch_size_at_step(2)==4 and bs.batch_size_at_step(3)==8)
ck('Boundary starts at offset twelve',bs.global_data_offset_by_step(3)==12)
ck('Offset twenty is not current batch times step',bs.global_data_offset_by_step(4)==20 and 4*8==32)
ck('Batch intervals are consecutive and their inverse returns owner',all(list(bs.batch_indices_at_step(s))==list(range(bs.global_data_offset_by_step(s),bs.global_data_offset_by_step(s+1))) and all(bs.find_step_containing_offset(i)==s for i in bs.batch_indices_at_step(s)) for s in range(12)))
j={'batch_schedule':rows,'block_size':8,'fixed_sequence_length':4096,'mixture_stages':[{'step':0,'weights':{'A':.34,'B':.33,'C':.33}},{'step':4,'weights':{'A':1}}],'previous_batch_schedule':[{'start':0,'value':8}],'resume_step':4}
r=a.audit(j)
ck('Step-four mixture offset twenty fails eight-sequence alignment',not r['construction_alignment_ok'] and r['stage_boundaries'][1]['block_remainder']==4)
ck('Configured fractions differ from whole-block allocation',r['stage_boundaries'][0]['whole_block_counts']=={'A':4,'B':2,'C':2})
ck('Changing past batch schedule moves next logical sequence by minus twelve',r['resume']['sequence_offset_delta']==-12 and not r['resume']['consumed_batch_prefix_equal'])
ck('Nominal positions are explicitly fixed-length arithmetic',r['stage_boundaries'][1]['nominal_positions']==81920 and r['actual_consumed_tokens'] is None)
equal=dict(j,previous_batch_schedule=[{'start':0,'value':4},{'start':2,'value':6}],resume_step=4)
eq=a.audit(equal)
ck('Equal endpoint offsets do not prove identical consumed batch prefix',eq['resume']['sequence_offset_delta']==0 and not eq['resume']['consumed_batch_prefix_equal'])
future=dict(j,previous_batch_schedule=rows+[{'start':10,'value':16}],resume_step=4)
ck('Future-only declaration change preserves consumed prefix',a.audit(future)['resume']['consumed_batch_prefix_equal'])
# Execute archived conversion and constructor's alignment loop, not its JAX initialization.
t=a.tree;f=next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name=='rescale_mixture_schedule_for_batch_schedule');env=dict(vars(a));env.update(Sequence=Sequence,Tuple=Tuple,List=List);exec(compile(ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[])),str(a.M),'exec'),env)
rescale=env[f.name];ck('Original conversion uses cumulative offset',rescale([(0,{'A':1}),(4,{'B':1})],bs)[1][0]==20)
ck('Original conversion preserves final negative sentinel',rescale([(0,{'A':1}),(-1,{'B':1})],bs)[1][0]==-1)
init=next(x for x in a.cls.body if isinstance(x,ast.FunctionDef) and x.name=='__init__');guard=next(x for x in init.body if isinstance(x,ast.For));guardcode=compile(ast.fix_missing_locations(ast.Module(body=[guard],type_ignores=[])),str(a.M),'exec')
def rejected(stages):
    try:exec(guardcode,{'weight_stages':stages,'block_size':8})
    except AssertionError:return True
    return False
ck('Original constructor rejects unaligned converted stage',rejected(rescale([(0,{'A':1}),(4,{'B':1})],bs)))
ck('Conversion negative sentinel is not supported by constructor alignment guard',rejected(rescale([(0,{'A':1}),(-1,{'B':1})],bs)))
# Original callback on a synthetic dataset; record exact logged keys.
p=R/'sources/scale_2026_10_05/train_hero_ep.py';tf=ast.parse(p.read_text());callback=next(x for x in tf.body if isinstance(x,ast.FunctionDef) and x.name=='_make_mixture_stage_callback')
callback.returns=None
for arg in callback.args.args:arg.annotation=None
logs=[];e={'levanter':types.SimpleNamespace(tracker=types.SimpleNamespace(log=lambda data,step:logs.append({'step':step,'data':data}))) };exec(compile(ast.fix_missing_locations(ast.Module(body=[callback],type_ignores=[])),str(p),'exec'),e)
ds=types.SimpleNamespace(block_size=8,weight_stages=[(0,{'A':.5,'B':.5}),(8,{'A':1.})],_get_stage_for_block=lambda b:0 if b<1 else 1)
cb=e[callback.name](ds,a.schedule([{'start':0,'value':4}]))
for s in [0,1,2,3]:cb(types.SimpleNamespace(step=s))
ck('Stage callback emits only on stage changes',[x['step'] for x in logs]==[0,2])
ck('Removed component key is omitted rather than explicitly logged zero','mixture/weight/B' not in logs[-1]['data'])
ck('Logged configured weights are not measured batch counts',logs[0]['data']['mixture/weight/A']==.5 and not any('count' in k for k in logs[0]['data']))
for name,change in [('negative_batch',{'batch_schedule':[{'start':0,'value':-1}]}),('missing_zero',{'batch_schedule':[{'start':1,'value':4}]}),('duplicate_start',{'batch_schedule':[{'start':0,'value':4},{'start':0,'value':8}]}),('invalid_block',{'block_size':65536}),('nan_weight',{'mixture_stages':[{'step':0,'weights':{'A':float('nan')}}]})]:
    try:a.audit(dict(j,**change))
    except ValueError:ck('Reviewer rejects '+name,True)
    else:ck('Reviewer rejects '+name,False)
small=dict(j,mixture_stages=[{'step':0,'weights':{'A':.99,'B':.01}}])
ck('Positive tiny component can round to zero with an explicit warning',a.audit(small)['stage_boundaries'][0]['whole_block_counts']['B']==0 and bool(a.audit(small)['stage_boundaries'][0]['warnings']))
aligned=dict(j,batch_schedule=[{'start':0,'value':4}],mixture_stages=[{'step':0,'weights':{'A':1}},{'step':2,'weights':{'B':1}}])
ck('Aligned step boundary passes declaration review',a.audit(aligned)['construction_alignment_ok'])
with tempfile.TemporaryDirectory() as temp:
    ip=pathlib.Path(temp)/'input.json';op=pathlib.Path(temp)/'output.json';ip.write_text(json.dumps(j));cmd=[sys.executable,str(R/'scripts/audit_batch_clock.py'),str(ip),str(op)]
    process=subprocess.run(cmd,capture_output=True,text=True);out=json.loads(op.read_text())
    ck('Actual CLI writes hashed input and no actual token claims',process.returncode==0 and out['input_sha256']==hashlib.sha256(ip.read_bytes()).hexdigest() and out['actual_consumed_tokens'] is None)
    before=op.read_bytes();again=subprocess.run(cmd,capture_output=True,text=True)
    ck('Actual CLI refuses output overwrite and preserves bytes',again.returncode!=0 and op.read_bytes()==before)
result={'scope':__doc__, 'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest()},'checks_passed':len(checks),'checks':checks,'synthetic_review':r,'equal_endpoint_counterexample':eq['resume'],'original_callback_logs':logs,'actual_GPU_behavior':None,'actual_historical_cursor_replay':None}
(R/'analysis/batch_clock_probe.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print('Batch clock bounded checks:',len(checks))
