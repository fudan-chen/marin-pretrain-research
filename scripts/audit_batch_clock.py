"""Review declared schedules, not actual consumed tokens. Uses archived pure-Python
BatchSchedule and mixture allocation helper; no JAX, loader, token replay or training.
Usage: python scripts/audit_batch_clock.py input.json output.json
"""
import ast,dataclasses,hashlib,json,pathlib,sys,types,warnings
import numpy as np
R=pathlib.Path(__file__).resolve().parents[1]
P=R/'sources/deepening_2026_10_04/schedule_production.py'
namespace={'__name__':__name__};exec(compile('from __future__ import annotations\n'+P.read_text(),str(P),'exec'),namespace)
BatchSchedule=namespace['BatchSchedule'];ScheduleStep=namespace['ScheduleStep']
M=R/'sources/deepening_2026_10_04/mixture_production.py'
tree=ast.parse(M.read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='MixtureDataset')
method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_compute_expected_counts_per_block')
exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(M),'exec'),globals())
def schedule(rows):
    if not isinstance(rows,list) or not rows:raise ValueError('batch_schedule must be a nonempty list')
    for i,row in enumerate(rows):
        if type(row.get('start')) is not int or type(row.get('value')) is not int or row['value']<=0 or row['start']<0:raise ValueError('nonnegative integer start and positive integer batch required')
        if (i==0 and row['start']!=0) or (i>0 and row['start']<=rows[i-1]['start']):raise ValueError('starts must begin at zero and increase')
    return BatchSchedule([ScheduleStep(x['start'],x['value']) for x in rows])
def audit(j):
    bs=schedule(j['batch_schedule']);k=j['block_size'];length=j.get('fixed_sequence_length')
    if type(k) is not int or not 0<k<2**16:raise ValueError('block_size must be an integer in [1,65535]')
    if length is not None and (type(length) is not int or length<=0):raise ValueError('fixed sequence length must be a positive integer or null')
    stages=j['mixture_stages'];out=[]
    if not isinstance(stages,list) or not stages:raise ValueError('mixture_stages cannot be empty')
    for i,s in enumerate(stages):
        step=s['step'];w=s['weights']
        if type(step) is not int or step<0 or (i==0 and step!=0) or (i>0 and step<=stages[i-1]['step']):raise ValueError('mixture steps must start at zero and increase; no negative sentinel')
        if not isinstance(w,dict) or not w or any(type(v) not in (int,float) or not np.isfinite(v) or v<0 for v in w.values()) or not np.isfinite(sum(w.values())) or sum(w.values())<=0:raise ValueError('finite nonnegative weights with positive sum required')
        norm={n:v/sum(w.values()) for n,v in w.items() if v>0};names=list(norm);offset=bs.global_data_offset_by_step(step)
        with warnings.catch_warnings(record=True) as caught:
            counts=_compute_expected_counts_per_block(types.SimpleNamespace(datasets=norm,dataset_index=names),norm,k)
        out.append({'step':step,'sequence_offset':offset,'nominal_positions':offset*length if length else None,'aligned_to_block':offset%k==0,'block_remainder':offset%k,'logged_weights':norm,'whole_block_counts':dict(zip(names,map(int,counts))),'whole_block_weights':dict(zip(names,map(lambda n:int(n)/k,counts))),'warnings':[str(x.message) for x in caught]})
    resume=None
    if 'previous_batch_schedule' in j:
        old=schedule(j['previous_batch_schedule']);r=j['resume_step']
        if type(r) is not int or r<0:raise ValueError('resume_step must be a nonnegative integer')
        a=old.global_data_offset_by_step(r);b=bs.global_data_offset_by_step(r)
        # Compare the entire consumed prefix, not just equal totals at one step.
        boundaries=sorted({0,r}|{x.start for x in old.segments+bs.segments if 0<x.start<r})
        equal=all(old.batch_size_at_step(x)==bs.batch_size_at_step(x) for x in boundaries[:-1])
        resume={'completed_steps':r,'old_next_sequence_offset':a,'new_next_sequence_offset':b,'sequence_offset_delta':b-a,'consumed_batch_prefix_equal':equal,'decision':'prefix_consistent_declaration_only' if equal else 'requires_cursor_and_prefix_review'}
    return {'scope':__doc__,'stage_boundaries':out,'resume':resume,'construction_alignment_ok':all(x['aligned_to_block'] for x in out),'actual_consumed_tokens':None,'actual_loader_cursor':None,'actual_historical_schedule':None,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [P,M]}}
if __name__=='__main__':
    if len(sys.argv)!=3:raise SystemExit(__doc__)
    target=pathlib.Path(sys.argv[2])
    if target.exists():raise SystemExit('Refusing to overwrite output')
    source=pathlib.Path(sys.argv[1]);raw=source.read_bytes();result=audit(json.loads(raw));result['input_sha256']=hashlib.sha256(raw).hexdigest()
    with target.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
