"""Strict record-shape and host-coverage layer over retained V93 consistency checker.
Neither version reads real data or approves training.
"""
import json,math,pathlib,runpy,sys
P=pathlib.Path(__file__).with_name('check_data_execution_record.py')
old_check=runpy.run_path(str(P))['check_record']
def name_list(value,allow_empty=False):
    return isinstance(value,list) and (allow_empty or bool(value)) and all(isinstance(n,str) and bool(n.strip()) for n in value) and len(set(value))==len(value)
def check_record(record):
    if not isinstance(record,dict):return old_check(record)
    conflicts=[];missing=[]
    actual=record.get('actual_children')
    if isinstance(actual,list):
        for child in actual:
            if not isinstance(child,dict):continue
            if not isinstance(child.get('name'),str) or not child['name'].strip():conflicts.append('Actual child requires nonempty name')
            kind=child.get('kind')
            if kind is not None and kind not in ['cached','direct','concat']:conflicts.append('Unsupported actual child kind')
            if kind=='concat' and child.get('children') is not None and not name_list(child['children']):conflicts.append('Concat requires unique nonempty child names')
    runtime=record.get('runtime')
    if isinstance(runtime,dict):
        for k in ['python','numpy','jax','normalization_policy','source_revision']:
            v=runtime.get(k)
            if v is not None and (not isinstance(v,str) or not v.strip()):conflicts.append('Runtime identity must be nonempty string: '+k)
    panel=record.get('evaluation_panel')
    if isinstance(panel,dict):
        for side in ['expected','actual']:
            domains=panel.get(side)
            if isinstance(domains,dict):
                for name,entry in domains.items():
                    if not isinstance(name,str) or not name.strip():conflicts.append('Invalid evaluation name')
                    if isinstance(entry,dict) and entry.get('children') is not None and not name_list(entry['children'],allow_empty=True):conflicts.append('Evaluation children must be unique name list: '+str(name))
    expected=record.get('expected_hosts');hosts=record.get('host_build_plans')
    if expected is None:missing.append('expected_hosts explicitly enumerating participating hosts')
    elif not name_list(expected):conflicts.append('expected_hosts requires unique nonempty names')
    elif isinstance(hosts,dict):
        if set(expected)-set(hosts):missing.append('Missing expected host plans: '+','.join(sorted(set(expected)-set(hosts))))
        if set(hosts)-set(expected):conflicts.append('Unexpected host plans: '+','.join(sorted(set(hosts)-set(expected))))
    if isinstance(hosts,dict):
        for host,plan in hosts.items():
            if not isinstance(host,str) or not host.strip():conflicts.append('Host requires nonempty name')
            if isinstance(plan,list):
                names=[x.get('name') for x in plan if isinstance(x,dict)]
                if not name_list(names,allow_empty=True):conflicts.append('Host plan requires unique nonempty names: '+str(host))
    # V93 math.isfinite converts ints to binary64; oversized JSON integers can overflow.
    oversized=False
    stages=record.get('weight_stages')
    if isinstance(stages,list):
        for w in stages:
            if not isinstance(w,dict):continue
            for value in w.values():
                if isinstance(value,int) and not isinstance(value,bool):
                    try:math.isfinite(value)
                    except OverflowError:oversized=True
    if oversized:conflicts.append('Weight integer cannot be represented in finite binary64')
    # Reject shape conflicts before calling legacy code; preserve its historical behavior.
    if conflicts:
        out={'scope':'Supplied record consistency only; strict shape layer. No training approval.',
             'gates':[],'production_execution_verified':False,'training_benefit_verified':False}
    else:out=old_check(record)
    out['gates'].append({'gate':'record_shape_and_host_coverage','status':'conflict' if conflicts else 'missing' if missing else 'consistent',
                         'conflicts':conflicts,'missing':missing,'notes':['Host enumeration is supplied evidence, not independent cluster discovery.']})
    out['status']='conflict' if any(g['status']=='conflict' for g in out['gates']) else 'needs_evidence' if any(g['status']=='missing' for g in out['gates']) else 'record_consistent_only'
    out['checker_version']='V94'
    return out
if __name__=='__main__':
    result=check_record(json.loads(pathlib.Path(sys.argv[1]).read_text()))
    pathlib.Path(sys.argv[2]).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(result['status'])
