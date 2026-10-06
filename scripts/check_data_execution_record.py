"""Check supplied data execution records, never attest production execution.
Usage: python scripts/check_data_execution_record.py INPUT.json OUTPUT.json
"""
import json, math, pathlib, re, sys

def valid_sha(value):
    return isinstance(value,str) and re.fullmatch(r'[0-9a-f]{64}',value) is not None

def check_record(record):
    if not isinstance(record,dict):
        return {'scope':'Supplied record consistency only. No training approval.','status':'conflict',
                'gates':[{'gate':'input','status':'conflict','conflicts':['Top-level object required'],'missing':[],'notes':[]}],
                'production_execution_verified':False,'training_benefit_verified':False}
    gates=[]
    def add(name,fail,missing,notes=()):
        gates.append({'gate':name,'status':'conflict' if fail else 'missing' if missing else 'consistent',
                      'conflicts':fail,'missing':missing,'notes':list(notes)})
    names=record.get('declared_components');stages=record.get('weight_stages');fail=[];missing=[]
    if names is None:missing.append('declared_components')
    elif not isinstance(names,list) or not names or any(not isinstance(n,str) or not n for n in names) or len(set(names))!=len(names):fail.append('Components must be unique nonempty names')
    known=set(names) if isinstance(names,list) and all(isinstance(n,str) for n in names) else set()
    active=set()
    if stages is None:missing.append('weight_stages')
    elif not isinstance(stages,list) or not stages:fail.append('Nonempty stage list required')
    else:
        for i,w in enumerate(stages):
            if not isinstance(w,dict) or not w:fail.append('Stage '+str(i)+' requires weights');continue
            if any(not isinstance(k,str) or not k for k in w):fail.append('Invalid bucket name in stage '+str(i))
            unknown=set(w)-known
            if unknown and names is not None:fail.append('Unknown declared keys: '+str(sorted(map(str,unknown))))
            numeric=all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and v>=0 for v in w.values())
            if not numeric or (numeric and not any(v>0 for v in w.values())):fail.append('Stage '+str(i)+' weights must be finite nonnegative with positive support');continue
            active.update(k for k,v in w.items() if v>0)
    add('declared_support',fail,missing)

    actual=record.get('actual_children');fail=[];missing=[]
    if actual is None:missing.append('actual_children from constructed objects, all-stage positive support')
    elif not isinstance(actual,list):fail.append('actual_children must be a list')
    else:
        seen=set()
        for child in actual:
            if not isinstance(child,dict) or not isinstance(child.get('name'),str):fail.append('Child requires name');continue
            name=child['name']
            if name in seen:fail.append('Duplicate actual child '+name)
            seen.add(name)
            if known and name not in known:fail.append('Undeclared actual child '+name)
            length=child.get('sequence_count')
            if length is None:missing.append(name+' sequence_count')
            elif not isinstance(length,int) or isinstance(length,bool) or length<=0:fail.append(name+' nonempty finite inventory required by this checker scope')
            if child.get('kind') is None:missing.append(name+' kind')
            elif child['kind']=='concat':
                sub=child.get('children')
                if sub is None:missing.append(name+' concat children')
                elif not isinstance(sub,list) or not sub:fail.append(name+' empty/invalid concat children')
            if not child.get('content_sha256'):missing.append(name+' content_sha256')
            elif not valid_sha(child['content_sha256']):fail.append(name+' malformed content_sha256')
        if active-seen:fail.append('Missing positive children: '+','.join(sorted(active-seen)))
    add('actual_support_and_inventory',fail,missing,['This checker scope is finite cached sequence datasets; supplied hashes are not independently read back.'])

    quotas=record.get('integer_quotas');order=record.get('dataset_index');block=record.get('block_size');fail=[];missing=[];notes=[]
    valid_order=isinstance(order,list) and bool(order) and all(isinstance(n,str) for n in order) and len(set(order))==len(order)
    if order is None:missing.append('dataset_index in actual order')
    elif not valid_order:fail.append('Invalid dataset_index')
    valid_block=isinstance(block,int) and not isinstance(block,bool) and 0<block<65536
    if block is None:missing.append('block_size')
    elif not valid_block:fail.append('Invalid packed-ID block_size')
    if quotas is None:missing.append('integer_quotas')
    elif not isinstance(quotas,list):fail.append('integer_quotas must be phase rows')
    else:
        if isinstance(stages,list) and len(quotas)!=len(stages):fail.append('Quota phase count differs')
        for i,q in enumerate(quotas):
            if not isinstance(q,list) or any(not isinstance(v,int) or isinstance(v,bool) or v<0 for v in q):fail.append('Invalid quota row '+str(i));continue
            if valid_order and len(q)!=len(order):fail.append('Quota width differs in stage '+str(i))
            if valid_block and sum(q)!=block:fail.append('Quota does not conserve block in stage '+str(i))
            if valid_order and len(q)==len(order) and isinstance(stages,list) and i<len(stages) and isinstance(stages[i],dict):
                for n,v in zip(order,q):
                    w=stages[i].get(n,0)
                    if isinstance(w,(int,float)) and w==0 and v>0:fail.append('Quota assigned to inactive '+n)
                    if isinstance(w,(int,float)) and w>0 and v==0:notes.append('Positive weight rounded to zero: phase '+str(i)+' '+n)
        if actual is not None and isinstance(actual,list) and valid_order:
            actual_names=[c.get('name') for c in actual if isinstance(c,dict)]
            if order!=actual_names:fail.append('dataset_index differs from ordered actual children')
    runtime=record.get('runtime')
    for field in ['python','numpy','jax','normalization_policy','source_revision']:
        if not isinstance(runtime,dict) or not runtime.get(field):missing.append('runtime.'+field)
    add('quota_and_runtime_identity',fail,missing,notes)

    hosts=record.get('host_build_plans');fail=[];missing=[]
    if hosts is None:missing.append('host_build_plans or explicit no-build plan per participating host')
    elif not isinstance(hosts,dict) or not hosts:fail.append('Host plan map must be nonempty')
    else:
        builds=[]
        for host,plan in hosts.items():
            if not isinstance(plan,list):fail.append(str(host)+' plan must be list');continue
            if any(not isinstance(x,dict) or not isinstance(x.get('name'),str) or x.get('action') not in ['load','build','skip'] for x in plan):fail.append(str(host)+' invalid actions');continue
            builds.append([(x['name'],x.get('cache_identity')) for x in plan if x['action']=='build'])
            if any(x['action']=='build' and not x.get('cache_identity') for x in plan):missing.append(str(host)+' build cache_identity')
        if builds and any(x!=builds[0] for x in builds[1:]):fail.append('Ordered build plans differ between supplied hosts')
    add('ordered_build_plan',fail,missing,['Equality covers supplied hosts only; real collective protocol remains outside this checker.'])

    panel=record.get('evaluation_panel');fail=[];missing=[]
    if panel is None:missing.append('evaluation_panel expected and actual identities')
    elif not isinstance(panel,dict):fail.append('evaluation_panel must be an object')
    else:
        expected=panel.get('expected');actual_panel=panel.get('actual')
        if expected is None or actual_panel is None:missing.append('expected/actual evaluation domains')
        elif not isinstance(expected,dict) or not isinstance(actual_panel,dict):fail.append('Domain maps required')
        else:
            if not expected:fail.append('Empty expected evaluation panel')
            if set(expected)!=set(actual_panel):fail.append('Evaluation domain sets differ')
            for name in set(expected)&set(actual_panel):
                a=actual_panel[name];e=expected[name]
                if not isinstance(a,dict) or not isinstance(e,dict):fail.append(name+' invalid evaluation record');continue
                for field in ['children','content_sha256']:
                    if field not in a or field not in e:missing.append(name+' '+field)
                    elif a[field]!=e[field]:fail.append(name+' evaluation '+field+' differs')
                if 'content_sha256' in a and not valid_sha(a['content_sha256']):fail.append(name+' malformed evaluation hash')
                if 'content_sha256' in e and not valid_sha(e['content_sha256']):fail.append(name+' malformed expected evaluation hash')
                n=a.get('valid_targets')
                if n is None:missing.append(name+' valid_targets')
                elif not isinstance(n,int) or isinstance(n,bool) or n<=0:fail.append(name+' positive effective denominator required')
    add('evaluation_panel_identity',fail,missing,['This checks identities/denominator records, not score quality or independent content verification.'])
    status='conflict' if any(x['status']=='conflict' for x in gates) else 'needs_evidence' if any(x['status']=='missing' for x in gates) else 'record_consistent_only'
    return {'scope':'Supplied record consistency only, finite cached datasets. No training approval.',
            'status':status,'gates':gates,'production_execution_verified':False,'training_benefit_verified':False}

if __name__=='__main__':
    data=json.loads(pathlib.Path(sys.argv[1]).read_text())
    result=check_record(data)
    pathlib.Path(sys.argv[2]).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(result['status'])
