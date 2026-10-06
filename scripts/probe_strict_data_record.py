"""New-tool regressions comparing retained V93 against strict V94, no Marin execution."""
import copy,hashlib,json,pathlib,runpy
R=pathlib.Path(__file__).resolve().parents[1]
P=R/'scripts/check_data_execution_record_v94.py';ns=runpy.run_path(str(P));check=ns['check_record'];old=ns['old_check']
good=json.loads((R/'templates/data_execution_record_synthetic.json').read_text());good['expected_hosts']=['h0','h1']
cases={};checks=[]
def run(name,r,expected):
    new=check(r);assert new['status']==expected,(name,new)
    assert new['production_execution_verified'] is False
    try:prior=old(r)['status']
    except Exception as e:prior=type(e).__name__
    cases[name]={'old_status':prior,'new':new};checks.append(name)
run('Complete synthetic fixture remains record consistency only',good,'record_consistent_only')
r=copy.deepcopy(good);r['actual_children'][0]['kind']='unsupported';run('Unsupported kind no longer self-consistent',r,'conflict')
r=copy.deepcopy(good)
for side in ['expected','actual']:r['evaluation_panel'][side]['eval']['children']='xy'
run('Matching malformed eval children rejected',r,'conflict')
r=copy.deepcopy(good);r['runtime']['python']=['3.12'];run('Runtime list rejected',r,'conflict')
r=copy.deepcopy(good);r.pop('expected_hosts');run('No expected host coverage is missing evidence',r,'needs_evidence')
r=copy.deepcopy(good);r['host_build_plans'].pop('h1');run('Missing expected host is missing evidence',r,'needs_evidence')
r=copy.deepcopy(good);r['host_build_plans']['h2']=r['host_build_plans']['h1'];run('Unexpected host conflicts',r,'conflict')
r=copy.deepcopy(good);r['host_build_plans']['h1'].append(r['host_build_plans']['h1'][0]);run('Duplicate dispatch names conflict',r,'conflict')
r=copy.deepcopy(good);r['weight_stages'][0]['A']=10**400;run('Oversized JSON integer reports conflict without crash',r,'conflict')
r=copy.deepcopy(good);r['actual_children'][0].update(kind='concat',children=['x','x']);run('Duplicate concat child names rejected',r,'conflict')
archive=json.loads((R/'analysis/data_execution_record_archived.json').read_text());run('Archive remains missing execution evidence',archive,'needs_evidence')
run('Empty input remains missing evidence',{},'needs_evidence')
(R/'templates/data_execution_record_v94.json').write_text(json.dumps(dict(json.loads((R/'templates/data_execution_record.json').read_text()),expected_hosts=None),indent=2)+'\n')
(R/'templates/data_execution_record_synthetic_v94.json').write_text(json.dumps(good,indent=2)+'\n')
out={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'cases':cases,
     'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [P,R/'scripts/check_data_execution_record.py',R/'templates/data_execution_record_synthetic.json',R/'analysis/data_execution_record_archived.json']},
     'actual_production_execution':None,'actual_training_benefit':None}
(R/'analysis/strict_data_record_probe.json').write_text(json.dumps(out,indent=2)+'\n')
print('Strict tool regression controls:',len(checks),'passed')
