"""Tamper controls for the proposed record checker and an incomplete public archive example.
These are checker tests, not original training-source execution or production validation.
"""
import copy,hashlib,json,pathlib,runpy
R=pathlib.Path(__file__).resolve().parents[1]
P=R/'scripts/check_data_execution_record.py';check=runpy.run_path(str(P))['check_record']
good={'scope':'SYNTHETIC checker fixture, no real execution',
 'declared_components':['A','B'],'weight_stages':[{'A':.5,'B':.5}],
 'actual_children':[{'name':n,'kind':'cached','sequence_count':3,'content_sha256':hashlib.sha256(('synthetic-'+n).encode()).hexdigest()} for n in ['A','B']],
 'dataset_index':['A','B'],'block_size':10,'integer_quotas':[[5,5]],
 'runtime':{'python':'fixture','numpy':'fixture','jax':'fixture','normalization_policy':'fixture','source_revision':'fixture'},
 'host_build_plans':{'h0':[{'name':'A','action':'build','cache_identity':'fixture-A'},{'name':'B','action':'build','cache_identity':'fixture-B'}],
                     'h1':[{'name':'A','action':'build','cache_identity':'fixture-A'},{'name':'B','action':'build','cache_identity':'fixture-B'}]},
 'evaluation_panel':{'expected':{'eval':{'children':['x','y'],'content_sha256':hashlib.sha256(b'fixture-panel').hexdigest()}},
                     'actual':{'eval':{'children':['x','y'],'content_sha256':hashlib.sha256(b'fixture-panel').hexdigest(),'valid_targets':6}}}}
checks=[];cases={}
def run(name,r,expected):
 o=check(r);assert o['status']==expected,(name,o);assert o['production_execution_verified'] is False and o['training_benefit_verified'] is False
 checks.append(name);cases[name]=o
run('Consistent synthetic fixture is not production approval',good,'record_consistent_only')
def altered():return copy.deepcopy(good)
r=altered();r['actual_children']=None;run('Missing actual objects remains evidence gap',r,'needs_evidence')
r=altered();r['actual_children']=r['actual_children'][:1];run('Missing positive child conflicts',r,'conflict')
r=altered();r['weight_stages'][0]['TYPO']=.1;run('Unknown declared key conflicts',r,'conflict')
r=altered();r['actual_children'][1].update(kind='concat',children=[]);run('Empty concat conflicts',r,'conflict')
r=altered();r['integer_quotas']=[[4,5]];run('Quota conservation fails',r,'conflict')
r=altered();r['dataset_index']=['B','A'];run('Actual order mismatch conflicts',r,'conflict')
r=altered();r['runtime']=None;run('Absent runtime is missing evidence',r,'needs_evidence')
r=altered();r['host_build_plans']['h1'].reverse();run('Different build dispatch order conflicts',r,'conflict')
r=altered();r['host_build_plans']['h1'][0]['cache_identity']='other';run('Different cache identity conflicts',r,'conflict')
r=altered();r['evaluation_panel']['actual']['eval']['children']=['x'];run('Same top-level eval with missing child conflicts',r,'conflict')
r=altered();r['evaluation_panel']['actual']['eval']['content_sha256']='changed';run('Eval content differs',r,'conflict')
r=altered();r['evaluation_panel']['actual']['eval']['valid_targets']=0;run('Zero eval denominator conflicts',r,'conflict')
r=altered();r['weight_stages'][0]['A']=float('nan');run('Nonfinite declaration conflicts',r,'conflict')
r=altered();r['weight_stages'][0]={'A':1,'B':.00001};r['integer_quotas']=[[10,0]];run('Positive zero quota is explicit diagnostic not hidden reject',r,'record_consistent_only');assert cases['Positive zero quota is explicit diagnostic not hidden reject']['gates'][2]['notes']
run('Empty template cannot pass',{},'needs_evidence')
run('Wrong top-level type rejected',[],'conflict')
meta=R/'sources/live_2026_10_07/meta.json';local=R/'analysis/normalization_python312.json'
d=json.loads(meta.read_text())['config']['data']['value'];nr=json.loads(local.read_text())
archive={'scope':'Archived declarations plus local quota control; actual constructed objects/host plans/panel not acquired',
 'declared_components':list(d['components']),'weight_stages':[x[1] for x in d['train_weights']],
 'actual_children':None,'dataset_index':None,'block_size':49152,
 'integer_quotas':[x['quotas'] for x in nr['cases'] if x['source']=='oct7_wandb' and x['summation']=='native'],
 'runtime':{'python':nr['runtime']['python'],'numpy':nr['runtime']['numpy'],'jax':None,
            'normalization_policy':'local native builtins.sum, not actual Hero replay','source_revision':'b65be4c9550c5097f0a3add08933531a1c24d534'},
 'host_build_plans':None,'evaluation_panel':None}
run('Public declarations and local quota do not prove actual execution',archive,'needs_evidence')
(R/'templates/data_execution_record.json').write_text(json.dumps({k:None for k in good if k!='scope'},indent=2)+'\n')
(R/'templates/data_execution_record_synthetic.json').write_text(json.dumps(good,indent=2)+'\n')
(R/'analysis/data_execution_record_archived.json').write_text(json.dumps(archive,indent=2)+'\n')
(R/'analysis/data_execution_record_archived_check.json').write_text(json.dumps(cases['Public declarations and local quota do not prove actual execution'],indent=2)+'\n')
out={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'cases':cases,
 'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [P,meta,local]},
 'actual_production_execution':None,'actual_token_stream':None,'actual_training_benefit':None}
(R/'analysis/data_record_checker_probe.json').write_text(json.dumps(out,indent=2)+'\n')
print('Proposed checker tamper controls:',len(checks),'passed; archived record needs_evidence')
