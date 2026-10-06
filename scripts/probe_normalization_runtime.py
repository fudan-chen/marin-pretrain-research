"""Original normalize/quota methods in actual Python runtimes plus isolated summation controls.
No dataset reads or actual Hero quota attribution.
"""
import ast,builtins,hashlib,json,math,pathlib,platform,sys,types,warnings
import numpy as np
R=pathlib.Path(__file__).resolve().parents[1];P=R/'sources/integer_exposure_2026_10_07/mixture.py';S=R/'sources/phase_budget_2026_10_07/harrier.json';W=R/'sources/wandb/hero-fa4sm100-nomask-step146k_meta.json';M=R/'sources/live_2026_10_07/meta.json';ns={'np':np,'warnings':warnings}
cl=next(n for n in ast.parse(P.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='MixtureDataset');nodes=[n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name in ['_normalize_weights','_compute_expected_counts_per_block']]
for n in nodes:n.decorator_list=[]
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])),str(P),'exec'),ns)
def sequential(values):
 total=0
 for v in values:total+=v
 return total
def main():
 raw=json.loads(S.read_text());keys=list(raw['available_tokens']);sources={'spec':[p['weights'] for p in raw['phases']],'historical_wandb':[p[1] for p in json.loads(W.read_text())['config']['data']['value']['train_weights']],'oct7_wandb':[p[1] for p in json.loads(M.read_text())['config']['data']['value']['train_weights']]};dummy=types.SimpleNamespace(dataset_index=keys,datasets=dict.fromkeys(keys));checks=[];cases=[]
 def ck(n,c):assert c,n;checks.append(n)
 for source,phases in sources.items():
  for mode,fn in [('native',builtins.sum),('sequential_control',sequential),('fsum_control',math.fsum)]:
   ns['sum']=fn
   for i,w in enumerate(phases):
    nw=ns['_normalize_weights'](w)
    with warnings.catch_warnings(record=True) as ws:
     warnings.simplefilter('always');q=ns['_compute_expected_counts_per_block'](dummy,nw,49152)
    ref=[int(nw.get(k,0)*49152) for k in keys];winner=max(range(len(keys)),key=lambda j:ref[j]);ref[winner]+=49152-builtins.sum(ref)
    ck(source+' '+mode+' '+str(i)+' block conserved',int(q.sum())==49152)
    ck(source+' '+mode+' '+str(i)+' independent integer reference matches',q.tolist()==ref)
    total=fn(w.values());cases.append({'source':source,'summation':mode,'phase_index':i,'raw_sum':total,'raw_sum_hex':float(total).hex(),'quotas':q.tolist(),'warning_count':len(ws),'c27q0_scaled_value':nw.get('c27q0',0)*49152,'c30q4_count':int(q[keys.index('c30q4')]),'positive_zero_count':builtins.sum(nw.get(k,0)>0 and int(n)==0 for k,n in zip(keys,q))})
 o={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'runtime':{'python':platform.python_version(),'numpy':np.__version__},'cases':cases,'dataset_keys':keys,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [P,S,W,M]},'declared_oct7_wandb_python':json.loads(M.read_text())['config']['_wandb']['value']['python_version'],'actual_Hero_quota_replay':None,'actual_Hero_python_binary':None,'actual_token_exposure':None,'controls':'Sequential/fsum substituted only for global sum dependency; original helper bodies unchanged. Actual native runs use builtins.sum.'}
 pathlib.Path(sys.argv[1]).write_text(json.dumps(o,indent=2)+'\n');print(o['runtime'],len(checks))
if __name__=='__main__':main()
