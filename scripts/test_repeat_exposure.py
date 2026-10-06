"""Synthetic recurrence and original remap/permutation helper tests; no Hero inventory."""
import ast,asyncio,collections,hashlib,json,pathlib,subprocess,sys,tempfile,types
from audit_repeat_exposure import audit
R=pathlib.Path(__file__).resolve().parents[1];checks=[]
def ck(n,v):assert v,n;checks.append(n)
def run(n,d,c):return audit({'index_policy':'fixed_order_modulo_restart','components':[{'name':'A','inventory_sequences':n,'draws':d,'start_sequence_index':c}]})['components'][0]
for n,d,c in [(3,0,0),(3,2,0),(3,8,0),(3,2,3),(5,8,4),(1,4,9)]:
 x=run(n,d,c);idx=[i%n for i in range(c,c+d)];old=set(i%n for i in range(c));counts=collections.Counter(idx);actual=[counts[i] for i in range(n)]
 ck('Independent enumeration n=%d draws=%d cursor=%d'%(n,d,c),x['window_distinct_sequence_indices']==len(set(idx)) and x['new_sequence_indices_since_lifetime_start']==len(set(idx)-old) and x['min_draws_per_inventory_index']==min(actual) and x['max_draws_per_inventory_index']==max(actual))
 extra=set(i for z in x['extra_draw_index_intervals'] for i in range(z['start'],z['stop']));ck('Extra intervals exactly represent count excess '+str((n,d,c)),extra=={i for i,v in enumerate(actual) if v>x['full_cycles_in_window']})
ck('Unknown inventory stays unknown',run(None,20,0)['window_distinct_sequence_indices'] is None)
ck('Unknown cursor does not invent lifetime coverage',run(5,8,None)['window_distinct_sequence_indices']==5 and run(5,8,None)['new_sequence_indices_since_lifetime_start'] is None)
for n,d,c in [(0,1,0),(3,-1,0),(3,1,True)]:
 try:run(n,d,c);rejected=False
 except ValueError:rejected=True
 ck('Reject invalid integer declaration '+str((n,d,c)),rejected)
# Original bodies with finite fake datasets and deterministic substitute permutation.
M=R/'sources/deepening_2026_10_04/mixture_production.py';D=R/'sources/deepening_2026_10_04/dataset_production.py';env={}
def install(path,cls,names):
 t=ast.parse(path.read_text());c=next(x for x in t.body if isinstance(x,ast.ClassDef) and x.name==cls);fs=[x for x in c.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name in names]
 for f in fs:f.decorator_list=[]
 mod=ast.Module(body=ast.parse('from __future__ import annotations').body+fs,type_ignores=[]);exec(compile(ast.fix_missing_locations(mod),str(path),'exec'),env)
install(M,'MixtureDataset',['_remap_indices']);env['StopStrategy']=types.SimpleNamespace(RESTART_STRATEGY='restart',ALL_STOP_STRATEGY='all_exhausted',FIRST_STOP_STRATEGY='first_exhausted')
class Fake:
 def is_finite(self):return True
 async def async_len(self):return 3
 async def get_batch(self,indices):return list(indices)
remap=asyncio.run(env['_remap_indices'](types.SimpleNamespace(stop_strategy='restart'),Fake(),list(range(8))));ck('Original restart helper remaps repeated ordinal indices',remap==[0,1,2,0,1,2,0,1])
install(D,'PermutationDataset',['_get_permutation','get_batch']);calls=[]
def make(kind,n,key):calls.append((kind,n,key));return lambda i:[2,0,1][i]
env['Permutation']=types.SimpleNamespace(make=make)
wrapper=types.SimpleNamespace(dataset=Fake(),_permutation=None,_perm_type='synthetic',key='synthetic');wrapper.async_len=wrapper.dataset.async_len
async def perm():return await env['_get_permutation'](wrapper)
wrapper._get_permutation=perm
first=asyncio.run(env['get_batch'](wrapper,remap));second=asyncio.run(env['get_batch'](wrapper,remap));ck('Original full-shuffle cache repeats the same supplied permutation',first==second==[2,0,1,2,0,1,2,0] and len(calls)==1)
meta=R/'sources/live_2026_10_06/meta.json';cfg=json.loads(meta.read_text())['config']['data']['value'];ck('Hero snapshot declares restart and block shuffle rather than full shuffle',cfg['stop_strategy']=='restart' and cfg['shuffle']=={'io_block_size':256,'perm_type':'feistel','window_blocks':512})
with tempfile.TemporaryDirectory() as tmp:
 p=pathlib.Path(tmp)/'in.json';o=pathlib.Path(tmp)/'out.json';p.write_text(json.dumps({'index_policy':'fixed_order_modulo_restart','components':[{'name':'A','inventory_sequences':3,'draws':8,'start_sequence_index':3}]}));cmd=[sys.executable,str(R/'scripts/audit_repeat_exposure.py'),str(p),str(o)];done=subprocess.run(cmd,capture_output=True);ck('Actual CLI writes a bounded projection with input digest',done.returncode==0 and len(json.loads(o.read_text())['input_sha256'])==64);before=o.read_bytes();again=subprocess.run(cmd,capture_output=True);ck('Actual CLI refuses overwrite',again.returncode!=0 and before==o.read_bytes())
(R/'analysis/repeat_exposure_validation.json').write_text(json.dumps({'checks_passed':len(checks),'checks':checks,'synthetic_after_one_cycle':run(3,2,3),'synthetic_order':first,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [M,D,meta]},'actual_Hero_inventory':None,'actual_Hero_cursor':None,'actual_JAX_permutation':None,'actual_block_shuffle_replay':None,'actual_training_benefit':None},ensure_ascii=False,indent=2)+'\n');print('Synthetic repeat exposure checks:',len(checks))
