"""Original callback runner, stage hook, loader methods and real CPU mixture.
Identity child data and host loader adapters; no full training, tokens or measured loss.
"""
import abc,ast,asyncio,collections,dataclasses,enum,hashlib,inspect,json,pathlib,types,typing
from abc import ABC
from typing import Any,Callable,Generic,Sequence,TypeVar
import jax,numpy as np
import probe_loader_resume as loader
import probe_mixture_identity_cpu as mix
R=loader.ROOT;T=R/'sources/scale_2026_10_05/train_hero_ep.py';C=R/'sources/state_2026_10_05/callback_core.py';A=R/'sources/state_2026_10_05/state_adapter.py'
ns=dict(globals(),dataclass=dataclasses.dataclass,field=dataclasses.field,StrEnum=enum.StrEnum,S=TypeVar('S'))
def extract(p,names):
 nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names];assert len(nodes)==len(names)
 mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[]);exec(compile(ast.fix_missing_locations(mod),str(p),'exec'),ns)
extract(C,['ProgressEvent','_ignore_progress_event','StepInfo','Callback','LambdaCallback'])
extract(A,['CallbackStateView','_Hook','StateCallbackRunner'])
extract(T,['_make_mixture_stage_callback'])
extract(mix.P,['rescale_mixture_schedule_for_batch_schedule'])
class Harness(loader.Harness):
 def __init__(self,bs,step,stages):
  super().__init__(bs,step);self.dl.data_store=mix.Mix({n:mix.Identity(n,10000) for n in ['A','B']},stages,4,key=7)
 def _batchify_local_data(self,b):return {'step':b.index,'offset':b.global_data_offset,'identities':[b.data_by_local_index[i] for i in sorted(b.data_by_local_index)]}
def run_case(bs,step,stages):
 h=Harness(bs,step,stages);batch=asyncio.run(loader.first(h));logs=[];seen=[]
 ns['levanter']=types.SimpleNamespace(tracker=types.SimpleNamespace(log=lambda values,step:logs.append({'step':step,'values':values})))
 hook=ns['_make_mixture_stage_callback'](h.dl.data_store,bs)
 runner=ns['StateCallbackRunner'](step_getter=lambda s:s.step,model_getter=lambda s:None,eval_model_getter=lambda s:None,opt_state_getter=lambda s:None)
 runner.add_hook(hook,every=1);runner.add_hook(lambda info:seen.append({'step':info.step,'next_step':info.next_step}),every=1)
 runner.run(types.SimpleNamespace(step=jax.numpy.array(step+1)),loss=0.,step_duration=0.)
 return {'batch':batch,'domain_counts':dict(collections.Counter(x.split(':')[0] for x in batch['identities'])),'callback_step_info':seen[0],'stage_log':logs[0]}
def main():
 checks=[]
 def check(n,c):assert c,n;checks.append(n)
 old=loader.Schedule([loader.Step(0,4),loader.Step(3,8)]);new=loader.Schedule(8);configured=[(0,{'A':1.}),(21,{'B':1.})];oldstages=ns['rescale_mixture_schedule_for_batch_schedule'](configured,old);newstages=ns['rescale_mixture_schedule_for_batch_schedule'](configured,new)
 rows={'normal_before':run_case(old,20,oldstages),'normal_after':run_case(old,21,oldstages),'frozen_stages_new_loader_crossing':run_case(new,19,oldstages),'reconverted_new_schedule_before':run_case(new,20,newstages),'reconverted_new_schedule_after':run_case(new,21,newstages)}
 check('Same-schedule conversion puts switch on batch boundary',oldstages[1][0]==156 and rows['normal_before']['domain_counts']=={'A':8} and rows['normal_after']['domain_counts']=={'B':8})
 check('Original StepInfo callback indexes completed batch not next batch',all(x['callback_step_info']['step']==x['batch']['step'] and x['callback_step_info']['next_step']==x['batch']['step']+1 for x in rows.values()))
 cross=rows['frozen_stages_new_loader_crossing'];check('Frozen old stage plus changed loader creates mixed crossing batch',cross['batch']['offset']==152 and cross['domain_counts']=={'A':4,'B':4})
 check('Original stage hook reports start domain for mixed crossing batch',cross['stage_log']['values']['mixture/stage']==0 and cross['stage_log']['values']['mixture/weight/A']==1 and 'mixture/weight/B' not in cross['stage_log']['values'])
 check('Reconversion aligns changed schedule without crossing',newstages[1][0]==168 and rows['reconverted_new_schedule_before']['domain_counts']=={'A':8} and rows['reconverted_new_schedule_after']['domain_counts']=={'B':8})
 check('Normal boundary callback stage label matches all-domain batches',rows['normal_before']['stage_log']['values']['mixture/stage']==0 and rows['normal_after']['stage_log']['values']['mixture/stage']==1)
 o={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'runtime':{'jax':jax.__version__,'numpy':np.__version__},'configured_switch_step':21,'converted_sequence_boundaries':{'old':156,'new':168},'observations':rows,'crossing_fixture':'deliberately frozen old sequence stages combined with changed loader schedule; not normal builder behavior','synthetic_callback_loss':0.,'actual_Hero_boundary_crossing':None,'actual_Hero_loss_change':None,'actual_full_training_loop':None,'explicit_substitutions':['identity child datasets','host loader layout/batchify/watchdog adapters','recording tracker','synthetic state exposing completed step','Generic type variable omits unused TrainerState bound'], 'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [T,C,A,loader.LP,loader.SP,mix.P]}}
 (R/'analysis/mixture_boundary_logging.json').write_text(json.dumps(o,indent=2)+'\n');print('Mixture boundary/logging checks:',len(checks))
if __name__=='__main__':main()
