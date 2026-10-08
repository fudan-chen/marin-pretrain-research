"""Original pinned scheduler CPU configuration boundaries and local candidate contracts.
No config parser, full optimizer, GPU, or historical Hero incident execution.
"""
import hashlib,json,math,pathlib,types
import jax,numpy as np,optax
R=pathlib.Path(__file__).resolve().parents[1]
prior=R/'scripts/probe_optimizer_schedule_resume_cpu.py'
# Definitions only: never run V133 cases or rewrite V133 results.
ns={'__file__':str(prior)}
exec(compile(prior.read_text().split('def flat(t):',1)[0],str(prior),'exec'),ns)
Config=ns['Config'];original_convert=ns['ns']['_convert_frac_or_steps']
checks=[];cases={}
def check(name,condition):
 if not condition:raise RuntimeError(name)
 checks.append({'name':name,'passed':True})
def run(name,n=100,**kw):
 c=Config()
 for k,v in kw.items():setattr(c,k,v)
 row={'N':n,'overrides':kw}
 try:
  row['cycle_points']=c._get_cycle_minima(n)
  row['warmup_each_cycle']=[original_convert(c.warmup if i==0 else c.rewarmup,b-a) for i,(a,b) in enumerate(zip(row['cycle_points'][:-1],row['cycle_points'][1:]))]
  f=c.lr_scheduler(n,override_lr=.01)
  points=sorted(set([0,1,2,5,10,n//2,max(n-1,0),n,n+1]))
  row.update(accepted=True,values=[{'count':p,'lr':float(f(p))} for p in points])
 except Exception as exc:row.update(accepted=False,error_type=type(exc).__name__,error=str(exc))
 cases[name]=row;return row
for name,n,kw in [
 ('baseline',100,{}),('warmup_one_int',100,{'warmup':1}),('warmup_one_float',100,{'warmup':1.}),
 ('warmup_true',100,{'warmup':True}),('warmup_two_steps',100,{'warmup':2}),
 ('tiny_warmup_small',100,{'warmup':.001}),('tiny_warmup_large',1000,{'warmup':.001}),
 ('tiny_cycle_small',50,{'cycle_length':.01}),('tiny_cycle_large',200,{'cycle_length':.01}),
 ('cycles_zero',100,{'cycles':0}),('cycles_negative',100,{'cycles':-2}),
 ('cycle_endpoints_unsorted',100,{'cycles':[80,20]}),('cycle_endpoint_beyond_N',100,{'cycles':[120]}),
 ('cycle_lengths_empty',100,{'cycle_length':[]}),('N_zero',0,{}),('negative_min_lr',100,{'min_lr_ratio':-.1})]:
 run(name,n,**kw)
def value(case,count):return next(x['lr'] for x in cases[case]['values'] if x['count']==count)
check('numeric 1 integer and float are full-cycle fractions',cases['warmup_one_int']['warmup_each_cycle']==cases['warmup_one_float']['warmup_each_cycle']==[100] and cases['warmup_one_int']['values']==cases['warmup_one_float']['values'])
check('boolean True also passes numeric fraction conversion',cases['warmup_true']['values']==cases['warmup_one_int']['values'])
check('warmup2 interpreted as absolute two steps',cases['warmup_two_steps']['warmup_each_cycle']==[2] and abs(value('warmup_two_steps',2)-.01)<1e-8)
check('tiny warmup floor changes onset across training scales',cases['tiny_warmup_small']['warmup_each_cycle']==[0] and cases['tiny_warmup_large']['warmup_each_cycle']==[1] and value('tiny_warmup_small',0)>.009 and value('tiny_warmup_large',0)==0)
check('tiny cycle truncation triggers zero division only in small control',not cases['tiny_cycle_small']['accepted'] and cases['tiny_cycle_small']['error_type']=='ZeroDivisionError' and cases['tiny_cycle_large']['accepted'])
check('zero and negative cycle counts silently behave as one cycle',cases['cycles_zero']['cycle_points']==cases['cycles_negative']['cycle_points']==[0,100] and cases['cycles_zero']['values']==cases['cycles_negative']['values']==cases['baseline']['values'])
check('unsorted endpoints accepted with nonmonotone boundaries',cases['cycle_endpoints_unsorted']['accepted'] and cases['cycle_endpoints_unsorted']['cycle_points']==[0,80,20,100])
check('endpoint beyond horizon accepted with backwards final interval',cases['cycle_endpoint_beyond_N']['accepted'] and cases['cycle_endpoint_beyond_N']['cycle_points']==[0,120,100])
check('empty cycle-length list raises index error',not cases['cycle_lengths_empty']['accepted'] and cases['cycle_lengths_empty']['error_type']=='IndexError')
check('zero horizon fails late while negative minimum ratio produces negative LR',not cases['N_zero']['accepted'] and cases['N_zero']['error_type']=='IndexError' and cases['negative_min_lr']['accepted'] and value('negative_min_lr',100)<0)

# Local candidate only: explicit unit representation, and preflight geometry validation.
# Raw numeric legacy values retain original semantics, including numeric 1.
def unit_convert(x,n):
 if isinstance(x,dict):
  if set(x)!={'unit','value'}:raise ValueError('quantity schema')
  unit,v=x['unit'],x['value']
  if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v):raise ValueError('finite numeric quantity required')
  if unit=='steps':
   if v<0 or int(v)!=v:raise ValueError('nonnegative integer steps required')
   return int(v)
  if unit=='fraction':
   if not 0<=v<=1:raise ValueError('fraction outside [0,1]')
   return int(v*n)
  raise ValueError('unknown quantity unit')
 if isinstance(x,bool):raise ValueError('boolean quantity rejected')
 return original_convert(x,n)

def preflight(c,n):
 if isinstance(n,bool) or not isinstance(n,int) or n<=0:raise ValueError('positive integer horizon required')
 if not math.isfinite(c.min_lr_ratio) or not 0<=c.min_lr_ratio<=1:raise ValueError('minimum ratio outside [0,1]')
 if isinstance(c.cycles,int) and (isinstance(c.cycles,bool) or c.cycles<=0):raise ValueError('positive cycle count required')
 if c.cooldown is not None:raise ValueError('candidate preflight supports cooldown=None only')
 if c.cycle_length is not None:
  if isinstance(c.cycle_length,list):
   if not c.cycle_length or any(isinstance(v,bool) or not isinstance(v,int) or v<=0 for v in c.cycle_length):raise ValueError('positive integer cycle lengths required')
  else:
   if unit_convert(c.cycle_length,n)<1:raise ValueError('cycle length rounds below one step')
 points=c._get_cycle_minima(n)
 if any(not isinstance(v,(int,np.integer)) or isinstance(v,bool) for v in points) or any(b<=a for a,b in zip(points[:-1],points[1:])):raise ValueError('strict increasing integer cycle endpoints required')
 return points
candidate_rejections={}
for name in ['tiny_cycle_small','cycles_zero','cycles_negative','cycle_endpoints_unsorted','cycle_endpoint_beyond_N','cycle_lengths_empty','N_zero','negative_min_lr']:
 row=cases[name];c=Config()
 for k,v in row['overrides'].items():setattr(c,k,v)
 try:preflight(c,row['N']);candidate_rejections[name]=None
 except Exception as exc:candidate_rejections[name]=type(exc).__name__
check('candidate preflight stops all eight invalid geometry or domain controls',all(v is not None for v in candidate_rejections.values()))
check('candidate preflight rejects all eight selected invalid inputs with explicit ValueError',all(v=='ValueError' for v in candidate_rejections.values()))
check('explicit units distinguish one step from full fraction and preserve legacy numeric1',unit_convert({'unit':'steps','value':1},100)==1 and unit_convert({'unit':'fraction','value':1},100)==unit_convert(1,100)==100)
ns['ns']['_convert_frac_or_steps']=unit_convert
c=Config();c.warmup={'unit':'steps','value':1};explicit=c.lr_scheduler(100,override_lr=.01);explicit_curve=[{'count':p,'lr':float(explicit(p))} for p in range(101)]
check('explicit one-step candidate runs original scheduler and reaches peak at count1',explicit_curve[0]['lr']==0 and abs(explicit_curve[1]['lr']-.01)<1e-8)
c=Config();preflight(c,100);candidate=c.lr_scheduler(100,override_lr=.01)
check('candidate unit converter and preflight preserve baseline schedule values',all(float(candidate(x['count']))==x['lr'] for x in cases['baseline']['values']))
ns['ns']['_convert_frac_or_steps']=original_convert
curves={}
for name,kw in [('fraction_1',{'warmup':1}),('steps_2',{'warmup':2}),('baseline',{})]:
 c=Config()
 for k,v in kw.items():setattr(c,k,v)
 f=c.lr_scheduler(100,override_lr=.01);curves[name]=[{'count':p,'lr':float(f(p))} for p in range(101)]
curves['explicit_steps_1_candidate']=explicit_curve
c=Config();c.cycles=[120];f=c.lr_scheduler(100,override_lr=.01)
curves['beyond_N_endpoint']=[{'count':p,'lr':float(f(p))} for p in range(131)]
archived_path=R/'sources/live_2026_10_07/meta.json'
record=json.loads(archived_path.read_text());recipe=record['config'];recipe=json.loads(recipe) if isinstance(recipe,str) else recipe
opt=recipe['optimizer'].get('value',recipe['optimizer']);trainer=recipe['trainer'].get('value',recipe['trainer']);trainer=trainer.get('trainer',trainer)
archived={'record_name':record['name'],'N':trainer['num_train_steps'],'fields':{k:opt.get(k) for k in ['warmup','decay','cooldown','cycles','cycle_length','lr_schedule','min_lr_ratio']},'execution_binding':'declared archived recipe only; not historical runtime verification'}
archived['warmup_steps_by_pinned_converter']=original_convert(opt['warmup'],archived['N'])
check('archived October7 declared recipe avoids selected unit and cycle counterexamples',archived['N']==390251 and archived['warmup_steps_by_pinned_converter']==3902 and archived['fields']['cycles'] is None and archived['fields']['cycle_length'] is None and archived['fields']['warmup']==.01 and archived['fields']['lr_schedule']=='linear')
out={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'cases':cases,'archived_recipe_control':archived,'candidate_preflight_rejections':candidate_rejections,'candidate_preflight_covers_all_config_forms':False,'candidate_upstream_applied':False,'curves':curves,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [prior,archived_path,R/'sources/optimizer_schedule_2026_10_08/config.py']},'probe_script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'runtime':{'jax':jax.__version__,'optax':optax.__version__},'actual_Hero_ambiguous_schedule_config':None,'actual_config_parser_acceptance':None,'actual_training_loss_effect':None,'actual_GPU_execution':None}
(R/'analysis/schedule_config_cpu.json').write_text(json.dumps(out,indent=2)+'\n');print('Schedule config controls:',len(checks))
