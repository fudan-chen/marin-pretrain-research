"""Execute original pure calibration/BME helpers with a synthetic scorer.
No deployed model, label parquet, tokenizer or historical corpus is evaluated.
"""
import ast,hashlib,json,pathlib,types
import numpy as np
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/quality_2026_10_05';env=dict(np=np,CHUNK_CHARS=2000,BUCKET_EDGES=(.2,.4,.6,.8),YK=[0,.2,.4,.6,.8,1]);checks=[]
def load(path,names):
 t=ast.parse(path.read_text());body=[]
 for x in t.body:
  if isinstance(x,ast.FunctionDef) and x.name in names:
   x.returns=None
   for a in x.args.args+x.args.kwonlyargs:a.annotation=None
   body.append(x)
 exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(path),'exec'),env)
def check(name,c):assert c,name;checks.append(name)
load(D/'calibrate.py',{'fit_cutpoints','calibration_knots'});load(D/'scorer.py',{'score_bme'})
levels=np.arange(1,6);good=np.array([.1,.3,.5,.7,.9]);knots=env['calibration_knots'](good,levels);check('Separated level medians yield strictly increasing knots',np.all(np.diff(knots['xk'])>0));cal=np.interp(good,knots['xk'],knots['yk']);check('Separated toy oracle levels map to five buckets',np.digitize(cal,env['BUCKET_EDGES']).tolist()==list(range(5)))
try:env['fit_cutpoints'](good[:-1],levels[:-1]);missing=False
except ValueError:missing=True
check('Missing oracle level is rejected',missing)
flat=env['calibration_knots'](np.full(5,.5),levels);check('Present levels do not ensure strictly increasing knots',not np.all(np.diff(flat['xk'])>0));flatcal=np.interp([.5],flat['xk'],flat['yk']);flatcase=dict(xk=flat['xk'],yk=flat['yk'],observed_numpy_result=flatcal.tolist(),valid_interp_contract=False)
# Character indices are encoded as distinct Unicode scalars to recover true slices.
class Recorder:
 def __init__(self):self.windows=[]
 def score(self,texts):self.windows=texts;return np.array([1. if '!' not in t else 0. for t in texts])
windows=[]
for n in [1999,2000,2001,3000,4000,6000,10000,20000]:
 text=''.join(chr(0x1000+i) for i in range(n));rec=Recorder();env['score_bme'](rec,[text]);spans=[(ord(t[0])-0x1000,ord(t[-1])-0x1000+1) for t in rec.windows];covered=len(set(i for a,b in spans for i in range(a,b)));windows.append(dict(chars=n,spans=spans,unique_chars=covered,coverage=covered/n,window_count=len(spans)))
check('2000 characters use one window; 2001 use three',windows[1]['window_count']==1 and windows[2]['window_count']==3);check('2001 repeats the first two windows',windows[2]['spans'][0]==windows[2]['spans'][1]);check('10000 characters cover only 6000 unique characters',windows[6]['unique_chars']==6000)
text='a'*10000;damaged=text[:3000]+'!'+text[3001:];rec=Recorder();score=env['score_bme'](rec,[text,damaged]);check('Synthetic unsampled damage is invisible to BME windows',score.tolist()==[1.,1.] and rec.windows[:3]==rec.windows[3:])
# Compact vocabulary mapping is original source; never call the real tokenizer.
load(D/'data.py',{'_build_vocab'});from collections import Counter
env.update(Counter=Counter,NUM_RESERVED=2,logger=types.SimpleNamespace(info=lambda *a:None));remap=env['_build_vocab']([[10,10,11],[10,12]],2);check('Rare raw token IDs are omitted from compact vocabulary',remap=={10:2})
edge=np.array([.2,.4,.6,.8]);check('Exact bucket edges enter the higher bucket',np.digitize(edge,edge).tolist()==[1,2,3,4])
# Monotone valid remap preserves order but can tie out-of-range values by clipping.
vals=np.array([-.5,-.1,.1,.3,.5,.7,.9,1.1,1.5]);mapped=np.interp(vals,knots['xk'],knots['yk']);check('Valid remap preserves nondecreasing order and clips endpoints',np.all(np.diff(mapped)>=0) and mapped[0]==mapped[1]==0 and mapped[-1]==mapped[-2]==1)
out=dict(checks_passed=len(checks),checks=checks,numpy_version=np.__version__,good_calibration=dict(raw=good.tolist(),levels=levels.tolist(),knots=knots,calibrated=cal.tolist(),buckets=np.digitize(cal,env['BUCKET_EDGES']).tolist()),collapsed_calibration=flatcase,bme_windows=windows,unsampled_damage=dict(char_position=3000,doc_chars=10000,scores=score.tolist(),scorer='synthetic ! detector',real_model_result=None),rare_vocab_remap=remap,source_sha256={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [D/'calibrate.py',D/'scorer.py',D/'data.py',D/'artifact.py']},actual_model=None,actual_calibration_file=None,historical_execution_sha=None)
(R/'analysis/quality_probe.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('Quality helper checks:',len(checks))
