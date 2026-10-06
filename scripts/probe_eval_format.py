"""Original TokenSeqDataset methods on a fake flat CPU cache; no model or real cache."""
import ast,asyncio,json,pathlib,hashlib
import numpy as np
R=pathlib.Path(__file__).resolve().parents[1];P=R/'sources/deepening_2026_10_04/datasets_production.py';env={'np':np};t=ast.parse(P.read_text());cl=next(x for x in t.body if isinstance(x,ast.ClassDef) and x.name=='TokenSeqDataset');cl.bases=[];cl.decorator_list=[]
m=ast.Module(body=ast.parse('from __future__ import annotations').body+[cl],type_ignores=[]);exec(compile(ast.fix_missing_locations(m),str(P),'exec'),env)
class Cache:
 def __init__(self,docs):self.flat=np.concatenate([np.asarray(x,dtype=np.int32) for x in docs]);self.requests=[]
 async def async_flat_field_length(self,key):return len(self.flat)
 async def get_flat_field_batch(self,key,offsets,length):
  self.requests.append({'key':key,'offsets':offsets.tolist(),'length':length,'offset_dtype':str(offsets.dtype)})
  return [self.flat[i:i+length] for i in offsets]
checks=[]
def ck(n,v):assert v,n;checks.append(n)
async def run(docs,L):
 c=Cache(docs);d=env['TokenSeqDataset'](c,L);n=await d.async_len();b=await d.get_batch(list(range(n)));return c,d,[x['input_ids'].tolist() for x in b]
a=[10,11,0];b=[20,21,22,23,24,0];c,d,ab=asyncio.run(run([a,b],4));rc,rd,ba=asyncio.run(run([b,a],4))
ck('Original length floors nine flat tokens into two length-four examples',asyncio.run(d.async_len())==2)
ck('Original slices cross document boundaries',ab==[[10,11,0,20],[21,22,23,24]])
ck('Short document prefix in next window loses its preceding token context',ab[0][-1]==20 and ab[1][0]==21)
ck('Original tail remainder is omitted from logical examples',sum(ab,[])==c.flat[:8].tolist() and c.flat[8:].tolist()==[0])
ck('Reversing same documents changes windows and the document whose tail is omitted',ba==[[20,21,22,23],[24,0,10,11]] and rc.flat[8:].tolist()==[0] and ab!=ba)
ck('Original offsets request int64 continuous windows',c.requests==[{'key':'input_ids','offsets':[0,4],'length':4,'offset_dtype':'int64'}])
sc,sd,sb=asyncio.run(run([[1,2,0]],4));ck('Fewer than one full window produces no logical examples',asyncio.run(sd.async_len())==0 and sb==[])
xc,xd,xb=asyncio.run(run([[1,2,3,0],[4,5,6,0]],4));ck('Exact multiple retains all cached tokens',sum(xb,[])==xc.flat.tolist())
ck('Empty request produces no cache fetch',asyncio.run(d.get_batch([]))==[] and len(c.requests)==1)
try:asyncio.run(d.get_batch([2]));bad=False
except ValueError:bad=True
ck('Request beyond floored length is rejected',bad)
config=json.loads((R/'sources/live_2026_10_06/meta.json').read_text())['config']['data']['value']
ck('Declared target sources have text format and null pack',all(config['components']['paloma/'+n]['pack'] is None and config['components']['paloma/'+n]['format']=={'text_key':'text'} for n in ['ptb-llama3','twitterAAE_HELM_fixed-llama3']))
f=next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name=='_effective_pack');ck('Archived text default resolves false',any(isinstance(x,ast.If) and ast.unparse(x.test)=='isinstance(fmt, TextLmDatasetFormat)' and any(isinstance(y,ast.Return) and isinstance(y.value,ast.Constant) and y.value.value is False for y in x.body) for x in f.body))
ck('Declared attention blocking is true but does not change stream slicing',config['block_cross_document_attention'] is True)
paths=[P,R/'sources/live_2026_10_06/meta.json'];out={'checks_passed':len(checks),'checks':checks,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'synthetic':{'documents':[a,b],'sequence_length':4,'forward_windows':ab,'reverse_windows':ba,'forward_omitted_tail':c.flat[8:].tolist(),'reverse_omitted_tail':rc.flat[8:].tolist()},'actual_Hero_cache_contents':None,'actual_Hero_tail_count':None,'actual_GPU_loss_difference':None,'actual_PTB_root_cause':None}
(R/'analysis/eval_format_probe.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('Original stream / declared format checks:',len(checks))
