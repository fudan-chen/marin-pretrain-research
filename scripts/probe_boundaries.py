# -*- coding: utf-8 -*-
"""Execute original mask and bounds helper bodies with NumPy and associative-scan substitutes."""
import ast,dataclasses,hashlib,json,pathlib,types
import numpy as np
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/boundaries_2026_10_05';checks=[]
def check(n,v):
 if not v:raise AssertionError(n)
 checks.append(n)
class A(np.ndarray):
 def __new__(cls,x):return np.asarray(x).view(cls)
 @property
 def at(self):return At(self)
class At:
 def __init__(self,a):self.a=a
 def __getitem__(self,k):self.key=k;return self
 def set(self,v):r=self.a.copy();r[self.key]=v;return r
jnp=types.SimpleNamespace(**{n:getattr(np,n) for n in dir(np) if not n.startswith('__')})
jnp.asarray=lambda x,*a,**k:A(np.asarray(x,*a,**k))
@dataclasses.dataclass(frozen=True)
class Mask:
 is_causal:bool=True
 segment_ids:object=None
 sliding_window:object=None
 fa4_bounds:object=None
 @classmethod
 def causal(cls,sliding_window=None):return cls(sliding_window=sliding_window)
 def with_segment_ids(self,x):return dataclasses.replace(self,segment_ids=(x,x))
@dataclasses.dataclass(frozen=True)
class Example:
 tokens:object
 loss_weight:object
 attn_mask:object
ns={'jnp':jnp,'np':np,'AttentionMask':Mask,'GrugAttentionMask':Mask,'GrugLmExample':Example}
extracted=[]
def extract(file,names,cls=None):
 tree=ast.parse(file.read_text());body=tree.body if cls is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls).body
 funcs=[n for n in body if isinstance(n,ast.FunctionDef) and n.name in names];assert len(funcs)==len(names)
 for n in funcs:n.decorator_list=[];extracted.append({'source':str(file.relative_to(R)),'function':n.name,'first_line':n.lineno,'last_line':n.end_lineno})
 mod=ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+funcs,type_ignores=[]));exec(compile(mod,str(file),'exec'),ns)
extract(D/'examples.py',['causal','causal_loss_mask','from_prompt_and_completion'],'GrugLmExample')
Example.causal=staticmethod(ns['causal']);Example.causal_loss_mask=staticmethod(ns['causal_loss_mask']);Example.from_prompt_and_completion=staticmethod(ns['from_prompt_and_completion'])
extract(D/'attention_core.py',['materialize_mask'],'AttentionMask');Mask.materialize_mask=ns['materialize_mask']
extract(D/'attention_core.py',['token_validity_from_attention_mask'])
def scan(op,values,axis=0,reverse=False):
 values=np.asarray(values);v=np.flip(values,axis=axis) if reverse else values
 out=op.accumulate(v,axis=axis)
 return A(np.flip(out,axis=axis) if reverse else out)
ns.update(jax=types.SimpleNamespace(lax=types.SimpleNamespace(associative_scan=scan)),_replicate_metadata=lambda x:x,_replicate_sequence_axis=lambda x:A(x))
extract(D/'fa4_cute.py',['_batched_segment_ids','_segment_starts','_packed_segment_start_positions','_packed_segment_causal_lower_bounds'])
rows=[]
for name,tokens,segs,eos,block,window in [('eos_separated',[11,99,22,23],None,99,True,None),('cross_attention_enabled',[11,99,22,23],None,99,False,None),('right_padding',[11,99,22,23,0,0],[0,0,1,1,-1,-1],None,True,None),('left_padding',[0,0,11,12,22,23,0],[-1,-1,0,0,1,1,-1],None,True,None),('window_two',[11,12,13,14],[0,0,0,0],None,True,2),('reused_segment_id',[11,12,13,14],[0,0,1,0],None,True,None)]:
 e=Example.causal(A(tokens),eos_id=eos,segment_ids=None if segs is None else A(segs),block_cross_document_attention=block,sliding_window=window)
 dense=e.attn_mask.materialize_mask(len(tokens),len(tokens));valid=ns['token_validity_from_attention_mask'](e.attn_mask,batch_size=1,sequence_length=len(tokens))
 row={'case':name,'tokens':tokens,'segments':None if e.attn_mask.segment_ids is None else e.attn_mask.segment_ids[0].tolist(),'loss_weights':e.loss_weight.tolist(),'dense_attention':dense.astype(int).tolist(),'token_valid':valid[0].tolist()}
 if e.attn_mask.segment_ids is not None:
  bounds,v=ns['_packed_segment_causal_lower_bounds'](A(row['segments']),batch_size=1,seq_len=len(tokens),sliding_window=window)
  q=np.arange(len(tokens))[:,None];k=np.arange(len(tokens))[None,:];compressed=(k>=bounds[0,:,None])&(k<=q)&v[0,:,None]&v[0,None,:]
  row.update(lower_bounds=bounds[0].tolist(),kernel_valid=v[0].tolist(),compressed_attention=compressed.astype(int).tolist())
  # Compare only real-token rows/keys: dense equality treats -1 padding as a segment.
  projected=dense&valid[0,:,None]&valid[0,None,:]
  row['real_token_mask_mismatches']=int(np.count_nonzero(projected!=compressed))
 rows.append(row)
a,b,c,d,w,reused=rows
check('EOS attention separation starts after EOS',a['segments']==[0,0,1,1] and a['dense_attention'][2]==[0,0,1,0])
check('Cross-document target still scored at EOS source position',a['loss_weights']==[1,1,1,0])
check('Disabling cross attention changes visibility but not target weights',b['dense_attention'][2]==[1,1,1,0] and b['loss_weights']==a['loss_weights'])
check('Right-padding successor is removed from target loss',c['loss_weights']==[1,1,1,0,0,0])
check('Router validity differs from next-token loss mask',c['token_valid']==[True,True,True,True,False,False])
check('Left-padding bound carries next valid query bound',d['lower_bounds']==[2,2,2,2,4,4,7])
check('Leading padding remains invalid despite carried lower bound',d['kernel_valid']==[False,False,True,True,True,True,False])
check('Loss constructor alone may score pad-to-first-real transition',d['loss_weights']==[0,1,1,1,1,0,0])
check('Contiguous segments match dense mask on real tokens',all(r['real_token_mask_mismatches']==0 for r in [c,d,w]))
check('Reused noncontiguous ID gives dense/compressed disagreement',reused['real_token_mask_mismatches']==2)
check('Window W includes self and W-1 predecessors',w['dense_attention'][3]==[0,0,1,1])
prompt=Example.from_prompt_and_completion(A([1,2,3,4,5]),prompt_length=3)
check('Prompt length three scores completion starting at source index two',prompt.loss_weight.tolist()==[0,0,1,1,0])
check('Dense all-false query rows imply invalid token',ns['token_validity_from_attention_mask'](A([[False,False],[True,False]]),batch_size=1,sequence_length=2).tolist()==[[False,True]])
check('Additive dense mask does not establish query validity',ns['token_validity_from_attention_mask'](A([[0.,-1e9],[-1e9,-1e9]]),batch_size=1,sequence_length=2).tolist()==[[True,True]])
for bad in [0,-1]:
 try:Mask(sliding_window=bad).materialize_mask(3,3)
 except ValueError:check('Nonpositive window rejects '+str(bad),True)
 else:raise AssertionError('Invalid window accepted')
configs=[]
for p in sorted((R/'sources/scale_2026_10_05').glob('config_*.json')):
 raw=json.loads(p.read_text())['data']['project']['run'];c0=json.loads(raw['config'])['data']['value'];configs.append({'run':raw['name'],'block_cross_document_attention':c0['block_cross_document_attention'],'source':str(p.relative_to(R))})
check('Six recorded ladder configs enable document separation',len(configs)==6 and all(r['block_cross_document_attention'] for r in configs))
result={'status':'local_original_helper_probe_with_substitutes','checks_passed':len(checks),'checks':checks,'cases':rows,'prompt_case':{'prompt_length':3,'loss_weights':prompt.loss_weight.tolist()},'ladder_configs':configs,'extracted':extracted,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(D.glob('*.py'))},'substitutions':['NumPy ndarray and functional .at.set adapter','associative_scan emulated with NumPy ufunc.accumulate','metadata replication is identity','lightweight Example/Mask classes instead of JAX/Equinox objects'],'historical_execution_binding':None,'actual_gpu_result':None,'actual_training_input_segments':None,'limitations':['No attention kernel, backward, batching or real token cache executed','Left-padding loss example tests constructor alone; actual upstream loss weights may already mask padding','Reused-ID counterexample does not prove such input exists in Hero','FA4 mask comparison explicitly excludes padding query/key rows']}
(R/'analysis/boundary_probe.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print('Boundary source-helper checks:',len(checks))
