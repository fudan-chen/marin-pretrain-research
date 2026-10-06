"""Compare declared global leaf-domain evaluation transcripts. Input hashes are
claims by the exporter, not independently verified arrays. Float64 reconstruction
of archived token-weighted BPB and nonempty-domain parent metrics; not root macro
with empty domains or root CE batch-denominator clamps. No GPU scoring or automatic root cause.
Usage: python scripts/compare_eval_replays.py first.json second.json output.json
"""
import hashlib,json,math,pathlib,re,sys
FIELDS=['checkpoint_sha256','execution_sha','parameter_view','pending_beta_policy','compute_dtype','backend','tokenizer_sha256','byte_table_sha256','input_digest_scheme']
def validate(j):
 if type(j.get('schema_version')) is not int or j.get('schema_version')!=1 or j.get('array_scope')!='global' or j.get('leaf_domains_disjoint') is not True:raise ValueError('schema 1, global sums and disjoint leaf-domain declaration required')
 for k in FIELDS:
  if not isinstance(j.get(k),str) or not j[k].strip():raise ValueError('Missing identity field '+k)
 for k in ['checkpoint_sha256','tokenizer_sha256','byte_table_sha256']:
  if not re.fullmatch('[0-9a-f]{64}',j[k]):raise ValueError('Expected lowercase sha256 '+k)
 if not re.fullmatch('[0-9a-f]{40}',j['execution_sha']):raise ValueError('Expected full execution SHA')
 if j['input_digest_scheme'] not in ('ordered_tokens_weights_masks_tags_v1','typed_global_eval_arrays_v2'):raise ValueError('unsupported digest declaration')
 rows=j.get('records');seen=set()
 if not isinstance(rows,list) or not rows:raise ValueError('records must be nonempty')
 for r in rows:
  if type(r.get('batch')) is not int or r['batch']<0 or not isinstance(r.get('domain'),str) or not r['domain']:raise ValueError('invalid batch/domain')
  key=(r['batch'],r['domain'])
  if key in seen:raise ValueError('duplicate batch/domain record')
  seen.add(key)
  if not re.fullmatch('[0-9a-f]{64}',r.get('input_sha256','')):raise ValueError('invalid input digest')
  for k in ['weighted_loss_sum','loss_weight_sum','weighted_byte_sum']:
   v=r.get(k)
   if type(v) not in (int,float) or not math.isfinite(v) or v<0:raise ValueError('invalid nonnegative finite '+k)
  if r['loss_weight_sum']==0 and (r['weighted_loss_sum']!=0 or r['weighted_byte_sum']!=0):raise ValueError('zero-weight record must have zero loss and bytes')
 return rows

def safe_sum(values):
 try:out=math.fsum(values)
 except (OverflowError,ValueError):raise ValueError('nonfinite or overflowing aggregate')
 if not math.isfinite(out):raise ValueError('nonfinite or overflowing aggregate')
 return out

def aggregate(rows):
 domains={}
 for r in rows:domains.setdefault(r['domain'],[]).append(r)
 leaves={}
 for d,rr in domains.items():
  n=safe_sum(x['weighted_loss_sum'] for x in rr);t=safe_sum(x['loss_weight_sum'] for x in rr);b=safe_sum(x['weighted_byte_sum'] for x in rr)
  leaves[d]={'weighted_loss_sum':n,'loss_weight_sum':t,'weighted_byte_sum':b,'CE':n/t if t>0 else None,'logged_token_weighted_BPB':safe_sum((x['loss_weight_sum']/t)*(x['weighted_loss_sum']/max(x['weighted_byte_sum'],1.))*math.log2(math.e) for x in rr) if t>0 else None,'byte_weighted_ratio_for_comparison':n/b*math.log2(math.e) if b>0 else None}
 for leaf in leaves.values():
  if any(v is not None and not math.isfinite(v) for v in leaf.values()):raise ValueError('nonfinite aggregate observation')
 active=[v for v in leaves.values() if v['loss_weight_sum']>0];t=safe_sum(x['loss_weight_sum'] for x in active)
 return {'leaves':leaves,'macro_CE':safe_sum(x['CE']/len(active) for x in active) if active else None,'macro_logged_BPB':safe_sum(x['logged_token_weighted_BPB']/len(active) for x in active) if active else None,'micro_CE':safe_sum(x['weighted_loss_sum'] for x in active)/t if t>0 else None,'micro_logged_BPB':safe_sum(x['logged_token_weighted_BPB']*(x['loss_weight_sum']/t) for x in active) if t>0 else None,'empty_weight_domains':[d for d,v in leaves.items() if v['loss_weight_sum']==0]}

def compare(a,b,atol=1e-6,rtol=1e-5):
 if not all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in [atol,rtol]):raise ValueError('finite nonnegative tolerances required')
 aa=validate(a);bb=validate(b);identity=[k for k in FIELDS if a[k]!=b[k]]
 def stream(r):return (r['batch'],r['domain'],r['input_sha256'],r['loss_weight_sum'],r['weighted_byte_sum'])
 stream_a=list(map(stream,aa));stream_b=list(map(stream,bb));aligned=stream_a==stream_b
 first=next((i for i,(x,y) in enumerate(zip(stream_a,stream_b)) if x!=y),min(len(aa),len(bb)) if len(aa)!=len(bb) else None)
 differences=[]
 if aligned:
  for i,(x,y) in enumerate(zip(aa,bb)):
   u=x['weighted_loss_sum'];v=y['weighted_loss_sum']
   if abs(v-u)>atol+rtol*max(abs(u),abs(v)):differences.append({'record':i,'batch':x['batch'],'domain':x['domain'],'loss_sum_delta':v-u})
 status='identity_or_input_review_required' if identity or not aligned else 'numeric_difference_under_matching_declarations' if differences else 'within_tolerance_under_matching_declarations'
 return {'scope':__doc__,'status':status,'identity_fields_differ':identity,'input_partition_and_denominator_declarations_match':aligned,'first_input_record_difference':first,'numeric_difference_records':differences,'atol':atol,'rtol':rtol,'first':aggregate(aa),'second':aggregate(bb),'input_hash_verification':'exporter declaration only; raw arrays not supplied','root_cause':None,'actual_GPU_replay':None}
if __name__=='__main__':
 if len(sys.argv)!=4:raise SystemExit(__doc__)
 p,q,o=map(pathlib.Path,sys.argv[1:]);a=p.read_bytes();b=q.read_bytes();result=compare(json.loads(a),json.loads(b));result['transcript_file_sha256']={'first':hashlib.sha256(a).hexdigest(),'second':hashlib.sha256(b).hexdigest()}
 with o.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
