"""Export already collected GLOBAL eval arrays into a comparison transcript.
No model scoring, rank gathering, checkpoint verification or historical Hero data.
Usage: python scripts/export_eval_arrays.py manifest.json output.json
"""
import hashlib,io,json,math,pathlib,sys
import numpy as np
from compare_eval_replays import validate,aggregate
SCHEME='typed_global_eval_arrays_v2'

def array_digest(arrays,context):
 """Canonical typed byte stream: sorted named arrays, JSON dtype/shape headers,
 little-endian C-order bytes; 8-byte little-endian length prefixes for each frame.
 Context is sorted-key UTF-8 JSON. Dtype/shape/order changes are meaningful.
 """
 h=hashlib.sha256()
 def frame(b):h.update(len(b).to_bytes(8,'little'));h.update(b)
 frame(json.dumps(context,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode())
 for name,a in sorted(arrays.items()):
  if a.dtype.kind not in 'bifu' or a.dtype.hasobject:raise ValueError('unsupported array type '+name)
  dtype=a.dtype.newbyteorder('<');canonical=np.ascontiguousarray(a.astype(dtype,copy=False))
  frame(json.dumps({'name':name,'dtype':dtype.str,'shape':list(a.shape)},sort_keys=True,separators=(',',':')).encode());frame(canonical.tobytes(order='C'))
 return h.hexdigest()

def export(manifest,base):
 j=dict(manifest['identity']);j.update(schema_version=1,array_scope='global',leaf_domains_disjoint=True,input_digest_scheme=SCHEME,records=[])
 if manifest.get('array_scope')!='global':raise ValueError('global arrays must be explicitly declared')
 batches=manifest.get('batches');domains=manifest.get('domains');mask_spec=manifest.get('mask_spec')
 if not isinstance(domains,list) or not domains or any(not isinstance(x,str) or not x for x in domains) or len(set(domains))!=len(domains):raise ValueError('unique ordered domain names required')
 if not isinstance(mask_spec,str) or not mask_spec.strip():raise ValueError('mask representation declaration required')
 if not isinstance(batches,list) or not batches:raise ValueError('nonempty batches required')
 sources=[];seen=set();table_hash=None
 for entry in batches:
  bn=entry['batch'];path=pathlib.Path(base)/entry['path']
  if type(bn) is not int or bn<0 or bn in seen:raise ValueError('unique nonnegative batch ids required')
  seen.add(bn);raw=path.read_bytes()
  with np.load(io.BytesIO(raw),allow_pickle=False) as f:arrays={name:f[name] for name in f.files}
  required={'input_tokens','scoring_token_ids','losses','weights','tags','bytes_per_token'}
  if not required<=arrays.keys() or any(k not in required and not k.startswith('mask_') for k in arrays):raise ValueError('required fields or mask leaves missing/unknown')
  if not any(k.startswith('mask_') for k in arrays):raise ValueError('at least one explicit mask leaf required')
  for k,a in arrays.items():
   if a.dtype.kind not in 'bifu' or a.dtype.hasobject or not np.isfinite(a).all():raise ValueError('finite numeric/boolean arrays only: '+k)
  tokens,ids,loss,w,tags,table=[arrays[k] for k in ['input_tokens','scoring_token_ids','losses','weights','tags','bytes_per_token']]
  if tokens.ndim!=2 or ids.ndim!=2 or loss.shape!=ids.shape or w.shape!=ids.shape or tokens.shape[0]!=ids.shape[0] or not all(tokens.shape):raise ValueError('rank-two batch arrays and matching scored shapes required')
  if tokens.dtype.kind not in 'iu' or ids.dtype.kind not in 'iu' or table.ndim!=1 or not len(table) or np.any(ids<0) or np.any(ids>=len(table)) or np.any(tokens<0) or np.any(tokens>=len(table)):raise ValueError('token ids outside byte table or invalid type')
  if loss.dtype.kind!='f' or w.dtype.kind!='f' or np.any(loss<0) or np.any(w<0) or np.any(table<0):raise ValueError('nonnegative losses, weights and bytes required')
  if tags.shape!=(tokens.shape[0],len(domains)) or not np.isin(tags,[0,1]).all() or np.any(tags.sum(axis=1)>1):raise ValueError('exclusive one-hot or padded zero domain tags required')
  if np.any((w.sum(axis=1)>0)&(tags.sum(axis=1)!=1)):raise ValueError('positive-weight row must belong to one domain')
  th=array_digest({'bytes_per_token':table},{'kind':'byte_table_v2'})
  if table_hash is not None and table_hash!=th:raise ValueError('byte table changed within transcript')
  table_hash=th
  inputs={k:a for k,a in arrays.items() if k!='losses' and k!='bytes_per_token'}
  batch_digest=array_digest(inputs,{'scheme':SCHEME,'domains':domains,'mask_spec':mask_spec})
  for i,d in enumerate(domains):
   digest=hashlib.sha256(json.dumps({'domain':d,'batch_input_sha256':batch_digest},sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
   selected=tags[:,i].astype(bool);ww=w[selected].astype(np.float64);ll=loss[selected].astype(np.float64);bb=table[ids[selected]].astype(np.float64)
   with np.errstate(over='raise',invalid='raise'):
    n=float(np.sum(ll*ww,dtype=np.float64));t=float(np.sum(ww,dtype=np.float64));b=float(np.sum(bb*ww,dtype=np.float64))
   if not all(map(math.isfinite,[n,t,b])):raise ValueError('nonfinite reductions')
   j['records'].append({'batch':bn,'domain':d,'input_sha256':digest,'weighted_loss_sum':n,'loss_weight_sum':t,'weighted_byte_sum':b})
  sources.append({'batch':bn,'path':entry['path'],'npz_sha256':hashlib.sha256(raw).hexdigest()})
 if j.get('byte_table_sha256') not in (None,table_hash):raise ValueError('declared byte-table digest disagrees with computed digest')
 j['byte_table_sha256']=table_hash;validate(j);aggregate(j['records']);j['array_export']={'sources':sources,'mask_spec':mask_spec,'input_hash_verified_against_supplied_arrays':True,'source_array_scope':'exporter-declared global; ranks not independently verified','model_checkpoint_and_tokenizer_identity':'declarations only','losses_origin':'supplied arrays; forward not executed'}
 return j
if __name__=='__main__':
 if len(sys.argv)!=3:raise SystemExit(__doc__)
 p,o=map(pathlib.Path,sys.argv[1:]);raw=p.read_bytes();result=export(json.loads(raw),p.resolve().parent);result['manifest_sha256']=hashlib.sha256(raw).hexdigest()
 with o.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
