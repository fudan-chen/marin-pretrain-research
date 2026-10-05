"""Read a normalized global-leaf NPZ and declared JSON contract; observations only.
Recomputes hyperball geometry in float64. It does not reproduce optimizer moments,
NS/BF16/SPMD, verify provenance or diagnose a root cause automatically.
"""
import argparse,hashlib,json,pathlib
import numpy as np

def group_norm(x,axes):
 axes=tuple(axes);x=np.asarray(x,dtype=np.float64);scale=np.max(np.abs(x),axis=axes,keepdims=True);safe=np.where(scale==0,1,scale)
 return scale*np.sqrt(np.sum((x/safe)**2,axis=axes,keepdims=True))
def values(x):return np.asarray(x).reshape(-1).tolist()
def analyze(arrays,meta):
 if meta.get('array_scope')!='global':raise ValueError('Requires declared global arrays; a local shard is insufficient')
 if meta.get('parameter_view')!='optimizer_parameter_before_update':raise ValueError('Parameter view must identify the actual optimizer input')
 if meta.get('update_convention')!='p_after=p_before+update':raise ValueError('Unspecified update sign convention')
 p=np.asarray(arrays['p_before']);update=np.asarray(arrays['update'])
 if p.ndim not in (2,3,4) or p.size==0 or update.shape!=p.shape:raise ValueError('Requires nonempty 2D/3D/4D matching arrays')
 axes=tuple(meta['norm_reduction_axes']);expected=tuple(range(2)) if p.ndim==2 else tuple(range(1,p.ndim))
 if axes!=expected:raise ValueError('Contract axes do not match the supported source hyperball grouping')
 for key,x in arrays.items():
  x=np.asarray(x)
  if x.dtype.kind!='f' or x.dtype.itemsize not in (4,8):raise ValueError('Only real float32/float64 normalized arrays supported')
  if key in {'direction','p_after','projection_intermediate'} and x.shape!=p.shape:raise ValueError('Optional tensor shape mismatch: '+key)
 tolerance=float(meta['relative_tolerance']);lr=float(meta['learning_rate']);epsilon=float(meta['projection_epsilon'])
 if not np.isfinite([tolerance,lr,epsilon]).all() or tolerance<=0 or lr<0 or epsilon<=0:raise ValueError('Invalid numeric contract')
 out={'status':'observations_not_root_cause','provenance':'metadata_declared_not_independently_verified','root_cause':None,'array_shape':list(p.shape),'norm_reduction_axes':list(axes),'array_nonfinite_counts':{k:int(np.count_nonzero(~np.isfinite(x))) for k,x in arrays.items()},'missing_evidence':[k for k in ['direction','projection_intermediate','computed_new_param_norm','p_after'] if k not in arrays],'execution_sha_declared':meta.get('execution_sha'),'geometry_precision':'float64 CPU recomputation; not original low precision or distributed arithmetic'}
 if any(out['array_nonfinite_counts'].values()):out['numerical_observation']='nonfinite_present_geometry_not_reconstructed';return out
 p=p.astype(np.float64);update=update.astype(np.float64);r=group_norm(p,axes);q=p+update;qr=group_norm(q,axes);ur=group_norm(update,axes);ratio=np.divide(qr,r,out=np.ones_like(r)*np.nan,where=r>0);out['zero_parameter_groups']=int(np.count_nonzero(r==0));out['parameter_norm_before']=values(r);out['parameter_norm_after_from_delta']=values(qr);out['norm_ratio']=[float(x) if np.isfinite(x) else None for x in ratio.reshape(-1)];out['norm_deviation_exceeds_declared_tolerance']=bool(np.any(np.abs(ratio-1)>tolerance));out['update_norm']=values(ur)
 if 'p_after' in arrays:out['p_after_disagrees_with_signed_delta']=bool(np.any(group_norm(arrays['p_after']-q,axes)>tolerance*np.maximum(r,1e-30)))
 if 'direction' in arrays:
  if meta.get('direction_stage')!='after_moment_or_NS_before_projection':raise ValueError('Direction stage declaration is required')
  u=arrays['direction'].astype(np.float64);s=group_norm(u,axes);v=p-lr*u*r/np.maximum(s,epsilon);vn=group_norm(v,axes);reference=v*r/np.maximum(vn,epsilon);out['reference_projection_norm']=values(group_norm(reference,axes));out['reference_update_disagrees']=bool(np.any(group_norm(q-reference,axes)>tolerance*np.maximum(r,1e-30)));out['reference_zero_intermediate_groups']=int(np.count_nonzero(vn==0))
 if 'projection_intermediate' in arrays and 'computed_new_param_norm' in arrays:
  recomputed=group_norm(arrays['projection_intermediate'],axes);recorded=np.asarray(arrays['computed_new_param_norm'])
  if recorded.size!=recomputed.size:raise ValueError('Recorded denominator must have one scalar per declared group')
  recorded=recorded.reshape(recomputed.shape)
  if np.any(recorded<0):raise ValueError('Recorded norm must be nonnegative')
  dr=np.divide(recorded,recomputed,out=np.ones_like(recomputed)*np.nan,where=recomputed>0);out['recorded_to_recomputed_denominator_ratio']=[float(x) if np.isfinite(x) else None for x in dr.reshape(-1)];out['denominator_disagreement']=bool(np.any(np.abs(dr-1)>tolerance))
 return out

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--arrays',type=pathlib.Path,required=True);parser.add_argument('--contract',type=pathlib.Path,required=True);parser.add_argument('--output',type=pathlib.Path,required=True);args=parser.parse_args()
 if args.output.exists():raise SystemExit('Refusing to overwrite existing result')
 with np.load(args.arrays,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
 meta=json.loads(args.contract.read_text());out=analyze(arrays,meta);out['input_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.arrays,args.contract]};args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
