"""Offline structural gate on supplied callback arrays; no forward/provenance proof.
Tag overlap is accepted only under an explicit overlap contract.
"""
import numpy as np

def check_callback_arrays(losses, weights, token_ids, tags, byte_table, *, num_tags, tag_policy):
 arrays=[np.asarray(x) for x in (losses,weights,token_ids,tags,byte_table)]
 loss,weight,ids,tag,table=arrays
 if tag_policy not in ('exclusive_leaf','overlapping_tags'):
  raise ValueError('explicit tag policy required')
 if any(x.ndim!=2 for x in arrays[:4]):
  raise ValueError('loss/weight/token_ids/tags must have rank two')
 if loss.shape!=weight.shape or ids.shape!=weight.shape or not all(weight.shape):
  raise ValueError('identical nonempty scored shapes required; no broadcasting')
 if tag.shape!=(weight.shape[0],num_tags):
  raise ValueError('tag rows and ordered column count must match batch')
 if ids.dtype.kind not in 'iu' or table.ndim!=1 or not len(table):
  raise ValueError('integer token ids and nonempty byte table required')
 if np.any(ids<0) or np.any(ids>=len(table)):
  raise ValueError('token ids must be within byte-table bounds')
 if any(x.dtype.kind not in 'bifu' or not np.isfinite(x).all() for x in arrays):
  raise ValueError('finite real callback arrays required')
 if np.any(weight<0) or np.any(table<0) or np.any(loss<0):
  raise ValueError('nonnegative NLL, weights and byte lengths required')
 if not np.isin(tag,[0,1]).all():
  raise ValueError('binary membership tags required')
 membership=tag.sum(axis=1)
 if tag_policy=='exclusive_leaf' and np.any(membership>1):
  raise ValueError('exclusive leaves cannot overlap')
 if np.any((weight.sum(axis=1)>0)&(membership==0)):
  raise ValueError('positive-weight rows need declared tag coverage')
 return {'scored_shape':list(weight.shape),'tag_policy':tag_policy,'target_mass':float(weight.sum()),'tag_mass':np.einsum('bt,bk->k',weight,tag).tolist(),'identity_verified':False,'target_alignment_verified':False}
