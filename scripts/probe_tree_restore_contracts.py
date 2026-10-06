"""Original tree restoration and extracted original Haliax NamedArray class.
Real local arrays, single-device sharding adapter; no complete Marin model or production restore.
"""
import ast,contextlib,dataclasses,functools as ft,hashlib,json,pathlib,tempfile,types,typing
from typing import Any,Callable,Mapping,Optional,Sequence,overload
import equinox as eqx,jax,jax.numpy as jnp,numpy as np,tensorstore as ts
import jax.tree_util as jtu
import probe_restore_candidate_real_io as base
R=base.R;D=R/'sources/tree_restore_2026_10_07';ns=base.ns
ns.update(globals());ns['_ENABLE_SHAPE_CHECKS']=True
def extract(p,names):
 nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 # Overloaded functions: only actual implementation is needed.
 nodes=[n for n in nodes if not any(ast.unparse(d) in ['overload','typing.overload'] for d in getattr(n,'decorator_list',[]))]
 assert len(nodes)==len(names),(p,[n.name for n in nodes])
 mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[]);exec(compile(ast.fix_missing_locations(mod),str(p),'exec'),ns)
extract(D/'haliax_util.py',['ensure_tuple'])
extract(D/'haliax_jax_utils.py',['is_jax_array_like','is_in_jit'])
extract(D/'haliax_axis.py',['Axis','axis_spec_to_shape_dict','_check_size_consistency'])
extract(D/'haliax_core.py',['NamedArrayMeta','NamedArray','enable_shape_checks','are_shape_checks_enabled'])
N=ns['NamedArray'];Axis=ns['Axis'];device=base.SingleDeviceSharding(jax.devices('cpu')[0])
ns['haliax']=ns['hax']=types.SimpleNamespace(NamedArray=N,partitioning=types.SimpleNamespace(sharding_for_axis=lambda *args:device))
extract(D/'haliax_util.py',['is_named_array'])
extract(D/'jax_utils.py',['leaf_key_paths','_isnamedtupleinstance','join_key'])
ns['jax_utils']=types.SimpleNamespace(leaf_key_paths=ns['leaf_key_paths'])
ns['record_transfer']=lambda *args:None
ns['partial']=ft.partial
extract(base.P,['_is_named_or_none','_sharding_from_leaf','_fully_replicated_sharding','tree_deserialize_leaves_tensorstore'])
ns['equinox']=eqx
extract(base.C,['load_checkpoint'])
def main():
 checks=[];cases=[];expected=jax.ShapeDtypeStruct((6,4),jnp.float32,sharding=device);axes=(Axis('rows',6),Axis('cols',4));config=ns['TensorStoreReadConfig'](replica_mode=ns['ReplicaRestoreMode'].EVERY_REPLICA)
 def check(n,c):assert c,(n,cases);checks.append(n)
 with tempfile.TemporaryDirectory(prefix='marin-tree-contract-') as tmp:
  for label,shape,dtype in [('matching',(6,4),'float32'),('wrong_shape',(2,4),'float32'),('wrong_dtype',(6,4),'int32')]:
   root=pathlib.Path(tmp)/label;root.mkdir();entry=ns['CheckpointArray'](path='w',shape=(6,4),dtype='float32',chunk_shape=(2,4));ns['write_manifest'](str(root),ns['build_manifest']([entry],array_driver='zarr3',kvstore_driver='ocdbt'))
   actual=ns['CheckpointArray'](path='w',shape=shape,dtype=dtype,chunk_shape=(2,4));store=ts.open(ns['_create_ocdbt_spec'](str(root),'w',entry=actual),create=True).result();store.write(np.arange(np.prod(shape),dtype=dtype).reshape(shape)).commit.result();(root/'metadata.json').write_text('{}')
   for named in [False,True]:
    exemplar={'w':N(expected,axes) if named else expected,'label':'diagnostic'};row={'fixture':label,'exemplar':'NamedArray' if named else 'ShapeDtypeStruct','expected_shape':[6,4],'expected_dtype':'float32'}
    try:
     tree=ns['load_checkpoint'](exemplar,str(root),allow_partial=False,read_config=config);leaf=tree['w'];value=leaf.array if named else leaf;row.update(error=None,restored_shape=list(value.shape),restored_dtype=str(value.dtype),non_array_label=tree['label'])
    except Exception as e:row.update(error=type(e).__name__+': '+str(e))
    cases.append(row)
 check('Original raw and named matching tree restore succeed',all(x['error'] is None and x['restored_shape']==[6,4] for x in cases[:2]))
 check('Raw exemplar shape does not constrain original tree restore',cases[2]['error'] is None and cases[2]['restored_shape']==[2,4])
 check('Original NamedArray rebuild rejects changed axis size',cases[3]['error'] is not None and 'different sizes' in cases[3]['error'])
 check('Raw and named restored dtype follow store rather than exemplar',all(x['error'] is None and x['restored_dtype']=='int32' for x in cases[4:]))
 with ns['enable_shape_checks'](False):unchecked=N(jnp.zeros((2,4)),axes)
 check('Explicitly disabled shape checks permit mismatched constructor control',unchecked.array.shape==(2,4))
 check('Shape-check context restores enabled state',ns['are_shape_checks_enabled']())
 check('Original load wrapper preserves non-array exemplar field',all(x['non_array_label']=='diagnostic' for x in cases if x['error'] is None))
 o={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'cases':cases,'runtime':{'jax':jax.__version__,'equinox':eqx.__version__},'original_tree_deserializer':True,'original_load_checkpoint':True,'original_NamedArray_class_extracted':True,'explicit_substitutions':{'sharding_for_axis':'single CPU device adapter; ignores mesh/axis mapping','record_transfer':'no-op cross-region accounting','StoragePath':'inherited local pathlib adapter','metadata':'manual empty object, only root traversal tested'},'actual_full_model_restore':None,'actual_Hero_leaf_types':None,'actual_Hero_shape_checks_disabled':None,'actual_production_dtype_incident':None,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [base.P,base.M,base.C,D/'jax_utils.py',D/'haliax_core.py',D/'haliax_axis.py',D/'haliax_jax_utils.py',D/'haliax_util.py']}}
 (R/'analysis/tree_restore_contracts.json').write_text(json.dumps(o,indent=2)+'\n');print('Original tree restore contract checks:',len(checks))
if __name__=='__main__':main()
