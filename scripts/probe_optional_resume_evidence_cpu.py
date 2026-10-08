"""Original pinned restore discovery/layout/reader with real local OCDBT fixtures.
Manual metadata and synthetic Grug state; no full training entry or production incident.
"""
import ast, dataclasses, hashlib, json, pathlib, tempfile
import jax, jax.numpy as jnp, numpy as np, optax
import probe_grug_state_restore_real_io as base
R=base.R;ns=base.ns;State=base.State
checks=[];cases={}
def check(name,condition):
 if not condition:raise RuntimeError(name)
 checks.append({'name':name,'passed':True})
w=jnp.array([1.,2.],jnp.float32);tx=optax.scale_by_adam();opt=tx.init({'w':w})
initial=State(step=jnp.array(0,jnp.int32),params={'w':w},master_params=None,opt_state=opt,ema_params=None,pending_qb_betas=jnp.array([[0.,0.]]))
_,nextopt=tx.update({'w':w*.1},opt,{'w':w})
saved=dataclasses.replace(initial,step=jnp.array(20,jnp.int32),params={'w':w+10},opt_state=nextopt)

def save(root,omit=None,marker='valid'):
 root.mkdir(parents=True)
 paths,arrays=ns['_flatten_serializable_leaves'](saved)
 included=[(p,a) for p,a in zip(paths,arrays) if p!=omit]
 entries=[ns['CheckpointArray'](path=p,shape=tuple(a.shape),dtype=np.dtype(a.dtype).name,chunk_shape=tuple(a.shape)) for p,a in included]
 ns['write_manifest'](str(root),ns['build_manifest'](entries,array_driver='zarr3',kvstore_driver='ocdbt'))
 manager=base.base.base.writer.array_ser.GlobalAsyncCheckpointManager()
 ns['_serialize_arrays']([np.asarray(a) for p,a in included],[ns['_create_ocdbt_spec'](str(root),p,entry=e) for (p,a),e in zip(included,entries)],[None]*len(included),manager,ns['TensorStoreWriteConfig'](max_staged_host_bytes=128),lambda:None,None,None)
 manager.wait_until_finished()
 if marker!='absent':
  text=json.dumps({'step':20,'timestamp':'2026-10-08T00:00:00Z'}) if marker=='valid' else '{bad' if marker=='malformed' else 'null'
  (root/'metadata.json').write_text(text)

def run(name,search,setting=None,template=None):
 if template is None:template=initial
 calls=[];barriers_before=len(base.barriers)
 row={'configured_path_kind':'concrete' if search.name=='step20' else 'parent','required':setting is True}
 try:
  candidates=ns['_checkpoint_candidates']([str(search)])
  row['candidates']=[str(pathlib.Path(p).relative_to(search.parent)) for p in candidates]
  row['written_count_by_original_predicate']=len([p for p in candidates if p!=str(search)])
  def load(exemplar,path,**kw):
   call={'path':str(pathlib.Path(path).relative_to(search.parent)),'legacy_wrapper':isinstance(exemplar,dict) and 'train_state' in exemplar};calls.append(call)
   try:return base.actual_load(exemplar,path,read_config=ns['TensorStoreReadConfig'](replica_mode=ns['ReplicaRestoreMode'].EVERY_REPLICA),**kw)
   except Exception as exc:call['error_type']=type(exc).__name__;raise
  restored=ns['restore_grug_state_from_checkpoint'](template,checkpoint_search_paths=[str(search)],load_checkpoint_setting=setting,mesh=None,allow_partial=False,_load_fn=load,template_for_candidate=lambda p:ns['template_for_candidate_layout'](template,p,ns['MasterParamMode'].DEVICE))
  row['returned_shape_template']=any(isinstance(v,jax.ShapeDtypeStruct) for v in jax.tree.leaves(restored))
  reinits=[]
  if row['returned_shape_template']:
   fn=next(n for n in ast.parse(base.T.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_run_grug_local')
   guard=next(n for n in ast.walk(fn) if isinstance(n,ast.If) and ast.unparse(n.test).startswith('released_initial_state and any'))
   env={'released_initial_state':True,'state':restored,'jax':jax,'_init_state':lambda key:(reinits.append('same_model_key') or initial),'model_key':'same_model_key'}
   exec(compile(ast.fix_missing_locations(ast.Module(body=[guard],type_ignores=[])),str(base.T),'exec'),env)
   restored=env['state']
  row.update(returned=True,step=int(restored.step),adam_count=int(restored.opt_state.count),same_object_as_initial=restored is initial,params=np.asarray(restored.params['w']).tolist(),entry_reinit_calls=reinits)
 except Exception as exc:
  row.update(returned=False,error_type=type(exc).__name__,error=str(exc).replace(str(search.parent),'<fixture>'))
 row['array_load_calls']=calls;row['barrier_calls']=len(base.barriers)-barriers_before;cases[name]=row
 return row

with tempfile.TemporaryDirectory(prefix='marin-resume-evidence-') as tmp:
 root=pathlib.Path(tmp)
 broken=root/'broken'/'step20';save(broken,omit='opt_state/nu/w')
 run('parent_valid_marker_missing_leaf_optional',broken.parent)
 run('concrete_valid_marker_missing_leaf_optional',broken)
 run('concrete_valid_marker_missing_leaf_required',broken,True)
 shape_template=jax.tree.map(lambda a:jax.ShapeDtypeStruct(a.shape,a.dtype,sharding=a.sharding),initial)
 run('concrete_broken_optional_shape_template_entry_guard',broken,template=shape_template)
 malformed=root/'malformed'/'step20';save(malformed,marker='malformed')
 run('parent_malformed_marker_complete_arrays_optional',malformed.parent)
 run('concrete_malformed_marker_complete_arrays_optional',malformed)
 unpublished=root/'unpublished'/'step20';save(unpublished,marker='absent')
 run('parent_no_marker_complete_arrays_optional',unpublished.parent)
 run('concrete_no_marker_complete_arrays_optional',unpublished)
 null=root/'null'/'step20';save(null,marker='null')
 run('parent_null_marker_complete_arrays_optional',null.parent)
 empty=root/'empty';empty.mkdir();run('empty_parent_optional',empty)
 good=root/'good'/'step20';save(good);run('parent_complete_optional',good.parent)

check('parent missing manifest leaf fails optional resume',not cases['parent_valid_marker_missing_leaf_optional']['returned'] and cases['parent_valid_marker_missing_leaf_optional']['error_type']=='FileNotFoundError')
check('same concrete broken checkpoint returns initial object under optional resume',cases['concrete_valid_marker_missing_leaf_optional']['same_object_as_initial'] and cases['concrete_valid_marker_missing_leaf_optional']['written_count_by_original_predicate']==0)
check('same concrete broken checkpoint fails when required',not cases['concrete_valid_marker_missing_leaf_required']['returned'])
check('concrete broken fixture executes real original reader and legacy retry',len(cases['concrete_valid_marker_missing_leaf_optional']['array_load_calls'])==2 and cases['concrete_valid_marker_missing_leaf_optional']['array_load_calls'][1]['legacy_wrapper'])
check('malformed child marker skips complete payload and returns initial',cases['parent_malformed_marker_complete_arrays_optional']['same_object_as_initial'] and cases['parent_malformed_marker_complete_arrays_optional']['written_count_by_original_predicate']==0)
check('direct same malformed-marker fixture restores actual step20',cases['concrete_malformed_marker_complete_arrays_optional']['step']==20 and cases['concrete_malformed_marker_complete_arrays_optional']['params']==[11.,12.])
check('parent unpublished child skipped while concrete path loads same payload',cases['parent_no_marker_complete_arrays_optional']['same_object_as_initial'] and cases['concrete_no_marker_complete_arrays_optional']['step']==20)
check('valid JSON null marker aborts scanner outside JSON parse exception guard',not cases['parent_null_marker_complete_arrays_optional']['returned'] and cases['parent_null_marker_complete_arrays_optional']['error_type']=='AttributeError')
check('empty first launch returns initial state',cases['empty_parent_optional']['same_object_as_initial'])
check('normal parent complete checkpoint restores state and optimizer',cases['parent_complete_optional']['step']==20 and cases['parent_complete_optional']['adam_count']==1 and cases['parent_complete_optional']['barrier_calls']==1)
check('no barrier is recorded for initial-state fallback',all(row['barrier_calls']==0 for row in cases.values() if row.get('same_object_as_initial')))
check('original entry guard reinitializes returned shape template',cases['concrete_broken_optional_shape_template_entry_guard']['returned_shape_template'] and cases['concrete_broken_optional_shape_template_entry_guard']['entry_reinit_calls']==['same_model_key'] and cases['concrete_broken_optional_shape_template_entry_guard']['step']==0)
paths={base.T,base.G,base.U,base.base.base.P,base.base.base.M,base.base.base.C,base.base.D/'jax_utils.py',base.base.D/'haliax_core.py',base.base.D/'haliax_axis.py',base.base.D/'haliax_util.py',base.base.D/'haliax_jax_utils.py'}
paths.update(R/'scripts'/name for name in ['probe_grug_state_restore_real_io.py','probe_tree_restore_contracts.py','probe_restore_candidate_real_io.py','probe_serialize_arrays_real_io.py'])
out={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'cases':cases,'runtime':{'jax':jax.__version__,'optax':optax.__version__},'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)},'probe_script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'metadata_publication':'manually constructed fixture after real local commit','collective':'barrier recorder only','actual_Hero_optional_resume_reset':None,'actual_production_metadata_corruption':None,'actual_full_training_entry_execution':None,'actual_training_loss_effect':None,'upstream_patch_applied':False}
(R/'analysis/optional_resume_evidence_cpu.json').write_text(json.dumps(out,indent=2)+'\n')
print('Optional resume evidence controls:',len(checks))
