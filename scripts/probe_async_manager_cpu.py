"""Uploaded JAX AsyncManager class with real Python threads; single-process dependency adapters.
Controlled standard-library futures and real TensorStore memory-only commit; no distributed save or model.
"""
import importlib.metadata
import ast,concurrent.futures,gc,hashlib,itertools,json,logging,pathlib,platform,threading,time,types,typing,weakref
import jax,numpy as np,tensorstore as ts
R=pathlib.Path(__file__).resolve().parents[1];P=R/'sources/async_manager_2026_10_08/serialization.py';checks=[]
def check(n,v):
 if not v:raise RuntimeError(n)
 checks.append({'name':n,'passed':True})
class RuntimeAdapterError(RuntimeError):pass
class BarrierTimeoutError(RuntimeError):pass
ns={'jax':jax,'distributed':types.SimpleNamespace(global_state=types.SimpleNamespace(client=None)),'logger':logging.getLogger(__name__),'time':time,'threading':threading,'_module_unique_count':itertools.count(),'_jax':types.SimpleNamespace(JaxRuntimeError=RuntimeAdapterError),'BarrierTimeoutError':BarrierTimeoutError,'_DISTRIBUTED_SYSTEM_MSG':'distributed adapter not supported','_BARRIER_TIMED_OUT_MSG':'synthetic timeout','_CHECKPOINT_SUCCESS':'success'}
node=next(n for n in ast.parse(P.read_text()).body if getattr(n,'name',None)=='AsyncManager')
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])),str(P),'exec'),ns);Manager=ns['AsyncManager']
def launch(future,callback):
 m=Manager();m._add_futures([future]);m._start_async_commit(callback);return m
def consume(m):
 try:m.wait_until_finished();return {'returned':True}
 except Exception as e:return {'returned':False,'error_type':type(e).__name__,'error':str(e)}
def owned_future():
 arr=np.arange(32,dtype=np.uint8);ref=weakref.ref(arr);f=concurrent.futures.Future();f.set_result(arr);return f,ref
f,ref=owned_future();m=launch(f,lambda:None);del f;m.wait_until_finished();gc.collect()
retained={'thread_cleared':m._thread is None,'completed_future_list_length':len(m._commit_futures),'payload_alive_after_wait':ref() is not None}
check('Original manager joins worker but retains completed controlled future and its payload',retained=={'thread_cleared':True,'completed_future_list_length':1,'payload_alive_after_wait':True})
f2=concurrent.futures.Future();f2.set_result(None);m._add_futures([f2]);gc.collect();retained['old_payload_alive_after_new_list']=ref() is not None
check('Replacing list releases old controlled payload, not unbounded historical list accumulation',not retained['old_payload_alive_after_new_list'] and len(m._commit_futures)==1)
arr=np.arange(16);cref=weakref.ref(arr);callback=lambda payload=arr:len(payload);f3=concurrent.futures.Future();f3.set_result(None);cm=launch(f3,callback);del arr,callback,f3;cm.wait_until_finished();cm._commit_futures=None;gc.collect();callback_alive=cref() is not None;cm._on_commit_callback=None;gc.collect()
check('Retained callback is independent ownership path in controlled closure',callback_alive and cref() is None)
callbacks=[];bad=concurrent.futures.Future();bad.set_exception(RuntimeError('local commit failed'));bm=launch(bad,lambda:callbacks.append('published'));first=consume(bm);second=consume(bm)
check('Commit exception skips callback and propagates once to waiting caller',first.get('error')=='local commit failed' and second=={'returned':True} and callbacks==[] and bm._exception is None)
callback_attempts=[]
def fail_callback():callback_attempts.append(1);raise RuntimeError('publish failed')
ok=concurrent.futures.Future();ok.set_result(None);pm=launch(ok,fail_callback);publish_first=consume(pm);publish_second=consume(pm)
check('Successful commit with failed callback also raises only once',publish_first.get('error')=='publish failed' and publish_second=={'returned':True} and callback_attempts==[1])
entered=threading.Event()
class GateFuture(concurrent.futures.Future):
 def result(self,*a,**kw):entered.set();return super().result(*a,**kw)
gate=GateFuture();events=[];gm=launch(gate,lambda:events.append('callback'))
waiting_observed=entered.wait(5) and gm._thread.is_alive() and events==[]
gate.set_result(None);gm.wait_until_finished()
check('Unfinished commit reaches original blocking result before callback',waiting_observed)
check('Resolving controlled commit releases worker and invokes callback once',events==['callback'] and gm._thread is None)
# Actual TensorStore write/commit, bounded to one 32-byte array and in-memory KV store.
store=ts.open({'driver':'zarr','kvstore':{'driver':'memory'},'metadata':{'shape':[32],'chunks':[32],'dtype':'|u1'}},create=True,open=True).result()
snapshot=np.arange(32,dtype=np.uint8);tref=weakref.ref(snapshot);writes=store.write(snapshot,can_reference_source_data_indefinitely=True);writes.copy.result();commit=writes.commit;tm=launch(commit,lambda:None);del snapshot,writes,commit;tm.wait_until_finished();gc.collect()
ts_observation={'bytes':32,'completed_future_list_length':len(tm._commit_futures),'snapshot_alive_after_wait':tref() is not None,'stored_values_equal':np.array_equal(store.read().result(),np.arange(32,dtype=np.uint8))};tm._commit_futures=None;gc.collect();ts_observation['snapshot_alive_after_explicit_future_clear']=tref() is not None
check('Real in-memory TensorStore commit completes and retains actual future list with correct data',ts_observation['completed_future_list_length']==1 and ts_observation['stored_values_equal'])
check('Single-process execution remains explicit',jax.process_count()==1 and jax.default_backend()=='cpu')
acq=json.loads((R/'analysis/async_manager_acquisition_v129.json').read_text());check('Dependency source matches both archived run manifests',acq['matches_both_run_manifests'] and acq['source_sha256']==hashlib.sha256(P.read_bytes()).hexdigest())
out={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'runtime':{'python':platform.python_version(),'jax':jax.__version__,'numpy':np.__version__,'tensorstore':importlib.metadata.version('tensorstore'),'backend':jax.default_backend()},'controlled_future_retention':retained,'controlled_callback_payload_alive_after_future_clear':callback_alive,'commit_failure':{'first_wait':first,'second_wait':second,'callback_calls':len(callbacks)},'callback_failure':{'first_wait':publish_first,'second_wait':publish_second,'callback_calls':len(callback_attempts)},'tensorstore_memory_observation':ts_observation,'source_sha256':{str(P.relative_to(R)):hashlib.sha256(P.read_bytes()).hexdigest()},'probe_script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'substitutions':['Original uploaded AsyncManager class only, not full JAX serialization module','Real installed JAX0.7.2 single-process queries/monitoring; uploaded class from recorded dependency','Distributed client None, runtime-error and barrier-error class adapters; no multi-rank branch','Controlled stdlib Future payloads and closure owners, distinct from opaque TensorStore futures','Real TensorStore write/commit to memory-only Zarr store,32bytes; no production filesystem or model checkpoint'],'actual_Hero_future_retained_bytes':None,'actual_RSS_effect':None,'actual_distributed_barrier':None,'actual_full_checkpoint_restore':None,'actual_Hero_runtime_import_binding':None}
(R/'analysis/async_manager_cpu.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('Async manager original class checks:',len(checks));print('TensorStore memory observation:',ts_observation)
