"""Original build_caches with real ThreadPoolExecutor/as_completed and gated IO adapters.
No distributed collectives, storage reads or cache builds executed.
"""
import ast, concurrent.futures as cf, contextlib, hashlib, json, pathlib, platform, threading, types

R=pathlib.Path(__file__).resolve().parents[1]
P=R/'sources/boundaries_2026_10_05/datasets.py'
cl=next(n for n in ast.parse(P.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='LmDataConfig')
node=next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name=='build_caches')

def run_case(order, completion, hits=(), error=None):
    gates={n:threading.Event() for n in order};started={n:threading.Event() for n in order}
    yielded={n:threading.Event() for n in order};done=threading.Event();error_cleanup=threading.Event()
    trace={'component_order':list(order),'forced_completion':list(completion),'observed_completion':[],
           'build_dispatch':[],'build_threads':[],'error':None,'error_observed_before_caller_exit':None}
    def path(name,component,cache_dir,split):
        started[name].set()
        if not gates[name].wait(5):raise TimeoutError('Probe gate '+name)
        if name==error:raise RuntimeError('Injected metadata failure '+name)
        return name
    class Storage:
        def __init__(self,p):self.p=p
        def exists(self):return self.p in hits
    def observe(fs):
        names=dict(zip(fs,order))
        for f in cf.as_completed(fs):
            name=names[f];trace['observed_completion'].append(name);yielded[name].set()
            yield f
    def build(cache_path,*args):
        trace['build_dispatch'].append(cache_path)
        trace['build_threads'].append(threading.current_thread().name)
        return 'built:'+cache_path
    class ObservedPool(cf.ThreadPoolExecutor):
        def __exit__(self,kind,value,tb):
            if kind is not None:
                trace['cleanup_exception_type']=kind.__name__;error_cleanup.set()
            return super().__exit__(kind,value,tb)
    ns={'ThreadPoolExecutor':ObservedPool,'as_completed':observe,'CACHE_METADATA_WORKERS':32,
        'log_time':lambda *a:contextlib.nullcontext(),'_component_cache_path':path,'StoragePath':Storage,
        'load_lm_dataset_cache':lambda p,*a:'loaded:'+p,'build_lm_dataset_cache':build}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])),str(P),'exec'),ns)
    components={n:types.SimpleNamespace(source=types.SimpleNamespace(get_shard_source=lambda split:'shards'),format='format') for n in order}
    config=types.SimpleNamespace(_cache_items=lambda split:list(components.items()),_loaded_cache_catalog=None,
        cache_dir='identity-cache',auto_build_caches=True,the_tokenizer=None,enforce_eos=True,cache_options=None)
    def caller():
        try:trace['returned_caches']=ns['build_caches'](config,'train')
        except Exception as e:trace['error']={'type':type(e).__name__,'message':str(e)}
        finally:done.set()
    thread=threading.Thread(target=caller,name='probe-caller');thread.start()
    try:
        assert all(e.wait(5) for e in started.values()),'All metadata workers started'
        for name in completion:
            gates[name].set()
            if error and name==error:
                assert yielded[name].wait(5),'Failed future surfaced'
                assert error_cleanup.wait(5),'Original error entered executor cleanup'
                # Original with-executor cleanup still joins blocked sibling workers.
                trace['error_observed_before_caller_exit']=not done.wait(.1)
                break
            assert yielded[name].wait(5),'Actual as_completed yield '+name
    finally:
        for e in gates.values():e.set()
        thread.join(5)
    assert not thread.is_alive(),'No live probe threads remain'
    return trace

def main():
    checks=[]
    def ck(n,c):assert c,n;checks.append(n)
    a=run_case(('A','B','C'),('C','B','A'))
    b=run_case(('A','B','C'),('B','A','C'))
    ck('Real completion traces differ',a['observed_completion']==['C','B','A'] and b['observed_completion']==['B','A','C'])
    ck('Original component order survives both real completion orders',a['build_dispatch']==b['build_dispatch']==['A','B','C'])
    ck('Actual build adapters run serially on caller thread',all(t=='probe-caller' for t in a['build_threads']+b['build_threads']))
    mixed=run_case(('A','B','C'),('C','A','B'),hits=('A',))
    ck('Cache hit excluded from serial build plan',mixed['build_dispatch']==['B','C'] and mixed['returned_caches']['A']=='loaded:A')
    reversed_case=run_case(('C','B','A'),('B','A','C'))
    ck('Component dictionary reorder changes build dispatch',reversed_case['build_dispatch']==['C','B','A'])
    other_visibility=run_case(('A','B','C'),('C','B','A'),hits=('B',))
    ck('Different cache visibility changes hypothetical host plan',other_visibility['build_dispatch']==['A','C'] and mixed['build_dispatch']==['B','C'])
    failure=run_case(('A','B','C'),('B',),error='B')
    ck('Completed error future surfaced while caller waits for cleanup',failure['observed_completion'][0]=='B' and failure['cleanup_exception_type']=='RuntimeError' and failure['error_observed_before_caller_exit'])
    ck('Original metadata error propagated after sibling release',failure['error']=={'type':'RuntimeError','message':'Injected metadata failure B'})
    ck('Metadata failure prevents all serial cache builds',failure['build_dispatch']==[])
    meta=R/'sources/live_2026_10_07/meta.json'
    data=json.loads(meta.read_text())['config']['data']['value']
    active={k for _,w in data['train_weights'] for k,v in w.items() if v>0}
    ck('Archived active children declare existing cache and no source',len(active)==200 and all(data['components'][k].get('source') is None and data['components'][k].get('cache_dir') for k in active))
    files=[P,R/'analysis/mixture_support_source_binding.json',meta]
    result={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'python':platform.python_version(),
        'cases':{'completion_cba':a,'completion_bac':b,'hit_A':mixed,'reordered_components':reversed_case,
                 'hit_B':other_visibility,'metadata_failure':failure},
        'archived_active_components':{'count':len(active),'source_none':True,'cache_dir_declared':True},
        'adapters':'Real executor subclass records __exit__ then delegates unchanged shutdown; real completion iterator observer records yields. Event-gated path/source/storage/cache adapters impose finite artificial IO states. Original build_caches body unchanged.',
        'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'actual_distributed_hang':None,'actual_cross_host_cache_visibility':None,'actual_Hero_execution':None}
    (R/'analysis/cache_dispatch_probe.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Original build_caches real-thread controls:',len(checks),'passed')
if __name__=='__main__':main()
