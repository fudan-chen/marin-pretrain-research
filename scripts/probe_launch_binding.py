"""Original launcher, forwarding filter and PJRT metadata guard with explicit stubs.
No job submission, child process, plugin load or backend initialization.
"""
import ast,hashlib,json,pathlib,types
from copy import deepcopy
import probe_runtime_defaults as runtime
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/launch_binding_2026_10_07';P=D/'dispatch.py';T=runtime.T
tree=ast.parse(P.read_text());ns={}
nodes=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id.startswith('_FORWARDED_ENV_') for t in n.targets)]
nodes += [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_forwarded_env_vars']
def execute(nodes,p,namespace):exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])),str(p),'exec'),namespace)
execute(nodes,P,ns)
tt=ast.parse(T.read_text());run=next(n for n in tt.body if isinstance(n,ast.FunctionDef) and n.name=='run_grug');guard=next(n for n in tt.body if isinstance(n,ast.FunctionDef) and n.name=='verify_ragged_pjrt')
execute([run],T,runtime.ns);execute([guard],T,ns)
class MissingPackage(Exception):pass
def main():
 checks=[]
 def check(n,c):assert c,n;checks.append(n)
 trainer=types.SimpleNamespace(id='synthetic-launch',watch=types.SimpleNamespace(is_enabled=False))
 config=types.SimpleNamespace(trainer=types.SimpleNamespace(trainer=trainer,watch_mode='inline'),model=types.SimpleNamespace(moe_implementation='ragged_all_to_all',remat_mode='offload_carry'),processes_per_task=1,resources=None,max_retries_failure=3,max_task_failures=10)
 inherited={'JAX_PLATFORMS':'cpu','JAX_ENABLE_PGLE':'false','XLA_FLAGS':'--xla_gpu_memory_limit_slop_factor=85','RAGGED_DOT_IMPL':'xla','TENSORSTORE_CURL_LOW_SPEED_TIME_SECONDS':'99','CUBLAS_WORKSPACE_CONFIG':':4096:8','GIT_COMMIT':'synthetic-parent-commit','NCCL_DEBUG':'INFO','LD_PRELOAD':'synthetic-library','MALLOC_CONF':'synthetic-config'}
 env=dict(inherited);runtime.ns['os']=types.SimpleNamespace(environ=env);ns['os']=types.SimpleNamespace(environ=env)
 runtime.ns['WatchMode']=types.SimpleNamespace(INLINE='inline');runtime.ns['_run_grug_local']=lambda config:None
 captures=[]
 def dispatch(**kwargs):captures.append({'kwargs_keys':sorted(kwargs),'forwarded':ns['_forwarded_env_vars']()})
 runtime.ns['dispatch_grug_training_run']=dispatch;runtime.ns['run_grug'](config)
 forwarded=captures[0]['forwarded'];check('Original run applies defaults before intercepted dispatch snapshot',len(captures)==1 and forwarded['XLA_PYTHON_CLIENT_MEM_FRACTION']=='0.78' and '--xla_gpu_experimental_parallel_collective_overlap_limit=1' in forwarded['XLA_FLAGS'])
 check('Dispatcher CPU-only platforms is intentionally excluded','JAX_PLATFORMS' in env and 'JAX_PLATFORMS' not in forwarded)
 check('Runtime prefixes and LD_PRELOAD forwarded exactly',all(forwarded[k]==env[k] for k in ['JAX_ENABLE_PGLE','NCCL_DEBUG','LD_PRELOAD','MALLOC_CONF','XLA_FLAGS']))
 omitted=['RAGGED_DOT_IMPL','TENSORSTORE_CURL_LOW_SPEED_TIME_SECONDS','CUBLAS_WORKSPACE_CONFIG','GIT_COMMIT']
 check('Four parent controls absent at forwarding stage',all(k in env and k not in forwarded for k in omitted))
 frozen=dict(forwarded);env['NCCL_DEBUG']='WARN';check('Returned forwarding dictionary is a snapshot, not a live environment view',forwarded==frozen and forwarded['NCCL_DEBUG']=='INFO')
 # Guard only receives metadata.version and __version__; no device/architecture surface supplied.
 ns['PJRT_DISTRIBUTION']='jax-cuda13-pjrt';ns['RAGGED_MOE_IMPLEMENTATION']='ragged_all_to_all';ns['jax']=types.SimpleNamespace(__version__='0.7.2')
 guard_cases=[]
 for installed in [None,'0.7.2','0.7.1+marin.1','0.7.2+marin.1','0.7.2+marin.not-an-artifact-proof','0.7.2+marin.']:
  def version(name):
   assert name=='jax-cuda13-pjrt'
   if installed is None:raise MissingPackage(name)
   return installed
  ns['importlib']=types.SimpleNamespace(metadata=types.SimpleNamespace(version=version,PackageNotFoundError=MissingPackage))
  try:ns['verify_ragged_pjrt']();passed=True;error=None
  except RuntimeError as e:passed=False;error=str(e)
  guard_cases.append({'installed_version':installed,'passed':passed,'error':error,'actual_plugin_loaded':None})
 check('Missing/unpatched/version-mismatched metadata rejected',all(not x['passed'] for x in guard_cases[:3]))
 check('Matching prefix including empty/artificial suffix passes lexical guard',all(x['passed'] for x in guard_cases[3:]))
 local=next(n for n in tt.body if isinstance(n,ast.FunctionDef) and n.name=='_run_grug_local')
 check('Child entrypoint statically guards ragged before trainer.initialize',isinstance(local.body[1],ast.If) and 'verify_ragged_pjrt()' in ast.unparse(local.body[1]) and next(i for i,n in enumerate(local.body) if 'trainer.initialize()'==ast.unparse(n))>1)
 a=json.loads((R/'analysis/launch_binding_acquisition.json').read_text())
 check('Three acquired full modules preserve 486 prior non-bookkeeping source bytes',len(a['records'])==3 and a['prior_bytes_preserved']==486 and all(hashlib.sha256((R/'sources'/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in a['records']) and all(hashlib.sha256((R/'sources'/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in a['prior_integrity_manifest']['files'] if x['file']!='source_manifest.json'))
 resolver=next(n for n in ast.parse((D/'training.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='resolve_training_env')
 check('Full downstream resolver visibly augments filtered environment','_add_default_env_variables' in ast.unparse(resolver) and 'add_run_env_variables' in ast.unparse(resolver) and '_add_gpu_collective_watchdog_env' in ast.unparse(resolver))
 rn=ast.parse((D/'run_environment.py').read_text());meta=next(n for n in rn.body if isinstance(n,ast.FunctionDef) and n.name=='add_run_env_variables')
 ns['deepcopy']=deepcopy;ns['logger']=types.SimpleNamespace(warning=lambda *a:None,info=lambda *a:None);execute([meta],D/'run_environment.py',ns)
 augmented=ns['add_run_env_variables'](forwarded)
 check('Original metadata helper can restore parent GIT_COMMIT after whitelist omission',augmented['GIT_COMMIT']==env['GIT_COMMIT'] and 'GIT_COMMIT' not in forwarded)
 tr=ast.parse((D/'training.py').read_text());constants=[n for n in tr.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['TENSORSTORE_CURL_LOW_SPEED_TIME_ENV','TENSORSTORE_CURL_LOW_SPEED_LIMIT_ENV','DEFAULT_TENSORSTORE_CURL_LOW_SPEED_TIME','DEFAULT_TENSORSTORE_CURL_LOW_SPEED_LIMIT','GPU_NCCL_TERMINATION_TIMEOUT_FLAG','DEFAULT_GPU_NCCL_TERMINATION_TIMEOUT'] for t in n.targets)]
 watchdog=next(n for n in tr.body if isinstance(n,ast.FunctionDef) and n.name=='_add_gpu_collective_watchdog_env');execute(constants+[watchdog],D/'training.py',ns)
 ts=[n for n in resolver.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='setdefault' and ast.unparse(n.value.args[0]).startswith('TENSORSTORE_CURL_')];assert len(ts)==2
 ns['env']=augmented;execute(ts,D/'training.py',ns);ns['_add_gpu_collective_watchdog_env'](augmented)
 check('Original resolver TensorStore defaults differ from omitted parent override',env['TENSORSTORE_CURL_LOW_SPEED_TIME_SECONDS']=='99' and augmented['TENSORSTORE_CURL_LOW_SPEED_TIME_SECONDS']=='60')
 check('Original GPU watchdog appends timeout while preserving forwarded NCCL_DEBUG','--xla_gpu_nccl_termination_timeout_seconds=600' in augmented['XLA_FLAGS'] and augmented['NCCL_DEBUG']=='INFO')
 o={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'parent_environment_after_helper':dict(env),'dispatch_capture':captures[0],'independently_augmented_environment':augmented,'omitted_at_forwarding_stage':omitted,'guard_cases':guard_cases,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [T,runtime.M,runtime.E,P,D/'training.py',D/'run_environment.py']},'explicit_substitutions':['os.environ isolated dictionary; no real environment writes','dispatch intercepted at call site, not Fray submission or full resolver execution','jax.__version__ synthetic 0.7.2; package metadata provided by controlled stub','constructor config represented by structural namespace','downstream metadata/watchdog helpers and two TensorStore statements independently executed; hardware config/merge not run','child init order inspected statically only'],'actual_child_environment':None,'actual_job_submission':None,'actual_plugin_loaded':None,'actual_architecture_check':None,'actual_GPU_execution':None}
 (R/'analysis/launch_binding.json').write_text(json.dumps(o,indent=2)+'\n');print('Launch binding checks:',len(checks));print(json.dumps({'forwarded_keys':sorted(forwarded),'omitted':omitted,'guard_results':[(x['installed_version'],x['passed']) for x in guard_cases]},indent=2))
if __name__=='__main__':main()
