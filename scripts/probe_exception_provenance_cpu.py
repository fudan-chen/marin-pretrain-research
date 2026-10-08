"""Original pinned loop/scope/event dispatch, synthetic faults and training services.
Not a historical Hero incident, complete training function, or checkpoint IO test.
"""
import ast
import contextlib
import hashlib
import json
import pathlib
import types

R = pathlib.Path(__file__).resolve().parents[1]
paths = ['sources/scale_2026_10_05/train_hero_ep.py',
         'sources/state_2026_10_05/callback_core.py',
         'sources/state_2026_10_05/state_adapter.py',
         'scripts/probe_failure_loop.py']
# Reuse only the old harness definitions; do not execute old cases or rewrite results.
base = (R / paths[-1]).read_text().split("clean=run('clean')", 1)[0]
base = base.replace('error=None', 'install(env, events, name)\n error=None')
base = base.replace('except BaseException as e:error=str(e)',
                    'except BaseException as e:error=exception_chain(e)')
ns = {'__file__': str(R / paths[-1])}
exec(compile(base, paths[-1], 'exec'), ns)

def original(path, name):
    tree = ast.parse((R / path).read_text())
    return next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)

def compile_nodes(nodes, env):
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), '<archived>', 'exec'), env)

env = {'contextmanager': contextlib.contextmanager}
# Annotations reference unavailable training packages, so postpone only annotations.
compile_nodes([ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
               original(paths[1], 'progress_event_scope')], env)
scope = env['progress_event_scope']
method = original(paths[2], 'emit_event')
e = {}
compile_nodes([ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), method], e)
class EventRunner:
    emit_event = e['emit_event']

# Keep the actual post-loop tracker.finish statement, with a recorder as tracker.
fn = original(paths[0], '_run_grug_local')
finish = next(n for n in fn.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
              and ast.unparse(n.value.func) == 'levanter.tracker.current_tracker().finish')
ns['code'] = compile(ast.fix_missing_locations(ast.Module(body=[ns['loop'], finish], type_ignores=[])), paths[0], 'exec')

def exception_chain(exc):
    chain = []
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        chain.append({'type': type(exc).__name__, 'message': str(exc)})
        exc = exc.__cause__ if exc.__cause__ is not None else exc.__context__
    return chain

def install(env, events, mode):
    event_fault = {'train_and_finish_fault': 'training_finished',
                   'save_and_scope_finish_fault': 'save_finished',
                   'save_start_fault': 'save_started'}.get(mode)
    runner = EventRunner()
    def observer(event):
        events.append(event)
        if event == event_fault:
            raise RuntimeError('event:' + event)
    runner._hooks = [types.SimpleNamespace(fn=types.SimpleNamespace(on_event=observer)),
                     types.SimpleNamespace(fn=types.SimpleNamespace(on_event=lambda v: events.append('tail:' + v)))]
    env['state_callbacks'].emit_event = runner.emit_event
    env['callbacks'].progress_event_scope = scope
    old_cb = env['state_callbacks'].run
    def cb(state, **kw):
        if kw.get('force') and mode == 'final_callback_fault':
            events.append('final_callback_attempt')
            raise RuntimeError('forced callback failed')
        return old_cb(state, **kw)
    env['state_callbacks'].run = cb
    def wait():
        events.append('wait_save')
        if mode == 'final_wait_fault':
            raise RuntimeError('final wait failed')
    env['checkpointer'].wait_until_finished = wait
    def log(msg):
        events.append('fatal_logged')
        if mode == 'train_and_logger_fault':
            raise RuntimeError('logger failed')
    env['logger'].exception = log
    env['levanter'].tracker.current_tracker = lambda: types.SimpleNamespace(finish=lambda: events.append('tracker_finish'))

ns.update(install=install, exception_chain=exception_chain)
run = ns['run']
cases = {}
for name in ['clean', 'train_fault', 'train_and_finish_fault', 'train_and_logger_fault',
             'save_fault', 'save_and_scope_finish_fault', 'save_start_fault',
             'final_callback_fault', 'final_wait_fault']:
    cases[name] = run(name, kind='bad_loss' if name.startswith('train') else 'clean',
                      save_fail=name in ('save_fault', 'save_and_scope_finish_fault'))
checks = []
def ck(name, condition):
    if not condition:
        raise RuntimeError(name)
    checks.append({'name': name, 'passed': True})
ck('clean path reaches tracker finish after wait', cases['clean']['error'] is None and cases['clean']['events'][-1]=='tracker_finish')
ck('nonfinite loss remains primary with successful logger and final event', cases['train_fault']['error'][0]['message'].startswith('Non-finite loss') and len(cases['train_fault']['error'])==1)
ck('final event fault becomes primary with loss in context', cases['train_and_finish_fault']['error'][0]['message']=='event:training_finished' and cases['train_and_finish_fault']['error'][1]['message'].startswith('Non-finite loss'))
ck('event dispatch stops before later registered hooks', 'tail:training_finished' not in cases['train_and_finish_fault']['events'])
ck('logger fault becomes primary with loss in context', cases['train_and_logger_fault']['error'][0]['message']=='logger failed' and cases['train_and_logger_fault']['error'][1]['message'].startswith('Non-finite loss'))
ck('checkpoint finished event emitted even after save body raises', cases['save_fault']['error'][0]['message']=='synthetic checkpoint handoff failure' and 'save_finished' in cases['save_fault']['events'])
ck('checkpoint finish event fault replaces save as primary but context remains', [x['message'] for x in cases['save_and_scope_finish_fault']['error']]==['event:save_finished','synthetic checkpoint handoff failure'])
ck('start event failure prevents save body and paired finished event', cases['save_start_fault']['checkpoint_handoffs']==[] and 'save_finished' not in cases['save_start_fault']['events'])
ck('forced callback fault bypasses preceding except and skips final save', 'fatal_logged' not in cases['final_callback_fault']['events'] and len(cases['final_callback_fault']['checkpoint_handoffs'])==1 and 'training_finished' in cases['final_callback_fault']['events'])
ck('final wait fault bypasses preceding except but still emits checkpoint finished', 'fatal_logged' not in cases['final_wait_fault']['events'] and cases['final_wait_fault']['events'].count('save_finished')==2 and cases['final_wait_fault']['error'][0]['message']=='final wait failed')
ck('every exception prevents selected post-loop tracker finish', all('tracker_finish' not in c['events'] for n,c in cases.items() if n!='clean'))
out = {'scope': __doc__, 'checks_passed': len(checks), 'checks': checks, 'cases': cases,
       'source_sha256': {p: hashlib.sha256((R/p).read_bytes()).hexdigest() for p in paths},
       'probe_script_sha256': hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
       'source_revision': '84869ae8c91ffe64e9f761c5bd714542eb1876e0',
       'actual_Hero_secondary_exception': None, 'actual_production_event_hook_failure': None,
       'actual_full_function_cleanup': None, 'actual_checkpoint_commit': None,
       'actual_training_loss_effect': None}
(R/'analysis/exception_provenance_cpu.json').write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n')
print('Exception provenance controls:', len(checks))
