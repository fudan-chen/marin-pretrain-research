"""Original routing block and portable combine with synthetic expert outputs/masks.
No capacity planner, collective, expert MLP, optimizer or Hero data execution.
"""
import ast, hashlib, json, pathlib, types
import jax
import jax.numpy as jnp
import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
M = R / 'sources/engineering_v77_2026_10_07/current_grug_moe.py'
E = R / 'sources/engineering_v77_2026_10_07/current_ep_ragged_all_to_all.py'
cls = next(n for n in ast.parse(M.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'QBRoutedMoE')
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__call__')
block = next(n for n in method.body if isinstance(n, ast.With) and ast.unparse(n.items[0].context_expr) == "jax.named_scope('moe_route')")
fn = ast.parse('def route(self,x,router,router_bias):\n pass').body[0]
fn.body = [block, ast.parse('return selected_experts,combine_weights').body[0]]
ns = {'jax': jax, 'jnp': jnp, 'reshard': lambda x, spec: x, 'P': lambda *args: args}
exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(M), 'exec'), ns)
combine = next(n for n in ast.parse(E.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == '_unpermute_from_global_expert')
ns.update(sonic_gather_sum_available=lambda: False, _sort_activations=lambda x, indices: x[indices])
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), combine], type_ignores=[])), str(E), 'exec'), ns)
entry = next(n for n in ast.parse(E.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == '_moe_mlp_ep_ragged_a2a_local')
mask_stmt = next(n for n in entry.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'weights' for t in n.targets))
mask_fn = ast.parse('def mask(accepted,combine_weights_local):\n pass').body[0]
mask_fn.body = [mask_stmt, ast.parse('return weights').body[0]]
exec(compile(ast.fix_missing_locations(ast.Module(body=[mask_fn], type_ignores=[])), str(E), 'exec'), ns)
recipe = types.SimpleNamespace(num_experts_per_token=2, routing_renorm_sum=2.5)
x = jnp.ones((1, 1), jnp.float32)
router = jnp.array([[2., 1., 0.]], jnp.float32)
bias = jnp.zeros(3, jnp.float32)

def evaluate(rr, mask, expert_outputs, renormalize=False):
    selected, w = ns['route'](recipe, x, rr, bias)
    used = ns['mask'](mask, w)
    if renormalize:
        # Deliberately different comparison function, not an upstream repair.
        used = used * (recipe.routing_renorm_sum / jnp.maximum(jnp.sum(used, axis=-1, keepdims=True), 1e-9))
    y = ns['_unpermute_from_global_expert'](expert_outputs, jnp.array([0, 1]), used, mask, tokens_per_shard=1, topk=2)
    return jnp.sum(y)

def main():
    checks = []
    def check(name, condition):
        assert condition, name
        checks.append(name)
    check('CPU backend', jax.default_backend() == 'cpu')
    ids, weights = ns['route'](recipe, x, router, bias)
    check('Fixed selected experts', ids.tolist() == [[0, 1]])
    cases = {}
    for name, accept in [('both', [True, True]), ('keep_first', [True, False]), ('keep_second', [False, True]), ('neither', [False, False])]:
        mask = jnp.array([accept]); y = jnp.array([[3.], [1.]])
        value, grad = jax.value_and_grad(evaluate)(router, mask, y)
        poisoned = jnp.where(mask.reshape(-1, 1), y, jnp.nan)
        pv, pg = jax.value_and_grad(evaluate)(router, mask, poisoned)
        check(name+' portable masking excludes dropped NaN rows from output and router gradient', bool(jnp.array_equal(value, pv)) and bool(jnp.array_equal(grad, pg)))
        used = ns['mask'](mask, weights)
        dw = jax.grad(lambda w: jnp.sum(ns['_unpermute_from_global_expert'](y, jnp.array([0,1]), ns['mask'](mask,w), mask, tokens_per_shard=1,topk=2)))(weights)
        cases[name] = {'accepted': accept, 'pre_clip_weights': weights.tolist(), 'retained_weight_mass': float(used.sum()), 'output': float(value), 'router_logit_gradient': grad.tolist(), 'pre_clip_weight_gradient': dw.tolist()}
        # Independent closed-form derivative including original epsilon.
        s = np.asarray(jax.nn.sigmoid(router))[0, :2]; sy = np.array(accept) * np.array([3.,1.]); denom = s.sum()+1e-9
        expected = 2.5*s*(1-s)*(sy*denom-(s*sy).sum())/denom**2
        check(name+' gradient agrees with independent selected sigmoid quotient', np.allclose(np.asarray(grad)[0,:2],expected,rtol=1e-5,atol=1e-6) and float(grad[0,2])==0)
    a = cases['keep_first']; b = cases['keep_second']
    check('Dropped assignment weight gradient zero but dropped selected logit gradient nonzero', a['pre_clip_weight_gradient'][0][1]==0 and abs(a['router_logit_gradient'][0][1])>0.1)
    check('Identical assignment drop counts retain different weight mass and outputs', a['retained_weight_mass']!=b['retained_weight_mass'] and a['output']!=b['output'])
    check('All dropped produces finite zero output and zero router gradient', cases['neither']['output']==0 and np.all(np.asarray(cases['neither']['router_logit_gradient'])==0))
    mask = jnp.array([[True,False]]); y=jnp.array([[3.],[1.]])
    cv,cg=jax.value_and_grad(lambda r:evaluate(r,mask,y,True))(router)
    check('Post-clip renormalization is a different forward and gradient function', abs(float(cv)-7.5)<1e-6 and np.max(np.abs(np.asarray(cg)))<1e-6 and abs(float(cv)-a['output'])>1)
    check('JIT agrees in keep-first control', bool(jnp.array_equal(jax.jit(evaluate)(router,mask,y),jnp.array(a['output']))))
    out={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'runtime':{'jax':jax.__version__,'backend':jax.default_backend()},'cases':cases,'post_clip_renormalized_control':{'output':float(cv),'router_logit_gradient':cg.tolist(),'upstream_fix':False},'extracted_mask_statement':ast.unparse(mask_stmt),'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [M,E]},'explicit_substitutions':['identity reshard and partition-spec placeholder','manual acceptance masks, fixed expert outputs 3 and 1','portable combine only; sonic branch forced unavailable','sort adapter is ordinary JAX indexing'],'actual_Hero_domain_effect':None,'actual_GPU_execution':None,'actual_optimizer_update':None}
    (R/'analysis/post_clip_router_cpu.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'checks_passed':len(checks),'cases':cases,'comparison':out['post_clip_renormalized_control']},indent=2))

if __name__=='__main__': main()
