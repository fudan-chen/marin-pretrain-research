"""Run original log-name/hierarchy helpers; inspect accumulator units statically.
No evaluator/JAX execution, real token counts, or historical code binding.
"""
import ast,json,pathlib,logging,types,hashlib,dataclasses
import numpy as np
R=pathlib.Path(__file__).resolve().parents[1];p=R/'sources/scale_2026_10_05/eval.py';tree=ast.parse(p.read_text());checks=[]
ns={'logger':logging.getLogger('eval-metric-audit')}
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('construct_log_dict','_join_prefix')]
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TaggedEvaluator')
hierarchy=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_construct_tag_hierarchy');nodes.append(hierarchy)
module=ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[]));exec(compile(module,str(p),'exec'),ns)
def check(name,condition):
 assert condition,name
 checks.append(name)
tags={'paloma/a':0,'paloma/b':1};ev=types.SimpleNamespace(tokenizer=object(),dataset=types.SimpleNamespace(tag_to_index=tags))
r=types.SimpleNamespace(micro_avg_loss=101.,macro_avg_loss=102.,total_eval_loading_time=0.,tag_micro_losses={'paloma':103.,'paloma/a':104.,'paloma/b':105.},tag_macro_losses={'paloma':106.,'paloma/a':999.},micro_bpb=201.,macro_bpb=202.,tag_micro_bpb={'paloma':203.,'paloma/a':204.,'paloma/b':205.},tag_macro_bpb={'paloma':206.})
out=ns['construct_log_dict'](ev,r,1.,'eval_dropless')
check('parent bpb is tag_micro_bpb',out['eval_dropless/paloma/bpb']==203.)
check('parent macro_bpb is separate tag_macro_bpb',out['eval_dropless/paloma/macro_bpb']==206.)
check('parent CE uses explicit micro_loss and macro_loss',out['eval_dropless/paloma/micro_loss']==103. and out['eval_dropless/paloma/macro_loss']==106.)
check('leaf CE loss and BPB retain leaf values',out['eval_dropless/paloma/a/loss']==104. and out['eval_dropless/paloma/a/bpb']==204.)
check('leaf macro_loss is not redundantly logged','eval_dropless/paloma/a/macro_loss' not in out)
check('root aggregate remains distinct',out['eval_dropless/bpb']==201. and out['eval_dropless/macro_bpb']==202.)
h=ns['_construct_tag_hierarchy'](ev)
check('parent maps to two leaf indices',h=={'paloma':[0,1]})
ev.tokenizer=None;without=ns['construct_log_dict'](ev,r,1.,'eval_dropless')
check('no tokenizer omits BPB only',not any(k.endswith('bpb') for k in without) and without['eval_dropless/paloma/micro_loss']==103.)
# AST records establish dataflow, not the actual distributed values.
expressions={}
for n in ast.walk(cls):
 if isinstance(n,ast.Assign):
  for target in n.targets:
   if isinstance(target,ast.Name) and target.id in ('bpb_per_tag_mean','total_bytes_per_tag_cpu','tag_micro_bpb','macro_avg_bpb'):
    expressions[target.id]={'line':n.lineno,'expression':ast.unparse(n.value)}
check('BPB running total accumulates token weights',expressions['bpb_per_tag_mean']['expression']=='state.bpb_per_tag.add(bpb_per_tag, this_weights_per_tag)')
check('variable named bytes total reads BPB running total',expressions['total_bytes_per_tag_cpu']['expression']=='np.array(state.bpb_per_tag.total)')
# Original RunningMean class, with eqx.Module/dataclass and hax.where replaced locally.
stat_path=R/'sources/eval_metrics_2026_10_05/stat_utils.py'
stat_tree=ast.parse(stat_path.read_text());rm=next(n for n in stat_tree.body if isinstance(n,ast.ClassDef) and n.name=='RunningMean')
stat_ns={'eqx':types.SimpleNamespace(Module=object),'hax':types.SimpleNamespace(where=np.where)}
stat_module=ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),rm],type_ignores=[]));exec(compile(stat_module,str(stat_path),'exec'),stat_ns)
RM=dataclasses.dataclass(stat_ns['RunningMean']);stat_ns['RunningMean']=RM
state=RM.zeros_like(np.array(0.))
for value,total in [(.1,2.),(1.,6.)]:state=state.add(np.array(value),np.array(total))
check('original RunningMean total is sum of passed token weights',float(state.total)==8.)
check('original RunningMean keeps token-weighted batch ratios',abs(float(state.mean)-.775)<1e-12)
byte_weighted=RM.zeros_like(np.array(0.))
for value,total in [(.1,20.),(1.,6.)]:byte_weighted=byte_weighted.add(np.array(value),np.array(total))
check('byte weighting recovers global NLL over bytes',abs(float(byte_weighted.mean)-8/26)<1e-12)
with np.errstate(divide='ignore',invalid='ignore'):
 zero=RM.zeros_like(np.array(0.)).add(np.array(99.),np.array(0.))
check('zero-total update preserves zero mean and total',float(zero.mean)==float(zero.total)==0.)
example={'scope':'artificial two-batch ratio example, not observed Hero data','N':[2,6],'T':[2,6],'B':[20,6],'token_weighted_ratio':float(state.mean),'token_weighted_total':float(state.total),'byte_weighted_ratio':float(byte_weighted.mean),'byte_weighted_total':float(byte_weighted.total),'log2e_not_applied':True}
x={'scope':'original log/hierarchy helpers with fabricated EvalResult and original RunningMean with NumPy substitutes; evaluator accumulator dataflow inspected by AST','checks':checks,'checks_passed':len(checks),'function_lines':{n.name:[n.lineno,n.end_lineno] for n in nodes},'log_mapping_example':out,'hierarchy_example':h,'static_expressions':expressions,'running_mean_example':example,'running_mean_substitutes':['eqx.Module replaced by object plus dataclass initializer','hax.where replaced by numpy.where'],'running_mean_lines':[rm.lineno,rm.end_lineno],'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest(),str(stat_path.relative_to(R)):hashlib.sha256(stat_path.read_bytes()).hexdigest()},'actual_evaluator_execution':None,'actual_distributed_reduce':None,'actual_token_byte_counts':None,'historical_execution_sha':None}
(R/'analysis/eval_metric_audit.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'original_helper_checks':len(checks),'scope':x['scope']},indent=2))
