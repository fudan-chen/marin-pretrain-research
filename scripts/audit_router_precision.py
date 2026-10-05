"""Static fork source audit and transparent BF16 output-rounding illustration.
No original model, GPU dot, gradients, fixture or checkpoint is executed.
"""
import ast,hashlib,json,pathlib
import numpy as np
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/router_precision_2026_10_05';m=D/'model.py';tree=ast.parse(m.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MoEMLP');call=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__call__');branch=next(n for n in call.body if isinstance(n,ast.If) and 'self.cfg.router_dot_precision' in ast.unparse(n.test));branches=[]
while isinstance(branch,ast.If):
 expr=next(n for n in branch.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='router_logits' for t in n.targets));branches.append({'condition':ast.unparse(branch.test),'expression':ast.unparse(expr.value),'line':expr.lineno});branch=branch.orelse[0]
assert len(branches)==3
# RNE BF16 output quantization only, represented in float32 storage.
def bf16_output_round(x):
 x=np.array(x,np.float32);bits=x.view(np.uint32);bits=(bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000);return bits.view(np.float32)
scores=np.array([1.001,1.002],np.float32);rounded=bf16_output_round(scores);illustration={'score_before_output_rounding':scores.tolist(),'bf16_rounded_output_represented_in_fp32':rounded.tolist(),'fp32_cast_after_rounding':rounded.astype(np.float32).tolist(),'numpy_argmax_before':int(np.argmax(scores)),'numpy_argmax_after':int(np.argmax(rounded)),'tie_rule':'NumPy first-maximum for this toy only; not a measured JAX top_k result'}
assert rounded.tolist()==[1,1] and np.argmax(scores)!=np.argmax(rounded)
launch=(D/'launch_diagnostics.py').read_text();lines=launch.splitlines();schedule_evidence=[{'line':i,'text':s.strip()} for i,s in enumerate(lines,1) if ('total_schedule_steps =' in s or 'num_train_steps=total_schedule_steps' in s or 'batch_size=optimizer_batch_size' in s)]
out={'fork_repository':'yonromai/marin','fork_revision':'a00cb77a491f4777a2c66edce54f47ff7b255c40','status':'experimental_not_live_deployment','router_branches':branches,'output_rounding_illustration':illustration,'schedule_source_evidence':schedule_evidence,'scope':'static AST/text inspection and NumPy BF16 output quantization illustration only','actual_gpu_dot':None,'actual_hlo_verification':None,'actual_fixture_loaded':None,'actual_training_result':None,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in D.glob('*.py')}};(R/'analysis/router_precision_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('Static router branches:',len(branches),'schedule source entries:',len(schedule_evidence))
