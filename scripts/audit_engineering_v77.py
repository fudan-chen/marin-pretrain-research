"""Offline audit of a separately acquired public engineering refresh.
Computational AST equality excludes docstrings/comments; no kernel execution.
"""
import ast,hashlib,json,pathlib
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/engineering_v77_2026_10_07';OLD=R/'sources/engineering_current_2026_10_07'
class StripDocs(ast.NodeTransformer):
 def trim(self,n):
  n=self.generic_visit(n)
  if n.body and isinstance(n.body[0],ast.Expr) and isinstance(n.body[0].value,ast.Constant) and isinstance(n.body[0].value.value,str):n.body=n.body[1:]
  return n
 visit_Module=trim;visit_ClassDef=trim;visit_FunctionDef=trim;visit_AsyncFunctionDef=trim
def read(p):return json.loads(p.read_text())
def main():
 checks=[];comparisons=[]
 def check(n,c):assert c,n;checks.append(n)
 for n in [8435,8506,8870]:
  before=read(OLD/f'issue_{n}.json');after=read(D/f'issue_{n}.json');old={x['id']:x for x in read(OLD/f'comments_{n}.json')};new={x['id']:x for x in read(D/f'comments_{n}.json')}
  check('Complete comments '+str(n),after['comments']==len(new))
  comparisons.append({'issue':n,'body_changed':before['body']!=after['body'],'comments':len(new),'added_ids':sorted(new.keys()-old.keys()),'removed_ids':sorted(old.keys()-new.keys()),'changed_body_ids':sorted(i for i in new.keys()&old.keys() if new[i]['body']!=old[i]['body'])})
 new_comment=next(x for x in read(D/'comments_8506.json') if x['id']==6025364160)
 check('New triage explicitly preserves unknown initiating condition',new_comment['user']['type']=='Bot' and 'remains unconfirmed' in new_comment['body'] and 'NoExecute' in new_comment['body'])
 pulls=[]
 for n in [9708,9832,9833]:
  before=read(OLD/f'pull_{n}.json');after=read(D/f'pull_{n}.json');pulls.append({'pull':n,'state':after['state'],'merged':after['merged'],'merged_at':after['merged_at'],'old_head':before['head']['sha'],'new_head':after['head']['sha'],'head_changed':before['head']['sha']!=after['head']['sha'],'body_changed':before['body']!=after['body']})
 check('Closed umbrella is not a merged production fix',pulls[0]['state']=='closed' and pulls[0]['merged'] is False and pulls[0]['merged_at'] is None)
 check('Both split proposals remain unmerged',all(p['state']=='open' and p['merged'] is False for p in pulls[1:]))
 compare=read(D/'compare_9833_heads.json');files=read(D/'pull_9833_files.json')
 check('Full revised PR file list covers metadata',len(files)==read(D/'pull_9833.json')['changed_files']==10)
 expected={'lib/levanter/src/levanter/grug/_moe/ep_ragged_all_to_all.py','lib/levanter/src/levanter/grug/grug_moe.py'}
 check('Head delta is one commit and exactly two files',compare['status']=='ahead' and compare['total_commits']==1 and {f['filename'] for f in compare['files']}==expected)
 ast_comparisons=[]
 for name in ['ep_ragged_all_to_all.py','grug_moe.py']:
  prior=D/('prior_'+name);current=D/('current_'+name)
  equal=ast.dump(StripDocs().visit(ast.parse(prior.read_text())))==ast.dump(StripDocs().visit(ast.parse(current.read_text())))
  check('Computational AST unchanged '+name,equal)
  ast_comparisons.append({'file':name,'computational_ast_equal':equal,'prior_sha256':hashlib.sha256(prior.read_bytes()).hexdigest(),'current_sha256':hashlib.sha256(current.read_bytes()).hexdigest()})
 code=ast.parse((D/'current_ep_ragged_all_to_all.py').read_text());portable=next(c for c in code.body if isinstance(c,ast.ClassDef) and c.name=='_RaggedDotExpertMlp')
 methods={f.name:f for f in portable.body if isinstance(f,ast.FunctionDef)}
 forward_returns=[ast.unparse(x.value) for x in ast.walk(methods['forward']) if isinstance(x,ast.Return)]
 backward=ast.unparse(methods['backward'])
 check('Portable residual retains output for output-scale row dot',any('active_group_sizes, out)' in x for x in forward_returns) and 'out.astype(jnp.float32) * cotangent.astype(jnp.float32)' in backward)
 acq=read(R/'analysis/engineering_v77_acquisition.json')
 check('Sixteen acquisitions bound to exact bytes',len(acq['records'])==16 and all(hashlib.sha256((R/'sources'/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in acq['records']))
 o={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'comparisons':comparisons,'pulls':pulls,'ast_comparisons':ast_comparisons,'portable_forward_returns':forward_returns,'new_comment':{'id':new_comment['id'],'url':new_comment['html_url'],'author':new_comment['user']['login'],'author_type':new_comment['user']['type'],'created_utc':new_comment['created_at'],'reported_task':16,'reported_node':'s14fys64','reported_training_step':215756,'underlying_taint_condition_confirmed':False},'actual_incident_logs_read':None,'actual_GPU_test':None,'actual_production_deployment':None,'actual_retry_cost':None}
 (R/'analysis/engineering_current_v77.json').write_text(json.dumps(o,indent=2)+'\n');print('Engineering refresh audit checks:',len(checks))
if __name__=='__main__':main()
