"""Independently recheck frozen issue comparisons and PR file/patch coverage.
Coverage is source inspection, not execution, correctness or deployment verification.
"""
import hashlib,json,pathlib
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/engineering_current_2026_10_07';OLD=R/'sources/engineering_2026_10_05'
def read(p):return json.loads(p.read_text())
checks=[]
def check(name,x):assert x,name;checks.append(name)
issues=[]
for n in [8435,8506,8870]:
 a=read(OLD/f'issue_{n}.json');b=read(D/f'issue_{n}.json');old={x['id']:x['body'] for x in read(OLD/f'issue_{n}_comments.json')};new={x['id']:x['body'] for x in read(D/f'comments_{n}.json')}
 check('Frozen issue full body and comment content unchanged '+str(n),a['body']==b['body'] and old==new and b['comments']==len(new))
 issues.append({'issue':n,'body_equal':True,'comment_map_equal':True,'comments':len(new)})
prs=[]
for n in [9832,9833]:
 p=read(D/f'pull_{n}.json');files=read(D/f'pull_{n}_files.json');check('PR complete file list '+str(n),p['changed_files']==len(files));rows=[]
 for f in files:
  patch=f.get('patch');assert patch is not None,f['filename'];lines=patch.splitlines();adds=sum(x.startswith('+') for x in lines);deletes=sum(x.startswith('-') for x in lines)
  check('PR patch changed-line coverage '+str(n)+' '+f['filename'],adds==f['additions'] and deletes==f['deletions'])
  rows.append({'file':f['filename'],'added_patch_lines':adds,'removed_patch_lines':deletes,'metadata_additions':f['additions'],'metadata_deletions':f['deletions']})
 prs.append({'number':n,'head_sha':p['head']['sha'],'base_sha':p['base']['sha'],'state':p['state'],'merged':p['merged'],'files':rows})
records=read(R/'analysis/engineering_v56_acquisition.json')['records']
check('All eleven acquired API bytes match provenance',len(records)==11 and all(hashlib.sha256((R/'sources'/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in records))
out={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'issues':issues,'PRs':prs,'source_sha256':{str((R/'sources'/x['file']).relative_to(R)):x['sha256'] for x in records},'actual_GPU_tests_executed_by_us':None,'actual_deployment':None}
(R/'analysis/moe_proposal_source_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('Independent frozen source audits:',len(checks),'passed')
