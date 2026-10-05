"""Original document reassembly/filter branches with real local Parquet fixtures.
StoragePath is a local pathlib adapter; upstream tables and sparse paths are
synthetic. This is not historical corpus verification or Zephyr execution.
"""
import ast,dataclasses,hashlib,json,pathlib,tempfile,types
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
R=pathlib.Path(__file__).resolve().parents[1];P=R/'sources/cache_2026_10_05/datakit_store.py'
ns=dict(np=np,pa=pa,pc=pc,pq=pq,StoragePath=pathlib.Path,CHUNK_INDEX_FIELD='chunk_index',_TOKENIZE_BATCH_SIZE=2,dataclasses=dataclasses)
ns['_read_columns']=lambda path,cols:pq.read_table(path,columns=cols)
tree=ast.parse(P.read_text());names={'_load_decon_table','_load_cluster_table','_load_quality_table','_load_verified_duplicates','_load_exact_duplicates','_iter_tokenized_documents','_iter_surviving_docs','_FilterStats','_SurvivingDocument'}
body=[x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name in names]
class Strip(ast.NodeTransformer):
 def visit_FunctionDef(self,node):
  node.returns=None
  for a in node.args.args+node.args.kwonlyargs:a.annotation=None
  return self.generic_visit(node)
 def visit_AnnAssign(self,node):
  node.annotation=ast.Name(id='object',ctx=ast.Load());return node
mod=ast.fix_missing_locations(Strip().visit(ast.Module(body=body,type_ignores=[])));exec(compile(mod,'original-store-helpers','exec'),ns)
checks=[];cases=[]
def check(name,condition):
 assert condition,name;checks.append(name)
def write(path,rows):pq.write_table(pa.Table.from_pylist(rows),path)
def rejects(call):
 try:call();return False
 except RuntimeError:return True
with tempfile.TemporaryDirectory() as tmp:
 d=pathlib.Path(tmp);t=d/'tokens.parquet';write(t,[dict(id='same',chunk_index=0,input_ids=[1,2]),dict(id='same',chunk_index=1,input_ids=[3]),dict(id='same',chunk_index=0,input_ids=[4]),dict(id='last',chunk_index=0,input_ids=[5])]);docs=[(i,v.tolist()) for i,v in ns['_iter_tokenized_documents'](str(t))]
 check('Repeated adjacent IDs remain two documents; chunks cross Arrow batches',docs==[('same',[1,2,3]),('same',[4]),('last',[5])]);cases.append(dict(case='chunk_reassembly',documents=docs))
 bad=d/'bad.parquet';write(bad,[dict(id='x',chunk_index=0,input_ids=[1]),dict(id='x',chunk_index=2,input_ids=[2])]);check('Skipped chunk index is rejected',rejects(lambda:list(ns['_iter_tokenized_documents'](str(bad)))))
 missing=d/'no_chunk.parquet';write(missing,[dict(id='x',input_ids=[1])]);check('Legacy token rows without chunk_index are rejected',rejects(lambda:list(ns['_iter_tokenized_documents'](str(missing)))))
 sparse=str(d/'absent.parquet');check('Missing exact and fuzzy attribute files read as empty sets',ns['_load_exact_duplicates'](sparse)==ns['_load_verified_duplicates'](sparse)==set())
 ids=['a','b','c','d'];write(t,[dict(id=i,chunk_index=0,input_ids=[n]) for n,i in enumerate(ids)])
 paths={k:str(d/(k+'.parquet')) for k in ['decontam','cluster','quality','exact_dedup','dedup']};write(paths['decontam'],[dict(id=i,contaminated=i=='a') for i in ids]);write(paths['cluster'],[dict(id=i,cluster=0) for i in ids]);write(paths['quality'],[dict(id=i,quality_bucket=4) for i in ids]);write(paths['exact_dedup'],[dict(id='b')]);write(paths['dedup'],[dict(id=i,dup_doc=True) for i in ['a','b','c']]);spec=dict(paths,tokenize=str(t),source_name='synthetic',basename='fixture')
 for label,path in [('normal',paths['dedup']),('fuzzy_exempt','')]:
  st=ns['_FilterStats']();out=list(ns['_iter_surviving_docs'](dict(spec,dedup=path),'cluster',stats=st));cases.append(dict(case=label,surviving_tokens=[x.input_ids.tolist() for x in out],counters=st.counters()));check(label+' conserves exclusive filter counters',st.records_in==st.records_out+st.contaminated_dropped+st.exact_duplicate_dropped+st.fuzzy_duplicate_dropped)
 check('Fuzzy exemption preserves exact and contamination filtering',cases[-1]['surviving_tokens']==[[2],[3]] and cases[-1]['counters']['datakit_store/exact_duplicate_dropped']==1)
 check('Overlap attributes are attributed to first rejecting filter',cases[-2]['counters']['datakit_store/exact_duplicate_dropped']==0 and cases[-2]['counters']['datakit_store/fuzzy_duplicate_dropped']==2)
 write(paths['cluster'],[dict(id=i,cluster=0) for i in reversed(ids)]);check('Equal row count with reordered IDs is rejected',rejects(lambda:list(ns['_iter_surviving_docs'](spec,'cluster',stats=ns['_FilterStats']()))))
# Pure source feature construction; no decision-rule execution.
fp=R/'sources/dedup_2026_10_05/fuzzy_verification.py';ft=ast.parse(fp.read_text());f=next(x for x in ft.body if isinstance(x,ast.FunctionDef) and x.name=='prepare_verification_text');env=dict(PreparedVerificationText=lambda **kw:types.SimpleNamespace(**kw));exec(compile(ast.fix_missing_locations(Strip().visit(ast.Module(body=[f],type_ignores=[]))),'original-ngram-preparation','exec'),env)
params=types.SimpleNamespace(ngram_size=3,minimum_distinct_ngram_ratio=.9,maximum_chars_per_token=20)
texts=['[DNA] [Region: coding sequence] '+'a'*200,'[DNA] [Region: coding sequence] '+'t'*200];prepared=[env['prepare_verification_text'](x,params) for x in texts];rawcontain=len(prepared[0].ngrams&prepared[1].ngrams)/len(prepared[0].ngrams);check('Unrelated single-token sequences share 2 of 3 word trigrams',rawcontain==2/3)
policy=R/'sources/dedup_2026_10_05/produce_store_policy.py';pt=ast.parse(policy.read_text());ass=next(x for x in pt.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FUZZY_DEDUP_EXEMPT_SOURCES' for t in x.targets));penv={};exec(compile(ast.fix_missing_locations(ast.Module(body=[ass],type_ignores=[])),'original-exemption-policy','exec'),penv);exempt=sorted(penv['FUZZY_DEDUP_EXEMPT_SOURCES']);check('Policy source contains sixteen registry names',len(exempt)==16)
old=json.loads((R/'sources/dedup_2026_10_05/old_spec.json').read_text());new=json.loads((R/'sources/dedup_2026_10_05/new_spec.json').read_text())
out=dict(checks_passed=len(checks),checks=checks,cases=cases,raw_feature_counterexample=dict(shared_word_trigrams=2,total_word_trigrams=3,containment=rawcontain,under_tokenized=[x.under_tokenized for x in prepared],decision_rule_executed=False),exempt_sources=exempt,spec_identity=dict(old_named_spec_store=old['candidate_store_uri'],new_named_spec_store=new['candidate_store_uri'],old_named_spec_total=sum(old['available_tokens'].values()),new_named_spec_total=sum(new['available_tokens'].values()),old_run_store='s3://marin-us-east-02a/marin/datakit/store_81e7e39a',old_spec_matches_old_run=False),source_sha256={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [P,fp,policy]},actual_corpus=None,historical_execution_sha=None,scope='local real Parquet; original source helpers; synthetic fixtures; pure feature counterexample, no fuzzy decision executed')
(R/'analysis/dedup_probe.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('Dedup/document checks:',len(checks))
