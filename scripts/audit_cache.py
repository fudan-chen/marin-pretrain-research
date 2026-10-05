"""Audit archived configurations and execute the original cache admission branch.
No cloud cache, tokenizer execution, JAX or historical run is reproduced.
"""
import ast,hashlib,json,pathlib,re,types
R=pathlib.Path(__file__).resolve().parents[1]; S=R/'sources'; D=S/'cache_2026_10_05'
rows=[]
for p in sorted((S/'scale_2026_10_05').glob('config_*.json')):
 run=json.loads(p.read_text())['data']['project']['run']; config={k:v['value'] for k,v in json.loads(run['config']).items() if 'value' in v}; data=config['data']
 for name,c in data['components'].items():
  if not re.fullmatch(r'c\d{2}q\d',name):continue
  rows.append(dict(run=p.stem[7:],component=name,cache_dir=c['cache_dir'],flat_cache=c['flat_cache'],format=c['format'],pack=c.get('pack'),source=c.get('source'),component_split=c.get('split'),tokenizer=data['tokenizer'],block_cross_document_attention=data['block_cross_document_attention']))
assert len(rows)==1200
assert all(x['flat_cache'] is True and x['format']=={'text_key':'text'} and x['pack'] is None and x['source'] is None for x in rows)
assert all(x['tokenizer']=='marin-community/marin-tokenizer' and x['block_cross_document_attention'] is True for x in rows)
# Source method executes with a deliberately supplied comparison result and fake constructor.
# This checks admission control, not DeepDiff or actual filesystem/token arrays.
tree=ast.parse((D/'tree_cache.py').read_text()); cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='TreeCache'); f=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='load_from_ledger'); f.decorator_list=[]; f.returns=None
for arg in f.args.args:arg.annotation=None
module=ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[])); warnings=[]
ns={'logger':types.SimpleNamespace(warning=warnings.append),'TreeCache':lambda *args:('opened',args)};exec(compile(module,'original-load-from-ledger','exec'),ns)
results=[]
for label,diff,finished in [('matching',{},True),('mismatch',{'tokenizer':'different'},True),('empty_metadata',{'preprocessor_metadata':'missing'},True),('unfinished',{},False)]:
 warnings.clear();ledger=types.SimpleNamespace(metadata=types.SimpleNamespace(compare_to=lambda other,d=diff:d),is_finished=finished)
 try:result=ns['load_from_ledger']('fake-cache',{},ledger,object());status=result[0]
 except FileNotFoundError:status='rejected_unfinished'
 results.append(dict(case=label,status=status,warnings=list(warnings)))
assert [x['status'] for x in results]==['opened','opened','opened','rejected_unfinished']
assert [len(x['warnings']) for x in results]==[0,1,1,0]
tok=json.loads((D/'tokenizer_config.json').read_text());markers={v['content']:int(k) for k,v in tok['added_tokens_decoder'].items() if v['content'] in ['<|begin_of_text|>','<|end_of_text|>','<|eot_id|>','<|finetune_right_pad_id|>']}
out=dict(configurations=6,component_instances=len(rows),unique_components=len({x['component'] for x in rows}),cache_roots=sorted({x['cache_dir'].rsplit('/cluster=',1)[0] for x in rows}),component_split_counts={k:sum(x['component_split']==k for x in rows) for k in sorted({x['component_split'] for x in rows})},configuration_sha256={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((S/'scale_2026_10_05').glob('config_*.json'))},components=rows,admission_branch_cases=results,current_tokenizer_markers=markers,observed_hf_revision=json.loads((R/'analysis/cache_acquisition.json').read_text())['hf_observed_revision'],historical_tokenizer_revision=None,actual_cache_ledger=None,actual_token_arrays=None,scope='original admission branch with stubbed comparison and constructor; configuration census; current public tokenizer metadata only')
(R/'analysis/cache_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['components','configuration_sha256']},ensure_ascii=False,indent=2))
