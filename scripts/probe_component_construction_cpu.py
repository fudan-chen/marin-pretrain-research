"""Original component dataclasses, config guards/build methods and mixture/concat classes.
Cache-to-dataset adapter returns finite identity children; no actual IO/tokenization.
"""
import ast, asyncio, dataclasses, hashlib, json, pathlib, runpy, types

R=pathlib.Path(__file__).resolve().parents[1]
P=R/'sources/boundaries_2026_10_05/datasets.py'
M=R/'sources/integer_exposure_2026_10_07/mixture.py'
previous=runpy.run_path(str(R/'scripts/probe_mixture_support_cpu.py'))
ns=previous['ns'];Identity=previous['Identity']
ns.update(dataclass=dataclasses.dataclass,field=dataclasses.field,
          DatasetComponentBase=type('ComponentBase',(),{}), TextLmDatasetFormat=lambda:None)
component_nodes=[]
for n in ast.parse(P.read_text()).body:
    if isinstance(n,ast.ClassDef) and n.name in ['DatasetComponent','DirectDatasetComponent','ConcatDatasetComponent']:
        n.decorator_list=[x for x in n.decorator_list if not isinstance(x,ast.Call) or not isinstance(x.func,ast.Attribute)]
        component_nodes.append(n)
cl=next(n for n in ast.parse(P.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='LmDataConfig')
methods=[n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name in ['__post_init__','_has_nonzero_weight','_cache_items','build_token_datasets']]
concat=next(n for n in ast.parse(M.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='ConcatDataset')
nodes=component_nodes+methods+[concat]
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])),str(P),'exec'),ns)
Config=type('Config',(object,),{n.name:ns[n.name] for n in methods})
def config(components,weights):
    c=Config();c.components=components;c.train_weights=weights
    c.max_train_batches=c.num_validation_sequences=c.experiment_budget=c.target_budget=None
    c.the_tokenizer=types.SimpleNamespace(eos_token_id=99);c.block_cross_document_attention=True
    c.__post_init__();return c
calls=[]
def dataset_for_component(component,Pos,cache,**kw):
    calls.append({'cache':cache,'eos_id':kw['eos_id'],'attention':kw['block_cross_document_attention']})
    return Identity(cache,3)
ns['dataset_for_component']=dataset_for_component

async def main():
    checks=[];cases={}
    def ck(n,c):assert c,n;checks.append(n)
    def attempt(c,caches,split):
        try:return {'datasets':c.build_token_datasets(caches,None,split=split),'error':None}
        except Exception as e:return {'datasets':None,'error':type(e).__name__,'message':str(e)}
    Cached=ns['DatasetComponent'];Direct=ns['DirectDatasetComponent'];Concat=ns['ConcatDatasetComponent']
    c=config({'A':Cached(),'B':Cached()},{'A':.5,'B':.5})
    missing=attempt(c,{'A':'cacheA'},'train')
    ck('Active ordinary train child missing cache raises',missing['error']=='ValueError' and 'B' in missing['message'])
    val=attempt(c,{'A':'cacheA'},'validation')
    ck('Ordinary missing validation cache omitted',list(val['datasets'])==['A'])
    future=config({'A':Cached(),'B':Cached()},[(0,{'A':1}),(20,{'B':1})])
    ck('Future-active child included in initial train cache inventory',[n for n,_ in future._cache_items('train')]==['A','B'])
    ck('Future-active child missing cache fails before phase switch',attempt(future,{'A':'cacheA'},'train')['error']=='ValueError')
    inactive=config({'A':Cached(),'B':Cached()},{'A':1,'B':0})
    ck('Permanently zero-weight train child omitted',list(attempt(inactive,{'A':'cacheA'},'train')['datasets'])==['A'])
    direct=config({'A':Direct({'validation':Identity('direct',3)})},{'A':1})
    ck('Direct missing train split rejected',attempt(direct,{},'train')['error']=='ValueError')
    ck('Direct missing validation split omitted',list(attempt(config({'A':Direct({'train':Identity('direct',3)})},{'A':1}),{},'validation')['datasets'])==[])
    cat=config({'A':Concat({'x':Cached(),'y':Cached()})},{'A':1})
    ck('Concat missing train child rejected',attempt(cat,{'A/x':'x'},'train')['error']=='ValueError')
    partial=attempt(cat,{'A/x':'x'},'validation')['datasets']['A']
    full=attempt(cat,{'A/x':'x','A/y':'y'},'validation')['datasets']['A']
    ck('Validation concat retains only available children',await partial.async_len()==3 and await full.async_len()==6)
    partial_stream=await partial.get_batch([0,1,2]);full_stream=await full.get_batch(list(range(6)))
    ck('Original concat reads shortened actual identity panel',partial_stream==full_stream[:3] and full_stream[3:]==['y:0','y:1','y:2'])
    empty=config({'A':Cached(),'B':Concat({})},{'A':.4,'B':.6})
    ds=empty.build_token_datasets({'A':'cacheA'},None,split='train')
    ck('Original config accepts empty concat and builder omits it',list(ds)==['A'])
    ck('Empty concat has no cache items',[n for n,_ in empty._cache_items('train')]==['A'])
    mix=ns['MixtureDataset'](ds,empty.train_weights,10,key=7)
    ck('Valid config empty concat reaches shifted all-A quota',mix.dataset_index==['A'] and mix._counts_per_block_per_stage[0].tolist()==[10])
    stream=await mix.get_batch(list(range(10)))
    ck('Original mixture reads only surviving child after construction',all(x.startswith('cacheA:') for x in stream))
    cases={'ordinary_train_missing':{k:v for k,v in missing.items() if k!='datasets'},
           'ordinary_validation_names':list(val['datasets']),
           'partial_concat_identity':partial_stream,'full_concat_identity':full_stream,
           'empty_concat_declared_weights':empty.train_weights,'empty_concat_actual_names':list(ds),
           'empty_concat_integer_quota':[10],'empty_concat_identity':stream,'adapter_calls':calls}
    meta=R/'sources/live_2026_10_07/meta.json';components=json.loads(meta.read_text())['config']['data']['value']['components']
    concat_names=[k for k,v in components.items() if 'children' in v or v.get('type')=='concat']
    ck('Archived components have no declared concat shapes',len(components)==223 and not concat_names)
    files=[P,M,meta,R/'scripts/probe_mixture_support_cpu.py',R/'scripts/probe_mixture_identity_cpu.py']
    out={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'runtime':previous['adapters']['platform'].python_version(),
         'cases':cases,'archived_components':{'count':len(components),'concat_shapes':concat_names},
         'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
         'adapters':'Base registry and format-default placeholder; dataset_for_component returns finite identity stores. Original component dataclass fields, guard/build bodies, concat/mixture class bodies executed.',
         'actual_Hero_missing_cache':None,'actual_Hero_empty_concat':None,'actual_token_stream':None,'actual_model_loss':None}
    (R/'analysis/component_construction_cpu.json').write_text(json.dumps(out,indent=2)+'\n')
    print('Original component construction CPU checks:',len(checks),'passed')
if __name__=='__main__':asyncio.run(main())
