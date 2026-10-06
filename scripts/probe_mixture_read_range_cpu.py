"""Current original MixtureDataset constructor and read paths on real JAX CPU.
Finite identity children only; no real token data, loader or production cursor.
"""
import asyncio,hashlib,json,pathlib,runpy,warnings
R=pathlib.Path(__file__).resolve().parents[1]
ad=runpy.run_path(str(R/'scripts/probe_mixture_support_cpu.py'))
Mix=ad['Mix'];Identity=ad['Identity'];np=ad['ns']['np'];jax=ad['ns']['jax']
LIMIT=2**31-1;K=8;N=(LIMIT//K+1)*K
checks=[];cases={}
def ck(n,c):assert c,n;checks.append(n)
def make(stages):return Mix({'A':Identity('A',1009)},stages,K,key=7)
def wide_identity(m,index):
    block=index//m.block_size;stage=m._get_stage_for_block(block)
    packed=int(m._get_block(block)[index%m.block_size]);dsid=packed>>16;local=packed&65535
    prefix=sum(int(m._counts_per_block_per_stage[i][dsid])*((m.weight_stages[i+1][0]-m.weight_stages[i][0])//m.block_size) for i in range(stage))
    offset=(block*m.block_size-m.weight_stages[stage][0])//m.block_size
    logical=prefix+offset*int(m._counts_per_block_per_stage[stage][dsid])+local
    return {'dataset':m.dataset_index[dsid],'logical_index':logical,
            'expected_identity':m.dataset_index[dsid]+':'+str(logical%1009)}
async def main():
    one=make({'A':1})
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always');single=await one.getitem_async(N)
        batch=(await one.get_batch([N]))[0]
    reference=wide_identity(one,N)
    ck('Same original source position yields different single/batch identities',single!=batch)
    ck('Batch path matches independent wide arithmetic in one-stage control',batch==reference['expected_identity'])
    ck('Single path finite restart hides current-stage overflow',single!=reference['expected_identity'] and single.startswith('A:'))
    cases['one_stage']={'index':N,'single':single,'batch':batch,'reference':reference,'warnings':[str(x.message) for x in ws]}
    phased=make([(0,{'A':1}),(N,{'A':1})])
    ss=await phased.getitem_async(N);bb=(await phased.get_batch([N]))[0];ref=wide_identity(phased,N)
    ck('Original constructor prefix has wrapped before read',int(phased._counts_after_stage[0][0])==-2**31)
    ck('Both read APIs agree but disagree with independent wide reference',ss==bb and ss!=ref['expected_identity'])
    cases['staged']={'index':N,'prefix':int(phased._counts_after_stage[0][0]),'single':ss,'batch':bb,'reference':ref}
    error=None
    try:make([(0,{'A':1}),((2**31)*K,{'A':1})])
    except Exception as e:error={'type':type(e).__name__,'message':str(e)}
    ck('Too-large Python block multiplier rejected in this NumPy runtime',error is not None and error['type']=='OverflowError')
    cases['too_large_multiplier']=error

    meta=R/'sources/live_2026_10_07/meta.json';config=json.loads(meta.read_text())['config'];data=config['data']['value'];trainer=config['trainer']['value']['trainer']
    weights=data['train_weights'];B=11264;starts=[int(step)*B for step,_ in weights]
    # Recorded constant batch declaration, not inference about actual historical loader.
    ck('Archived constant batch horizon and block match control',data['mixture_block_size']==49152 and trainer['train_batch_size']==B and trainer['num_train_steps']==390251)
    stages=[(start,w) for start,(_,w) in zip(starts,weights)]
    with warnings.catch_warnings(record=True) as hw:
        warnings.simplefilter('always');hero=Mix({n:Identity(n,1009) for n in data['components']},stages,49152,key=7)
    ck('Original constructor retains 200 positive support cells',len(hero.dataset_index)==200)
    wide=[0]*len(hero.dataset_index);prefix_rows=[]
    for i in range(len(stages)-1):
        blocks=(starts[i+1]-starts[i])//49152
        wide=[a+int(q)*blocks for a,q in zip(wide,hero._counts_per_block_per_stage[i])]
        ck('Hero declaration phase '+str(i)+' prefix agrees with original int32 array',wide==hero._counts_after_stage[i].tolist())
        prefix_rows.append({'phase':i,'max_wide_prefix':max(wide),'original_dtype':str(hero._counts_after_stage[i].dtype)})
    total=trainer['num_train_steps']*B
    suffix_blocks=(total-starts[-1]+49152-1)//49152
    full_bounds=[a+suffix_blocks*int(q) for a,q in zip(wide,hero._counts_per_block_per_stage[-1])]
    points=[0,starts[1]-1,starts[1],starts[2]-1,starts[2],total-1]
    singles=[await hero.getitem_async(i) for i in points];batches=await hero.get_batch(points)
    refs=[wide_identity(hero,i) for i in points]
    ck('Six Hero-declaration artificial reads agree across APIs',singles==batches)
    ck('Six Hero-declaration reads match independent wide arithmetic',singles==[x['expected_identity'] for x in refs])
    ck('Total global positions exceed int32 while all declared child full-block bounds fit',total>LIMIT and max(full_bounds)<=LIMIT and all(0<=x['logical_index']<=LIMIT for x in refs))
    cases['hero_declaration']={'artificial_key':7,'artificial_inventory_length':1009,'constant_declared_batch':B,
        'stage_sequence_starts':starts,'prefix_controls':prefix_rows,'global_total':total,
        'all_child_max_full_block_count_bound':max(full_bounds),'constructor_warning_count':len(hw),
        'positions':points,'single':singles,'batch':batches,'references':refs}
    files=[R/'sources/integer_exposure_2026_10_07/mixture.py',meta,R/'scripts/probe_mixture_support_cpu.py',R/'scripts/probe_mixture_identity_cpu.py']
    out={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'runtime':{'python':ad['adapters']['platform'].python_version(),'numpy':np.__version__,'jax':jax.__version__,'backend':jax.default_backend()},
         'cases':cases,'reference_scope':'Independent Python integer prefix/offset arithmetic reuses original packed-ID assignment. Does not independently verify permutation or actual token identity.',
         'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
         'actual_Hero_overflow':None,'actual_Hero_cursor':None,'actual_token_stream':None,'actual_training_benefit':None}
    (R/'analysis/mixture_read_range_cpu.json').write_text(json.dumps(out,indent=2)+'\n')
    print('Original mixture constructor/read CPU controls:',len(checks),'passed')
if __name__=='__main__':asyncio.run(main())
