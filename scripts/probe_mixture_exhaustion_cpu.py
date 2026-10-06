"""Original finite-mixture length and strict identity reads on real CPU permutations.
No real token inventory, loader or training; strict child enforces finite bounds.
"""
import asyncio,hashlib,json,pathlib,runpy
R=pathlib.Path(__file__).resolve().parents[1]
ad=runpy.run_path(str(R/'scripts/probe_mixture_support_cpu.py'));Mix=ad['Mix'];Identity=ad['Identity']
class StrictIdentity(Identity):
    async def get_batch(self,indices):
        if any(i<0 or i>=self.length for i in indices):raise IndexError(self.name+' finite inventory index out of bounds')
        return await super().get_batch(indices)
def mapping(m,index):
    block=index//m.block_size;stage=m._get_stage_for_block(block)
    packed=int(m._get_block(block)[index%m.block_size]);dsid=packed>>16;local=packed&65535
    before=sum(int(m._counts_per_block_per_stage[i][dsid])*((m.weight_stages[i+1][0]-m.weight_stages[i][0])//m.block_size) for i in range(stage))
    offset=(block*m.block_size-m.weight_stages[stage][0])//m.block_size
    logical=before+offset*int(m._counts_per_block_per_stage[stage][dsid])+local
    return {'position':index,'name':m.dataset_index[dsid],'logical_index':logical,'valid':logical<m._dataset_of_id(dsid).length}
def make(length,seed=7,randomize=True,stages=None,stop='first_exhausted'):
    return Mix({'A':StrictIdentity('A',length),'B':StrictIdentity('B',100)},stages or {'A':.5,'B':.5},8,
               key=seed,randomize_blocks=randomize,stop_strategy=stop)
async def main():
    checks=[];rows=[]
    def ck(n,c):assert c,n;checks.append(n)
    for randomize in [False,True]:
        for seed in [7,8]:
            for length in range(1,9):
                m=make(length,seed,randomize);reported=await m.async_len();trace=[mapping(m,i) for i in range(24)]
                first_bad=next(x['position'] for x in trace if not x['valid']);error=None
                try:await m.get_batch(list(range(reported)))
                except Exception as e:error={'type':type(e).__name__,'message':str(e)}
                rows.append({'randomize':randomize,'seed':seed,'A_length':length,'reported_length':reported,
                             'safe_prefix_length':first_bad,'within_reported_prefix_invalid':[x for x in trace[:reported] if not x['valid']],
                             'read_error':error,'first_block':trace[:8]})
    ck('Unrandomized finite controls have safe reported prefixes',all(r['safe_prefix_length']>=r['reported_length'] and r['read_error'] is None for r in rows if not r['randomize']))
    bad=[r for r in rows if r['read_error'] is not None]
    ck('Randomized reported finite range can contain invalid child indices',bool(bad) and all(r['randomize'] and r['within_reported_prefix_invalid'] for r in bad))
    example=next(r for r in bad if r['seed']==7)
    m=make(example['A_length']);single_error=None
    pos=example['within_reported_prefix_invalid'][0]['position']
    try:await m.getitem_async(pos)
    except Exception as e:single_error={'type':type(e).__name__,'message':str(e)}
    ck('Both actual APIs reject the invalid position inside reported length',single_error is not None and single_error['type']==example['read_error']['type']=='IndexError' and pos<example['reported_length'])
    staged=make(5,stages=[(0,{'A':.5,'B':.5}),(8,{'A':1})]);sl=await staged.async_len();st=[mapping(staged,i) for i in range(16)]
    st_error=None
    try:await staged.get_batch(list(range(sl)))
    except Exception as e:st_error={'type':type(e).__name__,'message':str(e)}
    ck('Stage change retains exhaustion mismatch under permuted local offsets',st_error is not None and any(not x['valid'] for x in st[:sl]))
    restart=make(1,stop='restart');rr=await restart.get_batch(list(range(16)))
    ck('Restart remaps finite child and avoids these finite bounds errors',all(x=='A:0' for x in rr if x.startswith('A:')))
    all_stop=Mix({'A':StrictIdentity('A',5)},{'A':1},8,key=7,stop_strategy='all_exhausted')
    al=await all_stop.async_len();ar=await all_stop.get_batch(list(range(al)))
    ck('All-stop count equals inventory but not unique inventory coverage',al==5 and len(set(ar))==3 and set(ar)!={f'A:{i}' for i in range(5)})
    all_unrandom=Mix({'A':StrictIdentity('A',5)},{'A':1},8,key=7,randomize_blocks=False,stop_strategy='all_exhausted')
    au=await all_unrandom.get_batch(list(range(await all_unrandom.async_len())))
    ck('Unrandomized all-stop covers this inventory without repeats',au==[f'A:{i}' for i in range(5)])
    meta=R/'sources/live_2026_10_07/meta.json';cfg=json.loads(meta.read_text())['config']['data']['value']
    ck('Archived Hero declares restart, not finite-first policy',cfg['stop_strategy']=='restart')
    files=[R/'sources/integer_exposure_2026_10_07/mixture.py',meta,R/'scripts/probe_mixture_support_cpu.py',R/'scripts/probe_mixture_identity_cpu.py']
    out={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'grid_cells':len(rows),'rows':rows,'failing_grid_cells':len(bad),
         'example_single_error':single_error,'staged':{'reported_length':sl,'trace':st,'read_error':st_error},
         'restart_identity':rr,'all_stop':{'reported_length':al,'identities':ar,'unique_count':len(set(ar)),'unrandomized':au},'runtime':{'python':ad['adapters']['platform'].python_version(),'numpy':ad['ns']['np'].__version__,'jax':ad['ns']['jax'].__version__},
         'reference_scope':'Original packed IDs reused, independently unpacked and accumulated with Python integers; no independent permutation or real token verification.',
         'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
         'actual_Hero_exhaustion_incident':None,'actual_token_stream':None,'actual_loader_behavior':None,'actual_training_benefit':None}
    (R/'analysis/mixture_exhaustion_cpu.json').write_text(json.dumps(out,indent=2)+'\n')
    print('Original finite exhaustion controls:',len(checks),'passed;',len(bad),'of',len(rows),'artificial grid cells expose read error')
if __name__=='__main__':asyncio.run(main())
