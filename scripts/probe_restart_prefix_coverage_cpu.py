"""Original restart reads versus conditional contiguous-index exposure formulas.
Artificial finite identities, no actual Hero inventory or model loss.
"""
import asyncio,collections,hashlib,json,pathlib,runpy
R=pathlib.Path(__file__).resolve().parents[1]
ad=runpy.run_path(str(R/'scripts/probe_mixture_support_cpu.py'));Mix=ad['Mix'];Identity=ad['Identity']
audit=runpy.run_path(str(R/'scripts/audit_repeat_exposure.py'))['audit']
def ledger(c,d):
    return audit({'index_policy':'fixed_order_modulo_restart','components':[{'name':'A','inventory_sequences':5,'draws':d,'start_sequence_index':c}]})['components'][0]
async def main():
    checks=[];cases={}
    def ck(n,c):assert c,n;checks.append(n)
    m=Mix({'A':Identity('A',5)},{'A':1},8,key=7,stop_strategy='restart')
    stream=await m.get_batch(list(range(16)))
    prefix=stream[:5];window=stream[5:8];history=set(prefix);new=set(window)-history
    ck('Actual partial prefix reads repeat before first full inventory coverage',len(set(prefix))==3 and len(prefix)==5)
    ck('Contiguous formula correctly states its own projection but mismatches actual partial prefix',ledger(0,5)['window_distinct_sequence_indices']==5 and len(set(prefix))!=5)
    ck('Actual later edge has new identities despite five prior draws',new=={'A:0','A:3'} and ledger(5,3)['new_sequence_indices_since_lifetime_start']==0)
    ck('Actual edge also has internal repeat despite three draws below inventory size',len(window)-len(set(window))==1 and ledger(5,3)['window_repeat_draws']==0)
    ck('Complete first block covers all inventory identities',set(stream[:8])=={f'A:{i}' for i in range(5)})
    ck('Complete subsequent block agrees with contiguous multiset counts',collections.Counter(stream[8:16])==collections.Counter(f'A:{i%5}' for i in range(8,16)))
    off=Mix({'A':Identity('A',5)},{'A':1},8,key=7,randomize_blocks=False,stop_strategy='restart')
    unrandom=await off.get_batch(list(range(8)))
    ck('Unrandomized partial window satisfies the contiguous projection',len(set(unrandom[:5]))==ledger(0,5)['window_distinct_sequence_indices'])
    cases['one_A']={'stream':stream,'partial_prefix':prefix,'edge_window':window,'actual_prefix_unique':len(history),
        'actual_window_unique':len(set(window)),'actual_new_against_history':sorted(new),'actual_window_internal_repeats':len(window)-len(set(window)),
        'conditional_prefix_projection':ledger(0,5),'conditional_window_projection':ledger(5,3),'unrandomized':unrandom}
    staged=Mix({'A':Identity('A',5),'B':Identity('B',7)},[(0,{'A':.5,'B':.5}),(16,{'A':.75,'B':.25})],8,key=7,stop_strategy='restart')
    ss=await staged.get_batch(list(range(32)))
    counts={'A':0,'B':0};full_rows=[]
    for block in range(4):
        phase=0 if block<2 else 1
        for name,q in zip(staged.dataset_index,staged._counts_per_block_per_stage[phase]):
            actual=collections.Counter(x for x in ss[block*8:(block+1)*8] if x.startswith(name+':'))
            n=5 if name=='A' else 7
            ref=collections.Counter(f'{name}:{i%n}' for i in range(counts[name],counts[name]+int(q)))
            full_rows.append({'block':block,'name':name,'count':int(q),'actual_multiset':dict(actual),'reference_multiset':dict(ref)})
            ck('Complete staged block '+str(block)+' '+name+' matches contiguous multiset',actual==ref)
            counts[name]+=int(q)
    ck('Staged complete-prefix coverage agrees with independent inventory sets',all(len({x for x in ss if x.startswith(name+':')})==min(counts[name],n) for name,n in [('A',5),('B',7)]))
    files=[R/'sources/integer_exposure_2026_10_07/mixture.py',R/'scripts/audit_repeat_exposure.py',R/'scripts/probe_mixture_support_cpu.py',R/'scripts/probe_mixture_identity_cpu.py']
    out={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'cases':cases,'staged_full_blocks':full_rows,
         'runtime':{'python':ad['adapters']['platform'].python_version(),'numpy':ad['ns']['np'].__version__,'jax':ad['ns']['jax'].__version__},
         'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
         'actual_Hero_coverage':None,'actual_Hero_inventory':None,'actual_training_benefit':None,'actual_unique_texts':None}
    (R/'analysis/restart_prefix_coverage_cpu.json').write_text(json.dumps(out,indent=2)+'\n')
    print('Original restart coverage and conditional formula controls:',len(checks),'passed')
if __name__=='__main__':asyncio.run(main())
