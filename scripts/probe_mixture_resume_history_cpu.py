"""Original fixed-head mixture replay with past-stage declarations changed.
Finite identity children, real JAX CPU; no loader cursor or model checkpoint restore.
"""
import asyncio, collections, hashlib, json, pathlib, runpy
R=pathlib.Path(__file__).resolve().parents[1]
a=runpy.run_path(str(R/'scripts/probe_mixture_support_cpu.py'))
Mix,Identity=a['Mix'],a['Identity']
def make(past=.5,lengths=(5,7)):
 return Mix({n:Identity(n,l) for n,l in zip(('A','B'),lengths)},[(0,{'A':past,'B':1-past}),(16,{'A':.5,'B':.5})],8,key=7,stop_strategy='restart')
def logical(m,i):
 b=i//8;packed=m._get_block(b)[i%8];d,j=m._index_into_dataset_for_id(packed,b)
 return m.dataset_index[d],int(j)
def names(xs):return [x.split(':')[0] for x in xs]
async def main():
 checks=[]
 def ck(n,c):assert c,n;checks.append(n)
 old,new=make(),make(.75);indices=list(range(16,40))
 x,y=await old.get_batch(indices),await new.get_batch(indices)
 lx,ly=[logical(old,i) for i in indices],[logical(new,i) for i in indices]
 ck('CPU original permutation',a['ns']['jax'].default_backend()=='cpu')
 ck('Current named quotas equal',old._counts_per_block_per_stage[1].tolist()==new._counts_per_block_per_stage[1].tolist()==[4,4])
 ck('Past stage prefixes differ',old._counts_after_stage[0].tolist()==[8,8] and new._counts_after_stage[0].tolist()==[12,4])
 ck('Current domain sequence unchanged',names(x)==names(y))
 ck('All logical A offsets shift plus four and B minus four',all(n==nn and jj-j==(4 if n=='A' else -4) for (n,j),(nn,jj) in zip(lx,ly)))
 ck('All current identities differ for inventory five seven',all(xx!=yy for xx,yy in zip(x,y)))
 rebuilt=make();ck('Fresh object reconstructs unchanged declaration',await rebuilt.get_batch(indices)==x)
 ck('Singleton and batch reads agree in bounded range',[await rebuilt.getitem_async(i) for i in indices]==x)
 pieces=[]
 for ids in [indices[:1],indices[1:9],indices[9:]]:pieces+=await rebuilt.get_batch(ids)
 ck('Different read partition preserves replay',pieces==x)
 req=[39,16,39,24,17]
 ck('Out of order duplicate reads preserve replay',await rebuilt.get_batch(req)==[x[i-16] for i in req])
 for b in range(100):rebuilt._get_block(b)
 ck('LRU eviction and access history preserve replay',await rebuilt.get_batch(indices)==x)
 small_old,small_new=make(lengths=(4,4)),make(.75,lengths=(4,4))
 so,sn=await small_old.get_batch(indices),await small_new.get_batch(indices)
 ck('Modulo inventory can mask changed logical offsets',so==sn)
 ck('Masked sequence still has distinct unwrapped history',[logical(small_old,i) for i in indices]!=[logical(small_new,i) for i in indices])
 # Starting the current mix anew is not equivalent to resuming its global index.
 fresh=Mix({'A':Identity('A',5),'B':Identity('B',7)},{'A':.5,'B':.5},8,key=7)
 reset=await fresh.get_batch(list(range(24)))
 ck('Resetting mixture index changes replay even with same current quotas',reset!=x)
 current_only=await fresh.get_batch(indices)
 ck('Same global index with flattened current schedule reproduces baseline coincidentally',current_only==x)
 ck('That flattened schedule does not reproduce changed past prefix',current_only!=y)
 files=[R/'sources/integer_exposure_2026_10_07/mixture.py',R/'scripts/probe_mixture_support_cpu.py',R/'scripts/probe_mixture_identity_cpu.py',pathlib.Path(__file__)]
 out={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'runtime':{'python':a['adapters']['platform'].python_version(),'numpy':a['ns']['np'].__version__,'jax':a['ns']['jax'].__version__,'backend':a['ns']['jax'].default_backend()},'indices':indices,'old_prefix':[8,8],'changed_prefix':[12,4],'same_current_quota':[4,4],'old_read':x,'changed_past_read':y,'old_logical':lx,'changed_logical':ly,'modulo_masked_old':so,'modulo_masked_changed':sn,'reset_cursor_read':reset,'actual_loader_resume':None,'actual_Hero_stage_rewrite':None,'actual_training_benefit':None,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
 (R/'analysis/mixture_resume_history_cpu.json').write_text(json.dumps(out,indent=2)+'\n')
 print('Original mixture historical-stage replay controls:',len(checks),'passed')
if __name__=='__main__':asyncio.run(main())
