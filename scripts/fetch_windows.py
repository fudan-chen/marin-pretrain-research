exec(open(__file__.replace('fetch_windows.py','fetch_wandb.py')).read().split('with concurrent')[0])
windows=[('ep_control','hero-wd-gate-router-p02-step58k',81716,81917),('ep_new','hero-ragged_a2a-nccl2307-ep-step81k',81716,81917),('mix_before','hero-ragged_a2a-nccl2307-ep-step81k',107800,108200),('mix_after','hero-mix-996f4891-step108k',108000,108200),('kernel_control','hero-main-step121638',146139,146339),('kernel_new','hero-fa4sm100-nomask-step146k',146139,146339)]
def fetch(w):
 name,r,a,b=w
 ks=['train/cross_entropy_loss','grad/norm/total','throughput/mfu','throughput/tokens_per_second','moe/drop_fraction']
 specs=[{'keys':['_step',k],'samples':1000,'minStep':a,'maxStep':b} for k in ks]
 h=gql('query($e:String!,$p:String!,$r:String!,$specs:[JSONString!]!){project(name:$p,entityName:$e){run(name:$r){sampledHistory(specs:$specs)}}}',{'e':'marin-community','p':'marin_moe','r':r,'specs':[json.dumps(x) for x in specs]})['sampledHistory']
 out={'run':r,'requested_range':[a,b],'specs':specs,'series':[[json.loads(x) if isinstance(x,str) else x for x in s] for s in h],'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 (S/'wandb'/('window_'+name+'.json')).write_text(json.dumps(out))
 return (name,[len(s) for s in out['series']],[[min(x['_step'] for x in s),max(x['_step'] for x in s)] if s else None for s in out['series']])
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
 for r in ex.map(fetch,windows):print(r)
