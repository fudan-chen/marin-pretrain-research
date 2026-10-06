"""Conditional reachability audit of declared text defaults; not an actual batch census."""
import ast,hashlib,json,pathlib,types,numpy as np
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/boundaries_2026_10_05';M=R/'sources/live_2026_10_07/meta.json';checks=[]
def ck(n,b):assert b,n;checks.append(n)
def extract(p,names,ns,from_class=None):
 t=ast.parse(p.read_text());body=t.body if from_class is None else next(n.body for n in t.body if isinstance(n,ast.ClassDef) and n.name==from_class);nodes=[n for n in body if isinstance(n,ast.FunctionDef) and n.name in names];assert {n.name for n in nodes}==set(names)
 for n in nodes:n.decorator_list=[]
 future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0);exec(compile(ast.fix_missing_locations(ast.Module(body=[future]+nodes,type_ignores=[])),str(p),'exec'),ns)
class TextFormat:pass
class ChatFormat:pass
class PrebuiltFormat:pass
ns={'TextLmDatasetFormat':TextFormat,'ChatLmDatasetFormat':ChatFormat,'PrebuiltLmDatasetFormat':PrebuiltFormat}
extract(D/'datasets.py',['_effective_pack'],ns)
def tokens(cache,length):return {'kind':'continuous_tokens','seq_len':length,'loss_weights_key':None}
def causal(ds,pos,**kw):return {'kind':'causal','dataset':ds,'options':kw}
ns.update(TokenSeqDataset=tokens,CausalLmDataset=causal);extract(D/'datasets.py',['dataset_for_component'],ns)
c=json.loads(M.read_text())['config'];comp=c['data']['value']['components'];L=c['model']['value']['max_seq_len'];B=c['trainer']['value']['trainer']['train_batch_size'];ck('Archived declaration has 223 plain text components',len(comp)==223 and all(x['pack'] is None and x['format']=={'text_key':'text'} for x in comp.values()))
obj=types.SimpleNamespace(pack=None,format=TextFormat());ck('Original default pack for text is false',ns['_effective_pack'](obj) is False)
path=ns['dataset_for_component'](obj,types.SimpleNamespace(size=L),'fake_cache',eos_id=0,block_cross_document_attention=True);ck('Original text builder requests continuous tokens without loss-weight field',path['dataset']['kind']=='continuous_tokens' and path['dataset']['loss_weights_key'] is None)
source=ast.parse((D/'datasets.py').read_text());klass=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='CausalLmDataset');calls=[n for n in ast.walk(klass) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='causal'];ck('Text causal wrapper supplies no ignore-id or external segment-id',len(calls)==1 and not {'ignore_id','segment_ids'} & {k.arg for k in calls[0].keywords})
mask_ns={'jnp':np};extract(D/'examples.py',['causal_loss_mask'],mask_ns,'GrugLmExample');mask=mask_ns['causal_loss_mask'](L);ck('Original default 4K mask has 4095 effective positions',L==4096 and np.count_nonzero(mask)==4095 and not mask[-1] and mask[:-1].all())
ck('Declared positive batch yields a conditional positive denominator',B==11264 and B*int(mask.sum())==46126080)
ck('Length-one constructor mask can be empty in a different recipe',mask_ns['causal_loss_mask'](1).sum()==0)
ck('Empty completion mask can be empty in a different recipe',mask_ns['causal_loss_mask'](4,prompt_length=4).sum()==0)
H=R/'sources/scale_2026_10_05/train_hero_ep.py';ck('Pinned Hero loader rejects nondivisible batch size','allow_nondivisible_batch_size=False' in H.read_text())
out={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'declared_components':223,'declared_batch_size':B,'declared_length':L,'conditional_default_effective_positions_per_sequence':int(mask.sum()),'conditional_default_global_T':B*int(mask.sum()),'selected_builder_path':path,'adapters':['Three format-type placeholders','Two dataset constructors replaced by route descriptors','Original causal_loss_mask with NumPy array operations','AST call-site argument inspection, no actual loader/cache/model execution'],'conditions':['Pinned code is the actual execution code','Declared plain text non-packing recipe is actually used','Full batch of sequences is delivered without external loss-weight mutation','Length and batch declarations are unchanged'],'actual_global_T':None,'actual_execution_binding':None,'actual_cached_inputs':None,'actual_Hero_zero_T_event':None,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [M,D/'datasets.py',D/'examples.py',H]}}
(R/'analysis/default_target_weights.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('Conditional default-mask checks:',len(checks),'conditional T:',out['conditional_default_global_T'])
