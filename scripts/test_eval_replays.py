"""Synthetic transcript tests only; no historical evaluation transcript acquired."""
import copy,json,math,pathlib,subprocess,sys,tempfile
from compare_eval_replays import compare,validate,aggregate,FIELDS
R=pathlib.Path(__file__).resolve().parents[1];checks=[]
def ck(n,v):assert v,n;checks.append(n)
j={'schema_version':1,'array_scope':'global','leaf_domains_disjoint':True,'checkpoint_sha256':'a'*64,'execution_sha':'b'*40,'parameter_view':'stored_params','pending_beta_policy':'stored_not_applied','compute_dtype':'bf16','backend':'synthetic','tokenizer_sha256':'c'*64,'byte_table_sha256':'d'*64,'input_digest_scheme':'ordered_tokens_weights_masks_tags_v1','records':[{'batch':0,'domain':'A','input_sha256':'e'*64,'weighted_loss_sum':2.,'loss_weight_sum':1.,'weighted_byte_sum':1.},{'batch':1,'domain':'A','input_sha256':'f'*64,'weighted_loss_sum':2.,'loss_weight_sum':1.,'weighted_byte_sum':3.}]}
clean=compare(j,j);ck('Identical declarations and sums compare within tolerance',clean['status']=='within_tolerance_under_matching_declarations')
ck('Identical result never asserts root cause or GPU replay',clean['root_cause'] is None and clean['actual_GPU_replay'] is None)
ck('Archived-style token-weighted BPB differs from byte ratio',math.isclose(clean['first']['micro_logged_BPB'],4/3*math.log2(math.e)) and math.isclose(clean['first']['leaves']['A']['byte_weighted_ratio_for_comparison'],math.log2(math.e)))
k=copy.deepcopy(j);k['records'][0]['weighted_loss_sum']=2.1;changed=compare(j,k);ck('Same declarations and changed loss are numeric difference',changed['status']=='numeric_difference_under_matching_declarations' and len(changed['numeric_difference_records'])==1)
k=copy.deepcopy(j);k['records'][0]['weighted_loss_sum']+=1e-8;ck('Declared tolerances admit small difference',compare(j,k)['status']=='within_tolerance_under_matching_declarations')
for field in FIELDS:
 k=copy.deepcopy(j);k[field]=('0'*64 if field.endswith('sha256') else '0'*40 if field=='execution_sha' else 'ordered_tokens_weights_masks_tags_v1' if field=='input_digest_scheme' else 'changed')
 if k[field]==j[field]:continue
 result=compare(j,k);ck('Changed '+field+' identity gates numeric conclusion',result['status']=='identity_or_input_review_required' and field in result['identity_fields_differ'])
for label,change in [('digest',lambda r:r.update(input_sha256='0'*64)),('denominator',lambda r:r.update(loss_weight_sum=2.)),('bytes',lambda r:r.update(weighted_byte_sum=2.))]:
 k=copy.deepcopy(j);change(k['records'][0]);result=compare(j,k);ck('Changed '+label+' requires input review',result['status']=='identity_or_input_review_required' and result['first_input_record_difference']==0)
k=copy.deepcopy(j);k['records'].reverse();ck('Record ordering difference is not silently aligned away',compare(j,k)['status']=='identity_or_input_review_required')
k=copy.deepcopy(j);k['records'].append({'batch':2,'domain':'empty','input_sha256':'1'*64,'weighted_loss_sum':0.,'loss_weight_sum':0.,'weighted_byte_sum':0.});agg=aggregate(validate(k));ck('Zero-weight domain is recorded and excluded from active macro',agg['empty_weight_domains']==['empty'] and agg['macro_CE']==2.)
# Same total N/T/B but different batch grouping changes logged BPB.
merged=copy.deepcopy(j);merged['records']=[{'batch':0,'domain':'A','input_sha256':'0'*64,'weighted_loss_sum':4.,'loss_weight_sum':2.,'weighted_byte_sum':4.}];result=compare(j,merged);ck('Batch regrouping is gated despite same total CE and bytes',result['status']=='identity_or_input_review_required' and result['first']['micro_CE']==result['second']['micro_CE']==2. and result['first']['micro_logged_BPB']!=result['second']['micro_logged_BPB'])
for label,mutate in [('local',lambda x:x.update(array_scope='local')),('NaN',lambda x:x['records'][0].update(weighted_loss_sum=float('nan'))),('duplicate',lambda x:x['records'].append(copy.deepcopy(x['records'][0]))),('negative',lambda x:x['records'][0].update(weighted_byte_sum=-1)),('zero weight nonzero loss',lambda x:x['records'][0].update(loss_weight_sum=0))]:
 k=copy.deepcopy(j);mutate(k)
 try:validate(k)
 except ValueError:ck('Reject '+label,True)
 else:ck('Reject '+label,False)
large=copy.deepcopy(j)
for row in large['records']:row['weighted_loss_sum']=1e308
try:aggregate(validate(large))
except ValueError:ck('Reject overflowing aggregate despite individually finite fields',True)
else:ck('Reject overflowing aggregate despite individually finite fields',False)
with tempfile.TemporaryDirectory() as d:
 p=pathlib.Path(d);a=p/'a.json';b=p/'b.json';o=p/'out.json';a.write_text(json.dumps(j));b.write_text(json.dumps(j));cmd=[sys.executable,str(R/'scripts/compare_eval_replays.py'),str(a),str(b),str(o)];done=subprocess.run(cmd,capture_output=True);ck('Real CLI writes file hashes and comparison',done.returncode==0 and len(json.loads(o.read_text())['transcript_file_sha256']['first'])==64);before=o.read_bytes();again=subprocess.run(cmd,capture_output=True);ck('Real CLI refuses overwrite and preserves bytes',again.returncode!=0 and o.read_bytes()==before)
(R/'analysis/eval_replay_validation.json').write_text(json.dumps({'scope':__doc__,'checks_passed':len(checks),'checks':checks,'synthetic_clean':clean,'synthetic_regrouped':result,'actual_historical_transcripts':None,'actual_GPU_replay':None},ensure_ascii=False,indent=2)+'\n');(R/'templates/eval_replay_synthetic.json').write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n');print('Synthetic evaluation transcript checks:',len(checks))
