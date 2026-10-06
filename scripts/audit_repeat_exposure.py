"""Project finite fixed-order modulo-restart sequence exposure from declarations.
No actual cache inventory, permutation, consumed cursor or duplicate-text verification.
Usage: python scripts/audit_repeat_exposure.py input.json output.json
"""
import hashlib,json,pathlib,sys

def integer(x,name,positive=False):
 if x is not None and (type(x) is not int or x<(1 if positive else 0)):raise ValueError(name+' must be a valid nonnegative integer or null')
 return x

def audit(j):
 if j.get('index_policy')!='fixed_order_modulo_restart':raise ValueError('explicit fixed-order modulo-restart policy required')
 rows=j.get('components')
 if not isinstance(rows,list) or not rows:raise ValueError('component list required')
 out=[];names=set()
 for row in rows:
  name=row.get('name')
  if not isinstance(name,str) or not name or name in names:raise ValueError('unique nonempty names required')
  names.add(name);n=integer(row.get('inventory_sequences'),'inventory',True);d=integer(row.get('draws'),'draws');c=integer(row.get('start_sequence_index'),'start cursor')
  result={'name':name,'inventory_sequences':n,'draws':d,'start_sequence_index':c,'status':'inventory_or_draw_count_unknown','window_distinct_sequence_indices':None,'window_repeat_draws':None,'new_sequence_indices_since_lifetime_start':None,'repeat_draws_against_lifetime_history':None,'extra_draw_index_intervals':None}
  if n is not None and d is not None:
   q,r=divmod(d,n);result.update(status='declared_projection_only',window_distinct_sequence_indices=min(d,n),window_repeat_draws=d-min(d,n),full_cycles_in_window=q,extra_draws=r,min_draws_per_inventory_index=q,max_draws_per_inventory_index=q+(r>0))
   if c is not None:
    start=c%n;first=min(r,n-start);intervals=[]
    if first:intervals.append({'start':start,'stop':start+first})
    if r>first:intervals.append({'start':0,'stop':r-first})
    new=min(c+d,n)-min(c,n);result.update(new_sequence_indices_since_lifetime_start=new,repeat_draws_against_lifetime_history=d-new,extra_draw_index_intervals=intervals,next_sequence_index=c+d,next_modulo_index=(c+d)%n)
  out.append(result)
 return {'scope':__doc__,'index_policy':j['index_policy'],'components':out,'assumptions':['contiguous ordinal component draws from lifetime origin zero','fixed finite inventory and unchanged index-to-sample mapping','distinct indices are not necessarily unique documents or unique text'],'actual_cache_inventory_verified':False,'actual_consumed_cursor_verified':False,'actual_quality_or_training_benefit':None}
if __name__=='__main__':
 if len(sys.argv)!=3:raise SystemExit(__doc__)
 p,o=map(pathlib.Path,sys.argv[1:]);raw=p.read_bytes();result=audit(json.loads(raw));result['input_sha256']=hashlib.sha256(raw).hexdigest()
 with o.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
