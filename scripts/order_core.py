"""Integer schedule ledger. No trainer, actual checkpoint or token-store access."""
import math,copy
from transfer_core import encode_counts

def window_ledger(start,end,boundaries,units):
    if not all(isinstance(x,int) for x in [start,end,units]+list(boundaries)) or units<=0 or start<0 or end<=start:
        raise ValueError('Integer positive window and block size required')
    if not boundaries or boundaries[0]!=0 or sorted(set(boundaries))!=list(boundaries) or any(x%units for x in boundaries):
        raise ValueError('Increasing block-aligned phase boundaries beginning at zero required')
    result=[]
    for phase,begin in enumerate(boundaries):
        finish=boundaries[phase+1] if phase+1<len(boundaries) else end
        lo,hi=max(start,begin),min(end,finish)
        if hi<=lo:continue
        first,last=lo//units,(hi-1)//units
        complete_begin=(lo+units-1)//units;complete_end=hi//units
        partials=[]
        for block in sorted({first,last}):
            a,b=max(lo,block*units)-block*units,min(hi,(block+1)*units)-block*units
            if b-a<units:partials.append({'block':block,'slot_begin':a,'slot_end':b,'read_sequences':b-a})
        full=max(0,complete_end-complete_begin)
        result.append({'phase':phase,'sequence_begin':lo,'sequence_end':hi,'complete_block_begin':complete_begin,
                       'complete_block_end':max(complete_begin,complete_end),'complete_blocks':full,'partials':partials})
    if sum(x['complete_blocks']*units+sum(p['read_sequences'] for p in x['partials']) for x in result)!=end-start:
        raise ValueError('Window ledger fails coverage')
    return result

def partial_difference_bounds(a,b,read,units):
    """Per-bucket bounds for arbitrary block permutations; not jointly tight."""
    if not 0<=read<=units:raise ValueError('Invalid partial block length')
    return {k:[max(0,b[k]-(units-read))-min(a[k],read),
               min(b[k],read)-max(0,a[k]-(units-read))] for k in a}

def capacities(base,receiver,donor,m):
    if receiver==donor or receiver not in base[0] or donor not in base[0]:raise ValueError('Distinct supported buckets required')
    g=math.gcd(*m);early,late=m[1]//g,m[0]//g
    forward=min(base[0][donor]//early,base[1][receiver]//late)
    reverse=min(base[0][receiver]//early,base[1][donor]//late)
    return {'early_units_per_k':early,'late_units_per_k':late,'forward_max_k':forward,
            'reverse_max_k':reverse,'symmetric_max_k':min(forward,reverse)}

def compile_order(base,initial,units,k,receiver='c39q4',donor='c26q4',start=402432,end=4024320,transition=3194880,lock_edges=True):
    if not isinstance(units,int) or isinstance(units,bool) or len(base)!=2 or any(set(c)!=set(initial) or any(not isinstance(n,int) or isinstance(n,bool) or n<0 for n in c.values()) or sum(c.values())!=units for c in [initial]+base):
        raise ValueError('Common normalized integer count maps required')
    ledger=window_ledger(start,end,[0,start//units*units,transition],units)
    phases=[x for x in ledger if x['phase'] in [1,2]]
    if len(phases)!=2 or any(len(x['partials'])!=1 for x in phases):raise ValueError('Compiler expects one leading and one trailing partial block')
    leading=phases[0]['partials'][0];trailing=phases[1]['partials'][0]
    if leading['slot_end']!=units or trailing['slot_begin']!=0:raise ValueError('Only outer edge blocks can be locked')
    m=[x['complete_blocks'] for x in phases]
    if min(m)<=0:raise ValueError('Both interior phases need complete blocks')
    cap=capacities(base,receiver,donor,m)
    if not isinstance(k,int) or isinstance(k,bool) or k<0 or k>cap['symmetric_max_k']:raise ValueError('Amplitude exceeds symmetric integer capacity; no clipping')
    starts=[0,leading['block']*units,(leading['block']+1)*units,transition,trailing['block']*units] if lock_edges else [0,leading['block']*units,transition]
    arms=[];cells=sorted(initial)
    for name,sign in [('baseline',0),('early_high',1),('late_high',-1)]:
        early,late=copy.deepcopy(base);u0,u1=sign*k*cap['early_units_per_k'],sign*k*cap['late_units_per_k']
        early[receiver]+=u0;early[donor]-=u0;late[receiver]-=u1;late[donor]+=u1
        stage_counts=[initial,base[0],early,late,base[1]] if lock_edges else [initial,early,late]
        stages=[{'sequence_begin':s,'counts':c.copy(),'weights':encode_counts(c,units)} for s,c in zip(starts,stage_counts)]
        residual={cell:m[0]*(early[cell]-base[0][cell])+m[1]*(late[cell]-base[1][cell]) for cell in cells}
        arms.append({'id':name,'sign':sign,'stages':stages,'interior_counts':[early,late],
                     'interior_cumulative_difference_sequences':residual,'interior_l1_difference_sequences':sum(abs(v) for v in residual.values()),
                     'receiver_forward_sequences':m[0]*u0,'receiver_early_delta_units':u0,'receiver_late_delta_units':-u1})
    return {'schema':'marin-prospective-order-ledger/1','status':'planned_not_executed','launchable':False,
            'block_size_sequences':units,'sequence_length':4096,'sequence_interval':[start,end],
            'physical_budget_tokens':(end-start)*4096,'complete_blocks':m,'capacity':cap,'amplitude_k':k,
            'receiver':receiver,'donor':donor,'lock_edge_blocks':lock_edges,'window_ledger':ledger,'arms':arms,
            'window_ledger_scope':'Reference two-phase partition; exact compiled stage boundaries are listed in each arm',
            'whole_window_count_match_by_construction':lock_edges or k==0,
            'count_match_assumptions':['Same recorded start/end sequence indices and batch schedule',
               'Same bucket order, common history schedule and underlying sequence datasets',
               'Same mixture and inner shuffle keys, deterministic mapping and restart semantics',
               'Entire interior blocks read once; common edge blocks retain identical weights and prefix offsets'],
            'actual_checkpoint_digest':None,'actual_data_cursor':None,'actual_shuffle_keys':None,
            'token_store_manifest':None,'independent_eval_manifest':None,'generation_results':None,
            'initial_counts_origin':'Common initial phase from an archived order configuration; illustrative history, not verified checkpoint exposure',
            'baseline_counts_origin':'V7 donor c26 M1Q1 planned integer recipe; not independently confirmed training',
            'native_loader_execution_verified':False,'actual_token_stream_verified':False}
