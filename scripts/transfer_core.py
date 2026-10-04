"""Continuous weight contracts and full-block arithmetic, never a trainer."""
import math

def validate(weights, budgets):
    if len(weights) != 2 or len(budgets) != 2 or any(not math.isfinite(x) or x <= 0 for x in budgets):
        raise ValueError('Two positive finite phase budgets are required')
    cells = sorted(weights[0])
    if len(cells) != 200 or set(weights[1]) != set(cells):
        raise ValueError('Exactly 200 common buckets are required')
    for phase in weights:
        if any(not math.isfinite(x) or x < 0 for x in phase.values()) or abs(math.fsum(phase.values())-1) > 1e-12:
            raise ValueError('Each phase must contain normalized finite nonnegative weights')
    return cells

def exposure(weights, budgets):
    cells = validate(weights, budgets)
    return {k: math.fsum(t*w[k] for t,w in zip(budgets,weights)) for k in cells}

def difference(a, b, budgets):
    ea, eb = exposure(a,budgets), exposure(b,budgets)
    deltas = {k: eb[k]-ea[k] for k in ea}
    domain = {c: math.fsum(v for k,v in deltas.items() if k.startswith('c%02dq'%c)) for c in range(40)}
    moved = math.fsum(abs(v) for v in deltas.values())/2
    cross = math.fsum(abs(v) for v in domain.values())/2
    phased = math.fsum(t*math.fsum(abs(wb[k]-wa[k]) for k in ea)/2 for t,wa,wb in zip(budgets,a,b))
    return {'cells':deltas,'domains':domain,'net':math.fsum(deltas.values()),'moved':moved,
            'cross_domain_net':cross,'within_domain_cancellation':moved-cross,
            'phase_moved_sum':phased,'temporal_cancellation':phased-moved}

def quantize(weights, units):
    if not isinstance(units,int) or units <= 0: raise ValueError('Block units must be a positive integer')
    cells=sorted(weights);z=math.fsum(weights.values())
    if z<=0 or any(not math.isfinite(v) or v<0 for v in weights.values()):raise ValueError('Invalid weights')
    counts={k:math.floor(weights[k]/z*units) for k in cells}
    recipient=max(cells,key=lambda k:counts[k]);remainder=units-sum(counts.values());counts[recipient]+=remainder
    return {'counts':counts,'weights':{k:n/units for k,n in counts.items()},'recipient':recipient,'remainder':remainder}

def apportion(values, units):
    """Allocate a fixed group total; all remainder stays inside that group."""
    total=math.fsum(values.values())
    if total<=0 or units<0:raise ValueError('Invalid group allocation')
    scaled={k:v/total*units for k,v in values.items()};counts={k:math.floor(v) for k,v in scaled.items()}
    for k in sorted(counts,key=lambda k:(-(scaled[k]-counts[k]),k))[:units-sum(counts.values())]:counts[k]+=1
    return counts

def encode_counts(counts, units):
    """Move positive weights inside floor intervals; the largest bucket receives its unit back."""
    if sum(counts.values())!=units or any(n<0 for n in counts.values()):raise ValueError('Invalid counts')
    recipient=max(sorted(counts),key=lambda k:counts[k]);epsilon=1e-9/units
    w={k:n/units+(epsilon if n>0 and k!=recipient else 0) for k,n in counts.items()}
    w[recipient]-=epsilon*sum(n>0 and k!=recipient for k,n in counts.items())
    w[recipient]+=1-math.fsum(w.values())
    if quantize(w,units)['counts']!=counts:raise ValueError('Encoded counts fail allocator roundtrip')
    return w

def controlled_integer_plan(base, continuous, receiver, donor, amount_delta, units):
    """Only two declared domains may change allocated counts; no global rounding donor."""
    rkeys=[k for k in base if k.startswith('c%02dq'%receiver)];dkeys=[k for k in base if k.startswith('c%02dq'%donor)]
    baseline=quantize(base,units)['counts'];counts=baseline.copy();transfer=round(amount_delta*units)
    rtotal=sum(baseline[k] for k in rkeys)+transfer;dtotal=sum(baseline[k] for k in dkeys)-transfer
    if min(rtotal,dtotal)<=0:raise ValueError('Integer domain capacity is insufficient')
    counts.update(apportion({k:continuous[k] for k in rkeys},rtotal))
    counts.update(apportion({k:continuous[k] for k in dkeys},dtotal))
    return {'target_counts':counts,'weights':encode_counts(counts,units),'transferred_units_per_block':transfer}

def factorial_plan(base, alternative, budgets, receiver=39, donor=26):
    cells=validate(base,budgets);validate(alternative,budgets)
    if receiver==donor or receiver not in range(40) or donor not in range(40):raise ValueError('Distinct valid donor and receiver are required')
    ea,eb=exposure(base,budgets),exposure(alternative,budgets);B=math.fsum(budgets)
    rkeys=[k for k in cells if k.startswith('c%02dq'%receiver)];dkeys=[k for k in cells if k.startswith('c%02dq'%donor)]
    r0=math.fsum(ea[k] for k in rkeys);r1=math.fsum(eb[k] for k in rkeys);d0=math.fsum(ea[k] for k in dkeys)
    if min(r0,r1,d0)<=0:raise ValueError('Zero support cannot define conditional quality proportions')
    increment=r1-r0
    conditional={0:{k:ea[k]/r0 for k in rkeys},1:{k:eb[k]/r1 for k in rkeys}}
    arms=[]
    for amount,quality in [(0,0),(1,0),(0,1),(1,1)]:
        total=r0+amount*increment;delta={k:0.0 for k in cells}
        for k in rkeys:delta[k]=(total*conditional[quality][k]-ea[k])/B
        for k in dkeys:delta[k]=-amount*increment/B*(ea[k]/d0)
        phases=[{k:w[k]+delta[k] for k in cells} for w in base]
        validate(phases,budgets)  # Invalid donor capacity fails; never clip or renormalize silently.
        arms.append({'id':'M%dQ%d'%(amount,quality),'amount_factor':amount,'quality_factor':quality,
                     'weights':phases,'constant_phase_delta':delta,'target_receiver_tokens':total,
                     'target_conditional_quality':conditional[quality],
                     'donor_exposure_delta_tokens':-amount*increment})
    return {'receiver_domain':receiver,'donor_domain':donor,'base_receiver_tokens':r0,
            'alternative_receiver_tokens':r1,'increment_tokens':increment,
            'constant_amount_delta_pp':100*increment/B,'arms':arms}
