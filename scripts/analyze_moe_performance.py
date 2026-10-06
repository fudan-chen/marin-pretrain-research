"""Recompute the frozen PR9708 performance table. Author measurements, not local GPU runs."""
import hashlib,json,pathlib,re
R=pathlib.Path(__file__).resolve().parents[1];P=R/'sources/engineering_current_2026_10_07/pull_9708.json';j=json.loads(P.read_text());rows=[]
for line in j['body'].splitlines():
 if not line.startswith('|'):continue
 cells=[x.strip() for x in line.strip('|').split('|')]
 if len(cells)!=5 or cells[0] in ['Change','---']:continue
 rows.append({'change':cells[0],'commit':cells[1],'seconds':float(cells[2]) if cells[2] else None,'MFU_percent':float(cells[3].rstrip('%')) if cells[3] else None,'author_gain_label':cells[4]})
timed=[r for r in rows if r['seconds'] is not None];first,last=timed[0],timed[-1]
for i,row in enumerate(timed):
 row['table_previous_seconds']=timed[i-1]['seconds'] if i else None
 row['table_time_delta_seconds']=row['seconds']-timed[i-1]['seconds'] if i else None
 row['table_relative_speed_gain']=timed[i-1]['seconds']/row['seconds']-1 if i else None
row=next(r for r in timed if r['change']=='Shared-expert SwiGLU in QuACK epilogues');checks=[]
def check(n,x):assert x,n;checks.append(n)
check('Complete eighteen-row author table parsed',len(rows)==18 and len(timed)==12)
check('Frozen endpoints preserved',first['seconds']==13.899 and last['seconds']==12.591)
check('Shared epilogue row is a sequential regression',abs(row['table_time_delta_seconds']-.014)<1e-12 and row['table_relative_speed_gain']<0)
check('Final-tip removal pairs have opposite effect in author description','by 0.35% and 0.22% in two pairs' in j['body'])
check('Last gain uses a fresh adjacent baseline','fresh run of the previous tip, 12.610 s/step' in j['body'])
check('Snapshot remains an unmerged proposal',j['state']=='open' and not j['merged'])
out={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'rows':rows,'timed_rows':timed,'aggregate':{'step_seconds_saved':first['seconds']-last['seconds'],'time_reduction_fraction':1-last['seconds']/first['seconds'],'relative_speed_gain':first['seconds']/last['seconds']-1,'MFU_percentage_point_change':last['MFU_percent']-first['MFU_percent']},'shared_epilogue':{'sequential_delta_seconds':row['table_time_delta_seconds'],'sequential_relative_speed_gain':row['table_relative_speed_gain'],'author_final_tip_removal_slowdowns':[.0035,.0022],'raw_pair_step_traces':None},'unidentified':{'component_independent_causal_effects':None,'full_factorial_interactions':None,'actual_device_trace':None,'cross_rack_scaling':None,'actual_GPU_reproduction':None},'source_sha256':{str(P.relative_to(R)):hashlib.sha256(P.read_bytes()).hexdigest()}}
(R/'analysis/moe_performance_attribution.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('Performance table source/arithmetic checks:',len(checks),'passed; timing rows:',len(timed))
