"""Plot observed sampled intervals, with no outage attribution."""
import pathlib,json,datetime,hashlib,re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as md
R=pathlib.Path(__file__).resolve().parents[1]
j=json.loads((R/'analysis/operational_progress.json').read_text()); rows=j['intervals']
x=[datetime.datetime.fromisoformat(r['end_utc']) for r in rows]
plt.rcParams.update({'svg.fonttype':'none','svg.hashsalt':'marin-v110-operational','font.size':10})
fig,ax=plt.subplots(figsize=(11,4.8),layout='constrained')
ax.scatter(x,[r['calendar_seconds_per_logged_step'] for r in rows],s=9,alpha=.65,label='Calendar seconds / logged step (sampled interval)',color='#ba6a20')
ax.plot(x,[r['end_logged_duration'] for r in rows],lw=1,label='Logged step duration at interval endpoint',color='#215f9a')
b=j['triage_alignment']['sampled_bracket']; ax.axvspan(datetime.datetime.fromisoformat(b['start_utc']),datetime.datetime.fromisoformat(b['end_utc']),alpha=.17,color='#6e5a99',label='Bracket containing public triage report (cause unassigned)')
ax.set_yscale('log');ax.set_ylabel('Seconds / step (log scale)');ax.set_xlabel('UTC; sample ends at step 219263, summary later at 219518')
ax.xaxis.set_major_formatter(md.DateFormatter('%m-%d',tz=datetime.timezone.utc));ax.grid(alpha=.2);ax.legend(loc='upper left',fontsize=8)
ax.set_title('Hero: 999 observed intervals from 1000 sampled records\nInterval rate and endpoint duration measure different scopes')
fig.savefig(R/'assets/operational_progress.svg',metadata={'Date':None});plt.close(fig)
p=R/'assets/operational_progress.svg';s=p.read_text()
for ident in re.findall(r'id="([^"]+)"',s):
 s=s.replace('id="'+ident+'"','id="operational-'+ident+'"').replace('#'+ident+'"','#operational-'+ident+'"').replace('#'+ident+')','#operational-'+ident+')')
at=s.index('>',s.index('<svg'))+1
s=s[:at]+'\n<title id="operational-progress-title">Observed calendar progress and logged step duration</title><desc id="operational-progress-desc">999 sampled intervals. Slow calendar progress does not identify fault downtime. Summary step 219518 is later than the sampled endpoint 219263.</desc>'+s[at:];s=s.replace('<svg ','<svg role="img" aria-labelledby="operational-progress-title operational-progress-desc" ',1);p.write_text(s)
(R/'analysis/operational_plot_binding.json').write_text(json.dumps({'rows':len(rows),'analysis_sha256':hashlib.sha256((R/'analysis/operational_progress.json').read_bytes()).hexdigest(),'figure_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'actual_browser_rendering':None},indent=2)+'\n')
