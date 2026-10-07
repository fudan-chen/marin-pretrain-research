"""Run unmodified upstream local SQLite controller tests with the safe default markers."""
import argparse,datetime,hashlib,json,pathlib,platform,subprocess,time,xml.etree.ElementTree as ET
R=pathlib.Path(__file__).resolve().parents[1]
a=argparse.ArgumentParser();a.add_argument('checkout',type=pathlib.Path);args=a.parse_args()
rev='eee467718515b2383fc3a433014afce4ab075b05';assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.checkout,text=True).strip()==rev
subprocess.run(['git','diff','--exit-code','HEAD'],cwd=args.checkout,check=True,stdout=subprocess.DEVNULL)
selection='coscheduled or stale_attempt or cumulative or max_task_failures_tolerance or timeout_charges or cancel_job_holds_resources or heartbeat_finalizes_stranded or worker_reported_killed or batch_success_and_failure'
xml=R/'analysis/gang_recovery_pytest.xml';log=R/'analysis/gang_recovery_pytest.txt'
cmd=['uv','run','--frozen','--no-sync','--package','marin-iris','--group','test','--no-default-groups','pytest','-n','0','--tb=short','lib/iris/tests/cluster/controller/test_transitions.py','-k',selection,'--junitxml='+str(xml)]
start=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.monotonic()
with log.open('w') as f:p=subprocess.run(cmd,cwd=args.checkout,stdout=f,stderr=subprocess.STDOUT,timeout=240)
tracked_source_unchanged=subprocess.run(['git','diff','--quiet','HEAD'],cwd=args.checkout).returncode==0
cases=[]
if xml.exists():
 for x in ET.parse(xml).getroot().iter('testcase'):
  outcome='failed' if x.find('failure') is not None else 'error' if x.find('error') is not None else 'skipped' if x.find('skipped') is not None else 'passed'
  cases.append({'classname':x.get('classname'),'name':x.get('name'),'seconds':float(x.get('time','0')),'outcome':outcome})
source={str(pathlib.Path('sources/gang_recovery_2026_10_07')/x['upstream_path']):x['sha256'] for x in json.loads((R/'analysis/gang_recovery_acquisition.json').read_text())['records']}
versions=subprocess.check_output(['uv','pip','freeze','--python',str(args.checkout/'.venv/bin/python')],text=True)
(R/'analysis/gang_recovery_environment.txt').write_text(versions)
data={'source_revision':rev,'command':cmd,'started_utc':start,'duration_seconds':time.monotonic()-t,'returncode':p.returncode,'tracked_source_unchanged':tracked_source_unchanged,'cases':cases,'passed':sum(x['outcome']=='passed' for x in cases),'source_sha256':source,'platform':platform.platform(),'python':subprocess.check_output([str(args.checkout/'.venv/bin/python'),'--version'],text=True).strip(),'safe_default_marker_expression':'not slow and not docker and not requires_cluster and not manual','test_boundary':'Original upstream tests, real local SQLite/controller; constructed workers and observations; no live backend, GPU or real training checkpoint','dependency_preparation':'Initial unfrozen sync interrupted during resolution; frozen existing upstream uv.lock used, no upstream source edits','artifact_sha256':{str(x.relative_to(R)):hashlib.sha256(x.read_bytes()).hexdigest() for x in [xml,log,R/'analysis/gang_recovery_environment.txt'] if x.exists()},'actual_Hero_deployed_revision':None,'actual_Hero_retry_parameters':None,'actual_Hero_recovery_trace':None,'actual_controller_crash_restart':None}
(R/'analysis/gang_recovery_native_tests.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');print('Native controller tests:',data['passed'],'passed;',len(cases),'cases; exit',p.returncode)
raise SystemExit(p.returncode)
