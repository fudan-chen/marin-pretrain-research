"""Preserve completed upstream run logs and JUnit case identities."""
import hashlib,json,pathlib,platform,shutil,subprocess,xml.etree.ElementTree as ET
R=pathlib.Path(__file__).resolve().parents[1];runs=[]
commands=[['lib/iris/tests/journeys/test_checkpoint.py','lib/iris/tests/cluster/controller/test_checkpoint.py','lib/iris/tests/cluster/controller/test_db.py','lib/iris/tests/cluster/controller/test_rollout.py','lib/iris/tests/cluster/controller/test_migration_runner.py','lib/iris/tests/cluster/controller/test_endpoints_projection.py','-k','not periodic_checkpoint and not begin_checkpoint'],['lib/iris/tests/journeys/test_restart.py','lib/iris/tests/journeys/test_endpoints.py::test_coordinator_endpoint_survives_controller_restart_and_moves_to_the_replacement_attempt']]
for name,args in zip(['native','restart'],commands):
 cases=[]
 for suffix in ['xml','txt']:shutil.copyfile('/tmp/marin-v103-'+name+'.'+suffix,R/'analysis'/('controller_restore_'+name+'.'+suffix))
 for x in ET.parse(R/'analysis'/('controller_restore_'+name+'.xml')).getroot().iter('testcase'):
  outcome='failed' if x.find('failure') is not None else 'error' if x.find('error') is not None else 'skipped' if x.find('skipped') is not None else 'passed'
  cases.append({'classname':x.get('classname'),'name':x.get('name'),'seconds':float(x.get('time','0')),'outcome':outcome})
 assert all(x['outcome']=='passed' for x in cases)
 runs.append({'observed_tool_exitcode':0,'command':['uv','run','--frozen','--no-sync','--package','marin-iris','--group','test','--no-default-groups','pytest','-n','0','--tb=short']+args+['--junitxml=/tmp/marin-v103-'+name+'.xml'],'stdout_and_stderr_file':'analysis/controller_restore_'+name+'.txt','cases':cases,'passed':len(cases)})
shutil.copyfile('/tmp/marin-v103-faults.txt',R/'analysis/controller_restore_fault_output.txt')
checkout=pathlib.Path('/tmp/marin-iris-source-eee467');assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()=='eee467718515b2383fc3a433014afce4ab075b05';assert subprocess.run(['git','diff','--quiet','HEAD'],cwd=checkout).returncode==0
freeze=subprocess.check_output(['uv','pip','freeze','--python',str(checkout/'.venv/bin/python')],text=True);(R/'analysis/controller_restore_environment.txt').write_text(freeze)
source={str(pathlib.Path('sources')/x['file']):x['sha256'] for x in json.loads((R/'analysis/controller_restore_acquisition.json').read_text())['records']}
source['sources/gang_recovery_2026_10_07/lib/iris/src/iris/cluster/controller/db.py']=hashlib.sha256((R/'sources/gang_recovery_2026_10_07/lib/iris/src/iris/cluster/controller/db.py').read_bytes()).hexdigest()
artifacts=['analysis/controller_restore_'+n+'.'+suffix for n in ['native','restart'] for suffix in ['xml','txt']]+['analysis/controller_restore_environment.txt','analysis/controller_restore_fault_output.txt']
data={'source_revision':'eee467718515b2383fc3a433014afce4ab075b05','runs':runs,'passed':sum(x['passed'] for x in runs),'tracked_source_unchanged':True,'source_sha256':source,'artifact_sha256':{p:hashlib.sha256((R/p).read_bytes()).hexdigest() for p in artifacts},'platform':platform.platform(),'safe_default_marker_expression':'not slow and not docker and not requires_cluster and not manual','boundary':'Original upstream local SQLite, file:// checkpoint and journey tests. Journey closes and rebuilds real Controller over SQLite; ScriptedTaskBackend, manual clock and local log stack. Not forced process death, real worker/GPU, cloud transport or training weights.','actual_production_incident':None,'actual_forced_process_kill_restore':None,'actual_training_checkpoint_restore':None}
(R/'analysis/controller_restore_native_tests.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');print('Original upstream passes:',data['passed'])
