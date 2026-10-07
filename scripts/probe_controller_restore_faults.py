"""Execute original Iris restore/upload paths with local files and explicit injected faults."""
import datetime,hashlib,json,pathlib,sqlite3,tempfile
from unittest.mock import patch
from iris.cluster.controller import checkpoint as cp
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.main import prepare_controller_state
from iris.cluster.controller.rollout import RolloutPhase,RolloutRecord,write_rollout_record,read_rollout_record
from iris.cluster.controller.writes import meta_value_set
R=pathlib.Path(__file__).resolve().parents[1];checks=[];cases={}
def check(name,value):
 assert value,name
 checks.append({'name':name,'passed':True})
def seed(p,marker):
 db=ControllerDB(db_dir=p)
 with db.transaction() as tx:meta_value_set(tx,'restore_probe_marker',marker)
 db.close()
def marker(p):
 if not (p/'controller.sqlite3').exists():return None
 with sqlite3.connect(p/'controller.sqlite3') as c:
  row=c.execute("SELECT value FROM meta WHERE key='restore_probe_marker'").fetchone()
 return int(row[0]) if row else None
def attempt(fn):
 try:return {'result':fn(),'exception':None}
 except Exception as e:return {'result':None,'exception':type(e).__name__,'message':str(e)}
with tempfile.TemporaryDirectory(prefix='iris-research-restore-') as temp:
 root=pathlib.Path(temp)
 for rollback in [False,True]:
  for kind in ['corrupt','missing']:
   base=root/(str(rollback)+'-'+kind);remote=base/'remote';target=remote/'controller-state'/'9999999999999';local=base/'local';seed(local,7)
   if kind=='corrupt':target.mkdir(parents=True);(target/'controller.sqlite3').write_bytes(b'not a sqlite database')
   else:remote.mkdir(parents=True)
   before_hash={n:hashlib.sha256((local/n).read_bytes()).hexdigest() for n in ['controller.sqlite3','auth.sqlite3']}
   remote_url=remote.as_uri();target_url=target.as_uri()
   if rollback:write_rollout_record(remote_url,RolloutRecord(phase=RolloutPhase.ROLLBACK_REQUESTED,image='constructed-image',rollback_checkpoint=target_url))
   outcome=attempt(lambda:prepare_controller_state(local,remote_url,fresh=False,checkpoint_path=None if rollback else target_url))
   outcome['local_main_exists']=(local/'controller.sqlite3').exists();outcome['local_marker']=marker(local) if outcome['local_main_exists'] else None
   outcome['before_file_sha256']=before_hash
   outcome['after_file_sha256']={n:hashlib.sha256((local/n).read_bytes()).hexdigest() if (local/n).exists() else None for n in before_hash}
   outcome['original_file_bytes_preserved']=before_hash==outcome['after_file_sha256']
   outcome['staging_dirs_remaining']=[p.name for p in base.glob('.local.restore-*')]
   outcome['rollout_phase']=str(read_rollout_record(remote_url).phase) if rollback else None
   key=('rollback_' if rollback else 'ordinary_')+kind;cases[key]=outcome
   check(key+' raises and cleanup finishes',outcome['exception'] is not None and not outcome['staging_dirs_remaining'])
   check(key+' retains original state' if not rollback else key+' removes original state before failing',outcome['local_marker']==7 and outcome['original_file_bytes_preserved'] if not rollback else not outcome['local_main_exists'])
 remote=root/'partial'/'remote';remote.mkdir(parents=True);source=root/'partial'/'source';seed(source,3);db=ControllerDB(db_dir=source)
 original_copy=cp._fsspec_copy
 def fault_copy(src,dst):
  if dst.endswith('auth.sqlite3.zst'):raise OSError('injected second-file upload failure')
  return original_copy(src,dst)
 try:
  with patch.object(cp,'_fsspec_copy',fault_copy):outcome=attempt(lambda:cp.write_checkpoint(db,remote.as_uri()))
  with db.read_snapshot() as q:ancestry=q.execute(__import__('sqlalchemy').text("SELECT value FROM meta WHERE key=:k"),{'k':cp.CHECKPOINT_EPOCH_META_KEY}).scalar_one_or_none()
 finally:db.close()
 latest=cp.latest_checkpoint_epoch_ms(remote.as_uri());directory=remote/'controller-state'/str(latest)
 outcome.update({'local_ancestry':ancestry,'latest_epoch_ms':latest,'main_uploaded':(directory/'controller.sqlite3.zst').exists(),'auth_uploaded':(directory/'auth.sqlite3.zst').exists()});cases['partial_upload']=outcome
 check('failed second-file upload leaves local ancestry unchanged',outcome['exception']=='OSError' and ancestry is None)
 check('partial checkpoint remains discoverable as latest',latest is not None and outcome['main_uploaded'] and not outcome['auth_uploaded'])
 local=root/'partial'/'restore';seed(local,7);before=cp.probe_database_dir(local);downloaded=cp.download_checkpoint_to_local(remote.as_uri(),local);after=cp.probe_database_dir(local)
 cases['partial_download']={'returned':downloaded,'before_healthy':before.healthy,'after_healthy':after.healthy,'detail':after.detail,'marker':marker(local),'auth_exists':(local/'auth.sqlite3').exists()}
 check('main-only checkpoint download returns true but pair probe unhealthy',downloaded is True and before.healthy and not after.healthy and not (local/'auth.sqlite3').exists() and marker(local)==3)
 local2=root/'partial'/'startup';seed(local2,7);startup=attempt(lambda:prepare_controller_state(local2,remote.as_uri(),fresh=False,checkpoint_path=None));startup['before_db_open_healthy']=cp.probe_database_dir(local2).healthy
 db2=ControllerDB(db_dir=local2);db2.close();startup['after_db_open_healthy']=cp.probe_database_dir(local2).healthy;startup['marker']=marker(local2);cases['partial_startup']=startup
 check('startup accepts partial download and DB construction creates missing auth file',startup['exception'] is None and not startup['before_db_open_healthy'] and startup['after_db_open_healthy'] and startup['marker']==3)
source={str(pathlib.Path('sources')/x['file']):x['sha256'] for x in json.loads((R/'analysis/controller_restore_acquisition.json').read_text())['records']}
source['sources/gang_recovery_2026_10_07/lib/iris/src/iris/cluster/controller/db.py']=hashlib.sha256((R/'sources/gang_recovery_2026_10_07/lib/iris/src/iris/cluster/controller/db.py').read_bytes()).hexdigest()
(R/'analysis/controller_restore_faults.json').write_text(json.dumps({'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'executed_script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'source_revision':'eee467718515b2383fc3a433014afce4ab075b05','source_sha256':source,'checks_passed':len(checks),'checks':checks,'cases':cases,'boundary':'Original installed Iris full modules, local file:// storage, original SQLite/zstd; patch only auth-file upload to raise OSError. Numeric checkpoint and marker values are constructed. No production cluster operations or user state touched.','actual_production_incident':None,'actual_auth_key_loss':None,'actual_training_checkpoint_restore':None},ensure_ascii=False,indent=2)+'\n');print('Fault controls:',len(checks),'passed')
