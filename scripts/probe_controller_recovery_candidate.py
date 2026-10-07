"""Local original-versus-candidate failure controls; no live cluster operations."""
import datetime,hashlib,json,pathlib,sqlite3,sys,tempfile
from unittest.mock import patch
R=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from candidates.controller_recovery import checkpoint as cp
from candidates.controller_recovery.main import prepare_controller_state
from iris.cluster.controller import checkpoint as original_cp
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.rollout import RolloutPhase,RolloutRecord,write_rollout_record,read_rollout_record
from iris.cluster.controller.writes import meta_value_set
from sqlalchemy import text
import zstandard
checks=[];cases={}
def check(name,value,details=None):
 assert value,name
 checks.append({'name':name,'passed':True,'details':details})
def seed(path,value):
 db=ControllerDB(db_dir=path)
 with db.transaction() as tx:meta_value_set(tx,'candidate_marker',value)
 db.close()
def hashes(path):
 return {n:hashlib.sha256((path/n).read_bytes()).hexdigest() if (path/n).exists() else None for n in ['controller.sqlite3','auth.sqlite3']}
def marker(path):
 with sqlite3.connect(path/'controller.sqlite3') as c:return int(c.execute("SELECT value FROM meta WHERE key='candidate_marker'").fetchone()[0])
def call(fn):
 try:return {'result':fn(),'error':None}
 except Exception as e:return {'result':None,'error':type(e).__name__,'message':str(e)}
def ancestry(db):
 with db.read_snapshot() as q:return q.execute(text("SELECT value FROM meta WHERE key=:key"),{'key':cp.CHECKPOINT_EPOCH_META_KEY}).scalar_one_or_none()
with tempfile.TemporaryDirectory(prefix='iris-candidate-') as temp:
 root=pathlib.Path(temp)
 for rollback in [False,True]:
  for kind in ['missing','corrupt']:
   base=root/(str(rollback)+'-'+kind);remote=base/'remote';remote.mkdir(parents=True);target=remote/'controller-state'/'9999999999999';local=base/'local';seed(local,7);before=hashes(local)
   if kind=='corrupt':target.mkdir(parents=True);(target/'controller.sqlite3').write_bytes(b'not sqlite')
   if rollback:write_rollout_record(remote.as_uri(),RolloutRecord(phase=RolloutPhase.ROLLBACK_REQUESTED,image='constructed-image',rollback_checkpoint=target.as_uri()))
   out=call(lambda:prepare_controller_state(local,remote.as_uri(),fresh=False,checkpoint_path=None if rollback else target.as_uri()));out.update({'original_bytes_preserved':hashes(local)==before,'marker':marker(local),'rollout_phase':str(read_rollout_record(remote.as_uri()).phase) if rollback else None});cases[str(rollback)+'-'+kind]=out
   check('candidate '+str(rollback)+' '+kind+' rejects without changing original pair',out['error'] is not None and out['original_bytes_preserved'] and out['marker']==7)
   if rollback:check('failed rollback request not acknowledged '+kind,out['rollout_phase']=='rollback_requested')
 remote=root/'publish'/'remote';remote.mkdir(parents=True);source=root/'publish'/'source';seed(source,3);db=ControllerDB(db_dir=source)
 old_path,_=cp.write_checkpoint(db,remote.as_uri());old_epoch=cp.parse_checkpoint_epoch_ms(old_path);old_dir=pathlib.Path(old_path.removeprefix('file://'));completion=json.loads((old_dir/cp.COMPLETION_FILE).read_text())
 check('complete publication records two actual compressed file hashes',set(completion['files'])==cp.REQUIRED_COMPRESSED_FILES and all(cp._file_signature(old_dir/n)==h for n,h in completion['files'].items()))
 check('ancestry updated only after completion publication',ancestry(db)==old_epoch and cp.latest_checkpoint_epoch_ms(remote.as_uri())==old_epoch)
 local=root/'roundtrip';seed(local,7);check('complete pair restores as healthy',cp.download_checkpoint_to_local(remote.as_uri(),local) and cp.probe_database_dir(local).healthy and marker(local)==3)
 original_copy=cp._fsspec_copy
 for suffix,label in [('auth.sqlite3.zst','second-file'),(cp.COMPLETION_FILE,'completion-file')]:
  before_ancestor=ancestry(db)
  def fail_copy(src,dst):
   if dst.endswith(suffix):raise OSError('injected '+label+' failure')
   return original_copy(src,dst)
  with patch.object(cp,'_fsspec_copy',fail_copy):out=call(lambda:cp.write_checkpoint(db,remote.as_uri()))
  out.update({'ancestry':ancestry(db),'latest':cp.latest_checkpoint_epoch_ms(remote.as_uri())});cases[label]=out
  check(label+' failure leaves ancestry unchanged',out['error']=='OSError' and out['ancestry']==before_ancestor)
  check(label+' partial directory not selected over completed older candidate',out['latest']==old_epoch)
 before_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in old_dir.iterdir() if p.is_file()}
 with patch.object(cp.Timestamp,'now',return_value=cp.Timestamp.from_ms(old_epoch)):
  collision=call(lambda:cp.write_checkpoint(db,remote.as_uri()))
 check('existing completed directory cannot be overwritten by same epoch',collision['error']=='ValueError' and before_files=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in old_dir.iterdir() if p.is_file()})
 cases['epoch-collision']=collision
 # Successful rollback is consumed once; a later local update must not be erased.
 rollback_local=root/'rollback-success';seed(rollback_local,7);write_rollout_record(remote.as_uri(),RolloutRecord(phase=RolloutPhase.ROLLBACK_REQUESTED,image='constructed-image',rollback_checkpoint=old_path))
 prepare_controller_state(rollback_local,remote.as_uri(),fresh=False,checkpoint_path=None)
 check('successful rollback restores selected pair and acknowledges once',marker(rollback_local)==3 and cp.probe_database_dir(rollback_local).healthy and read_rollout_record(remote.as_uri()).phase is RolloutPhase.ROLLED_BACK)
 db_local=ControllerDB(db_dir=rollback_local)
 with db_local.transaction() as tx:meta_value_set(tx,'candidate_marker',8)
 db_local.close();prepare_controller_state(rollback_local,remote.as_uri(),fresh=False,checkpoint_path=None)
 check('second startup reuses descendant and does not rewind again',marker(rollback_local)==8)
 db.close()
 # A committed candidate with bad content fails explicitly, retaining both local files.
 for kind in ['missing-auth','wrong-digest','corrupt-auth-valid-digest','corrupt-main-valid-digest','wrong-identity']:
  base=root/kind;newremote=base/'remote';newremote.mkdir(parents=True);src=base/'source';seed(src,3);d=ControllerDB(db_dir=src);path,_=cp.write_checkpoint(d,newremote.as_uri());d.close();directory=pathlib.Path(path.removeprefix('file://'));manifest=json.loads((directory/cp.COMPLETION_FILE).read_text());auth=directory/'auth.sqlite3.zst'
  if kind=='missing-auth':auth.unlink()
  elif kind=='wrong-digest':auth.write_bytes(auth.read_bytes()+b'altered')
  elif kind in ['corrupt-auth-valid-digest','corrupt-main-valid-digest']:
   bad=auth if kind=='corrupt-auth-valid-digest' else directory/'controller.sqlite3.zst';bad.write_bytes(zstandard.ZstdCompressor().compress(b'not sqlite'));manifest['files'][bad.name]=cp._file_signature(bad)
  else:manifest['epoch_ms']+=1
  (directory/cp.COMPLETION_FILE).write_text(json.dumps(manifest));local=base/'local';seed(local,7);before=hashes(local);out=call(lambda:cp.download_checkpoint_to_local(newremote.as_uri(),local,checkpoint_dir=path));cases[kind]=out
  check(kind+' fails closed retaining original bytes',out['error'] is not None and hashes(local)==before)
 # Valid old-format output is intentionally unsupported, never silently fresh.
 legacy=root/'legacy';legacy.mkdir();src=root/'legacy-source';seed(src,3);d=ControllerDB(db_dir=src);path,_=original_cp.write_checkpoint(d,legacy.as_uri());d.close();local=root/'legacy-local';seed(local,7);before=hashes(local);out=call(lambda:prepare_controller_state(local,legacy.as_uri(),fresh=False,checkpoint_path=None));cases['legacy']=out
 check('markerless legacy latest requires explicit migration and retains original state',out['error']=='ValueError' and hashes(local)==before)
model=json.loads((R/'analysis/controller_recovery_candidate.json').read_text());files=model['source_sha256']|model['candidate_sha256'];files['scripts/probe_controller_recovery_candidate.py']=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
(R/'analysis/controller_recovery_candidate_controls.json').write_text(json.dumps({'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks_passed':len(checks),'checks':checks,'cases':cases,'source_sha256':files,'status':'candidate_not_integrated','boundary':'Real SQLite/zstd and original full modules with explicit candidate source changes; local file://; constructed markers; injected file-copy failures only. Not production or forced process crash.','actual_production_fix':None,'actual_legacy_migration':None,'actual_cloud_transport':None},ensure_ascii=False,indent=2)+'\n');print('Candidate controls:',len(checks),'passed')
