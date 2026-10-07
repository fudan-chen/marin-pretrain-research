"""Demonstrate sequential snapshot skew with constructed cross-file data, not production auth rows."""
import hashlib,json,pathlib,sqlite3,sys,tempfile
from unittest.mock import patch
R=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from iris.cluster.controller import checkpoint as original
from iris.cluster.controller.db import ControllerDB
from candidates.controller_recovery import checkpoint as candidate
from sqlalchemy import text
checks=[];cases={}
def check(name,value):
 assert value,name
 checks.append({'name':name,'passed':True})
def read(path):
 with sqlite3.connect(path) as c:return c.execute('SELECT generation FROM research_generation').fetchone()[0]
with tempfile.TemporaryDirectory(prefix='iris-snapshot-semantics-') as tmp:
 root=pathlib.Path(tmp)
 fresh=ControllerDB(db_dir=root/'fresh')
 with sqlite3.connect(fresh.auth_db_path) as c:tables=[x[0] for x in c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
 check('fresh current auth schema has no application tables',tables==[]);fresh.close()
 for name,module,inject in [('original_quiet',original,False),('original_interleaved',original,True),('candidate_interleaved',candidate,True)]:
  db=ControllerDB(db_dir=root/name)
  with db.transaction() as tx:
   tx.execute(text('CREATE TABLE main.research_generation (generation INTEGER NOT NULL)'))
   tx.execute(text('CREATE TABLE auth.research_generation (generation INTEGER NOT NULL)'))
   tx.execute(text('INSERT INTO main.research_generation VALUES (0)'));tx.execute(text('INSERT INTO auth.research_generation VALUES (0)'))
  backup_fn=db.backup_to;interleavings=[]
  def backup_then_update(destination):
   backup_fn(destination)
   if inject:
    with db.transaction() as tx:
     tx.execute(text('UPDATE main.research_generation SET generation=1'));tx.execute(text('UPDATE auth.research_generation SET generation=1'))
    interleavings.append({'after_main_backup_live_main':read(db.db_path),'after_main_backup_live_auth':read(db.auth_db_path)})
  with patch.object(db,'backup_to',backup_then_update):
   backup=module.backup_databases(db)
  try:
   pair=[read(backup.main_path),read(backup.auth_path)];live=[read(db.db_path),read(db.auth_db_path)]
   checksums={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [backup.main_path,backup.auth_path]}
   cases[name]={'backup_pair':pair,'live_pair':live,'interleavings':interleavings,'backup_sha256':checksums}
   check(name+' captures expected snapshot generations',pair==([0,1] if inject else [0,0]) and live==([1,1] if inject else [0,0]))
   with sqlite3.connect(backup.main_path) as c:main_check=c.execute('PRAGMA quick_check').fetchone()[0]
   with sqlite3.connect(backup.auth_path) as c:auth_check=c.execute('PRAGMA quick_check').fetchone()[0]
   check(name+' both files individually valid',main_check==auth_check=='ok')
   if name=='candidate_interleaved':
    remote=(root/'candidate-remote');remote.mkdir();path,_=module.upload_checkpoint(db,backup,remote.as_uri());restored=root/'candidate-restored';result=module.download_checkpoint_to_local(remote.as_uri(),restored)
    pair2=[read(restored/'controller.sqlite3'),read(restored/'auth.sqlite3')];cases[name]['restored_pair']=pair2
    check('completion hashes and strict restore accept individually valid skewed pair',result and pair2==[0,1] and module.probe_database_dir(restored).healthy)
  finally:backup.cleanup();db.close()
files={str(pathlib.Path('sources')/x['file']):x['sha256'] for x in json.loads((R/'analysis/snapshot_semantics_acquisition.json').read_text())['records']};files.update(json.loads((R/'analysis/controller_recovery_candidate.json').read_text())['source_sha256']);files.update(json.loads((R/'analysis/controller_recovery_candidate.json').read_text())['candidate_sha256']);files['scripts/probe_snapshot_semantics.py']=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
(R/'analysis/snapshot_semantics.json').write_text(json.dumps({'checks_passed':len(checks),'checks':checks,'cases':cases,'fresh_auth_application_tables':tables,'source_sha256':files,'boundary':'Original full backup functions and unintegrated candidate, real SQLite, local storage. Constructed research_generation tables, transaction injected after original main backup before auth backup. Not production auth schema, crash atomicity, real concurrent threads or training-state reproduction.','actual_production_crossfile_invariant':None,'actual_signing_key_loss':None,'actual_training_state_skew':None},indent=2)+'\n');print('Snapshot controls:',len(checks))
