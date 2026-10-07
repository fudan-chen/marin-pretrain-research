"""Build an unintegrated strict-format candidate from the frozen original modules."""
import difflib,hashlib,json,pathlib
R=pathlib.Path(__file__).resolve().parents[1];B=R/'sources/controller_restore_2026_10_07/lib/iris/src/iris/cluster/controller';D=R/'candidates/controller_recovery';D.mkdir(parents=True,exist_ok=True)
(D.parent/'__init__.py').touch();(D/'__init__.py').touch()
s=(B/'checkpoint.py').read_text();original=s
s=s.replace('import logging\n','import hashlib\nimport json\nimport logging\n',1)
helpers='''
COMPLETION_FILE = "checkpoint.complete.json"
REQUIRED_COMPRESSED_FILES = {"controller.sqlite3.zst", "auth.sqlite3.zst"}


def _file_signature(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _validate_completion(record: dict, checkpoint_dir: str) -> dict:
    if not isinstance(record, dict) or record.get("format") != 1:
        raise ValueError("Unsupported checkpoint completion format")
    epoch = parse_checkpoint_epoch_ms(checkpoint_dir)
    if type(record.get("epoch_ms")) is not int or record["epoch_ms"] != epoch:
        raise ValueError("Checkpoint completion identity mismatch")
    files = record.get("files")
    if not isinstance(files, dict) or set(files) != REQUIRED_COMPRESSED_FILES:
        raise ValueError("Checkpoint completion requires main and auth")
    for spec in files.values():
        if not isinstance(spec, dict) or type(spec.get("bytes")) is not int or spec["bytes"] <= 0:
            raise ValueError("Invalid checkpoint size")
        digest = spec.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid checkpoint digest")
    return record


def _read_completion(checkpoint_dir: str) -> dict | None:
    try:
        with open_url(prefix_join(checkpoint_dir, COMPLETION_FILE), "rb") as stream:
            record = json.load(stream)
    except FileNotFoundError:
        return None
    return _validate_completion(record, checkpoint_dir)

'''
s=s.replace('CHECKPOINT_EPOCH_META_KEY = "last_checkpoint_epoch_ms"','CHECKPOINT_EPOCH_META_KEY = "last_checkpoint_epoch_ms"\n'+helpers,1)
s=s.replace('def _compress_and_upload_db(local_path: Path, remote_url: str, label: str) -> None:', 'def _compress_and_upload_db(local_path: Path, remote_url: str, label: str) -> dict:',1)
s=s.replace('        _fsspec_copy(str(tmp_zst), remote_url)\n', '        signature = _file_signature(tmp_zst)\n        _fsspec_copy(str(tmp_zst), remote_url)\n',1)
s=s.replace('        logger.info("checkpoint %s DB uploaded to %s", label, remote_url)\n','        logger.info("checkpoint %s DB uploaded to %s", label, remote_url)\n        return signature\n',1)
a=s.index('    main_url = prefix_join(checkpoint_dir,');b=s.index('    # The marker proves',a)
s=s[:a]+'''    if backup.auth_path is None:
        raise ValueError("Strict checkpoint requires an auth database")
    from rigging.filesystem.storage_path import StoragePath
    if StoragePath(checkpoint_dir).exists():
        raise ValueError("Checkpoint directory already exists; refusing overwrite")
    files = {}
    for name, path in [(ControllerDB.DB_FILENAME, backup.main_path), (ControllerDB.AUTH_DB_FILENAME, backup.auth_path)]:
        compressed = name + ".zst"
        files[compressed] = _compress_and_upload_db(path, prefix_join(checkpoint_dir, compressed), name)
    completion = {"format": 1, "epoch_ms": backup.created_at.epoch_ms(), "files": files}
    completion_path = backup.main_path.with_suffix(".complete.json")
    try:
        completion_path.write_text(json.dumps(completion, sort_keys=True) + "\\n")
        _fsspec_copy(str(completion_path), prefix_join(checkpoint_dir, COMPLETION_FILE))
    finally:
        completion_path.unlink(missing_ok=True)

'''+s[b:]
a=s.index('    # Return the most recent (highest timestamp)');b=s.index('\n\ndef latest_checkpoint_epoch_ms',a)
s=s[:a]+'''    for _, path in sorted(timestamp_dirs, reverse=True):
        reconstructed = _reconstruct_uri(remote_state_dir, path)
        if _read_completion(reconstructed) is not None:
            return reconstructed
    raise ValueError("Checkpoint directories exist but none has a completion record; legacy migration required")
'''+s[b:]
needle='        _sync_dir(source_dir, staging_dir)\n';replacement=needle+'''        completion_path = staging_dir / COMPLETION_FILE
        if not completion_path.exists():
            raise ValueError("Checkpoint has no completion record; refusing uncommitted or legacy restore")
        completion = _validate_completion(json.loads(completion_path.read_text()), source_dir)
        for name, expected in completion["files"].items():
            path = staging_dir / name
            if not path.exists() or _file_signature(path) != expected:
                raise ValueError("Checkpoint size/digest mismatch: " + name)
''';assert s.count(needle)==1;s=s.replace(needle,replacement,1)
s=s.replace('from rigging.filesystem.storage_path import prefix_join','from rigging.filesystem.storage_path import StoragePath, prefix_join',1).replace('    from rigging.filesystem.storage_path import StoragePath\n','',1)
(D/'checkpoint.py').write_text(s)
main=(B/'main.py').read_text();oldmain=main;main=main.replace('from iris.cluster.controller.checkpoint import (','from candidates.controller_recovery.checkpoint import (',1)
needle='    if db_dir.exists():\n        shutil.rmtree(db_dir)\n    db_dir.mkdir(parents=True, exist_ok=True)\n    if not download_checkpoint_to_local';assert main.count(needle)==1;main=main.replace(needle,'    if not download_checkpoint_to_local',1);(D/'main.py').write_text(main)
patch=''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='upstream/checkpoint.py',tofile='candidate/checkpoint.py'))+''.join(difflib.unified_diff(oldmain.splitlines(True),main.splitlines(True),fromfile='upstream/main.py',tofile='candidate/main.py'))
(D/'proposal.patch').write_text(patch)
record={'status':'candidate_not_integrated','base_revision':'eee467718515b2383fc3a433014afce4ab075b05','source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [B/'checkpoint.py',B/'main.py']},'candidate_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [D/'checkpoint.py',D/'main.py',D/'proposal.patch']},'legacy_policy':'Reject markerless checkpoints; production adoption needs explicit migration. No automatic fresh or silent auth initialization.','unverified':['cloud transport','concurrent publishers','forced process crash between renames','production integration','signing-key semantics','model training checkpoint recovery']}
(R/'analysis/controller_recovery_candidate.json').write_text(json.dumps(record,indent=2)+'\n');print('Candidate built; no original source changed')
