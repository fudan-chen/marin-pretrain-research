"""Create the shareable archive without environment files or Git metadata, then verify CRC."""
import pathlib,zipfile,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1];target=ROOT.with_suffix('.zip');tmp=target.with_suffix('.zip.tmp')
excluded={'.git','.venv','__pycache__','.DS_Store'}
paths=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and not any(part in excluded for part in p.relative_to(ROOT).parts) and p.suffix not in {'.pyc','.log'}]
with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in paths:z.write(p,pathlib.Path(ROOT.name)/p.relative_to(ROOT))
with zipfile.ZipFile(tmp) as z:
    assert z.testzip() is None
    assert len(z.infolist())==len(paths)
tmp.replace(target)
print('Archive:',target,'files:',len(paths),'bytes:',target.stat().st_size,'sha256:',hashlib.sha256(target.read_bytes()).hexdigest())
