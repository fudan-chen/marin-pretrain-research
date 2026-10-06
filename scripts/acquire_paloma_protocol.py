"""Archive public metadata/card and final paper anonymously; never request gated data."""
import pathlib,json,datetime,hashlib,subprocess
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/paloma_protocol_2026_10_06';assert not D.exists()
def fetch(u):
 return subprocess.run(['curl','--fail','--silent','--show-error','--location','--retry','2','--connect-timeout','15','--max-time','45',u],check=True,capture_output=True).stdout
u='https://huggingface.co/api/datasets/allenai/paloma';meta=fetch(u);sha=json.loads(meta)['sha'];assert len(sha)==40
items=[('hf_metadata.json',u,meta)]
card='https://huggingface.co/datasets/allenai/paloma/blob/'+sha+'/README.md';b=fetch(card);assert b'ptb' in b and b'twitterAAE_HELM_fixed' in b;items.append(('dataset_card.html',card,b))
paper='https://papers.neurips.cc/paper_files/paper/2024/file/760b2d94398aa61468aa3bc11506d9ea-Paper-Datasets_and_Benchmarks_Track.pdf';b=fetch(paper);assert b.startswith(b'%PDF');items.append(('paloma_neurips_2024.pdf',paper,b))
rows=[];D.mkdir()
for name,url,b in items:
 (D/name).write_bytes(b);rows.append({'file':str((D/name).relative_to(R/'sources')),'url':url,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
p=R/'sources/source_manifest.json';j=json.loads(p.read_text());j.extend(rows);p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n')
(R/'analysis/paloma_protocol_acquisition.json').write_text(json.dumps({'hf_revision':sha,'authentication':'anonymous HTTPS','data_payload_requested':False,'files':rows},ensure_ascii=False,indent=2)+'\n');print(sha,[(x['file'],x['bytes']) for x in rows])
