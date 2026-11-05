"""Read-only checks for current numbering, frozen science and public file scope."""
from pathlib import Path
import hashlib,json,re,subprocess,urllib.parse
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
expected={
 'ei2-vocabulary':'22d3ed6db1b79e7a4a7a03a32443c61ed9e248cd8c18d47eb67513b7db98b702',
 'ei2-pca-dimensions':'d93e942aec7135a7d4c6ee5741784ede7c481369c3aab1c1ea708ca8fe31e5f0',
 'ei2-pca-extended':'28debc00e0e3a033b4edd9722f35cb9ff405ca18611398ad0292094b1940d280',
 'ei2-pca-auto':'c4ad8320adb154970c8314866a15b59e8f53c67babbff6848e1935882193d3e6',
 'ei2-pca-select':'2a31cb393c62f88e8c3ea6bcc3d9eb28428a6c042b284e18574be8be9d29b449',
 'ei2-pca-uncapped':'92cd29ce2e4c4da69764c50ae4982214304222e673b4705c108c709939f2ee24',
 'ei2-pca-joint':'72db26c76714f910b4eb5b3c79b9579c36b21fd1178c4b371910e36c5ac87226',
 'ei2-centroid':'8acb936ae7222252bcd3bcdfb130c506341f243968bbcb239fc8aacb7b41a96b'}
for name,value in expected.items():assert sha(ROOT/'results'/name/'evaluation.json')==value,name
joint=ROOT/'results/ei2-pca-joint';final=json.loads((joint/'final-verification.json').read_text());assert final['passed'] and final['selected_entries_bound']==30 and final['selectors_described']==18
for name,value in final['files'].items():assert sha(joint/name)==value,name
assert sha(ROOT/'configs/ei2-pca-joint/audit_selected.py')==final['audit_code_sha256']
source=json.loads((joint/'source-lock.json').read_text())
for name,value in source['sources'].items():assert sha(ROOT/name)==value,name
audit=json.loads((joint/'selected-entry-audit.json').read_text());assert audit['passed'] and len(audit['records'])==30 and all(r['exact_content_match'] for r in audit['records'])
centroid=ROOT/'results/ei2-centroid';cp=json.loads((centroid/'verification.json').read_text());assert cp['passed'] and cp['selected_entries_bound']==9 and cp['trial_heads_reloaded']==27 and cp['primary_intervals_recomputed']==3 and cp['all_6582_F_verified']
for name,key in [('evaluation.json','evaluation_sha256'),('scores.npz','scores_sha256'),('source-lock.json','source_lock_sha256')]:assert sha(centroid/name)==cp[key]
for name,value in json.loads((centroid/'source-lock.json').read_text())['sources'].items():assert sha(ROOT/name)==value,name
registry=json.loads((ROOT/'docs/experiment-numbering.json').read_text(encoding='utf-8'));assert len(registry['readout'])==5 and len({r['id'] for r in registry['readout']})==5
for row in registry['readout']:assert (ROOT/row['source']).is_file()
repair=json.loads((ROOT/'results/ei2-pca-uncapped/verification-repair-source.json').read_text());proof=json.loads((ROOT/'results/ei2-pca-uncapped/verification.json').read_text());assert repair['evaluation_sha256_unchanged']==proof['evaluation_sha256'] and repair['scores_sha256_unchanged']==proof['scores_sha256']

images=ROOT/'results/ei2-images';pending=json.loads((ROOT/'docs/ei2-images-progress.json').read_text());assert pending['new_verified_results'] is False and pending['training_and_selection_sealed'] and pending['last_observed_completed']==1010
assert not (images/'evaluation.json').exists() and not (images/'verification.json').exists()
image_source=json.loads((images/'source-lock.json').read_text())
for name,value in image_source['sources'].items():assert sha(ROOT/name)==value,name
for name,key in [('manifest.json','manifest_sha256'),('grouping.json','grouping_sha256'),('preflight.json','preflight_sha256')]:assert sha(images/name)==image_source[key],name
image_seal=json.loads((images/'pre-evaluation-seal.json').read_text());assert image_seal['source_lock_sha256']==sha(images/'source-lock.json') and image_seal['search_sha256']==sha(images/'search.json') and len(image_seal['models'])==14 and len(image_seal['train_dev_indices'])==812
assert len(json.loads((images/'search.json').read_text()))==126
manifest=json.loads((images/'manifest.json').read_text());assert len(manifest['samples'])==1270
for row in manifest['samples']:assert not (set(row)&{'prompt','image_path','ocr','caption','text','image'})

paths=subprocess.check_output(['git','-C',str(ROOT),'ls-files','--cached','--others','--exclude-standard','-z']).decode('utf-8').split('\0');paths=sorted(set(filter(None,paths)));problems=[]
baseline=json.loads((ROOT/'viewer/public-baseline-objects.json').read_text())['already_public_binary_sha256']
credential=re.compile(r'(?:api[_-]?key|ssh[_-]?password|passwd|password)\s*[=:]\s*[\x22\x27]([^\x22\x27\r\n]{6,})[\x22\x27]',re.I)
for name in paths:
 path=ROOT/name
 if not path.is_file():continue
 if set(Path(name).parts)&{'cache','models','data','env','__pycache__','literature','readings'}:problems.append((name,'private directory'))
 if path.suffix.lower() in {'.joblib','.pkl','.pt','.pth','.safetensors','.pdf','.xlsx','.zip','.bundle','.pyc','.pem'} and (name not in baseline or sha(path)!=baseline[name]):problems.append((name,'new private/binary material'))
 if path.stat().st_size>95*1024**2:problems.append((name,'oversize blob'))
 if path.suffix.lower() in {'.py','.md','.json','.jsonl','.txt','.log','.html','.js','.yml','.yaml'}:
  text=path.read_text(encoding='utf-8',errors='replace')
  if re.search(r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}',text):problems.append((name,'credential-shaped literal'))
  for match in credential.finditer(text):
   value=match.group(1)
   if re.fullmatch(r'[A-Za-z0-9+/_=-]{6,}',value) and not re.search(r'example|placeholder|PASSWORD|environ',value,re.I):problems.append((name,'hardcoded credential candidate'))
assert not problems,problems
html=(ROOT/'viewer/experiments/detail.html').read_text(encoding='utf-8');assert 'CFG-2-3' in html and 'EI-2-7-2' in html and 'EI-3-1-2' in html
for name in ['README.md','research-map.md','docs/direction-registry.md','docs/experiment-numbering.md','viewer/README.md']:
 path=ROOT/name
 for link in re.findall(r'(?<!!)\[[^\]]+\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
  url=urllib.parse.urlsplit(link)
  if url.scheme or not url.path:continue
  assert (path.parent/urllib.parse.unquote(url.path)).exists(),(name,link)
print('PUBLICATION_SCOPE_NUMBERING_EIGHT_FROZEN_HASHES_AND_AUDITS_PASS',len(paths))
