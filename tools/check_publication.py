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
 'ei2-pca-uncapped':'92cd29ce2e4c4da69764c50ae4982214304222e673b4705c108c709939f2ee24'}
for name,value in expected.items():assert sha(ROOT/'results'/name/'evaluation.json')==value,name
registry=json.loads((ROOT/'docs/experiment-numbering.json').read_text(encoding='utf-8'));assert len(registry['readout'])==5 and len({r['id'] for r in registry['readout']})==5
for row in registry['readout']:assert (ROOT/row['source']).is_file()
repair=json.loads((ROOT/'results/ei2-pca-uncapped/verification-repair-source.json').read_text());proof=json.loads((ROOT/'results/ei2-pca-uncapped/verification.json').read_text());assert repair['evaluation_sha256_unchanged']==proof['evaluation_sha256'] and repair['scores_sha256_unchanged']==proof['scores_sha256']
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
print('PUBLICATION_SCOPE_NUMBERING_AND_SIX_FROZEN_HASHES_PASS',len(paths))
