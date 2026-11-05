import hashlib,importlib.util,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
ORIGIN=ROOT.parents[1];H1=ORIGIN/'cache/oc4-worktree';H2=ORIGIN/'cache/oc4c-worktree'
OUT=ROOT/'results/oc-retraining'
HEADS=['OC-4-1-1','OC-4-1-3','OC-4-4-1'];SEEDS=[20261104,20261105,20261106]
GROUPS=['old_full','old_small','or_small'];NAMES=['B']+[group+':'+name for group in GROUPS for name in HEADS]
PAIRS=[pair for j in range(3) for pair in [(7+j,1+j),(7+j,4+j),(4+j,1+j)]]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with Path(p).open(encoding='utf-8') as f:return [json.loads(line) for line in f if line.strip()]
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode())
def h1_core():
    spec=importlib.util.spec_from_file_location('h3_h1_core',ROOT/'configs/oc-heads/core.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def predict(artifact,x):
    import torch
    torch.set_num_threads(2);torch.backends.cuda.enable_flash_sdp(True);torch.backends.cuda.enable_mem_efficient_sdp(True);torch.backends.cuda.enable_math_sdp(True);torch.use_deterministic_algorithms(False)
    return h1_core().score(artifact,x)
def threshold(s):
    x=np.asarray(s)
    if x.shape!=(400,) or not np.isfinite(x).all():raise ValueError('400 safe scores required')
    return float(np.sort(x)[380])
def verify_source():
    seal=json.loads((OUT/'source-lock.json').read_text())
    for path,sha in seal['sha256'].items():assert digest(ROOT/path)==sha,path
    assert digest(OUT/'manifest.json')==seal['manifest_sha256']
    for base,key in [(H1,'h1_files'),(H2,'h2_files')]:
        for path,sha in seal[key].items():assert digest(base/path)==sha,path
    return seal
