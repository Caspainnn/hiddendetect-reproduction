import hashlib,importlib.util,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
ORIGIN=ROOT.parents[1]
H1=ORIGIN/'cache/oc4-worktree'
OUT=ROOT/'results/oc-confirmation'
NAMES=['B','OC-4-1-1','OC-4-1-3','OC-4-4-1']
SEEDS=[20261104,20261105,20261106]
PAIRS=[(1,0),(2,0),(3,0),(2,1),(3,1),(3,2)]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with Path(p).open(encoding='utf-8') as f:return [json.loads(line) for line in f if line.strip()]
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode())
def h1_core():
    spec=importlib.util.spec_from_file_location('h1_confirmation_core',ROOT/'configs/oc-heads/core.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def verify_source():
    seal=json.loads((OUT/'source-lock.json').read_text())
    for name,sha in seal['sha256'].items():assert digest(ROOT/name)==sha,name
    assert digest(OUT/'manifest.json')==seal['manifest_sha256']
    for name,sha in seal['models_sha256'].items():assert digest(H1/name)==sha
    for name,sha in seal['h1_result_sha256'].items():assert digest(H1/name)==sha
    return seal
def threshold(s):
    x=np.asarray(s)
    if x.shape!=(400,) or not np.isfinite(x).all():raise ValueError('400 safe calibration scores required')
    return float(np.sort(x)[380])
def predict(artifact,x):
    import torch
    torch.set_num_threads(2)
    # Match OC-H1's frozen evaluation context, separately from deterministic training.
    torch.backends.cuda.enable_flash_sdp(True);torch.backends.cuda.enable_mem_efficient_sdp(True);torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(False)
    return h1_core().score(artifact,x)
