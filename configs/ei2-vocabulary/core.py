import hashlib,json,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
ORIGIN=Path(os.environ.get('EI_ORIGIN',str(ROOT)))
OUT=ROOT/'results/ei2-vocabulary';CACHE=ROOT/'cache/ei2-vocabulary'
SEEDS=[20261104,20261105,20261106];GRID=[.1,1.,10.]
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
def read(p):
 with Path(p).open(encoding='utf-8') as f:return [json.loads(x) for x in f if x.strip()]
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def unit(z):
 z=np.asarray(z,np.float32);n=np.linalg.norm(z.astype(np.float64),axis=-1);return (z/np.maximum(n,1e-12)[...,None]).astype(np.float32),np.log(np.maximum(n,1e-12)).astype(np.float32)
def pack(f,extra,mean=None,scale=None,pad=True):
 x=np.column_stack([f,extra]).astype(np.float32)
 if mean is None:
  mean=x.mean(0,dtype=np.float64).astype(np.float32);scale=x.std(0,dtype=np.float64).astype(np.float32);scale[scale==0]=1
 x=(x-mean)/scale;x[:,:14]/=np.sqrt(14)
 if extra.shape[1]:x[:,14:]/=np.sqrt(extra.shape[1])
 if pad:
  assert x.shape[1]<=512;x=np.pad(x,((0,0),(0,512-x.shape[1])))
 return x,mean,scale
