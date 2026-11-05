import importlib.util,json,os,subprocess,time
from pathlib import Path
import joblib,numpy as np
ROOT=Path(__file__).resolve().parents[2];ORIGIN=Path(os.environ.get('EI_ORIGIN',str(ROOT)));PARENT=ORIGIN/'cache/ei2-worktree'
OUT=ROOT/'results/ei2-centroid';CACHE=ROOT/'cache/ei2-centroid';ARM='EI-2-7-2';NAMES=['B-F',ARM,'Noise']
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
v=module('vocabulary_core',ROOT/'configs/ei2-vocabulary/core.py');digest,write,SEEDS,GRID,pack=v.digest,v.write,v.SEEDS,v.GRID,v.pack;read_records=v.read
def unit(h):
 h=np.asarray(h,np.float32);return h/np.maximum(np.linalg.norm(h,axis=-1,keepdims=True),1e-12)
def fit_centroids(h,y):
 h=unit(h);assert h.ndim==3 and h.shape[1]==14
 assert set(np.unique(y))=={0,1};return unit(np.stack([h[np.asarray(y)==label].mean(0) for label in [0,1]]))
def features(h,refs):return np.einsum('nld,ld->nl',unit(h),refs[1]-refs[0])
def noise(ix):return np.vstack([np.random.default_rng(np.random.SeedSequence([20261104,7,int(i)])).normal(size=14) for i in ix]).astype(np.float32)
def data():
 proof=json.loads((PARENT/'results/ei2-vocabulary/verification.json').read_text());assert proof['passed'];source=json.loads((PARENT/'results/ei2-vocabulary/source-lock.json').read_text());mp=PARENT/'results/ei2-vocabulary/manifest.json';assert digest(mp)==source['manifest'];assert digest(PARENT/'cache/ei2-vocabulary/F.npy')==proof['sha256']['cache/ei2-vocabulary/F.npy']
 rows=json.loads(mp.read_text())['samples'];assert len(rows)==6582
 for key in ['id','group_id','prompt_sha256','input_ids_sha256']:assert len({r[key] for r in rows})==6582
 roles={role:np.array([i for i,r in enumerate(rows) if r['role']==role]) for role in ['train','dev','cal','C','RV','OR_cal','OR_test']};assert [len(x) for x in roles.values()]==[1600,600,400,1000,1082,400,1500]
 return rows,roles,np.load(PARENT/'cache/ei2-vocabulary/F.npy',mmap_mode='r'),np.array([r['label'] for r in rows])
def extra(name,h,refs,ix):return np.empty((len(ix),0),np.float32) if name=='B-F' else (features(h[ix],refs) if name==ARM else noise(ix))
