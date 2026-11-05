import importlib.util,os,json,hashlib,time
from pathlib import Path
import numpy as np,joblib
ROOT=Path(__file__).resolve().parents[2];ORIGIN=Path(os.environ.get('EI_ORIGIN',str(ROOT)));OUT=ROOT/'results/ei2-images';CACHE=ROOT/'cache/ei2-images';RAW=ORIGIN/'cache/ei2-image-acquisition/raw'
SEEDS=[20261104,20261105,20261106];GRID=[.1,1.,10.];ARMS=[f'EI-2-8-{i}' for i in range(1,6)];BASE='B-MM';NAMES=[BASE,'Noise14']+ARMS+['B-T','OCR-only','Caption-only','Neutral-description','ROI-uniform','ROI-random','Neutral-image-delta']
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
def write(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_bytes((json.dumps(value,indent=2)+'\n').encode());tmp.replace(p)
def module(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def unit(z):
 z=np.asarray(z,np.float32);norm=np.sqrt(np.sum(z.astype(np.float64)**2,axis=-1,keepdims=True));return (z/np.maximum(norm,1e-12)).astype(np.float32)
def assert_group_roles(rows):
 seen={}
 for row in rows:
  g=row['group_id'];assert g not in seen or seen[g]==row['role'];seen[g]=row['role']
def pack(f,extra,mean=None,scale=None):
 x=np.column_stack([f,extra]).astype(np.float32);assert f.shape[1]==14 and extra.shape[1] in [0,14]
 if mean is None:
  mean=x.mean(0,dtype=np.float64).astype(np.float32);scale=x.std(0,dtype=np.float64).astype(np.float32);scale[scale==0]=1
 x=(x-mean)/scale;x[:,:14]/=np.sqrt(14)
 if extra.shape[1]:x[:,14:]/=np.sqrt(14)
 return np.pad(x,((0,0),(0,512-x.shape[1]))),mean,scale
def weighted_auc(y,s,w):
 order=np.argsort(s,kind='stable');starts=np.r_[0,np.flatnonzero(np.diff(s[order]))+1];neg=np.add.reduceat(w[:,order]*(1-y[order])[None,:],starts,axis=1);pos=np.add.reduceat(w[:,order]*y[order][None,:],starts,axis=1);denom=neg.sum(1)*pos.sum(1);assert np.all(denom>0);return (pos*(np.cumsum(neg,axis=1)-.5*neg)).sum(1)/denom
def bootstrap(y,matrix,groups,draws=10000):
 names,inv=np.unique(groups,return_inverse=True);rng=np.random.default_rng(20261105);out=np.empty((draws,matrix.shape[1]));ng=len(names)
 for start in range(0,draws,100):
  b=min(100,draws-start);sample=rng.integers(ng,size=(b,ng));counts=np.zeros((b,ng));np.add.at(counts,(np.arange(b)[:,None],sample),1);w=counts[:,inv]
  for j in range(matrix.shape[1]):out[start:start+b,j]=weighted_auc(y,matrix[:,j],w)
 return out
def load():
 m=json.loads((OUT/'manifest.json').read_text());rows=m['samples'];assert_group_roles(rows);private=json.loads((CACHE/'records.json').read_text());assert [{k:v for k,v in r.items() if k not in ['prompt','image_path']} for r in private]==rows
 for row in private:assert hashlib.sha256(row['prompt'].encode()).hexdigest()==row['prompt_sha256']
 roles={role:np.array([i for i,r in enumerate(rows) if r['role']==role]) for role in ['train','dev','cal','test','task_test']};y=np.array([r['label'] for r in rows]);return rows,private,roles,y
def arrays(n,mode='r'):
 result={}
 for name,dtype,shape in [('signals',np.float64,(n,9,14)),('vision',np.float32,(n,1024))]+[(name,np.float32,(n,14,4096)) for name in ['roi','uniform','random']]:
  p=CACHE/(name+'.npy');result[name]=np.load(p,mmap_mode=mode) if p.exists() else np.lib.format.open_memmap(p,mode='w+',dtype=dtype,shape=shape)
 return result
def centroid(h,y):return unit(np.stack([unit(h)[y==label].mean(0) for label in [0,1]]))
def feature(name,ix,a,transforms):
 signals=a['signals'][ix];base=signals[:,0];empty=np.empty((len(ix),0),np.float32)
 if name==BASE:return base,empty
 if name=='B-T':return signals[:,1],empty
 if name=='Noise14':extra=np.vstack([np.random.default_rng(np.random.SeedSequence([20261105,8,int(i)])).normal(size=14) for i in ix]).astype(np.float32)
 elif name==ARMS[0]:extra=transforms['pca'].transform(unit(a['vision'][ix]))
 elif name in [ARMS[1],'OCR-only','Caption-only','Neutral-description']:extra=signals[:,{ARMS[1]:3,'OCR-only':4,'Caption-only':5,'Neutral-description':6}[name]]
 elif name in [ARMS[2],'Neutral-image-delta']:extra=base-signals[:,1 if name==ARMS[2] else 2]
 elif name in [ARMS[3],'ROI-uniform','ROI-random']:
  kind={ARMS[3]:'roi','ROI-uniform':'uniform','ROI-random':'random'}[name];refs=transforms[kind];extra=np.einsum('nld,ld->nl',unit(a[kind][ix]),refs[1]-refs[0])
 else:
  assert name==ARMS[4];extra=np.max(np.abs(signals[:,7:9]-base[:,None,:]),axis=1)
 assert extra.shape==(len(ix),14);return base,extra
