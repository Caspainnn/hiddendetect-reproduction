import hashlib,json,os,sys,time
from pathlib import Path
import joblib,numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(1,str(ROOT/'configs/ei2-vocabulary'))
from core import digest,write,unit,SEEDS,GRID
ORIGIN=Path(os.environ.get('EI_ORIGIN',str(ROOT)))
PARENT=ORIGIN/'cache/ei2-worktree';BASE=ORIGIN/'cache/ei2-pca-extended-worktree'
OUT=ROOT/'results/ei2-pca-auto';CACHE=ROOT/'cache/ei2-pca-auto'
CV='EI-2-3-2-1';VAR='EI-2-3-2-2';GATE='EI-2-3-2-3'
TAUS=[.8,.9,.95];LAMBDAS=[.01,.1,1.];STEPS=800

def pack(f,a,counts,mean=None,scale=None):
 counts=np.asarray(counts,dtype=int);assert counts.shape==(14,) and np.all((counts>=0)&(counts<=32))
 indices=np.concatenate([np.arange(l*32,l*32+k) for l,k in enumerate(counts)])
 x=np.column_stack([f,a.reshape(len(a),448)[:,indices]]).astype(np.float32)
 if mean is None:
  mean=x.mean(0,dtype=np.float64).astype(np.float32);scale=x.std(0,dtype=np.float64).astype(np.float32);scale[scale==0]=1
 x=(x-mean)/scale;x[:,:14]/=np.sqrt(14);x[:,14:]/=np.sqrt(56)
 return np.pad(x,((0,0),(0,512-x.shape[1]))),mean,scale

def variance_counts(basis,tau):
 counts=[];coverage=[];capped=[]
 for pca in basis:
  cumulative=np.cumsum(pca.explained_variance_ratio_,dtype=np.float64)
  k=min(32,int(np.searchsorted(cumulative,tau,side='left'))+1)
  counts.append(k);coverage.append(float(cumulative[k-1]));capped.append(bool(cumulative[-1]<tau))
 return counts,coverage,capped

def choose_cv(scores):
 assert scores.shape==(3,32,3) and np.isfinite(scores).all()
 means=scores.mean(0);best=min(np.ndindex(means.shape),key=lambda p:(-means[p],p[0],p[1]))
 se=float(scores[:,best[0],best[1]].std(ddof=1)/np.sqrt(3));threshold=float(means[best]-se)
 eligible=[p for p in np.ndindex(means.shape) if means[p]>=threshold-1e-12]
 selected=min(eligible,key=lambda p:(p[0],-means[p],p[1]))
 return {'k':selected[0]+1,'C':GRID[selected[1]],'best_k':best[0]+1,'best_C':GRID[best[1]],'best_mean':float(means[best]),'best_standard_error':se,'threshold':threshold,'selected_mean':float(means[selected])}

def prefix_gates(a):
 a=np.asarray(a,np.float64);return np.cumprod(1/(1+np.exp(-np.clip(a,-60,60))),axis=1)

def torch_prefix_gates(a):
 import torch
 probability=torch.sigmoid(a);previous=torch.ones_like(probability[:,0]);parts=[]
 for j in range(32):
  previous=previous*probability[:,j];parts.append(previous)
 return torch.stack(parts,dim=1)

def load_data(check_hashes=False):
 parent=json.loads((PARENT/'results/ei2-vocabulary/verification.json').read_text());base=json.loads((BASE/'results/ei2-pca-extended/verification.json').read_text());assert parent['passed'] and base['passed']
 assert digest(BASE/'results/ei2-pca-extended/evaluation.json')==base['evaluation_sha256']
 assert digest(BASE/'results/ei2-pca-extended/scores.npz')==base['scores_sha256']
 if check_hashes:
  for name in ['F.npy','Z.npy']:assert digest(PARENT/'cache/ei2-vocabulary'/name)==parent['sha256']['cache/ei2-vocabulary/'+name]
 pre=json.loads((BASE/'results/ei2-pca-extended/pre-evaluation-seal.json').read_text());assert digest(BASE/'cache/ei2-pca-extended/transforms.joblib')==pre['transform']
 rows=json.loads((PARENT/'results/ei2-vocabulary/manifest.json').read_text())['samples'];assert len(rows)==6582
 for key in ['id','group_id','prompt_sha256','input_ids_sha256']:assert len({r[key] for r in rows})==len(rows)
 roles={role:np.array([i for i,r in enumerate(rows) if r['role']==role]) for role in ['train','dev','cal','C','RV','OR_cal','OR_test']}
 coords=json.loads((ORIGIN/'cache/oc4c-worktree/results/oc23/token-audit.json').read_text())['token_ids'];rhat=np.zeros(32000,np.float32);rhat[coords]=1/np.sqrt(len(coords))
 return dict(rows=rows,roles=roles,z=np.load(PARENT/'cache/ei2-vocabulary/Z.npy',mmap_mode='r'),f=np.load(PARENT/'cache/ei2-vocabulary/F.npy',mmap_mode='r'),basis=joblib.load(BASE/'cache/ei2-pca-extended/transforms.joblib'),rhat=rhat)

def project(data,indices,basis=None):
 basis=data['basis'] if basis is None else basis
 return np.stack([pca.transform(unit(data['z'][indices,l])[0]-data['f'][indices,l,None].astype(np.float32)*data['rhat']) for l,pca in enumerate(basis)],axis=1).astype(np.float32)

def fit_gate(x,y,lam,seed):
 os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
 import torch
 torch.set_num_threads(4);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 device='cuda' if torch.cuda.is_available() else 'cpu'
 tx=torch.tensor(x[:,:462],dtype=torch.float32,device=device);ty=torch.tensor(y,dtype=torch.float32,device=device)
 w=torch.nn.Parameter(torch.randn(462,device=device)*.01);b=torch.nn.Parameter(torch.zeros((),device=device));a=torch.nn.Parameter(torch.full((14,32),float(np.log(99)),device=device)+torch.randn(14,32,device=device)*.01)
 optimizer=torch.optim.Adam([w,b,a],lr=.03);losses=[];started=time.monotonic()
 for step in range(STEPS):
  optimizer.zero_grad();g=torch_prefix_gates(a)
  score=tx[:,:14]@w[:14]+(tx[:,14:].reshape(-1,14,32)*g).flatten(1)@w[14:]+b
  loss=torch.nn.functional.binary_cross_entropy_with_logits(score,ty)+w.square().sum()/(2*10*len(y))+lam*g.mean()
  assert torch.isfinite(loss);loss.backward();optimizer.step()
  if step in [0,STEPS-1]:losses.append(float(loss.detach().cpu()))
 with torch.no_grad():g=torch_prefix_gates(a).cpu().numpy()
 counts=(g>=.5).sum(1).astype(int).tolist()
 result={'seed':seed,'lambda':lam,'device':device,'steps':STEPS,'seconds':time.monotonic()-started,'loss_start_end':losses,'a':a.detach().cpu().numpy(),'soft_gates':g,'counts':counts,'weight':w.detach().cpu().numpy(),'bias':float(b.detach().cpu()),'threshold':.5,'minimum_threshold_margin':float(np.min(np.abs(g-.5)))}
 del tx,ty,w,b,a,optimizer
 if device=='cuda':torch.cuda.empty_cache()
 return result

def references(data,role,a):
 ix=data['roles'][role];parent=np.load(PARENT/'results/ei2-vocabulary/scores.npz');base=np.load(BASE/'results/ei2-pca-extended/scores.npz')
 pre=json.loads((BASE/'results/ei2-pca-extended/pre-evaluation-seal.json').read_text());original_pre=json.loads((PARENT/'results/ei2-vocabulary/pre-evaluation-seal.json').read_text())
 scores={};maximum=0.
 for name,key,k in [('K8','K8',8),('K16','EI-2-3-4',16),('K32','EI-2-3-5',32),('B-F','B-F',0)]:
  folder=PARENT/'cache/ei2-vocabulary' if name=='B-F' else BASE/'cache/ei2-pca-extended'
  path=folder/(key+'.joblib');expected=original_pre['models'][key] if name=='B-F' else pre['models'][key];assert digest(path)==expected
  model=joblib.load(path);x,_,_=pack(data['f'][ix],a,[k]*14,model['mean'],model['scale']);value=np.array([h.decision_function(x) for h in model['models']])
  stored=parent[role+'__B-F'] if name=='B-F' else base[role+'__'+key]
  delta=float(np.max(np.abs(value-stored)));maximum=max(maximum,delta);assert delta<1e-7,(role,name,delta)
  scores[name]=value
 return scores,maximum
