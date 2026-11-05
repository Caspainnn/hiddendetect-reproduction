import importlib.util,json,os,subprocess,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import joblib,numpy as np
from scipy.linalg import eigh
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('parent_auto_common',ROOT/'configs/ei2-pca-auto/common.py');parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
digest,write,SEEDS,GRID=parent.digest,parent.write,parent.SEEDS,parent.GRID
ORIGIN,BASE,PARENT=parent.ORIGIN,parent.BASE,parent.PARENT
OUT=ROOT/'results/ei2-pca-uncapped';CACHE=ROOT/'cache/ei2-pca-uncapped'
FAMILIES=[parent.CV,parent.VAR,parent.GATE]
TAGS=[str(k) for k in range(1,33)]+[str(k) for k in [48,64,96,128,192,256,384,512,768,1024]]+['rank-75','rank-100']

def matmul(a,b,gpu=True):
 if not gpu:return a@b
 import torch
 assert torch.cuda.is_available(),'GPU requested but unavailable'
 torch.backends.cuda.matmul.allow_tf32=False
 with torch.no_grad():return (torch.as_tensor(a,device='cuda',dtype=torch.float64)@torch.as_tensor(b,device='cuda',dtype=torch.float64)).cpu().numpy()

def fit_dual(raw,gpu=True):
 raw=np.asarray(raw,np.float64);mean=raw.mean(0);centered=raw-mean;gram=matmul(centered,centered.T,gpu);gram=(gram+gram.T)*.5
 vals,vectors=eigh(gram,check_finite=True,driver='evr');order=np.argsort(vals)[::-1];vals,vectors=vals[order],vectors[:,order]
 tolerance=float(vals[0]*max(raw.shape)*np.finfo(np.float64).eps);keep=vals>tolerance;vals,vectors=vals[keep],vectors[:,keep]
 signs=np.sign(vectors[np.argmax(np.abs(vectors),axis=0),np.arange(len(vals))]);vectors*=np.where(signs==0,1,signs)
 total=float(np.sum(centered*centered));assert len(vals)<=min(len(raw)-1,raw.shape[1]) and len(vals)>0
 return {'mean':mean,'dual':vectors/np.sqrt(vals),'values':vals,'total_energy':total,'n_train':len(raw),'rank_tolerance':tolerance,'retained_energy':float(vals.sum()/total),'gram_eigen_residual_relative':float(np.max(np.abs(gram@vectors-vectors*vals))/max(1.,vals[0]))}

def transform_dual(raw,train_raw,basis,gpu=True):
 cross=matmul(np.asarray(raw,np.float64)-basis['mean'],(np.asarray(train_raw,np.float64)-basis['mean']).T,gpu)
 return cross@basis['dual']

def residual(data,indices,layer):
 return parent.unit(data['z'][indices,layer])[0]-data['f'][indices,layer,None].astype(np.float32)*data['rhat']

def fit_basis(data,indices,key):
 basis=[]
 for layer in range(14):
  start=time.monotonic();b=fit_dual(residual(data,indices,layer));b['train_indices']=np.asarray(indices);basis.append(b)
  print('PCA_FULL_RANK',key,layer+16,len(b['values']),b['retained_energy'],time.monotonic()-start,flush=True)
 path=CACHE/(key+'-basis.joblib');joblib.dump(basis,path);return basis,path

def project(data,indices,basis):
 return [transform_dual(residual(data,indices,l),residual(data,b['train_indices'],l),b).astype(np.float32) for l,b in enumerate(basis)]

def pack(f,a,counts,mean=None,scale=None):
 counts=np.asarray(counts,int);assert counts.shape==(14,) and all(0<=k<=v.shape[1] for k,v in zip(counts,a))
 raw=np.column_stack([f]+[v[:,:k] for v,k in zip(a,counts)]).astype(np.float32)
 if mean is None:
  mean=raw.mean(0,dtype=np.float64).astype(np.float32);scale=raw.std(0,dtype=np.float64).astype(np.float32);scale[scale==0]=1
 x=(raw-mean)/scale;x[:,:14]/=np.sqrt(14);x[:,14:]/=np.sqrt(56)
 return np.pad(x,((0,0),(0,max(0,512-x.shape[1])))),mean,scale

def count_for(tag,rank):
 return int(np.ceil(rank*int(tag.split('-')[1])/100)) if tag.startswith('rank-') else int(tag)

def variance_counts(basis,tau,cap=None):
 counts=[];coverage=[];capped=[]
 for b in basis:
  limit=len(b['values']) if cap is None else min(cap,len(b['values']));cumulative=np.cumsum(b['values'][:limit])/b['total_energy'];k=min(limit,int(np.searchsorted(cumulative,tau))+1)
  counts.append(k);coverage.append(float(cumulative[k-1]));capped.append(bool(cumulative[-1]<tau))
 return counts,coverage,capped

def choose_cv(scores,tags,counts):
 means=scores.mean(0);best=min(np.ndindex(means.shape),key=lambda p:(-means[p],float(np.mean(counts[:,p[0]])),p[0],p[1]));se=float(scores[:,best[0],best[1]].std(ddof=1)/np.sqrt(3));threshold=float(means[best]-se)
 allowed=[p for p in np.ndindex(means.shape) if means[p]>=threshold-1e-12];chosen=min(allowed,key=lambda p:(float(np.mean(counts[:,p[0]])),-means[p],p[0],p[1]))
 return {'tag':tags[chosen[0]],'C':GRID[chosen[1]],'best_tag':tags[best[0]],'best_C':GRID[best[1]],'threshold':threshold,'standard_error':se,'selected_mean':float(means[chosen])}

def scan_prefix(values,backend):
 result=values;step=1
 while step<values.shape[1]:
  parts=[result[:,:step],result[:,step:]*result[:,:-step]]
  result=backend.concatenate(parts,axis=1) if backend is np else backend.cat(parts,dim=1);step*=2
 return result

def fit_gate(x,y,ranks,lam,seed):
 import torch
 torch.set_num_threads(4);torch.manual_seed(seed);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
 device='cuda';ranks=np.asarray(ranks,int);maximum=int(ranks.max());offsets=np.r_[14,14+np.cumsum(ranks)];extra=np.zeros((len(x),14,maximum),np.float32)
 for l in range(14):extra[:,l,:ranks[l]]=x[:,offsets[l]:offsets[l+1]]
 f=torch.tensor(x[:,:14],device=device);tx=torch.tensor(extra,device=device);ty=torch.tensor(y,dtype=torch.float32,device=device);valid=torch.tensor(np.arange(maximum)[None,:]<ranks[:,None],device=device)
 wf=torch.nn.Parameter(torch.randn(14,device=device)*.01);w=torch.nn.Parameter(torch.randn(14,maximum,device=device)*.01);b=torch.nn.Parameter(torch.zeros((),device=device))
 p=.99**(32/ranks);initial=torch.tensor(np.log(p/(1-p)),dtype=torch.float32,device=device)[:,None];noise=torch.tensor(.01*32/ranks,dtype=torch.float32,device=device)[:,None]
 a=torch.nn.Parameter(initial+noise*torch.randn(14,maximum,device=device));optimizer=torch.optim.Adam([wf,w,b,a],lr=.03);losses=[];started=time.monotonic()
 for step in range(800):
  optimizer.zero_grad();g=scan_prefix(torch.sigmoid(a),torch)*valid;score=f@wf+(tx*(g*w)).sum((1,2))+b
  loss=torch.nn.functional.binary_cross_entropy_with_logits(score,ty)+(wf.square().sum()+(w*valid).square().sum())/(2*10*len(y))+lam*g.sum()/valid.sum();assert torch.isfinite(loss);loss.backward();optimizer.step()
  if step in [0,799]:losses.append(float(loss.detach().cpu()))
 with torch.no_grad():g=(scan_prefix(torch.sigmoid(a),torch)*valid).cpu().numpy()
 result={'seed':seed,'lambda':lam,'ranks':ranks.tolist(),'a':a.detach().cpu().numpy(),'soft_gates':g,'counts':(g>=.5).sum(1).tolist(),'steps':800,'threshold':.5,'loss_start_end':losses,'seconds':time.monotonic()-started,'device':device}
 del f,tx,ty,wf,w,b,a,optimizer;torch.cuda.empty_cache();return result
