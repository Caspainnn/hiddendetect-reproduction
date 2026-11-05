import importlib.util,json,os,subprocess,time
from pathlib import Path
import joblib,numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from scipy.linalg import cho_factor,cho_solve
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('parent_auto_common',ROOT/'configs/ei2-pca-auto/common.py');parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
digest,write,SEEDS,GRID=parent.digest,parent.write,parent.SEEDS,parent.GRID
ORIGIN,BASE,PARENT=parent.ORIGIN,parent.BASE,parent.PARENT
OUT=ROOT/'results/ei2-pca-select';CACHE=ROOT/'cache/ei2-pca-select'
FAMILIES=['EI-2-3-3-1','EI-2-3-3-2','EI-2-3-3-3'];LAMBDAS=[.0001,.001,.01];CUTS=[1e2,1e3,1e4]

def prefix(mask):
 return np.arange(32)[None,:]<np.asarray(mask).sum(1)[:,None]

def pack(f,a,mask,mean=None,scale=None):
 mask=np.asarray(mask,bool);assert mask.shape==(14,32)
 raw=np.column_stack([f,a.reshape(len(a),448)[:,mask.ravel()]]).astype(np.float32)
 if mean is None:
  mean=raw.mean(0,dtype=np.float64).astype(np.float32);scale=raw.std(0,dtype=np.float64).astype(np.float32);scale[scale==0]=1
 x=(raw-mean)/scale;x[:,:14]/=np.sqrt(14);x[:,14:]/=np.sqrt(56)
 return np.pad(x,((0,0),(0,512-x.shape[1]))),mean,scale

def sparse_gradient(x,y,w,b,lam,rho):
 n=len(y);alpha=1/(10*n);score=x@w+b
 loss=np.mean(np.logaddexp(0,score)-y*score)+alpha*(w@w)/2+lam*(1-rho)*(w[14:]@w[14:])/2
 residual=(expit(score)-y)/n;g=x.T@residual+alpha*w;g[14:]+=lam*(1-rho)*w[14:]
 return loss,g,float(residual.sum())

def sparse_kkt(x,y,w,b,lam,rho):
 _,g,gb=sparse_gradient(x,y,w,b,lam,rho);nz=np.abs(w[14:])>1e-8
 extra=np.where(nz,np.abs(g[14:]+lam*rho*np.sign(w[14:])),np.maximum(np.abs(g[14:])-lam*rho,0))
 return float(max(np.max(np.abs(g[:14])),abs(gb),np.max(extra,initial=0)))

def fit_sparse(x,y,lam,rho):
 started=time.monotonic();x=np.asarray(x,np.float64);y=np.asarray(y,np.float64);p=x.shape[1]-14
 def objective(v):
  w=np.r_[v[:14],v[15:15+p]-v[15+p:]];loss,g,gb=sparse_gradient(x,y,w,v[14],lam,rho)
  return loss+lam*rho*v[15:].sum(),np.r_[g[:14],gb,g[14:]+lam*rho,-g[14:]+lam*rho]
 result=minimize(objective,np.zeros(15+2*p),jac=True,method='L-BFGS-B',bounds=[(None,None)]*15+[(0,None)]*(2*p),options={'maxiter':5000,'maxls':50,'ftol':1e-14,'gtol':1e-7})
 v=result.x;w=np.r_[v[:14],v[15:15+p]-v[15+p:]];kkt=sparse_kkt(x,y,w,v[14],lam,rho);assert kkt<=2e-6,(result.message,kkt)
 return {'weight':w,'bias':float(v[14]),'mask':np.abs(w[14:])>1e-8,'lambda':lam,'rho':rho,'kkt':kkt,'iterations':int(result.nit),'optimizer_success':bool(result.success),'optimizer_message':str(result.message),'seconds':time.monotonic()-started}

def map_logistic(x,y,precision,w):
 def objective(v):
  score=x@v;return float(np.sum(np.logaddexp(0,score)-y*score)+.5*np.dot(precision*v,v))
 for step in range(60):
  probability=expit(x@w);g=x.T@(probability-y)+precision*w
  h=x.T@((probability*(1-probability))[:,None]*x)+np.diag(precision)
  factor=cho_factor(h,lower=True);gradient=float(np.max(np.abs(g)))
  if gradient<=1e-6:break
  direction=cho_solve(factor,g);old=objective(w);rate=1.
  while objective(w-rate*direction)>old-1e-4*rate*np.dot(g,direction):
   rate*=.5;assert rate>=2**-40,'MAP line search failed'
  w=w-rate*direction
 probability=expit(x@w);g=x.T@(probability-y)+precision*w;gradient=float(np.max(np.abs(g)));assert gradient<=2e-5,gradient
 h=x.T@((probability*(1-probability))[:,None]*x)+np.diag(precision);cov=cho_solve(cho_factor(h,lower=True),np.eye(len(w)))
 return w,np.diag(cov),gradient,step+1

def fit_ard(x,y):
 started=time.monotonic();x=np.column_stack([np.asarray(x,np.float64),np.ones(len(y))]);y=np.asarray(y,np.float64)
 p=x.shape[1]-1;active=np.ones(p+1,bool);alpha=np.r_[np.full(14,.1),np.ones(p-14),1e-6];w=np.zeros(p+1);history=[];stable=0;converged=False
 for outer in range(100):
  ids=np.flatnonzero(active);v,cov,gradient,steps=map_logistic(x[:,ids],y,alpha[ids],w[ids]);w[ids]=v
  extra=ids[(ids>=14)&(ids<p)];local=np.flatnonzero((ids>=14)&(ids<p));gamma=np.clip(1-alpha[extra]*cov[local],1e-12,1.)
  target=np.clip(gamma/(w[extra]**2+1e-24),1e-6,1e12);updated=np.sqrt(alpha[extra]*target);change=float(np.max(np.abs(np.log(updated/alpha[extra])),initial=0));alpha[extra]=updated
  pruned=extra[alpha[extra]>=1e10] if outer>=4 else np.array([],int);active[pruned]=False;alpha[pruned]=1e12;w[pruned]=0
  history.append({'outer':outer+1,'active_extra':int(active[14:p].sum()),'max_log_alpha_change':change,'map_gradient':gradient,'map_steps':steps,'pruned':pruned.tolist()})
  stable=stable+1 if not len(pruned) and change<1e-3 else 0
  if stable>=3:converged=True;break
 ids=np.flatnonzero(active);v,cov,gradient,steps=map_logistic(x[:,ids],y,alpha[ids],w[ids]);w[ids]=v;diagonal=np.zeros(p+1);diagonal[ids]=cov
 extra=np.flatnonzero(active & (np.arange(p+1)>=14)&(np.arange(p+1)<p));target=np.clip(np.clip(1-alpha[extra]*diagonal[extra],1e-12,1)/(w[extra]**2+1e-24),1e-6,1e12)
 return {'weight':w[:-1],'bias':float(w[-1]),'alpha':alpha[:-1],'active':active[:-1],'covariance_diagonal':diagonal,'map_gradient':gradient,'outer_converged':converged,'fixed_point_log_residual':float(np.max(np.abs(np.log(target/alpha[extra])),initial=0)),'history':history,'seconds':time.monotonic()-started}

def ard_mask(state,cut):
 return state['active'][14:] & (state['alpha'][14:]<=cut)

def jaccard(a,b):
 union=np.logical_or(a,b).sum();return float(np.logical_and(a,b).sum()/union) if union else 1.

def bootstrap(scores,labels):
 # Point calculations here use sklearn; the verifier independently counts weighted pairs.
 from scipy.stats import rankdata
 names=['K32']+FAMILIES+[n+'-prefix' for n in FAMILIES];matrix=np.column_stack([scores[n][0] for n in names]);rng=np.random.default_rng(20261105);groups=[np.flatnonzero(labels==v) for v in [0,1]];n0,n1=map(len,groups);dist=np.empty((10000,len(names)))
 for b in range(10000):
  ix=np.r_[rng.choice(groups[0],n0,replace=True),rng.choice(groups[1],n1,replace=True)];ranks=rankdata(matrix[ix],axis=0);dist[b]=(ranks[n0:].sum(0)-n1*(n1+1)/2)/(n0*n1)
 return {n:{control:{'ci991667':np.quantile(dist[:,1+j]-dist[:,(0 if control=='K32' else 4+j)],[.05/12,1-.05/12]).tolist()} for control in ['K32',n+'-prefix']} for j,n in enumerate(FAMILIES)}
