"""Original scoring contracts, with no fitted detector or candidate selection."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[2]
IDS=[f'EI-3-1-{i}' for i in range(1,7)]+['EI-3-2-1']
NAMES=['user_first','user_q25','user_q50','user_q75','user_q90','user_last','template_last']

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f: return [json.loads(x) for x in f if x.strip()]

def positions(user,length):
    user=np.asarray(user,dtype=int)
    if user.ndim!=1 or not len(user) or user[0]<0 or user[-1]>=length-1 or np.any(np.diff(user)<=0):
        raise ValueError('Expected sorted unique user tokens followed by template suffix')
    return [int(user[int(q*(len(user)-1))]) for q in (0,.25,.5,.75,.9,1)]+[length-1]

def project_readouts(state,indices,norm,head):
    # Preserve author GEMM shape; selecting tokens before lm_head can change fp16 numerics.
    return head(norm(state))[:,indices,:]

def integral(f,start=16,end=29):
    f=np.asarray(f,dtype=np.float64)
    if f.shape!=(32,7) or not np.isfinite(f).all() or not 0<=start<end<32:
        raise ValueError('Expected finite 32-layer, seven-position F')
    return np.array([np.trapz(f[start:end+1,j]) for j in range(7)])

def threshold(safe):
    safe=np.asarray(safe,dtype=float)
    if not safe.size or not np.isfinite(safe).all(): raise ValueError('Invalid calibration scores')
    return float(np.sort(safe)[math.ceil(.95*len(safe))-1])

def bootstrap_metrics(y,score,indices):
    # Weighted tied ROC contract from configs/swarm/core.py bootstrap_metrics.
    order=np.argsort(-score,kind='stable')
    starts=np.r_[0,np.flatnonzero(np.diff(score[order]))+1]
    weights=np.zeros((len(indices),len(y)),dtype=np.float64)
    np.add.at(weights,(np.arange(len(indices))[:,None],indices),1)
    tp=np.add.reduceat((weights*y)[:,order],starts,axis=1).cumsum(1)/(y==1).sum()
    fp=np.add.reduceat((weights*(1-y))[:,order],starts,axis=1).cumsum(1)/(y==0).sum()
    tp=np.column_stack([np.zeros(len(indices)),tp]);fp=np.column_stack([np.zeros(len(indices)),fp])
    width=np.diff(fp,axis=1)
    auc=(width*(tp[:,1:]+tp[:,:-1])/2).sum(1)
    clipped=np.clip(.1-fp[:,:-1],0,width)
    right=tp[:,:-1]+np.divide(clipped,width,out=np.zeros_like(width),where=width>0)*np.diff(tp,axis=1)
    area=(clipped*(tp[:,:-1]+right)/2).sum(1)
    return auc,.5*(1+(area-.005)/.095)

def bootstrap_auc(y,score,indices):return bootstrap_metrics(y,score,indices)[0]
def bootstrap_partial(y,score,indices):return bootstrap_metrics(y,score,indices)[1]

def paired_intervals(y,scores,repeats=5000,seed=20261030,family=6,thresholds=None):
    y=np.asarray(y,dtype=int);scores=np.asarray(scores,dtype=float)
    if scores.ndim!=2 or len(scores)!=len(y) or set(y)!={0,1} or not np.isfinite(scores).all():
        raise ValueError('Invalid paired observations')
    groups=[np.flatnonzero(y==k) for k in (0,1)];rng=np.random.default_rng(seed)
    indices=np.column_stack([rng.choice(g,(repeats,len(g))) for g in groups])
    distribution=np.empty((repeats,scores.shape[1]))
    partial=np.empty_like(distribution)
    for start in range(0,repeats,100):
        for j in range(scores.shape[1]):
            distribution[start:start+100,j],partial[start:start+100,j]=bootstrap_metrics(y,scores[:,j],indices[start:start+100])
    result=[];baseline=roc_auc_score(y,scores[:,-1])
    for j in range(scores.shape[1]-1):
        delta=distribution[:,j]-distribution[:,-1]
        result.append({'direction_id':IDS[j], 'delta':float(roc_auc_score(y,scores[:,j])-baseline),
                       'ci95':np.quantile(delta,[.025,.975]).tolist(),
                       'adjusted':np.quantile(delta,[.025/family,1-.025/family]).tolist()})
        pd=partial[:,j]-partial[:,-1]
        result[-1]['descriptive_pauc']={'delta':float(roc_auc_score(y,scores[:,j],max_fpr=.1)-roc_auc_score(y,scores[:,-1],max_fpr=.1)),
                                       'ci95':np.quantile(pd,[.025,.975]).tolist()}
        if thresholds is not None:
            for label,name in [(0,'fpr'),(1,'tpr')]:
                sample=indices[:,:len(groups[0])] if label==0 else indices[:,len(groups[0]):]
                difference=(scores[:,j]>thresholds[j]).astype(float)-(scores[:,-1]>thresholds[-1]).astype(float)
                result[-1]['descriptive_'+name]={'delta':float(difference[y==label].mean()),
                                                'ci95':np.quantile(difference[sample].mean(1),[.025,.975]).tolist()}
    return result
