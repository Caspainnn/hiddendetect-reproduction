"""OC-SA1 single-layer scoring and fixed cross-layer aggregation contracts."""
import hashlib,importlib.util,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[2]
NAMES=['B','OC-2-1','OC-2-2','OC-2-3','OC-3-1','OC-3-2','OC-3-3','OC-3-4','OC-3-5']
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with Path(p).open(encoding='utf-8') as f:return [json.loads(x) for x in f if x.strip()]
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode())
def logsumexp(x):
    x=np.asarray(x,dtype=np.float64);maximum=x.max();return float(maximum+np.log(np.exp(x-maximum).sum()))
def single_layer_scores(logits,ids):
    z=np.asarray(logits,dtype=np.float64);assert z.ndim==1 and np.isfinite(z).all()
    ids=np.asarray(ids,dtype=int);assert len(ids)>0 and len(set(ids))==len(ids) and ids.min()>=0 and ids.max()<len(z)
    selected=z[ids];ls=logsumexp(selected)
    return np.array([selected.mean(),ls-np.log(len(ids)),np.exp(ls-logsumexp(z))])
def aggregate(f,layer):
    f=np.asarray(f,dtype=np.float64);layer=np.asarray(layer,dtype=np.float64)
    assert f.shape==(32,) and layer.shape==(32,3) and np.isfinite(f).all() and np.isfinite(layer).all()
    x=f[16:30];g=np.sort(x)
    return np.r_[np.trapz(x),np.trapz(layer[16:30],axis=0),x.mean(),g[-4:].mean(),g[2:-2].mean(),
                 max(x[i:i+4].mean() for i in range(11)),x[7:].mean()-x[:7].mean()]
def threshold(s):
    s=np.asarray(s);assert s.shape==(400,) and np.isfinite(s).all();return float(np.sort(s)[380])
def wilson(k,n):
    z=1.959963984540054;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [float(c-h),float(c+h)]
def evaluate(rows,cut,batch):
    spec=importlib.util.spec_from_file_location('sa_bootstrap',ROOT/'configs/original-positions/core.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    y=np.array([r['label'] for r in rows]);s=np.array([r['scores'] for r in rows]);assert set(y)=={0,1} and s.shape==(len(y),len(NAMES))
    groups=[np.flatnonzero(y==k) for k in (0,1)];rng=np.random.default_rng(20261030)
    ix=np.column_stack([rng.choice(g,(10000,len(g))) for g in groups]);aucs=np.empty((10000,len(NAMES)));partials=np.empty_like(aucs)
    for start in range(0,10000,100):
        for j in range(len(NAMES)):aucs[start:start+100,j],partials[start:start+100,j]=old.bootstrap_metrics(y,s[:,j],ix[start:start+100])
    metrics=[];pred=s>np.asarray(cut)
    for j,name in enumerate(NAMES):
        m={'direction_id':name,'auroc':float(roc_auc_score(y,s[:,j])),'pauc':float(roc_auc_score(y,s[:,j],max_fpr=.1)),'threshold':cut[j]}
        for label,key in [(0,'fpr'),(1,'tpr')]:
            k=int(pred[y==label,j].sum());n=len(groups[label]);m.update({key:k/n,key+'_count':k,key+'_n':n,key+'_wilson95':wilson(k,n)})
            if j:
                sample=ix[:,:len(groups[0])] if label==0 else ix[:,len(groups[0]):];diff=pred[:,j].astype(float)-pred[:,0].astype(float)
                m[key+'_delta_ci95']=np.quantile(diff[sample].mean(1),[.025,.975]).tolist()
        if j:
            m.update(delta_auroc=float(roc_auc_score(y,s[:,j])-roc_auc_score(y,s[:,0])),auroc_delta_ci95=np.quantile(aucs[:,j]-aucs[:,0],[.025,.975]).tolist(),
                delta_pauc=float(roc_auc_score(y,s[:,j],max_fpr=.1)-roc_auc_score(y,s[:,0],max_fpr=.1)),
                pauc_delta_ci99375=np.quantile(partials[:,j]-partials[:,0],[.025/8,1-.025/8]).tolist())
            lo,hi=m['pauc_delta_ci99375'];m['decision']='positive_interval' if lo>0 else 'worse' if hi<0 else 'unconfirmed'
        metrics.append(m)
    mechanisms=[]
    for j,k in [(5,4),(6,4),(7,5)]:
        mechanisms.append({'comparison':NAMES[j]+' vs '+NAMES[k],'pauc_delta':metrics[j]['pauc']-metrics[k]['pauc'],
                           'ci95_descriptive':np.quantile(partials[:,j]-partials[:,k],[.025,.975]).tolist()})
    return {'run_id':'OC-SA1','batch':batch,'counts':[len(g) for g in groups],'metrics':metrics,'mechanisms_descriptive':mechanisms,
            'bootstrap':{'seed':20261030,'repeats':10000,'family':8,'coverage':.99375,'primary':'standardized pAUC FPR<=0.1'},
            'scope':'historical results are diagnostic only; conditional on fixed calibration and rules'}
