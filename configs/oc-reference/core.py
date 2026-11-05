"""OC-RV1: only the reference changes; author projection/cosine/integral stay fixed."""
import hashlib,importlib.util,json,math
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[2]
NAMES=['B','OC-1-1','OC-1-2','OC-1-3']
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with Path(p).open(encoding='utf-8') as f:return [json.loads(x) for x in f if x.strip()]
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode())
def references(rules,encoded,vocab=32000):
    sets=[rules['original'],rules['clean_keep'],rules['original']+rules['expand_add'],rules['original']]
    assert set(sets[1])<set(sets[0]) and not set(rules['expand_add'])&set(sets[0])
    w=np.zeros((4,vocab),dtype=np.float32)
    for j,words in enumerate(sets):
        for word in words:
            ix=encoded[word][0];assert 0<=ix<vocab
            value=rules['enhanced_weight'] if j==3 and word in rules['weight_two'] else rules['default_weight']
            w[j,ix]=max(w[j,ix],value)
    assert np.all(np.linalg.norm(w,axis=1)>0)
    return w
def integrate(f):
    f=np.asarray(f,dtype=np.float64);assert f.shape==(32,4) and np.isfinite(f).all()
    return np.trapz(f[16:30],axis=0)
def threshold(s):
    s=np.asarray(s);assert s.shape==(400,) and np.isfinite(s).all()
    return float(np.sort(s)[380])
def validate_confirmation(out):
    required=['manifest.json','token-audit.json','source-lock-v2.json','thresholds.json','historical-evaluation.json',
              'history-scores.jsonl','extraction-history.json','identity-history.json']
    pre=json.loads((out/'pre-confirmation-seal.json').read_text())
    assert pre['family']==NAMES[1:] and pre['status']=='frozen_before_confirmation_scoring'
    assert set(pre['sha256'])==set(required),'Incomplete pre-confirmation evidence'
    for name,h in pre['sha256'].items():assert digest(out/name)==h,name
    historical=json.loads((out/'historical-evaluation.json').read_text())
    assert abs(historical['metrics'][0]['auroc']-.746640)<1e-12
    identity=json.loads((out/'identity-history.json').read_text())
    assert identity['status']=='passed' and identity['maximum_original_F_difference']==identity['integral_difference']==0
    extraction=json.loads((out/'extraction-history.json').read_text())
    assert extraction['status']=='completed' and extraction['count']==1400
    assert extraction['scores_sha256']==digest(out/'history-scores.jsonl')
    return pre
def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n;c=(p+z*z/(2*n))/den
    d=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(c-d),float(c+d)]
def evaluate(rows,cut,batch):
    spec=importlib.util.spec_from_file_location('rv_bootstrap',ROOT/'configs/original-positions/core.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    y=np.array([r['label'] for r in rows]);s=np.array([r['scores'] for r in rows]);assert set(y)=={0,1} and s.shape==(len(y),4)
    groups=[np.flatnonzero(y==i) for i in (0,1)];rng=np.random.default_rng(20261030)
    ix=np.column_stack([rng.choice(g,(10000,len(g))) for g in groups]);auc=np.empty((10000,4));pauc=np.empty_like(auc)
    for start in range(0,10000,100):
        for j in range(4):auc[start:start+100,j],pauc[start:start+100,j]=old.bootstrap_metrics(y,s[:,j],ix[start:start+100])
    result=[];pred=s>np.array(cut)
    for j,name in enumerate(NAMES):
        m={'direction_id':name,'auroc':float(roc_auc_score(y,s[:,j])),'pauc':float(roc_auc_score(y,s[:,j],max_fpr=.1)),'threshold':cut[j]}
        for label,key in [(0,'fpr'),(1,'tpr')]:
            k=int(pred[y==label,j].sum());n=len(groups[label]);m.update({key:k/n,key+'_count':k,key+'_n':n,key+'_wilson95':wilson(k,n)})
            if j:
                sample=ix[:,:len(groups[0])] if label==0 else ix[:,len(groups[0]):]
                diff=pred[:,j].astype(float)-pred[:,0].astype(float)
                m[key+'_delta_ci95']=np.quantile(diff[sample].mean(1),[.025,.975]).tolist()
        if j:
            m.update(delta_auroc=float(roc_auc_score(y,s[:,j])-roc_auc_score(y,s[:,0])),auroc_delta_ci95=np.quantile(auc[:,j]-auc[:,0],[.025,.975]).tolist(),
                delta_pauc=float(roc_auc_score(y,s[:,j],max_fpr=.1)-roc_auc_score(y,s[:,0],max_fpr=.1)),
                pauc_delta_ci983333=np.quantile(pauc[:,j]-pauc[:,0],[.025/3,1-.025/3]).tolist())
            lo,hi=m['pauc_delta_ci983333'];m['decision']='positive_interval' if lo>0 else 'worse' if hi<0 else 'unconfirmed'
        result.append(m)
    return {'run_id':'OC-RV1','batch':batch,'counts':[len(g) for g in groups],'metrics':result,
        'bootstrap':{'seed':20261030,'repeats':10000,'family':3,'primary':'standardized pAUC FPR<=0.1','per_comparison_coverage':1-.05/3},
        'scope':'conditional on fixed calibration and manual rules; historical positive intervals are not new confirmation'}
