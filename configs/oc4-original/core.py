"""Matched original OC4 cohorts; reuse frozen S1/A2 estimators."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];ORIGIN=ROOT.parents[1]
OUT=ROOT/'results/oc4-original';CACHE=ROOT/'cache/oc4-original'
H1=ORIGIN/'cache/oc4-worktree';A2=ORIGIN/'cache/attention-worktree'
SEEDS=[20261104,20261105,20261106];REPEATS=20000
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_bytes((json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
def load(p):return json.loads(Path(p).read_text())
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
svm=module('o1_svm',ROOT/'configs/oc-svm/core.py')
attention=module('o1_attention',ROOT/'configs/oc-attention/core.py');attention.ROOT=A2
NEURAL=attention.NAMES;SVM=list(svm.GRIDS)
CANDIDATES=[svm.COST,svm.RBF,svm.PAUC,attention.GATE,attention.PLE_A,attention.PLE_M,attention.PAUC,attention.FT]
DIRECT={svm.COST:svm.BASE,svm.RBF:svm.BASE,svm.PAUC:svm.BASE,attention.GATE:attention.ATT,attention.PLE_A:attention.ATT,attention.PLE_M:attention.LIN_M,attention.PAUC:attention.ATT,attention.FT:attention.TRANS}
CONTRASTS=[(n+'-vs-B',[(n,1),('B',-1)]) for n in CANDIDATES]
CONTRASTS += [(n+'-vs-direct',[(n,1),(DIRECT[n],-1)]) for n in CANDIDATES]
CONTRASTS += [('rbf-vs-linear-svc',[(svm.RBF,1),(svm.CTRL,-1)])]
CONTRASTS += [(n,terms) for n,terms in attention.CONTRASTS if n in ['gate-vs-ungated','encoding-vs-linear-interface','encoding-attention-specific','ft-vs-attention']]
CONTRASTS += [(n+'-vs-SVM',[(n,1),(svm.BASE,-1)]) for n in [attention.GATE,attention.PLE_A,attention.PLE_M,attention.PAUC,attention.FT]]
CONTRASTS += [('attention-vs-SVM',[(attention.ATT,1),(svm.BASE,-1)])]
assert len(CONTRASTS)==27
HISTORICAL=['OC-4-1-3',attention.ATT,attention.TRANS,attention.MLP]
NAMES=['B']+SVM+NEURAL+['H1:'+n for n in HISTORICAL]
def check_lock():
    seal=load(OUT/'source-lock.json')
    for group in ['sources','inputs','vendor']:
        for name,sha in seal[group].items():
            p=Path(name);assert digest(p if p.is_absolute() else ROOT/p)==sha,name
    return seal
def score(art,x):
    if art['name'] in SVM:return svm.score(art,x)
    return attention.score(art,x)
def threshold(x):return attention.threshold(x)
def point(y,s,cuts):
    from sklearn.metrics import roc_auc_score
    rows=[]
    for j,n in enumerate(NAMES):
        pred=s[:,j]>cuts[j];fp=int(pred[y==0].sum());tp=int(pred[y==1].sum())
        rows.append({'id':n,'auroc':float(roc_auc_score(y,s[:,j])),'pauc':float(roc_auc_score(y,s[:,j],max_fpr=.1)),
            'tpr':tp/int((y==1).sum()),'fpr':fp/int((y==0).sum()),'tp':tp,'fp':fp,'threshold':float(cuts[j])})
    return rows
def bootstrap_draws(y):
    rng=np.random.default_rng(SEEDS[0]);return [rng.choice(np.flatnonzero(y==k),(REPEATS,int((y==k).sum()))) for k in (0,1)]
