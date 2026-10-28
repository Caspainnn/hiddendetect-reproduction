"""Cross-fitted expert fusion; legacy representation and metrics remain unchanged."""
import csv
import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'configs/highdim'))
from features import digest,read_jsonl,LAYOUT
from ridge import raw_matrix,fit_ridge,ridge_scores
from detectors import fit_detector,detector_scores
EXPERTS=('B','P','A','T')

def read_csv_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream: return list(csv.DictReader(stream))

def validate_fresh_rows(rows,old_ids,old_groups,probe_ids):
    assert len({x['id'] for x in rows})==len(rows)
    assert len({x['group_id'] for x in rows})==len(rows)
    for x in rows:
        if x['role']=='probe_train': assert x['id'] in probe_ids
        else: assert x['id'] not in old_ids and x['group_id'] not in old_groups
        if x['role']=='calibration_toxic': assert x['label']==0

def make_folds(labels):
    return list(StratifiedKFold(5,shuffle=True,random_state=20261030).split(np.zeros(len(labels)),labels))

def expert_matrix(cache,indices,kind):
    ix=np.asarray(indices)
    if kind=='B': return raw_matrix(cache,ix,[0])
    if kind=='P': return raw_matrix(cache,ix,[1,2,3,4])
    if kind=='A': return np.asarray(cache['A'][ix]).reshape(len(ix),-1)
    if kind=='T': return np.concatenate([np.asarray(cache['S'][ix]).reshape(len(ix),-1),np.asarray(cache['M'][ix]).reshape(len(ix),-1)],axis=1)
    raise ValueError('Unknown expert')

def fit_expert(x,y,kind): return fit_ridge(x,y) if kind in ('B','P') else fit_detector('linear',x,y)
def score_expert(head,x,kind): return ridge_scores(head,x) if kind in ('B','P') else detector_scores(head,x)

def pair_features(z):
    pairs=[z[:,i]*z[:,j] for i in range(z.shape[1]) for j in range(i+1,z.shape[1])]
    return np.column_stack([z,*pairs]) if pairs else z.copy()

def fit_meta(scores,y,mask,interactions=False):
    mask=np.asarray(mask,bool); assert mask.any()
    x=np.asarray(scores[:,mask],dtype=np.float64); mean=x.mean(0); scale=x.std(0); scale[scale==0]=1
    z=(x-mean)/scale; generated=pair_features(z) if interactions else z
    gmean=generated.mean(0); gscale=generated.std(0); gscale[gscale==0]=1
    model=LogisticRegression(C=.1,solver='liblinear',class_weight='balanced',random_state=0,max_iter=2000)
    model.fit((generated-gmean)/gscale,y); assert model.n_iter_.max()<2000
    return {'mask':mask,'interactions':interactions,'score_mean':mean,'score_scale':scale,
        'generated_mean':gmean,'generated_scale':gscale,'coef':model.coef_[0],'intercept':float(model.intercept_[0])}

def meta_scores(head,scores):
    z=(np.asarray(scores[:,head['mask']],dtype=np.float64)-head['score_mean'])/head['score_scale']
    generated=pair_features(z) if head['interactions'] else z
    return ((generated-head['generated_mean'])/head['generated_scale'])@head['coef']+head['intercept']
