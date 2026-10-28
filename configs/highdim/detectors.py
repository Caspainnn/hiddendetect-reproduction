"""Fixed group map and capacity-matched detector contracts."""
import sys
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from features import ROOT
sys.path.insert(0,str(ROOT/'configs/signal-selection'))
from train_search import fit_centroids, centroid_features

def schema(attention_enabled):
    groups=[]; cursor=0
    def add(name,family,width,layer=None,region=None):
        nonlocal cursor
        groups.append({'name':name,'family':family,'start':cursor,'stop':cursor+width,'layer':layer,'region':region}); cursor+=width
    for layer in range(32):
        for region in range(5): add('H_%d_%d'%(layer,region),'H',64,layer,region)
    for layer in range(32): add('trajectory_%d'%layer,'trajectory',16,layer)
    if attention_enabled:
        for layer in range(32): add('attention_%d'%layer,'attention',256,layer)
    for layer in range(32): add('module_%d'%layer,'module',8,layer)
    add('author_F32','F',32); add('centroid_C32','C',32); add('length_missing','length',6)
    return groups

def columns_for_mask(mask,groups):
    mask=np.asarray(mask,dtype=bool)
    if len(mask)!=len(groups) or not mask.any(): raise ValueError('Invalid/empty group mask')
    return np.concatenate([np.arange(g['start'],g['stop']) for take,g in zip(mask,groups) if take])

def make_matrix(cache,indices,references,attention_enabled):
    ix=np.asarray(indices)
    pieces=[np.asarray(cache['X'][ix]).reshape(len(ix),-1),np.asarray(cache['S'][ix]).reshape(len(ix),-1)]
    if attention_enabled: pieces.append(np.asarray(cache['A'][ix]).reshape(len(ix),-1))
    pieces += [np.asarray(cache['M'][ix]).reshape(len(ix),-1),np.asarray(cache['F'][ix]),
        centroid_features(np.asarray(cache['P'][ix,:,0,:]),references),np.asarray(cache['Q'][ix])]
    result=np.concatenate(pieces,axis=1).astype(np.float32)
    assert result.shape[1]==schema(attention_enabled)[-1]['stop'] and np.isfinite(result).all()
    return result

def raw_fit(kind,z,y,penalty='l2'):
    if kind=='linear':
        model=LogisticRegression(C=.01,penalty=penalty,solver='liblinear',class_weight='balanced',random_state=0,max_iter=2000)
    elif kind=='interaction':
        model=ExtraTreesClassifier(n_estimators=32,max_depth=3,min_samples_leaf=20,max_features='sqrt',
            class_weight='balanced',n_jobs=1,random_state=0)
    else: raise ValueError('Unknown detector')
    model.fit(z,y)
    if kind=='linear': assert model.n_iter_.max()<2000,'Non-converged detector'
    assert np.array_equal(model.classes_,[0,1])
    return model

def raw_scores(kind,model,z):
    return model.decision_function(z) if kind=='linear' else model.predict_proba(z)[:,1]

def fit_detector(kind,x,y,penalty='l2'):
    x=np.asarray(x,dtype=np.float32); mean=x.mean(0); scale=x.std(0); scale[scale==0]=1
    model=raw_fit(kind,(x-mean)/scale,y,penalty)
    return {'kind':kind,'mean':mean,'scale':scale,'model':model}

def detector_scores(detector,x):
    z=(np.asarray(x,dtype=np.float32)-detector['mean'])/detector['scale']
    return raw_scores(detector['kind'],detector['model'],z)
