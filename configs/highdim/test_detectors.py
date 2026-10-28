import numpy as np
from sklearn.metrics import roc_auc_score
from detectors import schema, columns_for_mask, fit_detector, detector_scores

for enabled in (False,True):
    groups=schema(enabled)
    cols=columns_for_mask(np.ones(len(groups),bool),groups)
    assert np.array_equal(cols,np.arange(groups[-1]['stop']))
    assert len(set(cols))==len(cols)
    try: columns_for_mask(np.zeros(len(groups),bool),groups)
    except ValueError: pass
    else: raise AssertionError('Empty group mask accepted')
rng=np.random.default_rng(9)
train=rng.uniform(-1,1,size=(1200,2)); y=(train[:,0]*train[:,1]>0).astype(int)
test=rng.uniform(-1,1,size=(600,2)); yt=(test[:,0]*test[:,1]>0).astype(int)
linear=fit_detector('linear',train,y); interaction=fit_detector('interaction',train,y)
assert np.allclose(linear['mean'],train.mean(axis=0))
assert roc_auc_score(yt,detector_scores(interaction,test))>.90
assert roc_auc_score(yt,detector_scores(linear,test))<.65
saved=interaction['mean'].copy(); test[:]=1e6
assert np.array_equal(saved,interaction['mean'])
print('PASS: group indexing, nonempty masks, train-only scaling and demonstrated nonlinear interaction capacity')
