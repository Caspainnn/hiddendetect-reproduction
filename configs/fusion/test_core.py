import tempfile
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from fusion_core import make_folds,pair_features,fit_meta,meta_scores,read_csv_rows,validate_fresh_rows

y=np.array([0]*50+[1]*50)
folds=make_folds(y); seen=[]
for train,valid in folds:
    assert not set(train)&set(valid) and len(train)==80 and len(valid)==20
    seen+=valid.tolist()
assert sorted(seen)==list(range(100))
assert np.array_equal(pair_features(np.array([[1.,2.,3.]])),[[1,2,3,2,3,6]])
rng=np.random.default_rng(3); x=rng.uniform(-1,1,size=(1200,2)); labels=(x[:,0]*x[:,1]>0).astype(int)
linear=fit_meta(x,labels,np.ones(2,bool),False); nonlinear=fit_meta(x,labels,np.ones(2,bool),True)
assert roc_auc_score(labels,meta_scores(linear,x))<.65 and roc_auc_score(labels,meta_scores(nonlinear,x))>.95
saved=nonlinear['score_mean'].copy(); meta_scores(nonlinear,x*100)
assert np.array_equal(saved,nonlinear['score_mean'])
with tempfile.TemporaryDirectory() as folder:
    path=Path(folder)/'a.csv'; path.write_bytes(b'id,text\n1,"line one\nline two"\n')
    assert read_csv_rows(path)[0]['text']=='line one\nline two'
validate_fresh_rows([{'id':'old','group_id':'g0','role':'probe_train','label':0},{'id':'new','group_id':'g1','role':'calibration_toxic','label':0}],{'old'},{'g0'},{'old'})
try: validate_fresh_rows([{'id':'old','group_id':'g0','role':'test','label':0}],{'old'},{'g0'},{'old'})
except AssertionError: pass
else: raise AssertionError('Old test reuse accepted')
print('PASS: cross-fit exclusion, interaction capacity, train-only transforms, CSV newlines and old-data/probe exception')
