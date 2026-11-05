import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from analysis import aggregate,calibrate,old
from sklearn.metrics import roc_auc_score

f=np.zeros((2,32));f[0,16:30]=np.arange(14);f[1,16:30]=[10,0,10,0,10,0,10,0,0,0,0,0,0,0]
s=aggregate(f)
assert np.array_equal(s[0],[84.5,6.5,11.5,6.5,11.5])
assert s[1,2]==10 and s[1,4]==5
assert calibrate(np.arange(400))==380
assert not (380>calibrate(np.arange(400)))
try:aggregate(np.zeros((1,14)))
except ValueError:pass
else:raise AssertionError('must reject ambiguous layer indexing')
# Repeated, tied resamples must agree with sklearn's clipped McClish pAUC.
y=np.array([0,0,0,1,1,1]);v=np.array([0,1,1,1,2,2.])
idx=np.array([[0,0,2,3,4,5],[2,1,0,3,3,5]])
a,p=old.bootstrap_metrics(y,v,idx)
for k,ii in enumerate(idx):
    assert abs(a[k]-roc_auc_score(y[ii],v[ii]))<1e-14
    assert abs(p[k]-roc_auc_score(y[ii],v[ii],max_fpr=.1))<1e-14
print('OC_CONTRACT_PASS')
