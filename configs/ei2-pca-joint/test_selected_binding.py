import copy
import numpy as np
from sklearn.linear_model import LogisticRegression
from audit_selected import assert_entry_matches

x=np.arange(80,dtype=float).reshape(20,4)/80
y=np.arange(20)%2
entry={'head':LogisticRegression(solver='liblinear').fit(x,y),'mean':np.zeros(4,dtype=np.float32),'scale':np.ones(4,dtype=np.float32),'indices':[np.array([0],dtype=int) for _ in range(14)],'C':1.,'seed':42,'variant':'test'}
assert_entry_matches(copy.deepcopy(entry),entry)
for field in ['head','mean','scale','indices']:
 bad=copy.deepcopy(entry)
 if field=='head':bad['head'].coef_[0,0]+=.001
 elif field=='indices':bad[field][0][0]=1
 else:bad[field][0]+=.001
 try:assert_entry_matches(bad,entry)
 except AssertionError:pass
 else:raise AssertionError('Undetected selected entry mutation: '+field)
print('SELECTED_ENTRY_BINDING_CONTRACT_PASS')
