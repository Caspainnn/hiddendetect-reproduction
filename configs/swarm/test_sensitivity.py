import numpy as np
from sensitivity import draw_policy
cal=np.array([-1.,0.,1.,2.]);s=np.array([-.5,.5,1.5,2.5])
draws=np.array([[0,1,2,3],[0,0,1,1]]);b0=np.array([[0,1],[1,1]]);b1=np.array([[2,3],[3,3]])
r=draw_policy(cal,s,b0,b1,draws)
for i in range(2):
    threshold=np.sort(cal[draws[i]])[3]
    assert r['threshold'][i]==threshold
    assert r['tpr'][i]==np.mean(s[b1[i]]>threshold) and r['fpr'][i]==np.mean(s[b0[i]]>threshold)
print('PASS: shared calibration draws and paired test resampling match direct computation')
