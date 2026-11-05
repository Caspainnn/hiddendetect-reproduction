import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from core import references,integrate,threshold
rules={'original':['a','b'],'clean_keep':['a'],'expand_add':['c'],'weight_two':['a'],'default_weight':1.,'enhanced_weight':2.}
encoded={'a':[1,7],'b':[2],'c':[2,9]}
w=references(rules,encoded,8)
assert w.shape==(4,8)
assert np.array_equal(w[0],[0,1,1,0,0,0,0,0])
assert np.array_equal(w[1],[0,1,0,0,0,0,0,0])
assert np.array_equal(w[2],w[0])
assert np.array_equal(w[3],[0,2,1,0,0,0,0,0])
f=np.zeros((32,4));f[16:30,:]=np.arange(14)[:,None]
assert np.array_equal(integrate(f),[84.5]*4)
assert threshold(np.arange(400))==380
print('RV_CONTRACT_PASS')

# A prospective phase must not run with a seal omitting calibration evidence.
import tempfile,json
from core import validate_confirmation
with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp);(p/'pre-confirmation-seal.json').write_text(json.dumps({'sha256':{},'family':['OC-1-1','OC-1-2','OC-1-3'],'status':'frozen_before_confirmation_scoring'}))
    try:validate_confirmation(p)
    except AssertionError:pass
    else:raise AssertionError('Incomplete calibration seal must fail')
