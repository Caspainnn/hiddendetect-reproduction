import importlib.util
from pathlib import Path
import numpy as np
from scipy.special import expit
from common import scan_prefix
path=Path(__file__).with_name('gate_audit.py');assert path.exists(),'Precision-aware independent gate audit is missing'
spec=importlib.util.spec_from_file_location('audit',path);audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
a=np.full((1,1599),8.,np.float32);p=expit(a.astype(float)).astype(np.float32);g=scan_prefix(p,np)
assert np.max(np.abs(np.cumprod(expit(a.astype(float)),axis=1)-g))>1e-5
result=audit.check_gate_float32(a,p,g,[1599]);assert result['hard_masks_identical'] and result['max_sigmoid64_difference']>1e-5
try:audit.check_gate_float32(a,p,g+.001,[1599])
except AssertionError:pass
else:raise AssertionError('Out-of-bound gate corruption passed')
print('GATE_PRECISION_CONTRACT_PASS')
