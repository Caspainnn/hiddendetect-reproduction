import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from verify import sparse_reference_scores
z=np.array([[1.3,2.7]],np.float32);norm=np.array([6.3],np.float32);weights=np.array([[1.,2.]],np.float32)
expected=(z.astype(np.float16)/norm.astype(np.float16)[:,None]).astype(float)@(weights.astype(float)/np.linalg.norm(weights.astype(float),axis=1)[:,None]).T
assert np.array_equal(sparse_reference_scores(z,norm,weights,np.linalg.norm(weights.astype(float),axis=1)),expected)
assert abs(float(expected[0,0])-((z/norm[:,None])@(weights/np.linalg.norm(weights,axis=1)[:,None]).T).item())>1e-6
print('RV_FP16_REPLAY_CONTRACT_PASS')
