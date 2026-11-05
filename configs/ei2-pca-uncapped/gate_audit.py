"""Independent product bounds for frozen float32 gates; no changes to masks/models."""
import numpy as np
from scipy.special import expit

def check_gate_float32(a,probabilities,stored,ranks):
 a=np.asarray(a);probabilities=np.asarray(probabilities);stored=np.asarray(stored);ranks=np.asarray(ranks)
 assert a.dtype==probabilities.dtype==stored.dtype==np.float32 and a.shape==probabilities.shape==stored.shape
 valid=np.arange(a.shape[1])[None,:]<ranks[:,None];ideal_probability=expit(a.astype(np.float64));eps=np.finfo(np.float32).eps
 np.testing.assert_allclose(probabilities,ideal_probability,atol=eps,rtol=eps)
 # A product of j+1 binary32 factors incurs j multiplies. Bound their accumulated
 # rounding by gamma_j=j*u/(1-j*u), u=eps/2; include an underflow allowance.
 reference=np.cumprod(probabilities.astype(np.float64),axis=1)*valid;j=np.arange(a.shape[1]);gamma=j*(eps/2)/(1-j*(eps/2))
 bound=gamma[None,:]*np.abs(reference)+np.finfo(np.float32).tiny*a.shape[1]
 error=np.abs(stored.astype(float)-reference);assert np.all(error<=bound+1e-14),(float(error.max()),float(bound.max()))
 ideal=np.cumprod(ideal_probability,axis=1)*valid
 np.testing.assert_array_equal(stored>=.5,reference>=.5)
 np.testing.assert_array_equal(stored>=.5,ideal>=.5)
 return {'hard_masks_identical':True,'max_rounded_probability_product_difference':float(error.max()),'max_sigmoid64_difference':float(np.max(np.abs(stored-ideal))),'max_analytic_product_bound':float(bound.max()),'minimum_threshold_margin':float(np.min(np.abs(stored[valid]-.5))),'u':eps/2}
