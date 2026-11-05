from pathlib import Path
import importlib.util
import os
import numpy as np
os.environ.pop('CUBLAS_WORKSPACE_CONFIG',None)
path=Path(__file__).with_name('common.py');assert path.exists(),'Full-rank implementation is missing'
spec=importlib.util.spec_from_file_location('uncapped',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') in [':4096:8',':16:8'],'Configure deterministic CUDA GEMM before importing torch'
rng=np.random.default_rng(31);x=rng.normal(size=(42,80));basis=m.fit_dual(x,False);scores=m.transform_dual(x,x,basis,False)
u,s,v=np.linalg.svd(x-x.mean(0),full_matrices=False)
assert len(basis['values'])==41>32
np.testing.assert_allclose(basis['values'],s[:41]**2,rtol=1e-10,atol=1e-10)
np.testing.assert_allclose(scores@scores.T,(x-x.mean(0))@(x-x.mean(0)).T,rtol=1e-10,atol=1e-10)
assert abs(basis['values'].sum()/basis['total_energy']-1)<1e-12
counts,coverage,capped=m.variance_counts([basis]*14,.95,None);assert not any(capped) and min(coverage)>=.95
limited=m.variance_counts([basis]*14,.95,2);assert all(limited[2])
f=rng.normal(size=(42,14));a=[scores]*14;packed,mean,scale=m.pack(f,a,[41]*14);assert packed.shape==(42,588)
prefix=m.pack(f,a,[2]*14)[0];np.testing.assert_array_equal(prefix[:,:14],packed[:,:14]);assert m.count_for('rank-100',1599)==1599 and m.count_for('rank-75',1065)==799
q=rng.uniform(.7,.99,size=(14,89));np.testing.assert_allclose(m.scan_prefix(q,np),np.cumprod(q,axis=1),rtol=1e-12,atol=1e-12)
assert len(m.TAGS)==44 and max(int(tag) for tag in m.TAGS if tag.isdigit())>32
cv=np.full((3,44,3),.5);cv[:,43,:]=.9;ks=np.array([[m.count_for(tag,1065) for tag in m.TAGS]]*3)
assert m.choose_cv(cv,m.TAGS,ks)['tag']=='rank-100'
assert m.choose_cv(cv[:,:32],m.TAGS[:32],ks[:,:32])['tag']=='1'
import verify as checker
assert hasattr(checker,'check_gate_ranges'),'Verifier must prove full-range gates saw the actual natural rank'
records=[{'scope':scope,'lambda':lam,'seed':seed,'ranks':([80]*14 if scope=='full' else [32]*14)} for scope in ['full','cap32'] for lam in [0.,.01,.1,1.] for seed in m.SEEDS]
checker.check_gate_ranges(records,[80]*14)
bad=[dict(record) for record in records];bad[0]['ranks']=[32]*14
try:checker.check_gate_ranges(bad,[80]*14)
except AssertionError:pass
else:raise AssertionError('Silently retained32 bound in full-range gate passed verification')
try:import torch
except ImportError:print('SKIP_CUDA_AND_TORCH_GRADIENT: torch not installed locally')
else:
 tensor=torch.tensor(q,requires_grad=True);v=m.scan_prefix(tensor,torch);v.sum().backward();reference=torch.tensor(q,requires_grad=True);torch.cumprod(reference,dim=1).sum().backward();np.testing.assert_allclose(tensor.grad.numpy(),reference.grad.numpy(),rtol=1e-10,atol=1e-10)
 if torch.cuda.is_available():
  torch.use_deterministic_algorithms(True);cuda=torch.tensor(q,device='cuda',requires_grad=True);m.scan_prefix(cuda,torch).sum().backward();np.testing.assert_allclose(cuda.grad.cpu().numpy(),reference.grad.numpy(),rtol=1e-10,atol=1e-10);np.testing.assert_array_equal(m.matmul(np.ones((2,3)),np.ones((3,4))),np.full((2,4),3.));print('CUDA_SCAN_GRADIENT_AND_DETERMINISTIC_GEMM_PASS')
 else:print('SKIP_CUDA_SCAN: no GPU available')
print('UNCAPPED_CPU_CONTRACT_PASS')
