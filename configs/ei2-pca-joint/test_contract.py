from pathlib import Path
import importlib.util,numpy as np
path=Path(__file__).with_name('common.py');assert path.exists(),'Joint selection implementation missing'
spec=importlib.util.spec_from_file_location('joint',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rng=np.random.default_rng(71);f=rng.normal(size=(50,14));a=[rng.normal(size=(50,6)) for _ in range(14)];ids=[np.array([1,5])]+[np.array([],int) for _ in range(13)]
x,mean,scale=m.pack(f,a,ids);assert x.shape==(50,512);np.testing.assert_allclose(x[:,14:16],(a[0][:,[1,5]].astype(np.float32)-mean[14:])/scale[14:]/np.sqrt(56),atol=1e-7)
empty=m.pack(f,a,[np.array([],int)]*14)[0];np.testing.assert_array_equal(x[:,:14],empty[:,:14]);assert not empty[:,14:].any()
weight=np.r_[np.zeros(14),np.tile([0.,2.,2.,0.,1.,0.],14)];order=m.rank_directions(weight,[6]*14);assert order[0].tolist()==[1,2,4,0,3,5]
top=m.top_indices(order,[2]*14);assert top[0].tolist()==[1,2]
mask=m.mask_indices(np.tile([False,True,False,True,False,False],14),[6]*14);assert mask[0].tolist()==[1,3]
remap=m.map_prefix(order,[4]*14);assert remap[0].tolist()==[0,1,2,4]
pooled=[np.array([0,1,2,3]) for _ in range(14)];inside=m.map_pool(pooled,np.tile([False,True,False,True],14));assert inside[0].tolist()==[1,3]
basis=[{'values':np.array([8.,1.,1.]),'total_energy':10.} for _ in range(14)];assert m.selected_coverage(basis,[np.array([1])]*14)==[.1]*14
p=np.full((1,1599),np.float32(.9998));g=m.u.scan_prefix(p,np);audit=m.gate_product_audit(np.full((1,1599),np.log(.9998/.0002),np.float32),p,g,[1599]);assert audit['rounded_hard_mask_mismatches']==0
try:m.gate_product_audit(np.full((1,1599),np.log(.9998/.0002),np.float32),p,g+.01,[1599])
except AssertionError:pass
else:raise AssertionError('Invalid long gate product accepted')
print('JOINT_SELECTION_CONTRACT_PASS')
