"""Seal both caches and verify repeated old-training probes have compatible features."""
import json
import sys
import numpy as np
from fusion_core import ROOT,digest,LAYOUT
from seal import verify_seal

if __name__=='__main__':
    old=ROOT/'cache/highdim'; new=ROOT/'cache/swarm'
    for directory in (old,new): verify_seal(directory,json.loads((directory/'seal.json').read_text()))
    prior=json.loads((ROOT/'configs/highdim/manifest.json').read_text())['samples']
    current=json.loads((ROOT/'configs/swarm/manifest.json').read_text())['samples']
    lookup={x['id']:x for x in prior}; probes=[x for x in current if x['role']=='probe_train']
    assert len(probes)==64
    previous=[lookup[x['id']]['index'] for x in probes]; present=[x['index'] for x in probes]
    assert all(lookup[x['id']]['role']=='train' and lookup[x['id']]['input_ids_sha256']==x['input_ids_sha256'] for x in probes)
    differences={}
    for name in LAYOUT:
        a=np.load(old/(name+'.npy'),mmap_mode='r')[previous].astype(np.float32)
        b=np.load(new/(name+'.npy'),mmap_mode='r')[present].astype(np.float32)
        differences[name]=float(np.max(np.abs(a-b)))
        if name in ('P','A','S','M'): assert np.allclose(a,b,rtol=1e-4,atol=1e-5),(name,differences[name])
        if name=='F': assert np.allclose(a,b,rtol=0,atol=1e-7)
        if name=='Q': assert np.array_equal(a,b)
    report={'status':'pass','old_training_probes':64,'maximum_absolute_differences':differences,
        'strict_checks':['P','A','S','M','F','Q'],'diagnostic_only':['X','L'],
        'old_cache_seal_sha256':digest(old/'seal.json'),'new_cache_seal_sha256':digest(new/'seal.json')}
    (ROOT/'logs/swarm-probe-compatibility.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PROBE_COMPATIBILITY_PASS',json.dumps(differences))
