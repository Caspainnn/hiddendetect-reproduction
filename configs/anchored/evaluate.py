"""Apply all committed heads, then compute preregistered heldout comparisons."""
import json,time,joblib
import numpy as np
from anchored_core import ROOT,FAMILIES,digest,write,verify_freeze,arrays,matrices,fuse,predict_kernel
from anchored_analysis import analyze

def main():
    import torch
    torch.backends.cuda.matmul.allow_tf32=False;began=time.monotonic();f=verify_freeze(True);out=ROOT/'results/anchored';assert not (out/'evaluation.json').exists()
    a=arrays();rows=a[0];ix=np.array([i for i,r in enumerate(rows) if r['role']!='meta_train']);raw=np.load(ROOT/'cache/anchored/base-scores.npy')[ix];heads={}
    for c in f['models']:
        if c['name']=='B_refit':continue
        p=c['parameters'];heads[c['name']]=fuse(raw[:,0],raw[:,FAMILIES.index(p['family'])],p,c['alignment'])
    h=joblib.load(out/'B_refit.joblib');oldB=np.load(ROOT/'cache/stability/B.npy',mmap_mode='r');bt=np.asarray(oldB[h['old_training_indices']],np.float64).reshape(1600,-1)/np.sqrt(32);bd,_=matrices(a,h['meta_indices']);train=np.concatenate([bt,bd]);del bt,bd
    x,_=matrices(a,ix);tx=torch.as_tensor(x,device='cuda');tt=torch.as_tensor(train,device='cuda');k=(tx@tt.T).cpu().numpy();del x,train,tx,tt
    heads['B_refit']=predict_kernel(h,k)
    published=[{**{key:rows[i][key] for key in ('id','group_id','role','label')},'scores':{n:float(s[j]) for n,s in heads.items()}} for j,i in enumerate(ix)]
    target=out/'heldout-scores.jsonl';target.write_bytes((''.join(json.dumps(r)+'\n' for r in published)).encode());e=analyze(published,f);e.update(scores_sha256=digest(target),seconds=time.monotonic()-began);write(out/'evaluation.json',e);print('EVALUATION_COMPLETE',len(published),flush=True)

if __name__=='__main__':main()
