"""Read four previously frozen heads as fixed features; no label-based tuning."""
import hashlib,json,subprocess,joblib
import numpy as np
from anchored_core import ROOT,FAMILIES,digest,stream_hash,write,old_cache,arrays,matrices,summary_features,predict_kernel

def main():
    import torch
    torch.backends.cuda.matmul.allow_tf32=False;old_cache(True)
    seal=json.loads((ROOT/'logs/anchored-cache-seal.json').read_text());assert seal['manifest_sha256']==digest(ROOT/'configs/anchored/manifest.json')
    for n,h in seal['files'].items():assert stream_hash(ROOT/n)==h,n
    a=arrays();rows=a[0];oldB=np.load(ROOT/'cache/stability/B.npy',mmap_mode='r');oldS=np.load(ROOT/'cache/stability/S.npy',mmap_mode='r');oldrows=json.loads((ROOT/'configs/stability/manifest.json').read_text())['samples']
    trained=json.loads((ROOT/'results/stability/freeze.json').read_text());model_shas={};heads={}
    for family in FAMILIES:
        c=next(c for c in trained['models'] if c['name']=='0_ridge_'+family);name='results/stability/'+c['file'];assert digest(ROOT/name)==c['sha256'];model_shas[name]=c['sha256'];heads[family]=joblib.load(ROOT/name)
    ix=heads['B']['training_indices'];ids=[oldrows[int(i)]['id'] for i in ix]
    assert not set(ids)&{r['id'] for r in rows}
    for h in heads.values():assert h['training_ids']==ids and np.array_equal(h['training_indices'],ix)
    xb,s=matrices(a,np.arange(len(rows)));bt=np.asarray(oldB[ix],np.float64).reshape(len(ix),-1)/np.sqrt(32);st=np.asarray(oldS[ix]);tx=torch.as_tensor(xb,device='cuda');tt=torch.as_tensor(bt,device='cuda');kb=(tx@tt.T).cpu().numpy();del xb,bt,tx,tt
    scores=[]
    for family in FAMILIES:
        k=kb.copy()
        if family!='B':
            x=summary_features(s,family);t=summary_features(st,family);tx=torch.as_tensor(x,device='cuda');tt=torch.as_tensor(t,device='cuda');k+=(tx@tt.T).cpu().numpy();del x,t,tx,tt
        scores.append(predict_kernel(heads[family],k))
    result=np.column_stack(scores);assert np.isfinite(result).all();target=ROOT/'cache/anchored/base-scores.npy';np.save(target,result)
    old={r['id']:r for r in __import__('anchored_core').read_jsonl(ROOT/'results/stability/heldout-scores.jsonl')};maximum=0.;checked=0
    for i,r in enumerate(rows):
        if 'stability_index' not in r:continue
        for j,family in enumerate(FAMILIES):maximum=max(maximum,abs(result[i,j]-old[r['id']]['scores']['0_ridge_'+family]))
        checked+=1
    assert checked==2200 and maximum<1e-7
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();names=['configs/anchored/prepare_scores.py','configs/anchored/anchored_core.py','configs/stability/stability_core.py','configs/heads/head_core.py'];shas={n:digest(ROOT/n) for n in names}
    for n,h in shas.items():assert hashlib.sha256(subprocess.check_output(['git','show',commit+':'+n])).hexdigest()==h,n
    write(ROOT/'logs/anchored-score-seal.json',{'manifest_sha256':digest(ROOT/'configs/anchored/manifest.json'),'commit':commit,'sources':shas,'files':{'cache/anchored/base-scores.npy':digest(target)},'model_shas':model_shas,'old_score_rows_checked':checked,'maximum_old_score_difference':maximum})
    print('FIXED_SCORES_PREPARED',len(rows),'OLD_MAX',maximum,flush=True)

if __name__=='__main__':main()
