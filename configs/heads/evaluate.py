"""Frozen linear heads and private-reference RBF inference, no refitting."""
import json,time,joblib
import numpy as np
from head_core import ROOT,digest,write,verify_freeze,arrays,matrix,squared_distances,predict_kernel,read_jsonl
from head_analysis import analyze

def main():
    import torch
    torch.backends.cuda.matmul.allow_tf32=False;out=ROOT/'results/heads';assert not (out/'evaluation.json').exists();began=time.monotonic();f=verify_freeze();a=arrays();rows=a[0]
    ti=np.array([i for i,r in enumerate(rows) if r['role']=='train']);ixall=np.array([i for i,r in enumerate(rows) if r['role'] not in ('train','dev')]);train=matrix(a,ti);t=torch.as_tensor(train,device='cuda');norm=(train*train).sum(1)
    models={c['name']:joblib.load(out/c['file']) for c in f['candidates']};values={n:np.empty(len(ixall)) for n in models}
    for h in models.values():assert h['training_ids']==[rows[i]['id'] for i in ti]
    for start in range(0,len(ixall),128):
        ix=ixall[start:start+128];x=matrix(a,ix);k=(torch.as_tensor(x,device='cuda')@t.T).cpu().numpy();d=squared_distances(k,(x*x).sum(1),norm)
        for n,h in models.items():
            s=predict_kernel(h,k,d) if h['method']=='rbf' else (x-h['mean'])@h['coef']+h['intercept'];values[n][start:start+len(ix)]=s
        print('SCORED',min(start+128,len(ixall)),'/',len(ixall),flush=True)
    old={r['id']:r for r in read_jsonl(ROOT/'results/routing/heldout-scores.jsonl')};maximum=0.;path=out/'heldout-scores.jsonl'
    with path.open('w',encoding='utf8',newline='\n') as stream:
        for j,i in enumerate(ixall):
            row=rows[i];v={n:float(s[j]) for n,s in values.items()};assert np.isfinite(list(v.values())).all()
            if row['id'] in old:maximum=max(maximum,abs(v['B_original']-old[row['id']]['scores']['fixed_template']))
            stream.write(json.dumps({'id':row['id'],'group_id':row['group_id'],'role':row['role'],'label':row['label'],'scores':v})+'\n')
    assert maximum<1e-7,maximum
    r=analyze(read_jsonl(path),f);r.update(freeze_sha256=digest(out/'freeze.json'),scores_sha256=digest(path),maximum_old_B_score_difference=maximum,seconds=time.monotonic()-began);write(out/'evaluation.json',r);print('HEAD_EVALUATION_COMPLETE',r['seconds'],flush=True)

if __name__=='__main__':main()
