"""Score only models whose freeze is committed before access to heldout scores."""
import json,time,joblib
import numpy as np
from stability_core import ROOT,verify_freeze,arrays,base,summary_features,predict_kernel,write,digest
from stability_analysis import analyze

def main():
    import torch
    torch.backends.cuda.matmul.allow_tf32=False;began=time.monotonic();f=verify_freeze(True);out=ROOT/'results/stability';assert not (out/'evaluation.json').exists()
    a=arrays();rows=a[0];pool=np.array([i for i,r in enumerate(rows) if r['role']=='pool']);evalix=np.array([i for i,r in enumerate(rows) if r['role']!='pool']);lookup={int(i):j for j,i in enumerate(pool)}
    xp=torch.as_tensor(base(a,pool),device='cuda');xt=torch.as_tensor(base(a,evalix),device='cuda');kb=(xt@xp.T).cpu().numpy();del xt,xp
    cross={'B':kb}
    for family in ('mean','dynamic','random'):
        xp=torch.as_tensor(summary_features(np.asarray(a[2][pool]),family),device='cuda');xt=torch.as_tensor(summary_features(np.asarray(a[2][evalix]),family),device='cuda');cross[family]=kb+(xt@xp.T).cpu().numpy();del xt,xp
    scores={}
    for c in f['models']:
        h=joblib.load(out/c['file']);pi=[lookup[int(i)] for i in h['training_indices']];scores[c['name']]=predict_kernel(h,cross[c['family']][:,pi])
    published=[{**{k:rows[i][k] for k in ('id','group_id','role','label')},'scores':{n:float(v[j]) for n,v in scores.items()}} for j,i in enumerate(evalix)]
    target=out/'heldout-scores.jsonl';target.write_bytes((''.join(json.dumps(r)+'\n' for r in published)).encode())
    e=analyze(published);e['scores_sha256']=digest(target);e['seconds']=time.monotonic()-began;write(out/'evaluation.json',e);print('EVALUATED',len(published),flush=True)

if __name__=='__main__':main()
