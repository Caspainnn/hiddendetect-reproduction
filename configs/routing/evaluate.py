"""Scoring only after byte/commit gates; fixed calibration and frozen selection."""
import json,time,joblib
import numpy as np
from routing_core import (ROOT,digest,write,verify_freeze,load_arrays,block,route,
    scores,read_jsonl)
from ridge import ridge_scores
from analyze import analyze

def main():
    out=ROOT/'results/routing';assert not (out/'evaluation.json').exists()
    began=time.monotonic();f=verify_freeze();arrays=load_arrays();rows=arrays[0]
    indices=np.array([i for i,r in enumerate(rows) if r['role'] not in ('train','dev')])
    heads={c['name']:joblib.load(out/c['file']) for c in f['candidates']};values={c['name']:np.empty(len(indices)) for c in f['candidates']}
    for start in range(0,len(indices),128):
        ix=indices[start:start+128];cached={}
        blocks={mode:block(arrays,ix,mode) for mode in ('variable','fixed')}
        for c in f['candidates']:
            signature=(c['mode'],str(c.get('suffix')),tuple(c.get('mask',[])),bool(c.get('legacy')))
            if signature not in cached:
                if c.get('suffix') is not None:x=np.asarray(arrays[1][ix,:,c['suffix'],:],np.float64).reshape(len(ix),-1)/np.sqrt(32)
                else:x=route(blocks[c['mode']],c['mask'])
                if c.get('legacy'):
                    # Preserve the preceding float32 normalization arithmetic exactly.
                    x=blocks['variable'][:,:,0,:].reshape(len(ix),-1).astype(np.float32)/np.sqrt(32)
                    cached[signature]=ridge_scores(heads[c['name']],x)
                else:cached[signature]=scores(heads[c['name']],x)
            values[c['name']][start:start+len(ix)]=cached[signature]
        print('SCORED',min(start+128,len(indices)),'/',len(indices),flush=True)
    old={r['id']:r for r in read_jsonl(ROOT/'results/positions/heldout-scores.jsonl')};max_delta=0.
    path=out/'heldout-scores.jsonl'
    with path.open('w',encoding='utf8',newline='\n') as stream:
        for j,i in enumerate(indices):
            row=rows[i];v={n:float(a[j]) for n,a in values.items()};assert np.isfinite(list(v.values())).all()
            if row['id'] in old:max_delta=max(max_delta,abs(v['legacy_B']-old[row['id']]['scores']['template_last']))
            stream.write(json.dumps({'id':row['id'],'group_id':row['group_id'],'role':row['role'],'label':row['label'],'scores':v})+'\n')
    assert max_delta<1e-4,max_delta
    r=analyze(read_jsonl(path),f);r.update(freeze_sha256=digest(out/'freeze.json'),scores_sha256=digest(path),maximum_legacy_score_difference=max_delta,seconds=time.monotonic()-began)
    write(out/'evaluation.json',r);print('ROUTING_EVALUATION_COMPLETE',r['seconds'],flush=True)

if __name__=='__main__':main()
