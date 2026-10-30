"""Small executable checks for the single-variable experiment contract."""
import numpy as np
from sklearn.metrics import roc_auc_score
from core import ROOT,read_jsonl,positions, integral, threshold, paired_intervals, project_readouts,bootstrap_auc,bootstrap_partial

def main():
    assert positions([8,9,10,11,12],17)==[8,9,10,11,11,12,16]
    assert positions([8],12)==[8]*6+[11]
    for bad in ([],[3,2],[2,2]):
        try: positions(bad,10)
        except ValueError: pass
        else: raise AssertionError('Invalid user sequence accepted')
    f=np.tile(np.arange(32,dtype=float)[:,None],(1,7))
    assert np.all(integral(f,16,29)==292.5)
    assert np.all(integral(f,17,29)==276.)
    assert threshold(np.arange(400))==379
    calls=[]
    def norm(x): calls.append(('norm',x.shape)); return x+1
    def head(x): calls.append(('head',x.shape)); return x*2
    # Preserve whole sequence projection and select afterwards, including repeated positions.
    out=project_readouts(np.zeros((1,10,3)),[2,2,9],norm,head)
    assert out.shape==(1,3,3) and np.all(out==2)
    assert calls==[('norm',(1,10,3)),('head',(1,10,3))]
    y=np.array([0,0,0,1,1,1]); a=np.array([0,1,1,1,2,3.]); b=np.array([0,0,1,2,2,3.])
    r=paired_intervals(y,np.column_stack([a,b,a]),repeats=100,seed=7,family=2)
    assert r[0]['delta']==0 and r[0]['ci95']==[0.,0.]
    assert r[1]['delta']==roc_auc_score(y,b)-roc_auc_score(y,a)
    assert r[1]['adjusted'][0]<=r[1]['ci95'][0]<=r[1]['ci95'][1]<=r[1]['adjusted'][1]
    resamples=np.array([[0,0,2,3,4,5],[1,2,2,3,3,5],[0,1,2,4,4,4]])
    assert np.allclose(bootstrap_auc(y,a,resamples),[roc_auc_score(y[ix],a[ix]) for ix in resamples],atol=1e-15)
    assert np.allclose(bootstrap_partial(y,a,resamples),[roc_auc_score(y[ix],a[ix],max_fpr=.1) for ix in resamples],atol=1e-15)
    # Replay all real R0 scores locally; this verifies integration, not new GPU extraction.
    rows=read_jsonl(ROOT/'results/xstest450-default-s16e29.jsonl')
    scores=[integral(np.tile(np.asarray(x['F'])[:,None],(1,7)))[-1] for x in rows]
    assert np.array_equal(scores,[x['score'] for x in rows])
    assert abs(roc_auc_score([x['label'] for x in rows],scores)-.85828)<1e-12
    print('ORIGINAL_POSITION_CONTRACT_PASS')

if __name__=='__main__': main()
