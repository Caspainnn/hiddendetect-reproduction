"""Public-artifact replay; never changes frozen scoring or rebrands history as fresh."""
import hashlib,json,subprocess
import numpy as np
from core import ROOT,NAMES,digest,read,write,aggregate,threshold,evaluate,logsumexp

def compare(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys();return sum(compare(a[k],b[k]) for k in a)
    if isinstance(a,list):
        assert len(a)==len(b);return sum(compare(x,y) for x,y in zip(a,b))
    if isinstance(a,(int,float)) and not isinstance(a,bool):assert abs(a-b)<1e-12;return 1
    assert a==b;return 0

def main():
    out=ROOT/'results/oc23';lock=json.loads((out/'source-lock.json').read_text())
    for n,h in lock['sha256'].items():assert hashlib.sha256(subprocess.check_output(['git','show',lock['source_commit']+':'+n],cwd=ROOT)).hexdigest()==h,n
    m=json.loads((out/'manifest.json').read_text());assert not m['confirmation_enabled']
    absent=json.loads((out/'confirmation-unavailable.json').read_text());assert absent['status']=='not_performed'
    assert not (out/'confirmation-scores.jsonl').exists()
    rows=read(out/'history-scores.jsonl');assert len(rows)==2482 and [{k:r[k] for k in x} for r,x in zip(rows,m['samples'])]==m['samples']
    assert digest(out/'history-scores.jsonl')==json.loads((out/'extraction-history.json').read_text())['scores_sha256']
    old={r['id']:np.array(r['F'])[:,0] for name in ['history','confirmation'] for r in read(ROOT/('results/oc-reference/'+name+'-scores.jsonl'))}
    # Existing four OC-3 methods are replayed against their earlier per-sample scores.
    previous={r['id']:r['scores'][1:] for r in read(ROOT/'results/pure-oc/historical-scores.jsonl')}
    maximum_cos=0.;maximum_layer=np.zeros(3)
    for r in rows:
        assert np.array_equal(np.array(r['F']),old[r['id']])
        assert np.array_equal(aggregate(r['F'],r['layer_scores']),r['scores'])
        if r['id'] in previous:assert np.array_equal(r['scores'][4:8],previous[r['id']])
        selected=np.asarray(r['refusal_logits'],np.float64);logz=np.asarray(r['full_logsumexp'],np.float64)
        lse=np.array([logsumexp(z) for z in selected])
        reconstructed=np.column_stack([selected.mean(1),lse-np.log(selected.shape[1]),np.exp(lse-logz)])
        diff=np.max(np.abs(reconstructed-r['layer_scores']),axis=0);assert np.all(diff<2e-5),(r['id'],diff)
        maximum_layer=np.maximum(maximum_layer,diff)
        norm=np.asarray(r['full_logit_norm']);half=(selected.astype(np.float16)/norm.astype(np.float16)[:,None]).astype(float)
        cosine=half.sum(1)/np.sqrt(selected.shape[1]);error=float(np.max(np.abs(cosine-r['F'])));assert error<2e-7
        maximum_cos=max(maximum_cos,error)
    cal=[r for r in rows if r['role']=='calibration'];cut=[threshold(np.array([r['scores'] for r in cal])[:,j]) for j in range(len(NAMES))]
    assert cut==json.loads((out/'thresholds.json').read_text())['thresholds']
    results={}
    for role in ['historical_C','historical_RV']:
        replay=evaluate([r for r in rows if r['role']==role],cut,role)
        results[role]=compare(replay,json.loads((out/(role.replace('_','-')+'-evaluation.json')).read_text()))
    write(out/'verification.json',{'status':'passed','numeric_values_checked':results,'all_2482_original_F_and_integrals_exact':True,
        'old_OC3_four_scores_exact':True,'max_original_cosine_reconstruction_difference':maximum_cos,
        'max_layer_score_reconstruction_differences':maximum_layer.tolist(),'layer_reconstruction_atol':2e-5,
        'no_independent_confirmation_performed':True,'scores_sha256':digest(out/'history-scores.jsonl')})
    print('OC23_ALL_REPLAYS_PASS',results,'COSINE',maximum_cos,'LAYER',maximum_layer.tolist())
if __name__=='__main__':main()
