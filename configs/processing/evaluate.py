"""Frozen scoring only; no refitting, tuning, or selection."""
import json,time,joblib
import numpy as np
from processing_core import ROOT,digest,write,verify_freeze,arrays,baseline,feature_matrix,predict_joint,read_jsonl
from analyze import analyze

def main():
    out=ROOT/'results/processing';assert not (out/'evaluation.json').exists();began=time.monotonic();f=verify_freeze()
    a=arrays();rows=a[0];indices=np.array([i for i,r in enumerate(rows) if r['role'] not in ('train','dev')]);values={c['name']:np.empty(len(indices)) for c in f['candidates']};heads={c['name']:joblib.load(out/c['file']) for c in f['candidates']}
    for start in range(0,len(indices),128):
        ix=indices[start:start+128];b=baseline(a,ix);raw=np.asarray(a[2][ix],np.float64);s=np.asarray(a[1][ix]);family_cache={}
        for c in f['candidates']:
            family=c['family']
            if family not in family_cache:family_cache[family]=np.zeros((len(ix),0)) if family=='B' else feature_matrix(s,family)
            values[c['name']][start:start+len(ix)]=predict_joint(heads[c['name']],b,family_cache[family],raw)
        print('SCORED',min(start+128,len(indices)),'/',len(indices),flush=True)
    old={r['id']:r for r in read_jsonl(ROOT/'results/routing/heldout-scores.jsonl')};maxdelta=0.;duplicate=0.;path=out/'heldout-scores.jsonl'
    with path.open('w',encoding='utf8',newline='\n') as stream:
        for j,i in enumerate(indices):
            row=rows[i];v={n:float(s[j]) for n,s in values.items()};assert np.isfinite(list(v.values())).all()
            if row['id'] in old:maxdelta=max(maxdelta,abs(v['B']-old[row['id']]['scores']['fixed_template']))
            for g in ('0p25','1p0'):duplicate=max(duplicate,abs(v['mean_g'+g]-v['duplicate_mean_g'+g]))
            stream.write(json.dumps({'id':row['id'],'group_id':row['group_id'],'role':row['role'],'label':row['label'],'scores':v})+'\n')
    assert maxdelta<1e-7 and duplicate<1e-7,(maxdelta,duplicate)
    result=analyze(read_jsonl(path),f);result.update(freeze_sha256=digest(out/'freeze.json'),scores_sha256=digest(path),maximum_old_B_score_difference=maxdelta,maximum_duplicate_mean_score_difference=duplicate,seconds=time.monotonic()-began)
    write(out/'evaluation.json',result);print('PROCESSING_EVALUATION_COMPLETE',result['seconds'],flush=True)

if __name__=='__main__':main()
