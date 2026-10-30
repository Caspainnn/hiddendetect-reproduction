"""Paired historical comparison; thresholds from independent safe calibration only."""
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from core import ROOT,IDS,NAMES,digest,read_jsonl,integral,threshold,paired_intervals

def main():
    directory=ROOT/'results/original-positions';mp=ROOT/'configs/original-positions/manifest.json'
    manifest=json.loads(mp.read_text());extraction=json.loads((directory/'extraction.json').read_text())
    assert extraction['manifest_sha256']==digest(mp) and extraction['scores_sha256']==digest(directory/'scores.jsonl')
    for name,sha in manifest['source_sha256'].items():assert digest(ROOT/name)==sha,name
    identity=json.loads((ROOT/'logs/original-positions-identity.json').read_text())
    assert identity['status']=='passed' and identity['maximum_F_difference']==0 and identity['manifest_sha256']==digest(mp)
    rows=read_jsonl(directory/'scores.jsonl')
    assert [x['id'] for x in rows]==[x['id'] for x in manifest['samples']]
    output={'run_id':'ORP-U1','scope':'historical benchmark v1 C; not unseen confirmation','manifest_sha256':digest(mp),
            'scores_sha256':digest(directory/'scores.jsonl'),'bootstrap_repeats':5000,'bootstrap_seed':20261030,
            'family':'six user-position AUROC comparisons to EI-3-2-1, per primary interval',
            'threshold_uncertainty':'conditional on fixed calibration; no population FPR guarantee','analyses':{}}
    for name,interval in [('primary',[16,29]),('sensitivity',[17,29])]:
        scores=np.asarray([integral(x['F'],*interval) for x in rows])
        stored=np.asarray([x['score_default' if name=='primary' else 'score_fewshot'] for x in rows])
        assert np.array_equal(scores,stored)
        cal=np.array([x['role']=='calibration_aegis' for x in rows]);test=np.array([x['role']=='test_aegis_holdout' for x in rows])
        assert cal.sum()==400 and test.sum()==1000 and all(x['label']==0 for x in np.asarray(rows,dtype=object)[cal])
        y=np.array([x['label'] for x in rows])[test];s=scores[test];cut=[threshold(scores[cal,j]) for j in range(7)]
        metrics=[]
        for j,(identifier,label) in enumerate(zip(IDS,NAMES)):
            predicted=s[:,j]>cut[j]
            metrics.append({'direction_id':identifier,'name':label,'record_type':'control' if j==6 else 'candidate',
                            'baseline_auroc':float(roc_auc_score(y,s[:,-1])),'auroc':float(roc_auc_score(y,s[:,j])),
                            'standardized_pauc_fpr0_1':float(roc_auc_score(y,s[:,j],max_fpr=.1)),
                            'threshold':cut[j],'tpr':float(predicted[y==1].mean()),'fpr':float(predicted[y==0].mean())})
        output['analyses'][name]={'interval':interval,'metrics':metrics,'paired_auroc':paired_intervals(y,s,thresholds=cut),
                                 'interpretation':'Sensitivity is descriptive; no interval chosen by test performance.'}
    (directory/'evaluation.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output['analyses']['primary'],indent=2))

if __name__=='__main__':main()
