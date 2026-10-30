"""Two template neighbors, new prespecified family, original-score framework."""
import json,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'configs/original-positions'))
from core import digest,read_jsonl,threshold,paired_intervals
IDS=['EI-3-2-2','EI-3-2-3','EI-3-2-1']

def main():
    directory=ROOT/'results/original-template';mp=ROOT/'configs/original-template/manifest.json'
    manifest=json.loads(mp.read_text());ex=json.loads((directory/'extraction.json').read_text())
    assert ex['manifest_sha256']==digest(mp) and ex['scores_sha256']==digest(directory/'scores.jsonl')
    assert ex['maximum_same_batch_baseline_F_difference']==0 and ex['both_integral_baselines_exact']
    for name,sha in manifest['source_sha256'].items():assert digest(ROOT/name)==sha,name
    rows=read_jsonl(directory/'scores.jsonl');assert [x['id'] for x in rows]==[x['id'] for x in manifest['samples']]
    output={'run_id':'ORP-T1','scope':'historical benchmark C; no unseen confirmation','manifest_sha256':digest(mp),
            'scores_sha256':digest(directory/'scores.jsonl'),'bootstrap_repeats':5000,'bootstrap_seed':20261030,
            'family':'two template-neighbor AUROC differences; Bonferroni family2; sensitivity descriptive',
            'threshold_uncertainty':'conditional on fixed calibration; no population FPR guarantee','analyses':{}}
    for name,interval in [('primary',[16,29]),('sensitivity',[17,29])]:
        score=np.array([[float(np.trapz(np.asarray(x['F'])[interval[0]:interval[1]+1,j])) for j in range(3)] for x in rows])
        assert np.array_equal(score,np.array([x['score_default' if name=='primary' else 'score_fewshot'] for x in rows]))
        cal=np.array([x['role']=='calibration_aegis' for x in rows]);test=np.array([x['role']=='test_aegis_holdout' for x in rows])
        assert cal.sum()==400 and test.sum()==1000 and all(x['label']==0 for x in np.asarray(rows,dtype=object)[cal])
        y=np.array([x['label'] for x in rows])[test];s=score[test];cuts=[threshold(score[cal,j]) for j in range(3)]
        paired=paired_intervals(y,s,family=2,thresholds=cuts)
        for j,row in enumerate(paired):row['direction_id']=IDS[j]
        metrics=[]
        for j,identifier in enumerate(IDS):
            pred=s[:,j]>cuts[j]
            metrics.append({'direction_id':identifier,'record_type':'control' if j==2 else 'candidate','baseline_auroc':float(roc_auc_score(y,s[:,-1])),
                 'auroc':float(roc_auc_score(y,s[:,j])),'standardized_pauc_fpr0_1':float(roc_auc_score(y,s[:,j],max_fpr=.1)),
                 'threshold':cuts[j],'tpr':float(pred[y==1].mean()),'fpr':float(pred[y==0].mean())})
        output['analyses'][name]={'interval':interval,'metrics':metrics,'paired_auroc':paired}
    (directory/'evaluation.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output['analyses']['primary'],indent=2))

if __name__=='__main__':main()
