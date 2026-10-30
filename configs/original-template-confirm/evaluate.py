import json,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'configs/original-positions'))
from core import digest,read_jsonl,paired_intervals

def main():
    directory=ROOT/'results/original-template-confirm';mp=ROOT/'configs/original-template-confirm/manifest.json';m=json.loads(mp.read_text())
    ex=json.loads((directory/'extraction.json').read_text());lock=json.loads((ROOT/'configs/original-template-confirm/run-lock.json').read_text())
    assert ex['scores_sha256']==digest(directory/'scores.jsonl') and ex['manifest_sha256']==digest(mp)
    for name,sha in lock['sha256'].items():assert digest(ROOT/name)==sha,name
    identity=json.loads((ROOT/'logs/original-template-confirm-identity.json').read_text())
    assert identity['status']=='passed' and identity['manifest_sha256']==digest(mp) and identity['maximum_F_difference']==0
    rows=read_jsonl(directory/'scores.jsonl');assert [x['id'] for x in rows]==[x['id'] for x in m['samples']]
    assert len(rows)==1000 and len({x['group_id'] for x in rows})==1000
    prior=json.loads((ROOT/'results/original-template/evaluation.json').read_text());y=np.array([x['label'] for x in rows]);assert (y==0).sum()==500
    output={'run_id':'ORP-T2','scope':'unused same-source confirmation at selection; historical after scoring','manifest_sha256':digest(mp),
            'scores_sha256':digest(directory/'scores.jsonl'),'thresholds_frozen_from_sha256':digest(ROOT/'results/original-template/evaluation.json'),
            'bootstrap_repeats':5000,'bootstrap_seed':20261030,'family':'one frozen candidate AUROC comparison; sensitivity descriptive','analyses':{}}
    for name,interval in [('primary',[16,29]),('sensitivity',[17,29])]:
        s=np.array([[float(np.trapz(np.asarray(x['F'])[interval[0]:interval[1]+1,j])) for j in range(2)] for x in rows])
        assert np.array_equal(s,np.array([x['score_default' if name=='primary' else 'score_fewshot'] for x in rows]))
        old={x['direction_id']:x for x in prior['analyses'][name]['metrics']};ids=['EI-3-2-2','EI-3-2-1'];cut=[old[k]['threshold'] for k in ids]
        paired=paired_intervals(y,s,family=1,thresholds=cut);paired[0]['direction_id']=ids[0]
        metrics=[]
        for j,k in enumerate(ids):
            pred=s[:,j]>cut[j];metrics.append({'direction_id':k,'record_type':'candidate' if j==0 else 'control','baseline_auroc':float(roc_auc_score(y,s[:,-1])),
                'auroc':float(roc_auc_score(y,s[:,j])),'standardized_pauc_fpr0_1':float(roc_auc_score(y,s[:,j],max_fpr=.1)),
                'threshold':cut[j],'tpr':float(pred[y==1].mean()),'fpr':float(pred[y==0].mean())})
        output['analyses'][name]={'interval':interval,'metrics':metrics,'paired_auroc':paired}
    (directory/'evaluation.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output['analyses']['primary'],indent=2))

if __name__=='__main__':main()
