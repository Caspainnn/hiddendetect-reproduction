"""Supplemental joint calibration/test bootstrap; never replaces fixed point thresholds."""
import json,math,hashlib
from pathlib import Path
import numpy as np

def draw_policy(cal,score,b0,b1,draws):
    index=math.ceil(.95*len(cal))-1;threshold=np.sort(cal[draws],axis=1)[:,index]
    return {'threshold':threshold,'tpr':(score[b1]>threshold[:,None]).mean(1),'fpr':(score[b0]>threshold[:,None]).mean(1)}

def main():
    root=Path(__file__).resolve().parents[2];out=root/'results/swarm';path=out/'heldout-scores.jsonl'
    scores=[json.loads(line) for line in path.read_text().splitlines() if line.strip()];evaluation=json.loads((out/'evaluation.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==evaluation['scores_sha256']
    names=list(scores[0]['scores']);result={'iterations':1000,'seed':20261102,
        'scope':'supplemental analysis specified before new scoring; resamples safe calibration and test groups, conditional on fitted candidates; no threshold retuning or guarantee',
        'scores_sha256':evaluation['scores_sha256'],'conditions':{}}
    for role,policy in [('test_aegis_holdout','global'),('confirmation_aegis','global'),('test_toxic_holdout','global'),('test_toxic_holdout','source')]:
        calrole='calibration_aegis' if policy=='global' else 'calibration_toxic'
        cal=[x for x in scores if x['role']==calrole and x['label']==0];test=[x for x in scores if x['role']==role]
        assert len(cal)==400
        y=np.array([x['label'] for x in test]);g0=np.flatnonzero(y==0);g1=np.flatnonzero(y==1);rng=np.random.default_rng(20261102)
        draws=rng.choice(len(cal),size=(1000,len(cal)),replace=True)
        b0=rng.choice(g0,size=(1000,len(g0)),replace=True);b1=rng.choice(g1,size=(1000,len(g1)),replace=True)
        distributions={n:draw_policy(np.array([x['scores'][n] for x in cal]),np.array([x['scores'][n] for x in test]),b0,b1,draws) for n in names}
        record={}
        for name,d in distributions.items():
            record[name]={'joint_ci95':{k:np.quantile(v,[.025,.975]).tolist() for k,v in d.items()},
                'paired_delta_vs_B_ci95':{k:np.quantile(d[k]-distributions['expert_B'][k],[.025,.975]).tolist() for k in ('tpr','fpr')}}
        for method in ('random','bpso','ga','de'):
            mean={k:np.mean([distributions[method+'_seed'+str(i)][k] for i in range(5)],axis=0) for k in ('tpr','fpr')}
            record[method+'_mean5']={'joint_ci95':{k:np.quantile(v,[.025,.975]).tolist() for k,v in mean.items()},
                'paired_delta_vs_B_ci95':{k:np.quantile(mean[k]-distributions['expert_B'][k],[.025,.975]).tolist() for k in mean}}
        result['conditions'][role+':'+policy]=record
    (out/'calibration-sensitivity.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8',newline='\n');print('JOINT_CALIBRATION_TEST_SENSITIVITY_COMPLETE')

if __name__=='__main__':main()
