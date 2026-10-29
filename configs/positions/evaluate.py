"""Fixed-position models only; no selection or refitting after benchmark access."""
import json,time,joblib
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from position_core import ROOT,digest,verify_freeze,matrix,ridge_scores,p_auc,bootstrap_metrics,upper_bounds,conservative_threshold,read_jsonl
from evaluate_frozen import metrics as standard_metrics,wilson

def main():
    out=ROOT/'results/positions';target=out/'evaluation.json'
    assert not target.exists(),'Already evaluated'
    began=time.monotonic();f=verify_freeze(out,actual=True);rows=json.loads((ROOT/'configs/positions/manifest.json').read_text())['samples']
    data=np.load(ROOT/'cache/positions/P.npy',mmap_mode='r')
    roles=['calibration_aegis','calibration_toxic','test_aegis_holdout','test_aegis_official','test_toxic_holdout','test_toxicchat','historical_confirmation','confirmation_aegis']
    indices={role:np.array([i for i,x in enumerate(rows) if x['role']==role]) for role in roles};scores={}
    for role,ix in indices.items():
        scores[role]={}
        for c in f['candidates']:
            value=ridge_scores(joblib.load(out/c['file']),matrix(data,ix,c['positions']));assert np.isfinite(value).all()
            scores[role][c['name']]=value
        print('SCORED',role,len(ix),flush=True)
    thresholds={}
    for policy,role in [('global','calibration_aegis'),('source','calibration_toxic')]:
        y=np.array([rows[i]['label'] for i in indices[role]]);assert (y==0).sum()==400
        thresholds[policy]={c['name']:conservative_threshold(scores[role][c['name']][y==0],.05) for c in f['candidates']}
    previous={x['id']:x for x in read_jsonl(ROOT/'results/swarm/heldout-scores.jsonl')};maximum_old=0.;maximum_duplicate=0.
    path=out/'heldout-scores.jsonl'
    with path.open('w',encoding='utf8',newline='\n') as stream:
        for role,ix in indices.items():
            for j,i in enumerate(ix):
                x=rows[i];v={name:float(value[j]) for name,value in scores[role].items()}
                if x['id'] in previous:maximum_old=max(maximum_old,abs(v['template_last']-previous[x['id']]['scores']['expert_B']))
                maximum_duplicate=max(maximum_duplicate,abs(v['template_last']-v['template_duplicated']))
                stream.write(json.dumps({'id':x['id'],'group_id':x['group_id'],'role':role,'label':x['label'],'scores':v})+'\n')
    assert maximum_old<=1e-4 and maximum_duplicate<=1e-4,(maximum_old,maximum_duplicate)
    r={'freeze_sha256':digest(out/'freeze.json'),'manifest_sha256':f['manifest_sha256'],'code_sha256':digest(__file__),
        'thresholds':thresholds,'tests':{},'restricted_optimality':{},'selection':f['selection'],
        'bootstrap':{'iterations':5000,'seed':20261119,'paired':True,'stratified':True,'conditional_on':'fixed trained heads and safe-calibration thresholds'},
        'maximum_old_B_score_difference':maximum_old,'maximum_duplicate_score_difference':maximum_duplicate,
        'scope':'fixed previously observed benchmark plus historical supplement; fresh same-source confirmation separate; no all-position optimum claim'}
    alternatives=[c['name'] for c in f['candidates'] if c['real_alternative']];assert len(alternatives)==12
    for role in roles[2:]:
        ix=indices[role];y=np.array([rows[i]['label'] for i in ix]);rng=np.random.default_rng(20261119)
        g0=np.flatnonzero(y==0);g1=np.flatnonzero(y==1)
        b0=rng.choice(g0,size=(5000,len(g0)),replace=True);b1=rng.choice(g1,size=(5000,len(g1)),replace=True)
        policies=['global','source'] if role.startswith('test_toxic') else ['global'];boot={};observed={}
        for c in f['candidates']:
            name=c['name'];s=scores[role][name];rank=bootstrap_metrics(y,s,b0,b1)
            for policy in policies:
                t=thresholds[policy][name];pred=s>t;distribution={**rank,'tpr':pred[b1].mean(1),'fpr':pred[b0].mean(1)}
                v={**standard_metrics(y,s,t),'partial_auroc':p_auc(y,s)}
                v['ci95']={k:np.quantile(a,[.025,.975]).tolist() for k,a in distribution.items()}
                v['wilson_ci95']={'tpr':wilson(v['tp'],v['tp']+v['fn']),'fpr':wilson(v['fp'],v['fp']+v['tn'])}
                observed[(policy,name)]=v;boot[(policy,name)]=distribution
        for policy in policies:
            condition=role+':'+policy;r['tests'][condition]={}
            for c in f['candidates']:
                name=c['name'];v=observed[(policy,name)]
                v['paired_delta_vs_template_ci95']={k:np.quantile(boot[(policy,name)][k]-boot[(policy,'template_last')][k],[.025,.975]).tolist() for k in ('auroc','partial_auroc','tpr','fpr')}
                r['tests'][condition][name]=v
            r['restricted_optimality'][condition]=upper_bounds({n:boot[(policy,n)]['auroc']-boot[(policy,'template_last')]['auroc'] for n in alternatives})
        print('TEST_COMPLETE',role,flush=True)
    r['scores_sha256']=digest(path);r['seconds']=time.monotonic()-began;target.write_text(json.dumps(r,indent=2)+'\n')
    print('POSITION_EVALUATION_COMPLETE',r['seconds'],flush=True)

if __name__=='__main__':main()
