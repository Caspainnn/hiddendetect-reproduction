"""Seven preregistered paired contrasts per task and per metric."""
import numpy as np
from anchored_core import contrasts,bootstrap_metrics,conservative_threshold,p_auc
from evaluate_frozen import metrics,wilson

def analyze(rows,f):
    names=[c['name'] for c in f['models']];out={'tests':{},'thresholds':{},'contrasts':{},'selection':f['selection'],'bootstrap':{'iterations':5000,'seed':20270311,'family':7,'quantiles':[.05/14,1-.05/14],'scope':'paired stratified test inputs, conditional on frozen base/meta/refit training and calibration; separate metrics and tasks'}}
    for policy,role in [('global','calibration_aegis'),('source','calibration_toxic')]:
        safe=[r for r in rows if r['role']==role and r['label']==0];assert len(safe)==400
        out['thresholds'][policy]={n:conservative_threshold(np.array([r['scores'][n] for r in safe]),.05) for n in names}
    for role in ('test_aegis_holdout','test_toxic_holdout','confirmation_aegis','confirmation_toxic'):
        cohort=[r for r in rows if r['role']==role];y=np.array([r['label'] for r in cohort]);g0=np.flatnonzero(y==0);g1=np.flatnonzero(y==1);rng=np.random.default_rng(20270311)
        assert len(g0)>0 and len(g1)>0;b0=rng.choice(g0,(5000,len(g0)),replace=True);b1=rng.choice(g1,(5000,len(g1)),replace=True);policies=['global','source'] if 'toxic' in role else ['global'];boots={}
        for n in names:
            s=np.array([r['scores'][n] for r in cohort]);rank=bootstrap_metrics(y,s,b0,b1)
            for policy in policies:
                t=out['thresholds'][policy][n];pred=s>t;dist={k:rank[k] for k in ('auroc','partial_auroc')};dist.update(tpr=pred[b1].mean(1),fpr=pred[b0].mean(1));boots[(policy,n)]=dist
                v={**metrics(y,s,t),'partial_auroc':p_auc(y,s),'changed_fraction_vs_B':float(np.mean(s!=np.array([r['scores']['B'] for r in cohort])))}
                v['ci95']={k:np.quantile(z,[.025,.975]).tolist() for k,z in dist.items()};v['wilson_ci95']={'tpr':wilson(v['tp'],v['tp']+v['fn']),'fpr':wilson(v['fp'],v['fp']+v['tn'])}
                out['tests'].setdefault(role+':'+policy,{})[n]=v
        for policy in policies:
            key=role+':'+policy;out['contrasts'][key]={}
            for label,(a,b) in contrasts().items():
                out['contrasts'][key][label]={}
                for metric in ('auroc','partial_auroc','tpr','fpr'):
                    d=boots[(policy,a)][metric]-boots[(policy,b)][metric]
                    out['contrasts'][key][label][metric]={'delta':out['tests'][key][a][metric]-out['tests'][key][b][metric],'ci95':np.quantile(d,[.025,.975]).tolist(),'family7_ci':np.quantile(d,[.05/14,1-.05/14]).tolist()}
        print('ANALYZED',role,flush=True)
    return out
