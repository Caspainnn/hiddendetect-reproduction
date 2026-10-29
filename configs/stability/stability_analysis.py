"""Paired test resampling, averaged across fixed overlapping training draws."""
import numpy as np
from stability_core import variants,contrasts,bootstrap_metrics,conservative_threshold,p_auc
from evaluate_frozen import metrics

def analyze(rows):
    names=[str(i)+'_'+v['name'] for i in range(5) for v in variants()];result={'tests':{},'aggregate':{},'contrasts':{},'thresholds':{},'bootstrap':{'iterations':5000,'seed':20270217,'family':6,'quantiles':[.05/12,1-.05/12],'scope':'paired stratified test IDs, mean metric differences across five fixed overlapping training draws; conditional on training and calibration'}}
    for policy,role in [('global','calibration_aegis'),('source','calibration_toxic')]:
        safe=[r for r in rows if r['role']==role and r['label']==0];assert len(safe)==400
        result['thresholds'][policy]={n:conservative_threshold(np.array([r['scores'][n] for r in safe]),.05) for n in names}
    for role in ('test_aegis_holdout','test_toxic_holdout','confirmation_aegis'):
        cohort=[r for r in rows if r['role']==role];y=np.array([r['label'] for r in cohort]);g0=np.flatnonzero(y==0);g1=np.flatnonzero(y==1);rng=np.random.default_rng(20270217)
        b0=rng.choice(g0,(5000,len(g0)),replace=True);b1=rng.choice(g1,(5000,len(g1)),replace=True)
        policies=['global','source'] if role=='test_toxic_holdout' else ['global'];boots={}
        for n in names:
            s=np.array([r['scores'][n] for r in cohort]);rank=bootstrap_metrics(y,s,b0,b1)
            for policy in policies:
                t=result['thresholds'][policy][n];pred=s>t;dist={k:rank[k] for k in ('auroc','partial_auroc')};dist.update(tpr=pred[b1].mean(1),fpr=pred[b0].mean(1));boots[(policy,n)]=dist
                result['tests'].setdefault(role+':'+policy,{})[n]={**metrics(y,s,t),'partial_auroc':p_auc(y,s)}
        for policy in policies:
            key=role+':'+policy;points=result['tests'][key];agg={};ab={}
            for v in variants():
                label=v['name'];ns=[str(i)+'_'+label for i in range(5)]
                agg[label]={k:float(np.mean([points[n][k] for n in ns])) for k in ('auroc','partial_auroc','tpr','fpr')}
                agg[label]['draw_range']={k:[float(min(points[n][k] for n in ns)),float(max(points[n][k] for n in ns))] for k in ('auroc','partial_auroc','tpr','fpr')}
                ab[label]={k:np.mean([boots[(policy,n)][k] for n in ns],axis=0) for k in ('auroc','partial_auroc','tpr','fpr')}
            result['aggregate'][key]=agg;result['contrasts'][key]={}
            for label,(left,right) in contrasts().items():
                result['contrasts'][key][label]={}
                for metric in ('auroc','partial_auroc','tpr','fpr'):
                    d=ab[left][metric]-ab[right][metric];each=[points[str(i)+'_'+left][metric]-points[str(i)+'_'+right][metric] for i in range(5)]
                    result['contrasts'][key][label][metric]={'delta':agg[left][metric]-agg[right][metric],'draw_deltas':each,'positive_draws':sum(x>0 for x in each),'ci95':np.quantile(d,[.025,.975]).tolist(),'family6_ci':np.quantile(d,[.05/12,1-.05/12]).tolist()}
        print('ANALYZED',role,flush=True)
    return result
