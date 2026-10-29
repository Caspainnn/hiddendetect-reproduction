"""Fixed five-contrast family, per-metric multiplicity correction."""
import numpy as np
from head_core import p_auc,bootstrap_metrics,conservative_threshold
from evaluate_frozen import metrics,wilson

def contrasts(selection):
    return {'ridge_tuning':('ridge_tuned','B_original'),'logistic_vs_ridge':('logistic','ridge_tuned'),'svm_vs_ridge':('svm','ridge_tuned'),'rbf_vs_ridge':('rbf','ridge_tuned'),'selected_vs_ridge':(selection['selected_head'],'ridge_tuned')}

def analyze(rows,f):
    names=[c['name'] for c in f['candidates']];thresholds={};tests={}
    for policy,role in [('global','calibration_aegis'),('source','calibration_toxic')]:
        safe=[r for r in rows if r['role']==role and r['label']==0];assert len(safe)==400
        thresholds[policy]={n:conservative_threshold(np.array([r['scores'][n] for r in safe]),.05) for n in names}
    result={'thresholds':thresholds,'tests':tests,'contrasts':{},'selection':f['selection'],'bootstrap':{'iterations':5000,'seed':20270109,'family':5,'quantiles':[.005,.995],'conditional_on':'fixed training/selection/calibration; correction separately within each metric'},'scope':'fixed observed benchmark and fresh same-source confirmation, no pooled task average'}
    for role in ('test_aegis_holdout','test_aegis_official','test_toxic_holdout','test_toxicchat','confirmation_aegis'):
        cohort=[r for r in rows if r['role']==role];y=np.array([r['label'] for r in cohort]);g0=np.flatnonzero(y==0);g1=np.flatnonzero(y==1);rng=np.random.default_rng(20270109)
        b0=rng.choice(g0,(5000,len(g0)),replace=True);b1=rng.choice(g1,(5000,len(g1)),replace=True);boot={};policies=['global','source'] if role.startswith('test_toxic') else ['global']
        for n in names:
            s=np.array([r['scores'][n] for r in cohort]);rank=bootstrap_metrics(y,s,b0,b1)
            for policy in policies:
                t=thresholds[policy][n];pred=s>t;dist={**rank,'tpr':pred[b1].mean(1),'fpr':pred[b0].mean(1)};v={**metrics(y,s,t),'partial_auroc':p_auc(y,s)}
                v['ci95']={k:np.quantile(d,[.025,.975]).tolist() for k,d in dist.items()};v['wilson_ci95']={'tpr':wilson(v['tp'],v['tp']+v['fn']),'fpr':wilson(v['fp'],v['fp']+v['tn'])}
                tests.setdefault(role+':'+policy,{})[n]=v;boot[(policy,n)]=dist
        for policy in policies:
            condition=role+':'+policy;result['contrasts'][condition]={}
            for n in names:tests[condition][n]['paired_delta_vs_tuned_ridge_ci95']={k:np.quantile(boot[(policy,n)][k]-boot[(policy,'ridge_tuned')][k],[.025,.975]).tolist() for k in ('auroc','partial_auroc','tpr','fpr')}
            for label,(left,right) in contrasts(f['selection']).items():
                result['contrasts'][condition][label]={}
                for metric in ('auroc','partial_auroc','tpr','fpr'):
                    d=boot[(policy,left)][metric]-boot[(policy,right)][metric];result['contrasts'][condition][label][metric]={'delta':tests[condition][left][metric]-tests[condition][right][metric],'ci95':np.quantile(d,[.025,.975]).tolist(),'family5_ci':np.quantile(d,[.005,.995]).tolist()}
        print('ANALYZED',role,flush=True)
    return result
