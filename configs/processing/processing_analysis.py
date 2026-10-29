"""Ten frozen contrasts, same bootstrap and calibration policy as prior rounds."""
import numpy as np
from processing_core import bootstrap_metrics,conservative_threshold,p_auc
from evaluate_frozen import metrics,wilson

def contrasts(selection):
    return {'mean_vs_B':('mean_g1p0','B'),'dynamic_vs_B':('dynamic_g1p0','B'),
        'dynamic_vs_mean':('dynamic_g1p0','mean_g1p0'),'dynamic_vs_random':('dynamic_g1p0','random_g1p0'),
        'processed_vs_B':('processed_g1p0','B'),'processed_vs_dynamic':('processed_g1p0','dynamic_g1p0'),
        'relations_vs_B':('relations_g1p0','B'),'relations_vs_processed':('relations_g1p0','processed_g1p0'),
        'selected_vs_B':(selection['selected'],'B'),'duplicate_vs_mean':('duplicate_mean_g1p0','mean_g1p0')}

def analyze(rows,f):
    names=[c['name'] for c in f['candidates']];thresholds={};tests={};contr=contrasts(f['selection'])
    for policy,role in [('global','calibration_aegis'),('source','calibration_toxic')]:
        safe=[r for r in rows if r['role']==role and r['label']==0];assert len(safe)==400
        thresholds[policy]={n:conservative_threshold(np.array([r['scores'][n] for r in safe]),.05) for n in names}
    result={'thresholds':thresholds,'tests':tests,'contrasts':{},'selection':f['selection'],'bootstrap':{'iterations':5000,'seed':20261229,'family':10,'quantiles':[.05/20,1-.05/20],'conditional_on':'fixed training and calibration fits; per-metric family correction'},'scope':'v1 observed comparison; fresh same-source confirmation; no test selection'}
    for role in ('test_aegis_holdout','test_aegis_official','test_toxic_holdout','test_toxicchat','confirmation_aegis'):
        cohort=[r for r in rows if r['role']==role];y=np.array([r['label'] for r in cohort]);g0=np.flatnonzero(y==0);g1=np.flatnonzero(y==1);rng=np.random.default_rng(20261229)
        b0=rng.choice(g0,(5000,len(g0)),replace=True);b1=rng.choice(g1,(5000,len(g1)),replace=True);boots={}
        policies=['global','source'] if role.startswith('test_toxic') else ['global']
        for n in names:
            s=np.array([r['scores'][n] for r in cohort]);rank=bootstrap_metrics(y,s,b0,b1)
            for policy in policies:
                t=thresholds[policy][n];pred=s>t;dist={**rank,'tpr':pred[b1].mean(1),'fpr':pred[b0].mean(1)}
                v={**metrics(y,s,t),'partial_auroc':p_auc(y,s)};v['ci95']={k:np.quantile(a,[.025,.975]).tolist() for k,a in dist.items()}
                v['wilson_ci95']={'tpr':wilson(v['tp'],v['tp']+v['fn']),'fpr':wilson(v['fp'],v['fp']+v['tn'])}
                tests.setdefault(role+':'+policy,{})[n]=v;boots[(policy,n)]=dist
        for policy in policies:
            condition=role+':'+policy;result['contrasts'][condition]={}
            for n in names:
                tests[condition][n]['paired_delta_vs_B_ci95']={k:np.quantile(boots[(policy,n)][k]-boots[(policy,'B')][k],[.025,.975]).tolist() for k in ('auroc','partial_auroc','tpr','fpr')}
            for label,(left,right) in contr.items():
                v={}
                for metric in ('auroc','partial_auroc','tpr','fpr'):
                    delta=boots[(policy,left)][metric]-boots[(policy,right)][metric]
                    v[metric]={'delta':tests[condition][left][metric]-tests[condition][right][metric],'ci95':np.quantile(delta,[.025,.975]).tolist(),'family10_ci':np.quantile(delta,[.05/20,1-.05/20]).tolist()}
                result['contrasts'][condition][label]=v
        print('ANALYZED',role,flush=True)
    return result
