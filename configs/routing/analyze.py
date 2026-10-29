"""Archive-portable fixed contrasts; no fitting or model choice here."""
import numpy as np
from routing_core import conservative_threshold,p_auc,bootstrap_metrics
from evaluate_frozen import metrics,wilson

def contrasts(selection):
    names=lambda method:[f'fixed_{method}_{seed}' for seed in range(20261201,20261206)]
    return {
        'suffix_minus1':(['fixed_suffix_minus1'],['fixed_template']),
        'suffix_minus2':(['fixed_suffix_minus2'],['fixed_template']),
        'selected_fixed':(['fixed_'+selection['selected_label']],['fixed_template']),
        'selected_variable':(['variable_'+selection['selected_label']],['variable_template']),
        'ga_vs_random':(names('ga'),names('random')),
        'ga_vs_scan':(names('ga'),names('scan')),
        'de_vs_random':(names('de'),names('random')),
        'bpso_vs_random':(names('bpso'),names('random'))}

def analyze(rows,f):
    candidates=f['candidates'];thresholds={};tests={};family=contrasts(f['selection'])
    for policy,role in [('global','calibration_aegis'),('source','calibration_toxic')]:
        safe=[r for r in rows if r['role']==role and r['label']==0];assert len(safe)==400
        thresholds[policy]={c['name']:conservative_threshold(np.array([r['scores'][c['name']] for r in safe]),.05) for c in candidates}
    result={'thresholds':thresholds,'tests':tests,'contrasts':{},'selection':f['selection'],
        'bootstrap':{'iterations':5000,'seed':20261209,'paired':True,'stratified':True,'conditional_on':'fixed heads and calibration thresholds','family':8,'two_sided_quantiles':[.05/16,1-.05/16]},
        'scope':'v1 previously observed benchmark; confirmation fresh same-source, not unseen domain; no test selection'}
    for role in ['test_aegis_holdout','test_aegis_official','test_toxic_holdout','test_toxicchat','confirmation_aegis']:
        cohort=[r for r in rows if r['role']==role];y=np.array([r['label'] for r in cohort]);g0=np.flatnonzero(y==0);g1=np.flatnonzero(y==1)
        rng=np.random.default_rng(20261209);b0=rng.choice(g0,(5000,len(g0)),replace=True);b1=rng.choice(g1,(5000,len(g1)),replace=True)
        policies=['global','source'] if role.startswith('test_toxic') else ['global'];boots={};memo={}
        for c in candidates:
            n=c['name'];s=np.array([r['scores'][n] for r in cohort]);signature=s.tobytes()
            if signature not in memo:memo[signature]=bootstrap_metrics(y,s,b0,b1)
            rank=memo[signature]
            for policy in policies:
                t=thresholds[policy][n];pred=s>t;dist={**rank,'tpr':pred[b1].mean(1),'fpr':pred[b0].mean(1)}
                v={**metrics(y,s,t),'partial_auroc':p_auc(y,s)};v['ci95']={k:np.quantile(a,[.025,.975]).tolist() for k,a in dist.items()}
                v['wilson_ci95']={'tpr':wilson(v['tp'],v['tp']+v['fn']),'fpr':wilson(v['fp'],v['fp']+v['tn'])}
                condition=role+':'+policy;tests.setdefault(condition,{})[n]=v;boots[(policy,n)]=dist
        for policy in policies:
            condition=role+':'+policy
            for c in candidates:
                n=c['name'];base=c['mode']+'_template';d=boots[(policy,n)];b=boots[(policy,base)]
                tests[condition][n]['paired_delta_vs_matched_template_ci95']={k:np.quantile(d[k]-b[k],[.025,.975]).tolist() for k in ('auroc','partial_auroc','tpr','fpr')}
            result['contrasts'][condition]={}
            for name,(left,right) in family.items():
                v={}
                for metric in ('auroc','partial_auroc','tpr','fpr'):
                    delta=np.mean([boots[(policy,n)][metric] for n in left],axis=0)-np.mean([boots[(policy,n)][metric] for n in right],axis=0)
                    observed=float(np.mean([tests[condition][n][metric] for n in left])-np.mean([tests[condition][n][metric] for n in right]))
                    v[metric]={'delta':observed,'ci95':np.quantile(delta,[.025,.975]).tolist(),'family8_ci':np.quantile(delta,[.05/16,1-.05/16]).tolist()}
                result['contrasts'][condition][name]=v
        print('ANALYZED',role,'distinct_scores',len(memo),flush=True)
    return result
