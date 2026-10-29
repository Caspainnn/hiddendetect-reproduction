"""Fixed public benchmark plus fresh confirmation; candidates never retuned here."""
import json,time,sys,importlib.util,joblib
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from core import ROOT,LAYOUT,digest,verify_freeze,cache as load_cache,standardized,joined,p_auc,bootstrap_metrics
from detectors import columns_for_mask,detector_scores
from provenance import verify_pre_evaluation_seal
from experiment_core import conservative_threshold,confusion
from evaluate_frozen import metrics as original_metrics,wilson
spec=importlib.util.spec_from_file_location('frozen_fusion_evaluation',ROOT/'configs/fusion/evaluate.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)

def metrics(y,s,t):return {**original_metrics(y,s,t),'partial_auroc':p_auc(y,s)}

def all_scores(cache,indices,freeze,out):
    old=json.loads((ROOT/'results/fusion/freeze.json').read_text())
    reference=legacy.all_scores(cache,indices,old,ROOT/'results/fusion')
    z=standardized(cache,indices);groups=json.loads((ROOT/'results/highdim/schema.json').read_text());values={}
    for c in freeze['candidates']:
        name=c['name']
        if c['kind']=='anchored':
            x=joined(reference['expert_B'],z,np.array(c['mask'],bool),groups,freeze['base_mean'],freeze['base_scale'])
            score=joblib.load(out/c['file']).decision_function(x)
        elif c['kind']=='fusion_reference':score=reference[c['reference']]
        else:
            entry=c['reference']
            if entry['kind']=='interval':score=np.trapz(np.asarray(cache['F'][indices,entry['lo']:entry['hi']+1],dtype=np.float64),axis=1)
            elif entry['kind']=='logit':score=detector_scores(joblib.load(ROOT/'results/highdim'/entry['file']),np.asarray(cache['L'][indices]))
            else:
                model=joblib.load(ROOT/'results/highdim'/entry['file']);cols=columns_for_mask(np.array(entry['mask'],bool),groups)
                score=model.decision_function(z[:,cols]) if entry['head']=='linear' else model.predict_proba(z[:,cols])[:,1]
        assert np.isfinite(score).all();values[name]=score
    return values

def method_comparison(observed,boot,policy):
    names=('random','bpso','ga','de');result={}
    for method in names:
        members=[method+'_seed'+str(i) for i in range(5)];result[method]={}
        for metric in ('auroc','partial_auroc','tpr','fpr'):
            values=[observed[(policy,n)][metric] for n in members]
            result[method][metric]={'mean':float(np.mean(values)),'seed_std':float(np.std(values,ddof=1)),'seeds':values}
            for ref in ('random','bpso'):
                deltas=[observed[(policy,n)][metric]-observed[(policy,ref+'_seed'+str(i))][metric] for i,n in enumerate(members)]
                distribution=np.mean([boot[(policy,n)][metric]-boot[(policy,ref+'_seed'+str(i))][metric] for i,n in enumerate(members)],axis=0)
                result[method][metric]['vs_'+ref]={'mean_delta':float(np.mean(deltas)),'positive_seed_pairs':int((np.array(deltas)>0).sum()),
                    'paired_ci95':np.quantile(distribution,[.025,.975]).tolist()}
    return result


def main():
    started=time.monotonic(); out=ROOT/'results/swarm'; target=out/'evaluation.json'
    if target.exists(): print('ALREADY_EVALUATED'); return
    freeze=verify_freeze(out,actual_caches=True); fh=digest(out/'freeze.json')
    manifest=json.loads((ROOT/'configs/fusion/manifest.json').read_text()); rows=manifest['samples']
    cache=load_cache('fusion'); confirmation_cache=load_cache('swarm')
    confirmation=json.loads((ROOT/'configs/swarm/manifest.json').read_text())['samples']
    rows=rows+[{**x,'_cache_index':x['index']} for x in confirmation if x['role']!='probe_train']
    roles=['calibration_aegis','calibration_toxic','test_aegis_holdout','test_aegis_official','test_toxic_holdout','test_toxicchat','confirmation_aegis']
    indices={role:np.array([i for i,x in enumerate(rows) if x['role']==role]) for role in roles}
    scores={role:all_scores(confirmation_cache if role=='confirmation_aegis' else cache,np.array([rows[i].get('_cache_index',i) for i in ix]),freeze,out) for role,ix in indices.items()}
    thresholds={}; calibration={}
    for policy,role in [('global','calibration_aegis'),('source','calibration_toxic')]:
        labels=np.array([rows[i]['label'] for i in indices[role]]); assert (labels==0).sum()==400
        thresholds[policy]={}; calibration[policy]={}
        for c in freeze['candidates']:
            name=c['name']; threshold=conservative_threshold(scores[role][name][labels==0],.05)
            thresholds[policy][name]=threshold; counts=confusion(labels,scores[role][name],threshold)
            calibration[policy][name]={**counts,'fpr':counts['fp']/(counts['fp']+counts['tn'])}
            assert calibration[policy][name]['fpr']<=.05
    path=out/'heldout-scores.jsonl'
    with path.open('w',encoding='utf-8',newline='\n') as stream:
        for role,ix in indices.items():
            for j,i in enumerate(ix): stream.write(json.dumps({'id':rows[i]['id'],'group_id':rows[i]['group_id'],'role':role,'label':rows[i]['label'],
                'scores':{name:float(value[j]) for name,value in scores[role].items()}})+'\n')
    report={'freeze_sha256':fh,'manifest_sha256':freeze['manifest_sha256'],'code_sha256':digest(__file__),
        'method_comparison':{},'thresholds':thresholds,'calibration':calibration,'tests':{},'best_dev_subset':freeze['best_dev_method'],
        'bootstrap':{'iterations':1000,'seed':20261030,'paired':True,'stratified_by_label':True,'unit':'one row per deduplicated prompt group',
            'scope':'conditional on frozen expert/meta training and calibration; no multiplicity adjustment'},
        'primary':'fixed benchmark partialAUROC, paired mean5 GA/DE versus random/BPSO; separate fresh confirmation','policy_scope':'source policy applies only to separate toxicity proxy; rankings unchanged'}
    for role in roles[2:]:
        ix=indices[role]; y=np.array([rows[i]['label'] for i in ix]); assert len({rows[i]['group_id'] for i in ix})==len(ix)
        policies=('global','source') if role.startswith('test_toxic') else ('global',)
        rng=np.random.default_rng(20261030); g0=np.flatnonzero(y==0); g1=np.flatnonzero(y==1)
        b0=rng.choice(g0,size=(1000,len(g0)),replace=True); b1=rng.choice(g1,size=(1000,len(g1)),replace=True)
        samples=np.concatenate([b0,b1],axis=1); yb=np.array([0]*len(g0)+[1]*len(g1)); boot={}; observed={}
        for c in freeze['candidates']:
            name=c['name']; values=scores[role][name]
            rank_boot=bootstrap_metrics(y,values,b0,b1)
            for policy in policies:
                threshold=thresholds[policy][name]; predicted=values>threshold
                distribution={**rank_boot,'tpr':predicted[b1].mean(1),'fpr':predicted[b0].mean(1)}
                boot[(policy,name)]=distribution
                value=metrics(y,values,threshold)
                value['ci95']={key:np.quantile(array,[.025,.975]).tolist() for key,array in distribution.items()}
                value['binomial_wilson_ci95']={'tpr':wilson(value['tp'],value['tp']+value['fn']),'fpr':wilson(value['fp'],value['fp']+value['tn'])}
                observed[(policy,name)]=value
        for policy in policies:
            report['method_comparison'][role+':'+policy]=method_comparison(observed,boot,policy)
            report['tests'][role+':'+policy]={}
            for c in freeze['candidates']:
                name=c['name']; value=observed[(policy,name)]; distribution=boot[(policy,name)]
                value['paired_delta_ci95']={ref:{key:np.quantile(distribution[key]-boot[(policy,ref)][key],[.025,.975]).tolist()
                    for key in ('auroc','partial_auroc','tpr','fpr')} for ref in ('expert_B','anchor_B_only','linear_BPAT')}
                if policy=='source':
                    value['policy_delta_vs_global_ci95']={key:np.quantile(distribution[key]-boot[('global',name)][key],[.025,.975]).tolist() for key in ('auroc','partial_auroc','tpr','fpr')}
                    assert value['auroc']==observed[('global',name)]['auroc']
                report['tests'][role+':'+policy][name]=value
        print('TEST_COMPLETE',role,len(ix),','.join(policies),flush=True)
    assert digest(out/'freeze.json')==fh
    report['scores_sha256']=digest(path); report['seconds']=time.monotonic()-started
    tmp=target.with_suffix('.tmp'); tmp.write_text(json.dumps(report,indent=2)+'\n'); tmp.replace(target)
    print('SEALED_EVALUATION_COMPLETE',report['seconds'],flush=True)

if __name__=='__main__': main()
