"""All frozen late-fusion candidates, fresh holdouts, two independent calibration policies."""
import json
import sys
import time
import joblib
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from fusion_core import ROOT,LAYOUT,EXPERTS,digest,expert_matrix,score_expert,meta_scores
from provenance import verify_freeze
from ridge import raw_matrix,ridge_scores
from detectors import centroid_features
sys.path.insert(0,str(ROOT/'configs/signal-selection'))
from experiment_core import conservative_threshold,confusion
from evaluate_frozen import metrics,wilson

def all_scores(cache,indices,freeze,out):
    matrix=np.zeros((len(indices),4))
    for column,expert in enumerate(freeze['experts']):
        matrix[:,column]=score_expert(joblib.load(out/expert['file']),expert_matrix(cache,indices,expert['name']),expert['name'])
    with np.load(out/'equal.npz') as item: mean=item['mean']; scale=item['scale']
    values={}; refs=np.load(ROOT/'results/highdim/centroids.npy')
    with np.load(ROOT/'results/highdim/scaler.npz') as item: oldmean=item['mean']; oldscale=item['scale']
    oldschema=json.loads((ROOT/'results/highdim/schema.json').read_text())
    fc_columns=np.concatenate([np.arange(g['start'],g['stop']) for g in oldschema if g['family'] in ('F','C')])
    for candidate in freeze['candidates']:
        name=candidate['name']; kind=candidate['kind']
        if kind=='expert': score=matrix[:,candidate['column']]
        elif kind=='meta': score=meta_scores(joblib.load(out/candidate['file']),matrix)
        elif kind=='equal': score=((matrix-mean)/scale).mean(1)
        elif kind=='legacy_raw_positions': score=ridge_scores(joblib.load(out/candidate['file']),raw_matrix(cache,indices,[0,1,2,3,4]))
        elif kind=='legacy_FC64':
            x=np.concatenate([np.asarray(cache['F'][indices]),centroid_features(np.asarray(cache['P'][indices,:,0,:]),refs)],axis=1)
            score=joblib.load(out/candidate['file']).decision_function((x-oldmean[fc_columns])/oldscale[fc_columns])
        else: raise ValueError(kind)
        assert np.isfinite(score).all(); values[name]=score
    return values

def main():
    started=time.monotonic(); out=ROOT/'results/fusion'; target=out/'evaluation.json'
    if target.exists(): print('ALREADY_EVALUATED'); return
    freeze=verify_freeze(out,cache_bytes=True); fh=digest(out/'freeze.json')
    manifest=json.loads((ROOT/'configs/fusion/manifest.json').read_text()); rows=manifest['samples']
    cache={name:np.load(ROOT/'cache/fusion'/(name+'.npy'),mmap_mode='r') for name in LAYOUT}
    roles=['calibration_aegis','calibration_toxic','test_aegis_holdout','test_aegis_official','test_toxic_holdout','test_toxicchat']
    indices={role:np.array([i for i,x in enumerate(rows) if x['role']==role]) for role in roles}
    scores={role:all_scores(cache,ix,freeze,out) for role,ix in indices.items()}
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
        'thresholds':thresholds,'calibration':calibration,'tests':{},'best_dev_subset':freeze['best_dev_additive_subset'],
        'bootstrap':{'iterations':1000,'seed':20261030,'paired':True,'stratified_by_label':True,'unit':'one row per deduplicated prompt group',
            'scope':'conditional on frozen expert/meta training and calibration; no multiplicity adjustment'},
        'primary':'test_aegis_holdout global, linear_BPAT versus expert_B','policy_scope':'source policy applies only to separate toxicity proxy; rankings unchanged'}
    for role in roles[2:]:
        ix=indices[role]; y=np.array([rows[i]['label'] for i in ix]); assert len({rows[i]['group_id'] for i in ix})==len(ix)
        policies=('global','source') if role.startswith('test_toxic') else ('global',)
        rng=np.random.default_rng(20261030); g0=np.flatnonzero(y==0); g1=np.flatnonzero(y==1)
        b0=rng.choice(g0,size=(1000,len(g0)),replace=True); b1=rng.choice(g1,size=(1000,len(g1)),replace=True)
        samples=np.concatenate([b0,b1],axis=1); yb=np.array([0]*len(g0)+[1]*len(g1)); boot={}; observed={}
        for c in freeze['candidates']:
            name=c['name']; values=scores[role][name]
            auc=np.array([roc_auc_score(yb,values[b]) for b in samples]); ap=np.array([average_precision_score(yb,values[b]) for b in samples])
            for policy in policies:
                threshold=thresholds[policy][name]; predicted=values>threshold
                distribution={'auroc':auc,'average_precision':ap,'tpr':predicted[b1].mean(1),'fpr':predicted[b0].mean(1)}
                boot[(policy,name)]=distribution
                value=metrics(y,values,threshold)
                value['ci95']={key:np.quantile(array,[.025,.975]).tolist() for key,array in distribution.items()}
                value['binomial_wilson_ci95']={'tpr':wilson(value['tp'],value['tp']+value['fn']),'fpr':wilson(value['fp'],value['fp']+value['tn'])}
                observed[(policy,name)]=value
        for policy in policies:
            report['tests'][role+':'+policy]={}
            for c in freeze['candidates']:
                name=c['name']; value=observed[(policy,name)]; distribution=boot[(policy,name)]
                value['paired_delta_ci95']={ref:{key:np.quantile(distribution[key]-boot[(policy,ref)][key],[.025,.975]).tolist()
                    for key in ('auroc','tpr','fpr')} for ref in ('expert_B','linear_BPAT')}
                if policy=='source':
                    value['policy_delta_vs_global_ci95']={key:np.quantile(distribution[key]-boot[('global',name)][key],[.025,.975]).tolist() for key in ('auroc','tpr','fpr')}
                    assert value['auroc']==observed[('global',name)]['auroc']
                report['tests'][role+':'+policy][name]=value
        print('TEST_COMPLETE',role,len(ix),','.join(policies),flush=True)
    assert digest(out/'freeze.json')==fh
    report['scores_sha256']=digest(path); report['seconds']=time.monotonic()-started
    tmp=target.with_suffix('.tmp'); tmp.write_text(json.dumps(report,indent=2)+'\n'); tmp.replace(target)
    print('SEALED_EVALUATION_COMPLETE',report['seconds'],flush=True)

if __name__=='__main__': main()
