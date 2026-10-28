"""One sealed evaluation of every frozen candidate, separate label scopes."""
import json
import sys
import time
from pathlib import Path
import joblib
import numpy as np
from features import ROOT,digest,LAYOUT
from detectors import make_matrix,columns_for_mask,raw_scores,detector_scores
from seal import verify_seal
from ridge import raw_matrix,ridge_scores
from seal_code import verify_code_seal
sys.path.insert(0,str(ROOT/'configs/signal-selection'))
from experiment_core import conservative_threshold
from evaluate_frozen import metrics,wilson

def main():
    began=time.monotonic(); out=ROOT/'results/highdim'; target=out/'evaluation.json'
    if target.exists(): print('ALREADY_EVALUATED'); return
    freeze_path=out/'freeze.json'; freeze=json.loads(freeze_path.read_text()); freeze_hash=digest(freeze_path)
    verify_code_seal(out)
    mp=Path(__file__).with_name('manifest.json'); manifest=json.loads(mp.read_text()); rows=manifest['samples']
    assert digest(mp)==freeze['manifest_sha256']
    assert digest(Path(__file__).with_name('detectors.py'))==freeze['detectors_code_sha256']
    for name,key in [('schema.json','schema_sha256'),('centroids.npy','centroids_sha256'),('scaler.npz','scaler_sha256'),('search.json','search_sha256')]:
        assert digest(out/name)==freeze[key]
    for c in freeze['candidates']:
        if 'file' in c: assert digest(out/c['file'])==c['sha256']
    directory=ROOT/'cache/highdim'; assert digest(directory/'metadata.json')==freeze['cache_metadata_sha256']
    assert digest(directory/'seal.json')==freeze['cache_seal_sha256']
    verify_seal(directory,json.loads((directory/'seal.json').read_text()))
    assert json.loads((directory/'checkpoint.json').read_text())['next_index']==len(rows)
    cache={name:np.load(directory/(name+'.npy'),mmap_mode='r') for name in LAYOUT}
    groups=json.loads((out/'schema.json').read_text()); refs=np.load(out/'centroids.npy')
    with np.load(out/'scaler.npz') as artifact: mean=artifact['mean']; scale=artifact['scale']
    scores={}; indices={}
    for role in ('calibration','test_aegis','test_toxicchat'):
        ix=np.array([i for i,x in enumerate(rows) if x['role']==role]); indices[role]=ix
        x=make_matrix(cache,ix,refs,freeze['attention_enabled']); z=(x-mean)/scale
        for c in freeze['candidates']:
            if c['kind']=='interval': values=np.trapz(np.asarray(cache['F'][ix,c['lo']:c['hi']+1],dtype=np.float64),axis=1)
            elif c['kind']=='logit': values=detector_scores(joblib.load(out/c['file']),cache['L'][ix])
            elif c['kind']=='raw_ridge': values=ridge_scores(joblib.load(out/c['file']),raw_matrix(cache,ix,c['regions']))
            else:
                columns=columns_for_mask(c['mask'],groups)
                values=raw_scores(c['head'],joblib.load(out/c['file']),z[:,columns])
            assert np.isfinite(values).all()
            scores[(role,c['name'])]=values
    cy=np.array([rows[i]['label'] for i in indices['calibration']]); thresholds={}
    for c in freeze['candidates']:
        thresholds[c['name']]=conservative_threshold(scores[('calibration',c['name'])][cy==0],.05)
    report={'freeze_sha256':freeze_hash,'manifest_sha256':freeze['manifest_sha256'],'code_sha256':digest(__file__),
        'pre_evaluation_seal_sha256':digest(out/'pre-evaluation-seal.json'),
        'threshold_rule':'strict score > independent safe calibration95% order statistic; no test retuning',
        'calibration_safe_rows':int((cy==0).sum()),'thresholds':thresholds,'calibration':{},'tests':{},
        'bootstrap':{'iterations':500,'seed':20261029,'unit':'one row per deduplicated prompt group',
            'paired':True,'stratified_by_label':True,'scope':'conditional on fitted training/selection/calibration; no multiplicity adjustment'},
        'label_scope':{'test_aegis':'human prompt harmfulness, unused official test','test_toxicchat':'human toxicity proxy, fresh prompts, separate task'},
        'delta_reference':'linear_last_vector'}
    path=out/'heldout-scores.jsonl'
    with path.open('w',encoding='utf-8',newline='\n') as stream:
        for role,ix in indices.items():
            for j,i in enumerate(ix):
                stream.write(json.dumps({'id':rows[i]['id'],'group_id':rows[i]['group_id'],'role':role,'label':rows[i]['label'],
                    'scores':{c['name']:float(scores[(role,c['name'])][j]) for c in freeze['candidates']}})+'\n')
    for c in freeze['candidates']:
        name=c['name']; report['calibration'][name]=metrics(cy,scores[('calibration',name)],thresholds[name])
        assert report['calibration'][name]['fpr']<=.05
    for role in ('test_aegis','test_toxicchat'):
        ix=indices[role]; y=np.array([rows[i]['label'] for i in ix])
        assert len(set(rows[i]['group_id'] for i in ix))==len(ix)
        rng=np.random.default_rng(20261029); labels=[np.flatnonzero(y==label) for label in (0,1)]
        resamples=[np.concatenate([rng.choice(g,len(g),replace=True) for g in labels]) for _ in range(500)]
        # Named references fixed before test inspection: information, interaction and full-pool controls.
        references=['linear_last_vector','linear_position_vectors','linear_all','interaction_all','raw_ridge_last']
        reference_boot={name:[metrics(y[b],scores[(role,name)][b],thresholds[name]) for b in resamples] for name in references}
        report['tests'][role]={}
        for c in freeze['candidates']:
            name=c['name']; value=scores[(role,name)]; observed=metrics(y,value,thresholds[name])
            boot=[metrics(y[b],value[b],thresholds[name]) for b in resamples]
            observed['ci95']={key:np.quantile([v[key] for v in boot],[.025,.975]).tolist() for key in ('auroc','average_precision','tpr','fpr')}
            observed['binomial_wilson_ci95']={'tpr':wilson(observed['tp'],observed['tp']+observed['fn']),
                'fpr':wilson(observed['fp'],observed['fp']+observed['tn'])}
            observed['paired_delta_ci95']={ref:{key:np.quantile([v[key]-r[key] for v,r in zip(boot,rb)],[.025,.975]).tolist()
                for key in ('auroc','tpr','fpr')} for ref,rb in reference_boot.items()}
            report['tests'][role][name]=observed
        print('TEST_COMPLETE',role,len(ix),flush=True)
    assert digest(freeze_path)==freeze_hash
    report['seconds']=time.monotonic()-began; report['scores_sha256']=digest(path)
    temporary=target.with_suffix('.tmp'); temporary.write_text(json.dumps(report,indent=2)+'\n'); temporary.replace(target)
    print('SEALED_EVALUATION_COMPLETE',report['seconds'],flush=True)

if __name__=='__main__': main()
