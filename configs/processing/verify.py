"""Replay source gates, unchanged benchmark, folds/selection and all point metrics."""
import argparse,json
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix
from processing_core import ROOT,digest,write,read_jsonl,verify_freeze,make_folds,select,candidates
from analyze import analyze

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--actual',action='store_true');parser.add_argument('--bootstrap',action='store_true');args=parser.parse_args()
    out=ROOT/'results/processing';f=verify_freeze(args.actual);m=json.loads((ROOT/'configs/processing/manifest.json').read_text());rows=m['samples'];byid={r['id']:r for r in rows}
    assert len(byid)==len(rows)==5960
    oldids=set()
    for family in ('signal-selection','highdim','fusion','swarm','positions','routing'):
        oldids.update(r['id'] for r in json.loads((ROOT/f'configs/{family}/manifest.json').read_text())['samples'])
    fresh=[r for r in rows if r['role']=='confirmation_aegis'];assert len(fresh)==1000 and not oldids&{r['id'] for r in fresh}
    bench=json.loads((ROOT/'configs/benchmarks/v1.json').read_text())['samples']
    for r in bench:
        for key in ('id','group_id','role','label'):assert byid[r['id']][key]==r[key]
    cv=json.loads((out/'cv.json').read_text());archive=np.load(out/'oof.npz');training=[r for r in rows if r['role']=='train'];labels=np.array([r['label'] for r in training])
    assert archive['training_ids'].tolist()==[r['id'] for r in training] and archive['labels'].tolist()==labels.tolist()
    assert archive['names'].tolist()==[c['name'] for c in candidates()]==[c['name'] for c in f['candidates']]
    folds=make_folds(labels,np.array([r['group_id'] for r in training]));assert len(folds)==len(cv['folds'])==10
    for a,b in zip(cv['folds'],folds):assert a=={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in b.items()}
    metrics=0
    for ci,c in enumerate(f['candidates']):
        vals=[];aucs=[]
        for fi,fold in enumerate(folds):
            y=labels[fold['valid']];s=archive['scores'][ci,fold['repeat'],fold['valid']];auc=roc_auc_score(y,s);pauc=roc_auc_score(y,s,max_fpr=.1)
            expected=c['folds'][fi];assert abs(auc-expected['auroc'])<1e-12 and abs(pauc-expected['pauc'])<1e-12
            vals.append(pauc);aucs.append(auc);metrics+=1
        assert abs(np.mean(vals)-c['cv_mean'])<1e-12 and abs(np.std(vals,ddof=1)-c['cv_std'])<1e-12 and abs(np.mean(aucs)-c['cv_auroc'])<1e-12
    assert metrics==130 and select(f['candidates'])==f['selection']==json.loads((out/'selection.json').read_text())
    names=archive['names'].tolist()
    for g in ('0p25','1p0'):assert np.max(np.abs(archive['scores'][names.index('mean_g'+g)]-archive['scores'][names.index('duplicate_mean_g'+g)]))<1e-7
    e=json.loads((out/'evaluation.json').read_text());scored=read_jsonl(out/'heldout-scores.jsonl');assert len(scored)==3760 and digest(out/'heldout-scores.jsonl')==e['scores_sha256']
    assert {r['id'] for r in scored}=={r['id'] for r in rows if r['role'] not in ('train','dev')}
    for r in scored:
        for key in ('group_id','role','label'):assert r[key]==byid[r['id']][key]
    checks=0
    for condition,table in e['tests'].items():
        role,policy=condition.split(':');cohort=[r for r in scored if r['role']==role];y=np.array([r['label'] for r in cohort])
        safe=[r for r in scored if r['role']==('calibration_aegis' if policy=='global' else 'calibration_toxic') and r['label']==0];assert len(safe)==400
        for name,v in table.items():
            s=np.array([r['scores'][name] for r in cohort]);threshold=sorted(r['scores'][name] for r in safe)[379];assert threshold==e['thresholds'][policy][name]
            assert abs(roc_auc_score(y,s)-v['auroc'])<1e-12 and abs(average_precision_score(y,s)-v['average_precision'])<1e-12 and abs(roc_auc_score(y,s,max_fpr=.1)-v['partial_auroc'])<1e-12
            tn,fp,fn,tp=confusion_matrix(y,s>threshold,labels=[0,1]).ravel();assert [tn,fp,fn,tp]==[v[k] for k in ('tn','fp','fn','tp')];checks+=1
    assert checks==91
    if args.bootstrap:
        replay=analyze(scored,f)
        for k in ('tests','contrasts','thresholds'):assert replay[k]==e[k],k
    report={'status':'passed','actual_private_cache_checked':args.actual,'point_metric_conditions':checks,'cv_fold_metrics_replayed':metrics,'fresh_rows':len(fresh),'benchmark_rows':len(bench),'bootstrap_replayed':args.bootstrap,'freeze_sha256':digest(out/'freeze.json'),'evaluation_sha256':digest(out/'evaluation.json')}
    write(out/('verification-actual.json' if args.actual else 'verification-archive.json'),report);print(report)

if __name__=='__main__':main()
