"""Independent fold/grid/selection and score-archive replay."""
import argparse,json
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix
from head_core import ROOT,digest,write,read_jsonl,verify_freeze,grid,selection,make_folds
from head_analysis import analyze

def main():
    p=argparse.ArgumentParser();p.add_argument('--actual',action='store_true');p.add_argument('--bootstrap',action='store_true');args=p.parse_args()
    out=ROOT/'results/heads';f=verify_freeze(args.actual);m=json.loads((ROOT/'configs/heads/manifest.json').read_text());rows=m['samples'];byid={r['id']:r for r in rows};assert len(byid)==len(rows)==5960
    oldids=set()
    for family in ('signal-selection','highdim','fusion','swarm','positions','routing','processing'):oldids.update(r['id'] for r in json.loads((ROOT/f'configs/{family}/manifest.json').read_text())['samples'])
    fresh=[r for r in rows if r['role']=='confirmation_aegis'];assert len(fresh)==1000 and not oldids&{r['id'] for r in fresh}
    bench=json.loads((ROOT/'configs/benchmarks/v1.json').read_text())['samples']
    for r in bench:
        for key in ('id','group_id','role','label'):assert byid[r['id']][key]==r[key]
    cv=json.loads((out/'cv.json').read_text());o=np.load(out/'oof.npz');train=[r for r in rows if r['role']=='train'];y=np.array([r['label'] for r in train]);assert o['training_ids'].tolist()==[r['id'] for r in train] and o['labels'].tolist()==y.tolist()
    expected=grid();assert len(cv['candidates'])==len(expected)==36 and o['names'].tolist()==[c['name'] for c in expected]
    folds=make_folds(y,np.array([r['group_id'] for r in train]));assert len(folds)==len(cv['folds'])==10
    for a,b in zip(cv['folds'],folds):assert a=={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in b.items()}
    checks=0
    for ci,(c,want) in enumerate(zip(cv['candidates'],expected)):
        for key,value in want.items():assert c[key]==value
        vals=[];aucs=[]
        for fi,fold in enumerate(folds):
            v=fold['valid'];s=o['scores'][ci,fold['repeat'],v];auc=roc_auc_score(y[v],s);pauc=roc_auc_score(y[v],s,max_fpr=.1);rec=c['folds'][fi]
            assert rec['repeat']==fold['repeat'] and rec['fold']==fold['fold'];assert abs(auc-rec['auroc'])<1e-12 and abs(pauc-rec['pauc'])<1e-12
            vals.append(pauc);aucs.append(auc);checks+=1
        assert abs(c['cv_mean']-np.mean(vals))<1e-12 and abs(c['cv_std']-np.std(vals,ddof=1))<1e-12 and abs(c['cv_auroc']-np.mean(aucs))<1e-12
    sel=selection(cv['candidates']);sel['selected_head']='ridge_tuned' if sel['selected']==sel['winners']['ridge'] else next(method for method,name in sel['winners'].items() if name==sel['selected'])
    assert sel==f['selection']==json.loads((out/'selection.json').read_text())
    for c in f['candidates']:
        assert c['name']=='B_original' and c['parameters']['value']==1. or c['source_candidate']==sel['winners'][c['method']]
    for s in cv['span_checks']:assert s['relative_gram_error']<1e-9 and s['rank']<=s['fit_rows']-1
    scores=read_jsonl(out/'heldout-scores.jsonl');e=json.loads((out/'evaluation.json').read_text());assert digest(out/'heldout-scores.jsonl')==e['scores_sha256'] and len(scores)==3760
    assert {r['id'] for r in scores}=={r['id'] for r in rows if r['role'] not in ('train','dev')}
    for r in scores:
        for key in ('group_id','role','label'):assert r[key]==byid[r['id']][key]
    pointchecks=0
    for condition,heads in e['tests'].items():
        role,policy=condition.split(':');cohort=[r for r in scores if r['role']==role];y=np.array([r['label'] for r in cohort]);safe=[r for r in scores if r['role']==('calibration_aegis' if policy=='global' else 'calibration_toxic') and r['label']==0];assert len(safe)==400
        for n,v in heads.items():
            s=np.array([r['scores'][n] for r in cohort]);t=sorted(r['scores'][n] for r in safe)[379];assert t==e['thresholds'][policy][n]
            assert abs(roc_auc_score(y,s)-v['auroc'])<1e-12 and abs(average_precision_score(y,s)-v['average_precision'])<1e-12 and abs(roc_auc_score(y,s,max_fpr=.1)-v['partial_auroc'])<1e-12
            tn,fp,fn,tp=confusion_matrix(y,s>t,labels=[0,1]).ravel();assert [tn,fp,fn,tp]==[v[k] for k in ('tn','fp','fn','tp')];pointchecks+=1
    assert checks==360 and pointchecks==35
    if args.bootstrap:
        replay=analyze(scores,f)
        for key in ('tests','contrasts','thresholds'):assert replay[key]==e[key],key
    report={'status':'passed','actual_private_cache_checked':args.actual,'settings':36,'cv_fold_metrics_replayed':checks,'point_metric_conditions':pointchecks,'fresh_rows':len(fresh),'benchmark_rows':len(bench),'bootstrap_replayed':args.bootstrap,'freeze_sha256':digest(out/'freeze.json'),'evaluation_sha256':digest(out/'evaluation.json')}
    write(out/('verification-actual.json' if args.actual else 'verification-archive.json'),report);print(report)

if __name__=='__main__':main()
