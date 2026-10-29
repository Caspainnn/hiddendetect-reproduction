"""Independently replay all portable point metrics, manifests, budgets and folds."""
import argparse,json
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix
from routing_core import ROOT,digest,write,read_jsonl,verify_freeze,warm_masks,key,make_folds,verify_seal,replay_selection
from analyze import analyze

def main():
    p=argparse.ArgumentParser();p.add_argument('--actual',action='store_true');p.add_argument('--bootstrap',action='store_true');args=p.parse_args()
    out=ROOT/'results/routing';f=verify_freeze(actual=args.actual);m=json.loads((ROOT/'configs/routing/manifest.json').read_text());old=json.loads((ROOT/'configs/positions/manifest.json').read_text())['samples']
    verify_seal(actual=args.actual)
    rows=m['samples'];ids={r['id'] for r in rows};assert len(ids)==len(rows)==5960
    oldids={r['id'] for r in old};fresh=[r for r in rows if r['role']=='confirmation_aegis'];assert len(fresh)==1000 and not oldids&{r['id'] for r in fresh}
    benchmark=json.loads((ROOT/'configs/benchmarks/v1.json').read_text());br=benchmark.get('samples',benchmark.get('rows'))
    if isinstance(br,dict):br=list(br.values())
    byid={r['id']:r for r in rows}
    for r in br:
        current=byid[r['id']]
        for k in ('label','role','group_id'):assert current[k]==r[k],(r['id'],k)
    cv=json.loads((out/'cv.json').read_text());archive=np.load(out/'oof.npz');keys=archive['keys'].tolist();oof=archive['scores'];labels=archive['labels'];checks=0
    training=[r for r in rows if r['role']=='train'];assert archive['training_ids'].tolist()==[r['id'] for r in training]
    assert labels.tolist()==[r['label'] for r in training]
    expected_folds=make_folds(labels,np.array([r['group_id'] for r in training]))
    for a,b in zip(cv['folds'],expected_folds):
        assert a=={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in b.items()}
    for k,spec in cv['evaluations'].items():
        a=oof[keys.index(k)]
        for mi,mode in enumerate(('variable','fixed')):
            vals=[]
            for fi,fold in enumerate(cv['folds']):
                fit=set(fold['fit']);valid=fold['valid'];assert not fit&set(valid) and len(fit)+len(valid)==1600
                s=a[mi,fold['repeat'],valid];y=labels[valid];v=spec['stats'][mode]['folds'][fi]
                auc=roc_auc_score(y,s);pauc=roc_auc_score(y,s,max_fpr=.1)
                assert abs(auc-v['auroc'])<1e-12 and abs(pauc-v['pauc'])<1e-12
                vals.append(pauc);checks+=1
            assert abs(np.mean(vals)-spec['stats'][mode]['mean'])<1e-12
            assert abs(np.std(vals,ddof=1)-spec['stats'][mode]['std'])<1e-12
        fitness=min(v['mean']-.25*v['std'] for v in spec['stats'].values())-.002*k.count('1')/32
        assert abs(fitness-spec['fitness'])<1e-12
    assert len(cv['runs'])==25 and sum(r['unique_evaluations'] for r in cv['runs'])==1600
    expected=[key(x) for x in warm_masks(32)]
    assert {(r['method'],r['seed']) for r in cv['runs']}=={(method,seed) for method in ('random','bpso','ga','de','scan') for seed in range(20261201,20261206)}
    for run in cv['runs']:
        assert run['unique_evaluations']==len(set(run['evaluated_masks']))==64 and run['evaluated_masks'][:34]==expected
        for k,v in zip(run['evaluated_masks'],run['fitness_values']):assert v==cv['evaluations'][k]['fitness']
        assert run['best_fitness']==max(run['fitness_values']) and cv['evaluations'][key(run['best_mask'])]['fitness']==run['best_fitness']
    assert replay_selection(cv['evaluations'],cv['runs'])==f['selection']==json.loads((out/'selection.json').read_text())
    e=json.loads((out/'evaluation.json').read_text());scored=read_jsonl(out/'heldout-scores.jsonl');assert digest(out/'heldout-scores.jsonl')==e['scores_sha256'] and len(scored)==3760
    assert {r['id'] for r in scored}=={r['id'] for r in rows if r['role'] not in ('train','dev')}
    for r in scored:
        for k in ('role','label','group_id'):assert r[k]==byid[r['id']][k]
    metric_checks=0
    for condition,models in e['tests'].items():
        role,policy=condition.split(':');cohort=[r for r in scored if r['role']==role];y=np.array([r['label'] for r in cohort])
        safe=[r for r in scored if r['role']==('calibration_aegis' if policy=='global' else 'calibration_toxic') and r['label']==0]
        for n,v in models.items():
            s=np.array([r['scores'][n] for r in cohort]);t=sorted(r['scores'][n] for r in safe)[379];assert t==e['thresholds'][policy][n]
            assert abs(roc_auc_score(y,s)-v['auroc'])<1e-12 and abs(average_precision_score(y,s)-v['average_precision'])<1e-12
            assert abs(roc_auc_score(y,s,max_fpr=.1)-v['partial_auroc'])<1e-12
            tn,fp,fn,tp=confusion_matrix(y,s>t,labels=[0,1]).ravel();assert [tn,fp,fn,tp]==[v[k] for k in ('tn','fp','fn','tp')]
            metric_checks+=1
    if args.bootstrap:
        replay=analyze(scored,f)
        for k in ('tests','contrasts','thresholds'):assert replay[k]==e[k],k
    report={'status':'passed','actual_private_cache_checked':args.actual,'point_metric_conditions':metric_checks,'cv_fold_metrics_replayed':checks,'charged_masks':1600,'distinct_masks':len(cv['evaluations']),'fresh_rows':1000,'benchmark_rows':len(br),'bootstrap_replayed':args.bootstrap,'freeze_sha256':digest(out/'freeze.json'),'evaluation_sha256':digest(out/'evaluation.json')}
    write(out/('verification-actual.json' if args.actual else 'verification-archive.json'),report);print(report)

if __name__=='__main__':main()
