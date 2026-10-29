"""Replay all 3600 fold metrics, all predictions' point metrics and bootstrap."""
import argparse,json
import numpy as np
from sklearn.metrics import roc_auc_score,confusion_matrix
from stability_core import ROOT,verify_freeze,digest,write,read_jsonl,draws,variants,params,make_folds,p_auc
from stability_analysis import analyze

def main():
    p=argparse.ArgumentParser();p.add_argument('--actual',action='store_true');p.add_argument('--bootstrap',action='store_true');args=p.parse_args();f=verify_freeze(args.actual);out=ROOT/'results/stability'
    m=json.loads((ROOT/'configs/stability/manifest.json').read_text());rows=m['samples'];sets=draws(rows);cv=json.loads((out/'cv.json').read_text());o=np.load(out/'oof.npz')['scores'];vs=variants();assert [x.tolist() for x in sets]==m['draws']==cv['draws'] and o.shape==(5,8,9,2,1600)
    assert len({r['id'] for r in rows})==len({r['group_id'] for r in rows})==6400
    old=json.loads((ROOT/'configs/heads/manifest.json').read_text())['samples'];oldids={r['id'] for r in old}
    assert all(r['id'] not in oldids for r in rows if r.get('origin') in ('new_train','fresh_confirmation'))
    bench=json.loads((ROOT/'configs/benchmarks/v1.json').read_text())['samples'];byid={r['id']:r for r in rows}
    for r in bench:
        if r['id'] in byid:
            for k in ('group_id','role','label'):assert byid[r['id']][k]==r[k]
    checks=0
    for si,ix in enumerate(sets):
        y=np.array([rows[i]['label'] for i in ix]);folds=make_folds(y,np.array([rows[i]['group_id'] for i in ix]));assert cv['folds'][si]==[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in f.items()} for f in folds]
        for vi,v in enumerate(vs):
            rec=next(r for r in cv['records'] if r['draw']==si and r['variant']==v['name']);means=[]
            for ci,value in enumerate(params(v['method'])):
                c=rec['settings'][ci];assert c['value']==value;vals=[]
                for fi,fold in enumerate(folds):
                    ixv=fold['valid'];s=o[si,vi,ci,fold['repeat'],ixv];assert np.isfinite(s).all()
                    auc=float(roc_auc_score(y[ixv],s));pa=p_auc(y[ixv],s);assert abs(auc-c['folds'][fi]['auroc'])<1e-12 and abs(pa-c['folds'][fi]['pauc'])<1e-12;vals.append(pa);checks+=1
                assert abs(np.mean(vals)-c['mean_pauc'])<1e-12;means.append(c['mean_pauc'])
            best=max(range(9),key=lambda j:means[j]);assert best==rec['selected_index']
            model=next(c for c in f['models'] if c['draw']==si and c['variant']==v['name']);assert model['parameter_index']==best and model['value']==params(v['method'])[best]
    for s in cv['span_checks']:assert s['relative_gram_error']<1e-9
    scores=read_jsonl(out/'heldout-scores.jsonl');e=json.loads((out/'evaluation.json').read_text());assert digest(out/'heldout-scores.jsonl')==e['scores_sha256'];assert len(scores)==3200 and {r['id'] for r in scores}=={r['id'] for r in rows if r['role']!='pool'}
    for r in scores:
        for k in ('group_id','role','label'):assert r[k]==byid[r['id']][k]
    pointchecks=0
    for condition,heads in e['tests'].items():
        role,policy=condition.split(':');cohort=[r for r in scores if r['role']==role];y=np.array([r['label'] for r in cohort]);safe=[r for r in scores if r['role']==('calibration_aegis' if policy=='global' else 'calibration_toxic') and r['label']==0];assert len(safe)==400
        for n,v in heads.items():
            s=np.array([r['scores'][n] for r in cohort]);t=sorted(r['scores'][n] for r in safe)[379];assert t==e['thresholds'][policy][n]
            assert abs(roc_auc_score(y,s)-v['auroc'])<1e-12 and abs(p_auc(y,s)-v['partial_auroc'])<1e-12
            tn,fp,fn,tp=confusion_matrix(y,s>t,labels=[0,1]).ravel();assert [tn,fp,fn,tp]==[v[k] for k in ('tn','fp','fn','tp')];pointchecks+=1
    assert checks==3600 and pointchecks==160
    if args.bootstrap:
        replay=analyze(scores)
        for k in ('tests','aggregate','contrasts','thresholds'):assert replay[k]==e[k],k
    result={'status':'passed','actual_private_cache_checked':args.actual,'fold_metrics':checks,'test_conditions':pointchecks,'bootstrap_replayed':args.bootstrap,'freeze_sha256':digest(out/'freeze.json'),'evaluation_sha256':digest(out/'evaluation.json')}
    write(out/('verification-actual.json' if args.actual else 'verification-archive.json'),result);print(result)

if __name__=='__main__':main()
