"""Replay meta transforms, 640 CV fold metrics, calibration and bootstrap."""
import argparse,json
import numpy as np
from sklearn.metrics import roc_auc_score,confusion_matrix
from anchored_core import ROOT,FAMILIES,digest,write,read_jsonl,verify_freeze,candidates,fit_alignment,fuse,make_folds,p_auc,select
from anchored_analysis import analyze

def main():
    p=argparse.ArgumentParser();p.add_argument('--actual',action='store_true');p.add_argument('--bootstrap',action='store_true');args=p.parse_args();f=verify_freeze(args.actual);out=ROOT/'results/anchored'
    m=json.loads((ROOT/'configs/anchored/manifest.json').read_text());rows=m['samples'];byid={r['id']:r for r in rows};meta=read_jsonl(out/'meta-scores.jsonl');assert len(meta)==600 and {r['id'] for r in meta}=={r['id'] for r in rows if r['role']=='meta_train'}
    oldrows=json.loads((ROOT/'configs/stability/manifest.json').read_text())['samples'];draw=json.loads((ROOT/'configs/stability/manifest.json').read_text())['draws'][0];assert not {oldrows[i]['id'] for i in draw}&set(byid)
    history=set()
    for name in ('signal-selection','highdim','fusion','swarm','positions','routing','processing','heads','stability'):
        path=ROOT/'configs'/name/'manifest.json'
        if path.exists():history.update(r['id'] for r in json.loads(path.read_text())['samples'])
    assert all(r['id'] not in history for r in rows if r['role'].startswith('confirmation_'))
    assert len({r['group_id'] for r in rows})==len(rows)
    for r in meta:
        for key in ('group_id','role','label'):assert r[key]==byid[r['id']][key]
    y=np.array([r['label'] for r in meta]);groups=np.array([r['group_id'] for r in meta]);raw=np.array([[r['scores'][n] for n in FAMILIES] for r in meta]);folds=make_folds(y,groups);cv=json.loads((out/'cv.json').read_text());o=np.load(out/'oof.npz');checks=0
    serialize=lambda fs:[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in f.items()} for f in fs]
    assert serialize(folds)==cv['folds']
    for ci,c in enumerate(candidates()):
        rec=cv['candidates'][ci];vals=[];e=raw[:,FAMILIES.index(c['family'])]
        for key,v in c.items():assert rec[key]==v
        for fi,fold in enumerate(folds):
            ix,v=fold['fit'],fold['valid'];h=fit_alignment(raw[ix,0],e[ix],y[ix]);s=fuse(raw[v,0],e[v],c,h);np.testing.assert_allclose(s,o['scores'][ci,fold['repeat'],v],rtol=0,atol=1e-12)
            auc=float(roc_auc_score(y[v],s));pa=p_auc(y[v],s);assert abs(auc-rec['folds'][fi]['auroc'])<1e-12 and abs(pa-rec['folds'][fi]['pauc'])<1e-12;vals.append(pa);checks+=1
        assert abs(np.mean(vals)-rec['cv_mean'])<1e-12 and abs(np.std(vals,ddof=1)-rec['cv_std'])<1e-12
    selection=select(cv['candidates']);selection['selected_head']='B' if selection['selected']=='B' else next(c['name'] for c in f['models'] if c['parameters'].get('name')==selection['selected'])
    assert selection==f['selection']==cv['selection']==json.loads((out/'selection.json').read_text())
    for c in f['models']:
        if c['parameters'].get('mode') in ('gated','global'):
            assert c['parameters']['name']==selection['winners'][c['name']]
            np.testing.assert_equal(c['alignment'],fit_alignment(raw[:,0],raw[:,FAMILIES.index(c['parameters']['family'])],y))
    ry=np.array(cv['refit_labels']);rg=np.array(cv['refit_groups']);assert len(ry)==len(set(rg))==2200
    expected_groups=[oldrows[i]['group_id'] for i in draw]+groups.tolist();expected_labels=[oldrows[i]['label'] for i in draw]+y.tolist();assert expected_groups==cv['refit_groups'] and expected_labels==cv['refit_labels']
    rf=make_folds(ry,rg);assert serialize(rf)==cv['refit_folds'];means=[]
    for ci,c in enumerate(cv['refit_candidates']):
        vals=[]
        for fi,fold in enumerate(rf):
            v=fold['valid'];s=o['refit_scores'][ci,fold['repeat'],v];pa=p_auc(ry[v],s);assert abs(roc_auc_score(ry[v],s)-c['folds'][fi]['auroc'])<1e-12 and abs(pa-c['folds'][fi]['pauc'])<1e-12;vals.append(pa);checks+=1
        assert abs(np.mean(vals)-c['cv_mean'])<1e-12;means.append(c['cv_mean'])
    assert max(range(9),key=lambda i:means[i])==cv['refit_selected_index'] and checks==640
    scores=read_jsonl(out/'heldout-scores.jsonl');e=json.loads((out/'evaluation.json').read_text());assert digest(out/'heldout-scores.jsonl')==e['scores_sha256'];assert {r['id'] for r in scores}=={r['id'] for r in rows if r['role']!='meta_train'}
    for r in scores:
        for key in ('group_id','role','label'):assert r[key]==byid[r['id']][key]
    points=0
    for condition,heads in e['tests'].items():
        role,policy=condition.split(':');cohort=[r for r in scores if r['role']==role];truth=np.array([r['label'] for r in cohort]);safe=[r for r in scores if r['role']==('calibration_aegis' if policy=='global' else 'calibration_toxic') and r['label']==0];assert len(safe)==400
        for n,v in heads.items():
            s=np.array([r['scores'][n] for r in cohort]);t=sorted(r['scores'][n] for r in safe)[379];assert t==e['thresholds'][policy][n]
            assert abs(roc_auc_score(truth,s)-v['auroc'])<1e-12 and abs(p_auc(truth,s)-v['partial_auroc'])<1e-12
            tn,fp,fn,tp=confusion_matrix(truth,s>t,labels=[0,1]).ravel();assert [tn,fp,fn,tp]==[v[k] for k in ('tn','fp','fn','tp')];points+=1
    assert points==66
    if args.bootstrap:
        replay=analyze(scores,f)
        for key in ('tests','contrasts','thresholds'):assert replay[key]==e[key],key
    result={'status':'passed','actual_private_cache_checked':args.actual,'cv_fold_metrics':checks,'test_conditions':points,'bootstrap_replayed':args.bootstrap,'freeze_sha256':digest(out/'freeze.json'),'evaluation_sha256':digest(out/'evaluation.json')};write(out/('verification-actual.json' if args.actual else 'verification-archive.json'),result);print(result)

if __name__=='__main__':main()
