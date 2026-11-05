import importlib.util,json,joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from core import ROOT,OUT,CACHE,H2,H3,SEEDS,BASE,PRIMARY,GRIDS,digest,write,verify_lock,score,threshold

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
    verify_lock();assert not (OUT/'evaluation.json').exists()
    training=json.loads((OUT/'training.json').read_text());sealed=digest(OUT/'training.json')
    data=np.load(H2/'cache/oc-confirmation/features.npz');x=data['test_X'];y=data['test_y'];cal=data['cal_X']
    assert x.shape==(1500,14) and cal.shape==(400,14) and (data['cal_y']==0).all()
    assert [(y==k).sum() for k in [0,1]]==[1000,500]
    names=['B']+list(GRIDS)+['H3-C10','cost-fixed-C-unweighted'];metrics={};all_scores={};cuts={}
    for seed in SEEDS:
        values={'cal':[np.trapz(cal,axis=1)],'test':[np.trapz(x,axis=1)]}
        for name in names[1:]:
            item=next(r for r in training['runs'] if r['id']==name and r['seed']==(SEEDS[0] if name not in GRIDS else seed))
            path=ROOT/item['path'];assert digest(path)==item['sha256'];a=joblib.load(path)
            values['cal'].append(score(a,cal));values['test'].append(score(a,x))
        values={k:np.column_stack(v) for k,v in values.items()}
        cut=np.array([threshold(values['cal'][:,j]) for j in range(len(names))]);cuts[str(seed)]=cut.tolist()
        all_scores.update({str(seed)+'_'+role:v for role,v in values.items()})
        rows=[]
        for j,name in enumerate(names):
            s=values['test'][:,j];pred=s>cut[j]
            rows.append({'id':name,'auroc':float(roc_auc_score(y,s)),'pauc':float(roc_auc_score(y,s,max_fpr=.1)),
                'tpr':float(pred[y==1].mean()),'fpr':float(pred[y==0].mean()),'tp':int(pred[y==1].sum()),'fp':int(pred[y==0].sum()),'threshold':float(cut[j])})
        metrics[str(seed)]=rows
    history=json.loads((H3/'results/oc-retraining/retraining.json').read_text())
    old=next(r for r in history['metrics'] if r['id']=='or_small:'+BASE)
    reproduced=next(r for r in metrics[str(SEEDS[0])] if r['id']=='H3-C10')
    for key in ['auroc','pauc','tpr','fpr']:np.testing.assert_allclose(old[key],reproduced[key],atol=1e-12,rtol=0)
    assert reproduced['tp']==old['tpr_count'] and reproduced['fp']==old['fpr_count']
    # Also replay every historical H3 score, not only aggregate metrics.
    old_scores=np.load(H3/'cache/oc-retraining/frozen-scores.npz');column=history['names'].index('or_small:'+BASE)
    for role in ['cal','test']:
        np.testing.assert_allclose(all_scores[str(SEEDS[0])+'_'+role][:,names.index('H3-C10')],old_scores[str(SEEDS[0])+'_'+role][:,column],atol=1e-12,rtol=0)
    s=all_scores[str(SEEDS[0])+'_test'];groups=[np.flatnonzero(y==k) for k in [0,1]];rng=np.random.default_rng(SEEDS[0])
    indices=np.column_stack([rng.choice(g,(10000,len(g))) for g in groups]);auc=np.empty((10000,len(names)));partial=np.empty_like(auc)
    implementation=module('s1_boot',ROOT/'configs/original-positions/core.py')
    for j,name in enumerate(names):
        for start in range(0,10000,100):auc[start:start+100,j],partial[start:start+100,j]=implementation.bootstrap_metrics(y,s[:,j],indices[start:start+100])
        print('OC_S1_BOOT',name,flush=True)
    comparisons=[];point=metrics[str(SEEDS[0])];pred=s>np.array(cuts[str(SEEDS[0])])
    for left,right in PRIMARY:
        j,k=names.index(left),names.index(right);ci=np.quantile(partial[:,j]-partial[:,k],[.025/4,1-.025/4]);lo,hi=ci
        row={'left':left,'right':right,'delta_pauc':point[j]['pauc']-point[k]['pauc'],'pauc_delta_ci9875':ci.tolist(),
             'auroc_delta_ci95':np.quantile(auc[:,j]-auc[:,k],[.025,.975]).tolist(),
             'decision':'positive' if lo>0 else 'worse' if hi<0 else 'unconfirmed'}
        for label,key in [(0,'fpr'),(1,'tpr')]:
            ix=indices[:,:1000] if label==0 else indices[:,1000:]
            delta=pred[:,j].astype(float)-pred[:,k].astype(float)
            row[key+'_delta_ci95']=np.quantile(delta[ix].mean(1),[.025,.975]).tolist()
        comparisons.append(row)
    np.savez(CACHE/'frozen-scores.npz',y=y,**all_scores)
    np.savez(CACHE/'bootstrap-distributions.npz',auc=auc,pauc=partial)
    write(OUT/'evaluation.json',{'run_id':'OC-S1','evidence':'historical_diagnostic','names':names,'counts':[1000,500],
        'primary_seed':SEEDS[0],'metrics':point,'seed_metrics':metrics,'thresholds':cuts,'comparisons':comparisons,
        'family':4,'repeats':10000,'bootstrap_seed':SEEDS[0],'training_sha256':sealed,'source_lock_sha256':digest(OUT/'source-lock.json'),
        'scores_sha256':digest(CACHE/'frozen-scores.npz'),'bootstrap_sha256':digest(CACHE/'bootstrap-distributions.npz'),
        'h3_reproduced_scores_and_metrics':True,'independent_confirmation':False})
    print('OC_S1_SCORED',flush=True)
if __name__=='__main__':main()
