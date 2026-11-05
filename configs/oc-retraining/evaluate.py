import importlib.util,json,joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from core import ROOT,H1,H2,OUT,NAMES,HEADS,SEEDS,GROUPS,PAIRS,digest,write,verify_source,predict,threshold,h1_core

def point(y,s,cuts):
    result=[]
    for j,name in enumerate(NAMES):
        m={'id':name,'auroc':float(roc_auc_score(y,s[:,j])),'pauc':float(roc_auc_score(y,s[:,j],max_fpr=.1)),'threshold':float(cuts[j])}
        for label,key in [(0,'fpr'),(1,'tpr')]:
            n=int((y==label).sum());k=int((s[y==label,j]>cuts[j]).sum());p=k/n;z=1.959963984540054;d=1+z*z/n
            center=(p+z*z/(2*n))/d;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
            m.update({key:p,key+'_count':k,key+'_n':n,key+'_wilson95':[float(center-half),float(center+half)]})
        result.append(m)
    return result

def main():
    verify_source();assert not (OUT/'retraining.json').exists()
    training=json.loads((OUT/'training.json').read_text());selection=json.loads((H1/'results/oc-heads/selection.json').read_text())
    data=np.load(H2/'cache/oc-confirmation/features.npz');old_scores=np.load(H2/'cache/oc-confirmation/frozen-scores.npz');y=data['test_y'];scores={};metrics={};cuts={}
    for seed in SEEDS:
        values={role:[h1_core().baseline(data[role+'_X'])] for role in ['cal','test']}
        for group in GROUPS:
            for name in HEADS:
                if group=='old_full':
                    entry=next(m for m in selection['selected'][name]['models'] if m['seed']==seed);path=H1/entry['path']
                else:
                    entry=next(m for m in training['runs'] if m['seed']==seed and m['group']==group and m['id']==name);path=ROOT/entry['path']
                assert digest(path)==entry['sha256'];artifact=joblib.load(path)
                for role in values:values[role].append(predict(artifact,data[role+'_X']))
        values={role:np.column_stack(v) for role,v in values.items()}
        for role in values:np.testing.assert_array_equal(values[role][:,:4],old_scores[f'{seed}_{role}'])
        scores[str(seed)]=values;cuts[str(seed)]=[threshold(values['cal'][:,j]) for j in range(len(NAMES))];metrics[str(seed)]=point(y,values['test'],cuts[str(seed)])
    s=scores[str(SEEDS[0])]['test'];groups=[np.flatnonzero(y==k) for k in (0,1)];rng=np.random.default_rng(20261104)
    ix=np.column_stack([rng.choice(g,(10000,len(g))) for g in groups]);auc=np.empty((10000,len(NAMES)));partial=np.empty_like(auc)
    spec=importlib.util.spec_from_file_location('h3_bootstrap',ROOT/'configs/original-positions/core.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for j in range(len(NAMES)):
        for start in range(0,10000,100):auc[start:start+100,j],partial[start:start+100,j]=module.bootstrap_metrics(y,s[:,j],ix[start:start+100])
    comparisons=[];pred=s>np.array(cuts[str(SEEDS[0])])
    for j,k in PAIRS:
        ci=np.quantile(partial[:,j]-partial[:,k],[.025/9,1-.025/9]);lo,hi=ci
        item={'left':NAMES[j],'right':NAMES[k],'delta_pauc':metrics[str(SEEDS[0])][j]['pauc']-metrics[str(SEEDS[0])][k]['pauc'],'pauc_delta_ci994444':ci.tolist(),'auroc_delta_ci95':np.quantile(auc[:,j]-auc[:,k],[.025,.975]).tolist(),'decision':'positive' if lo>0 else 'worse' if hi<0 else 'unconfirmed'}
        for label,key in [(0,'fpr'),(1,'tpr')]:
            indices=ix[:,:len(groups[0])] if label==0 else ix[:,len(groups[0]):];delta=pred[:,j].astype(float)-pred[:,k].astype(float)
            item[key+'_delta_ci95']=np.quantile(delta[indices].mean(1),[.025,.975]).tolist()
        comparisons.append(item)
    np.savez(ROOT/'cache/oc-retraining/frozen-scores.npz',**{f'{seed}_{role}':v for seed,roles in scores.items() for role,v in roles.items()})
    write(OUT/'retraining.json',{'run_id':'OC-H3','evidence':'historical_test_retraining_diagnostic','counts':[len(g) for g in groups],'names':NAMES,'primary_seed':SEEDS[0],'metrics':metrics[str(SEEDS[0])],'seed_metrics':metrics,'thresholds':cuts,'comparisons':comparisons,'family':9,'repeats':10000,'bootstrap_seed':20261104,'pauc_ci_coverage':1-.05/9,'training_sha256':digest(OUT/'training.json'),'source_lock_sha256':digest(OUT/'source-lock.json'),'frozen_scores_sha256':digest(ROOT/'cache/oc-retraining/frozen-scores.npz'),'old_full_H2_scores_exact':True,'independent_confirmation':False})
    print('OC_H3_SCORED',flush=True)
if __name__=='__main__':main()
