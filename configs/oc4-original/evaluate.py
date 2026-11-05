import joblib
from core import *
def main():
    check_lock();assert not (OUT/'evaluation.json').exists()
    training=load(OUT/'training.json');selection=load(H1/'results/oc-heads/selection.json')
    data=np.load(H1/'cache/oc-heads/features.npz');oldcore=module('o1_h1',ROOT/'configs/oc-heads/core.py')
    historical=[json.loads(line) for line in (H1/'results/oc-heads/scores.jsonl').read_text().splitlines()]
    frozen={};metrics={};cuts={}
    for seed in SEEDS:
        values={key:[np.trapz(data[key+'_X'],axis=1)] for key in ['cal','RV','C']}
        for name in SVM+NEURAL:
            entry=next(i for i in training['runs'] if i['id']==name and i['seed']==(SEEDS[0] if name in SVM else seed))
            assert digest(ROOT/entry['path'])==entry['sha256'];art=joblib.load(ROOT/entry['path'])
            for key in values:values[key].append(score(art,data[key+'_X']))
        for name in HISTORICAL:
            entry=next(i for i in selection['selected'][name]['models'] if i['seed']==seed);assert digest(H1/entry['path'])==entry['sha256'];art=joblib.load(H1/entry['path'])
            for key in values:
                import torch
                torch.set_num_threads(2);torch.backends.cuda.enable_flash_sdp(True);torch.backends.cuda.enable_mem_efficient_sdp(True);torch.backends.cuda.enable_math_sdp(True);torch.use_deterministic_algorithms(False)
                s=oldcore.score(art,data[key+'_X']);attention.setup(seed)
                expected=np.array([r['scores'][(['B']+oldcore.IDS).index(name)] for r in historical if r['seed']==seed and r['role']==key])
                np.testing.assert_array_equal(s,expected);values[key].append(s)
        values={key:np.column_stack(v) for key,v in values.items()};cut=np.array([threshold(values['cal'][:,j]) for j in range(len(NAMES))]);cuts[str(seed)]=cut.tolist()
        for key,s in values.items():frozen[f'{seed}_{key}']=s
        metrics[str(seed)]={key:point(data[key+'_y'],values[key],cut) for key in ['RV','C']}
    assert abs(metrics[str(SEEDS[0])]['RV'][0]['auroc']-.7565473145780051)<1e-12
    assert abs(metrics[str(SEEDS[0])]['C'][0]['auroc']-.746640)<1e-12
    boot=module('o1_boot',ROOT/'configs/original-positions/core.py');distributions={};cohorts={}
    for key in ['RV','C']:
        y=data[key+'_y'];s=frozen[f'{SEEDS[0]}_{key}'];draws=bootstrap_draws(y);ix=np.column_stack(draws)
        auc=np.empty((REPEATS,len(NAMES)));partial=np.empty_like(auc)
        for j,name in enumerate(NAMES):
            for start in range(0,REPEATS,100):auc[start:start+100,j],partial[start:start+100,j]=boot.bootstrap_metrics(y,s[:,j],ix[start:start+100])
            print('OC_O1_BOOT',key,name,flush=True)
        distributions[key+'_auc']=auc;distributions[key+'_pauc']=partial
        pred=s>np.array(cuts[str(SEEDS[0])]);rows=metrics[str(SEEDS[0])][key];comparisons=[]
        for label,terms in CONTRASTS:
            weights=np.array([sum(w for n,w in terms if n==name) for name in NAMES]);tail=.025/len(CONTRASTS) if key=='RV' else .025
            ci=np.quantile(partial@weights,[tail,1-tail]);lo,hi=ci
            item={'id':label,'terms':terms,'delta_pauc':float(sum(w*row['pauc'] for w,row in zip(weights,rows))),'pauc_delta_ci':ci.tolist(),'decision':'positive' if lo>0 else 'worse' if hi<0 else 'unconfirmed','auroc_delta_ci95':np.quantile(auc@weights,[.025,.975]).tolist()}
            d=pred.astype(float)@weights
            for k,name in [(0,'fpr'),(1,'tpr')]:item[name+'_delta_ci95']=np.quantile(d[draws[k]].mean(1),[.025,.975]).tolist()
            comparisons.append(item)
        cohorts[key]={'counts':[int((y==k).sum()) for k in [0,1]],'metrics':rows,'comparisons':comparisons,'pauc_ci_coverage':1-.05/27 if key=='RV' else .95,'descriptive_only':key=='C'}
    np.savez(CACHE/'scores.npz',**frozen);np.savez(CACHE/'bootstrap.npz',**distributions)
    write(OUT/'evaluation.json',{'run_id':'OC-O1','primary_cohort':'RV','cohorts':cohorts,'seed_metrics':metrics,'thresholds':cuts,'names':NAMES,'family':len(CONTRASTS),'repeats':REPEATS,'primary_seed':SEEDS[0],
       'training_sha256':digest(OUT/'training.json'),'source_lock_sha256':digest(OUT/'source-lock.json'),'scores_sha256':digest(CACHE/'scores.npz'),'bootstrap_sha256':digest(CACHE/'bootstrap.npz'),'h1_scores_reproduced':True,'independent_confirmation':False})
    print('OC_O1_SCORED',flush=True)
if __name__=='__main__':main()
