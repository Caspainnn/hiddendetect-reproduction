import json,importlib.util,joblib,time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from core import ROOT,OUT,IDS,SEEDS,digest,write,score,threshold,baseline,pauc

def wilson(k,n):
    z=1.959963984540054;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [float(c-h),float(c+h)]
def calculate(y,s,cut):
    result=[];prediction=s>cut
    for j,name in enumerate(['B']+IDS):
        m={'id':name,'auroc':float(roc_auc_score(y,s[:,j])),'pauc':pauc(y,s[:,j]),'threshold':float(cut[j])}
        for label,key in [(0,'fpr'),(1,'tpr')]:
            n=int((y==label).sum());k=int(prediction[y==label,j].sum())
            m.update({key:k/n,key+'_count':k,key+'_n':n,key+'_wilson95':wilson(k,n)})
        result.append(m)
    return result
def uncertainty(y,s,cut):
    spec=importlib.util.spec_from_file_location('head_bootstrap',ROOT/'configs/original-positions/core.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    groups=[np.flatnonzero(y==label) for label in (0,1)];rng=np.random.default_rng(20261104)
    ix=np.column_stack([rng.choice(g,(10000,len(g))) for g in groups]);auc=np.empty((10000,9));partial=np.empty_like(auc)
    for start in range(0,len(ix),100):
        for j in range(9):auc[start:start+100,j],partial[start:start+100,j]=module.bootstrap_metrics(y,s[:,j],ix[start:start+100])
    metrics=calculate(y,s,cut);prediction=s>cut
    for j,m in enumerate(metrics[1:],1):
        m.update(delta_auroc=m['auroc']-metrics[0]['auroc'],delta_pauc=m['pauc']-metrics[0]['pauc'],
            auc_delta_ci95=np.quantile(auc[:,j]-auc[:,0],[.025,.975]).tolist(),
            pauc_delta_ci99375=np.quantile(partial[:,j]-partial[:,0],[.025/8,1-.025/8]).tolist())
        lo,hi=m['pauc_delta_ci99375'];m['historical_decision']='positive_interval' if lo>0 else 'worse' if hi<0 else 'unconfirmed'
        for label,key in [(0,'fpr'),(1,'tpr')]:
            sample=ix[:,:len(groups[0])] if label==0 else ix[:,len(groups[0]):]
            difference=prediction[:,j].astype(float)-prediction[:,0].astype(float)
            m[key+'_delta_ci95']=np.quantile(difference[sample].mean(1),[.025,.975]).tolist()
    pairs=[(IDS[4],IDS[0]),(IDS[5],IDS[0]),(IDS[1],IDS[0]),(IDS[2],IDS[0]),(IDS[6],IDS[2]),(IDS[6],IDS[0]),(IDS[3],IDS[1]),(IDS[3],IDS[0]),(IDS[7],IDS[3]),(IDS[7],IDS[1])]
    names=['B']+IDS;mechanisms=[]
    for left,right in pairs:
        j,k=names.index(left),names.index(right)
        mechanisms.append({'left':left,'right':right,'pauc_delta':metrics[j]['pauc']-metrics[k]['pauc'],'descriptive_ci95':np.quantile(partial[:,j]-partial[:,k],[.025,.975]).tolist()})
    return metrics,mechanisms
def main():
    torch.set_num_threads(2);assert json.loads((OUT/'identity-training.json').read_text())['status']=='passed'
    selection=json.loads((OUT/'selection.json').read_text());lock=json.loads((OUT/'source-lock.json').read_text())
    for name,sha in lock['sha256'].items():assert digest(ROOT/name)==sha
    assert digest(OUT/'source-lock.json')==selection['source_lock_sha256']
    assert not (OUT/'evaluation-C.json').exists(),'Evaluation already exists; use verify.py'
    write(OUT/'pre-evaluation-seal.json',{'selection_sha256':digest(OUT/'selection.json'),'source_lock_sha256':digest(OUT/'source-lock.json'),'model_sha256':{m['path']:m['sha256'] for v in selection['selected'].values() for m in v['models']},'before_loading_calibration_or_evaluation_arrays':True})
    data=np.load(ROOT/'cache/oc-heads/features.npz');manifest=json.loads((OUT/'manifest.json').read_text());all_scores=[];began=time.monotonic()
    for seed in SEEDS:
        scores={key:[baseline(data[key+'_X'])] for key in ['cal','C','RV']}
        for name in IDS:
            entry=next(m for m in selection['selected'][name]['models'] if m['seed']==seed);path=ROOT/entry['path'];assert digest(path)==entry['sha256']
            artifact=joblib.load(path)
            for key in scores:scores[key].append(score(artifact,data[key+'_X']))
        scores={k:np.column_stack(v) for k,v in scores.items()}
        cuts=np.array([threshold(scores['cal'][:,j]) for j in range(9)])
        for key in ['cal','C','RV']:
            values=[r for r in manifest['samples'] if r['cohort']==key]
            for i,row in enumerate(values):all_scores.append({'id':row['id'],'role':key,'label':row['label'],'seed':seed,'scores':scores[key][i].tolist()})
        write(OUT/f'thresholds-{seed}.json',{'names':['B']+IDS,'values':cuts.tolist(),'ordinal':381,'strict_greater':True,'calibration_count':400})
        for key in ['C','RV']:
            y=data[key+'_y'];s=scores[key]
            if seed==SEEDS[0]:
                metrics,mechanisms=uncertainty(y,s,cuts)
                if key=='C':assert abs(metrics[0]['auroc']-.746640)<1e-12
                else:assert abs(metrics[0]['auroc']-.7565473145780051)<1e-12
                write(OUT/f'evaluation-{key}.json',{'run_id':'OC-H1','evidence':'historical_diagnostic_only','cohort':key,'counts':[int((y==i).sum()) for i in (0,1)],'primary_seed':seed,'metrics':metrics,'mechanisms':mechanisms,'family':8,'repeats':10000,'bootstrap_seed':20261104,'pauc_ci_coverage':.99375,'conditioning':'fixed trained head, selection and calibration'})
            write(OUT/f'seed-{seed}-{key}.json',calculate(y,s,cuts))
    path=OUT/'scores.jsonl';path.write_bytes(''.join(json.dumps(r)+'\n' for r in all_scores).encode())
    write(OUT/'evaluation-complete.json',{'scores_sha256':digest(path),'selection_sha256':digest(OUT/'selection.json'),'seconds':time.monotonic()-began,'status':'completed','independent_confirmation':False})
    print('HEAD_ALL_EVALUATIONS_COMPLETE',flush=True)
if __name__=='__main__':main()
