"""Frozen pure OC rules; all inference is conditional on the fixed calibration."""
import hashlib,importlib.util,json,math,time
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('original_core',ROOT/'configs/original-positions/core.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
NAMES=['B','M','top4','trim2','window4']
IDS=['original-B','OC-1-M','OC-1-1','OC-1-2','OC-1-3']
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with Path(p).open(encoding='utf-8') as stream:return [json.loads(x) for x in stream if x.strip()]
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def aggregate(f):
    f=np.asarray(f,dtype=np.float64)
    if f.ndim!=2 or f.shape[1]!=32 or not np.isfinite(f).all():raise ValueError('finite exact original 32-layer F required')
    x=f[:,16:30];g=np.sort(x,axis=1)
    return np.column_stack([np.trapz(x,axis=1),x.mean(1),g[:,-4:].mean(1),g[:,2:-2].mean(1),np.stack([x[:,i:i+4].mean(1) for i in range(11)],axis=1).max(1)])
def calibrate(x):
    x=np.asarray(x);assert x.shape==(400,) and np.isfinite(x).all()
    return float(np.sort(x)[math.ceil(.95*(len(x)+1))-1])
def evaluate(rows,cut,run):
    s=aggregate([r['F'] for r in rows]);y=np.array([r['label'] for r in rows]);assert set(y)=={0,1}
    groups=[np.flatnonzero(y==k) for k in (0,1)];rng=np.random.default_rng(20261030)
    indices=np.column_stack([rng.choice(g,(10000,len(g))) for g in groups])
    auc=np.empty((10000,5));pauc=np.empty_like(auc)
    for start in range(0,10000,100):
        for j in range(5):auc[start:start+100,j],pauc[start:start+100,j]=old.bootstrap_metrics(y,s[:,j],indices[start:start+100])
    def ci(v,adjust=False):return np.quantile(v,[.00625,.99375] if adjust else [.025,.975]).tolist()
    metrics=[];pred=s>np.array(cut)
    for j in range(5):
        item={'name':NAMES[j],'direction_id':IDS[j],'auroc':float(roc_auc_score(y,s[:,j])),
              'pauc':float(roc_auc_score(y,s[:,j],max_fpr=.1)),'threshold':cut[j],
              'tpr':float(pred[y==1,j].mean()),'fpr':float(pred[y==0,j].mean())}
        if j:
            item.update(delta_auroc=float(roc_auc_score(y,s[:,j])-roc_auc_score(y,s[:,0])),auroc_ci95=ci(auc[:,j]-auc[:,0]),
                delta_pauc=float(roc_auc_score(y,s[:,j],max_fpr=.1)-roc_auc_score(y,s[:,0],max_fpr=.1)),pauc_ci9875=ci(pauc[:,j]-pauc[:,0],True))
            for label,name in [(0,'fpr'),(1,'tpr')]:
                sample=indices[:,:len(groups[0])] if label==0 else indices[:,len(groups[0]):]
                diff=pred[:,j].astype(float)-pred[:,0].astype(float)
                item['delta_'+name]=float(diff[y==label].mean());item[name+'_delta_ci95']=ci(diff[sample].mean(1))
            lo,hi=item['pauc_ci9875'];item['decision']='confirmed_improvement' if lo>0 else 'worse' if hi<0 else 'unconfirmed'
        metrics.append(item)
    mechanism=[]
    for j,k in [(2,1),(3,1),(4,2)]:
        mechanism.append({'comparison':NAMES[j]+' vs '+NAMES[k],'delta_pauc':metrics[j]['pauc']-metrics[k]['pauc'],
            'pauc_ci95_descriptive':ci(pauc[:,j]-pauc[:,k]),'delta_auroc':metrics[j]['auroc']-metrics[k]['auroc']})
    return {'run_id':run,'counts':[len(g) for g in groups],'metrics':metrics,'mechanism_descriptive':mechanism,
        'bootstrap':{'repeats':10000,'seed':20261030,'family':4,'primary':'standardized pAUC FPR<=0.1','primary_interval':.9875,
        'conditional':'fixed calibration and rules; no calibration population uncertainty'},'scores':s.tolist()}
def historical():
    out=ROOT/'results/pure-oc';p=ROOT/'results/original-positions/scores.jsonl'
    rows=read(p);ex=json.loads((p.parent/'extraction.json').read_text());assert digest(p)==ex['scores_sha256']
    manifest=json.loads((ROOT/'configs/original-positions/manifest.json').read_text());assert ex['manifest_sha256']==digest(ROOT/'configs/original-positions/manifest.json')
    assert [r['id'] for r in rows]==[r['id'] for r in manifest['samples']]
    f=np.array([np.asarray(r['F'])[:,-1] for r in rows]);s=aggregate(f)
    assert np.array_equal(s[:,0],np.array([r['score_default'][-1] for r in rows]))
    c=[i for i,r in enumerate(rows) if r['role']=='test_aegis_holdout'];q=[i for i,r in enumerate(rows) if r['role']=='calibration_aegis']
    assert len(q)==400 and all(rows[i]['label']==0 for i in q) and not set(rows[i]['id'] for i in c)&set(rows[i]['id'] for i in q)
    baseline=roc_auc_score([rows[i]['label'] for i in c],s[c,0]);assert abs(baseline-.746640)<1e-12
    cut=[calibrate(s[q,j]) for j in range(5)]
    identity={'status':'passed','C1000_original_auroc':baseline,'integral_max_difference':0.,'dtype':'float64 reduction of exact stored F',
        'layer_interval_0based_inclusive':[16,29],'old_scores_sha256':digest(p),'old_manifest_sha256':digest(ROOT/'configs/original-positions/manifest.json'),
        'upstream_identity_sha256':digest(ROOT/'logs/original-positions-identity.json'),'new_calibration_order_statistic':381,'old_order_statistic':380}
    write(out/'identity.json',identity)
    write(out/'thresholds.json',{'names':NAMES,'thresholds':cut,'n':400,'order_statistic':381,'alarm':'strictly greater',
        'calibration_ids':[rows[i]['id'] for i in q],'calibration_scores':s[q].tolist(),'scope':'historical calibration shared by all rules'})
    selected=[{**rows[i],'F':f[i].tolist()} for i in c]
    result=evaluate(selected,cut,'OC-A1-H');result.pop('scores');write(out/'historical-C.json',result)
    t=read(ROOT/'results/original-template-confirm/scores.jsonl')
    assert digest(ROOT/'results/original-template-confirm/scores.jsonl')==json.loads((ROOT/'results/original-template-confirm/extraction.json').read_text())['scores_sha256']
    t=[{**r,'F':np.asarray(r['F'])[:,-1].tolist()} for r in t]
    result=evaluate(t,cut,'OC-A1-H-ORP-T2');result.pop('scores');write(out/'historical-ORP-T2.json',result)
    score_rows=[]
    for batch,items in [('C',selected),('ORP-T2',t)]:
        for r,v in zip(items,aggregate([r['F'] for r in items])):score_rows.append({k:r[k] for k in ('id','label','prompt_sha256') if k in r}|{'batch':batch,'F':r['F'],'scores':v.tolist()})
    (out/'historical-scores.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in score_rows),encoding='utf-8')
    benchmark=np.tile(f[c],(10,1));timings=[]
    for _ in range(30):
        tick=time.perf_counter();aggregate(benchmark);timings.append((time.perf_counter()-tick)/len(benchmark))
    write(out/'cost.json',{'cpu_all5_seconds_per_input_median':float(np.median(timings)),'repeats':30,'rows_per_repeat':len(benchmark),'model_forwards_for_historical':0})
    print('IDENTITY_AND_HISTORICAL_COMPLETE',baseline,cut)
if __name__=='__main__':historical()
