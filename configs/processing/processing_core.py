"""Label-free user processing and finite additive-kernel controls."""
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'configs/routing'))
from routing_core import digest,stream_hash,read_jsonl,write,make_folds,p_auc,bootstrap_metrics,conservative_threshold,centered_predict,pad_ids
ANCHORS=[7,15,23,31]
FAMILIES={'mean':[0],'dynamic':[1,2],'processed':[1,2,3,4,5],'relations':[1,2,3,4,5],'random':[6],'duplicate_mean':[0]*5}

def unit(a):
    a=np.asarray(a,dtype=np.float64);return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)

def weights(values):
    a=np.asarray(values,dtype=np.float64);assert a.ndim==1 and len(a)>0 and np.isfinite(a).all()
    z=(a-a.mean())/max(float(a.std()),1e-12);z=np.clip(z,-5,5);z=np.exp(z-z.max());return z/z.sum()

def summarize(h,pre,attention,mlp,global_h,attention_weights,sample_id,layer):
    h,pre,attention,mlp=[np.asarray(x,dtype=np.float64) for x in (h,pre,attention,mlp)]
    assert h.shape==pre.shape==attention.shape==mlp.shape and len(h)>0
    aw=np.asarray(attention_weights,dtype=float);assert aw.shape==(len(h),) and (aw>=0).all()
    aw=aw/aw.sum() if aw.sum()>0 else np.ones(len(h))/len(h)
    delta=h-pre;rel=np.linalg.norm(delta,axis=1)/np.maximum(np.linalg.norm(pre,axis=1),1e-12);cw=weights(rel)
    seed=int(hashlib.sha256((sample_id+':processing-random:'+str(layer)).encode()).hexdigest()[:16],16)
    rw=weights(np.random.default_rng(seed).normal(size=len(h)));u=unit(h)
    pooled=np.stack([u.mean(0),aw@u,cw@u,cw@unit(delta),cw@unit(attention),cw@unit(mlp),rw@u])
    g=unit(global_h);norms=np.linalg.norm(pooled,axis=1);cos=(unit(pooled)*g).sum(1);distance=np.linalg.norm(pooled-g,axis=1)
    entropy=lambda w:float(-(w*np.log(np.maximum(w,1e-30))).sum()/np.log(max(2,len(w))))
    misc=[entropy(aw),entropy(cw),entropy(rw),rel.mean(),rel.std(),rel.max(),float(unit(aw)@unit(cw)),float(unit(pooled[4])@unit(pooled[5]))]
    relations=np.r_[np.column_stack([cos,distance,norms]).reshape(-1),misc]
    assert pooled.shape==(7,h.shape[1]) and relations.shape==(29,) and np.isfinite(relations).all()
    return pooled,relations

def feature_matrix(a,family):
    groups=FAMILIES[family];x=np.asarray(a[:,:,groups,:],np.float64)
    return x.reshape(len(x),-1)/np.sqrt(a.shape[1]*len(groups))

def relation_transform(raw,fit):
    mu=raw[fit].mean(0);sd=raw[fit].std(0);sd=np.where(sd>1e-12,sd,1.)
    allz=(raw-mu)/sd/np.sqrt(raw.shape[1]);return allz[fit],allz,mu,sd

def candidates():
    return [{'name':'B','family':'B','gamma':0.}]+[{'name':f'{f}_g{str(g).replace(".","p")}','family':f,'gamma':g} for f in FAMILIES for g in (.25,1.)]

def select(specs):
    eligible=[c for c in specs if c['family'] in ('B','mean','dynamic','processed','relations')]
    best=max(eligible,key=lambda c:c['cv_mean']);base=next(c for c in eligible if c['name']=='B')
    se=best['cv_std']/np.sqrt(10);chosen=base if base['cv_mean']>=best['cv_mean']-se else best
    return {'selected':chosen['name'],'raw_best':best['name'],'heuristic_se':se,'scope':'train-only repeatedCV pAUC, B preferred within oneSE heuristic'}

def sources():
    paths=list((ROOT/'configs/processing').glob('*.py'))+[ROOT/p for p in ('configs/routing/routing_core.py','configs/positions/position_core.py','configs/swarm/search.py','configs/swarm/core.py','configs/fusion/fusion_core.py','configs/highdim/features.py','configs/highdim/ridge.py','configs/highdim/seal.py','configs/signal-selection/experiment_core.py','configs/signal-selection/evaluate_frozen.py')]
    return {p.relative_to(ROOT).as_posix():digest(p) for p in paths}

def check_sources(snapshot):
    for name,sha in snapshot.items():assert digest(ROOT/name)==sha,name

def arrays():
    rows=json.loads((ROOT/'configs/processing/manifest.json').read_text())['samples']
    cache=ROOT/'cache/processing'
    return rows,np.load(cache/'summaries.npy',mmap_mode='r'),np.load(cache/'relations.npy',mmap_mode='r'),np.load(cache/'fresh_B.npy',mmap_mode='r'),np.load(ROOT/'cache/routing/fixed.npy',mmap_mode='r')

def baseline(a,indices):
    rows,_,_,fresh,old=a
    result=[old[rows[i]['baseline_index'],:,3,:] if 'baseline_index' in rows[i] else fresh[rows[i]['fresh_index']] for i in indices]
    return np.asarray(result,np.float64).reshape(len(indices),-1)/np.sqrt(32)

def verify_cache(actual=True):
    p=ROOT/'logs/processing-cache-seal.json';s=json.loads(p.read_text());m=json.loads((ROOT/'configs/processing/manifest.json').read_text());meta=s['metadata'];n=len(m['samples'])
    assert s['manifest_sha256']==meta['manifest_sha256']==digest(ROOT/'configs/processing/manifest.json')
    assert s['state']['next_index']==n and s['state']['old_B_checked']==4960 and s['state']['maximum_old_B_difference']<=1e-5
    assert s['state']['prefix_difference']==0.
    assert meta['shapes']=={'summaries.npy':[n,4,7,4096],'relations.npy':[n,116],'fresh_B.npy':[m['fresh_class_count']*2,32,4096]}
    for name,sha in s['extraction_sources'].items():
        assert hashlib.sha256(subprocess.check_output(['git','show',s['extraction_commit']+':'+name],cwd=ROOT)).hexdigest()==sha,name
    previous=json.loads((ROOT/'logs/routing-cache-seal.json').read_text());assert s['files']['cache/routing/fixed.npy']==previous['files']['cache/routing/fixed.npy']
    if actual:
        assert digest(ROOT/'data/processing/records.jsonl')==m['records_sha256']
        assert json.loads((ROOT/'cache/processing/metadata.json').read_text())==meta
        assert json.loads((ROOT/'cache/processing/checkpoint.json').read_text())==s['state']
        for name,sha in s['files'].items():assert stream_hash(ROOT/name)==sha,name
        for name,shape in meta['shapes'].items():
            x=np.load(ROOT/'cache/processing'/name,mmap_mode='r');assert list(x.shape)==shape and str(x.dtype)==('float32' if name=='relations.npy' else 'float16')
    return s

def verify_freeze(actual=True):
    out=ROOT/'results/processing';f=json.loads((out/'freeze.json').read_text());check_sources(f['sources'])
    for name,sha in f['dependencies'].items():assert digest(ROOT/name)==sha,name
    for c in f['candidates']:assert digest(out/c['file'])==c['sha256']
    pre=json.loads((out/'pre-evaluation-seal.json').read_text());assert pre['freeze_sha256']==digest(out/'freeze.json')
    verify_cache(actual)
    if actual:
        expected={'results/processing/freeze.json':pre['freeze_sha256'],**f['sources'],**f['dependencies'],**{'results/processing/'+c['file']:c['sha256'] for c in f['candidates']}}
        for name,sha in expected.items():assert hashlib.sha256(subprocess.check_output(['git','show',pre['commit']+':'+name],cwd=ROOT)).hexdigest()==sha,name
        subprocess.run(['git','merge-base','--is-ancestor',pre['commit'],'HEAD'],cwd=ROOT,check=True)
    return f
