"""Frozen score alignment and compact-support anchor corrections."""
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'configs/stability'))
from stability_core import (digest,stream_hash,write,read_jsonl,make_folds,p_auc,bootstrap_metrics,
    conservative_threshold,base,summary_features,fit_kernel,predict_kernel,ANCHORS,pad_ids,user_indices,summarize,weights,unit)
from stability_core import sources as old_sources,verify_cache as old_cache
FAMILIES=('B','mean','dynamic','random')

def candidates():
    result=[{'name':'B','mode':'raw','family':'B','alpha':0.}]
    for mode in ('global','gated'):
        for family in FAMILIES[1:]:
            settings=[(a,None) for a in (.0625,.125,.1875,.25,.375,.5,.625,.75,1.)] if mode=='global' else [(a,w) for a in (.25,.5,1.) for w in (.5,1.,2.)]
            result += [{'name':f'{mode}_{family}_{i}','mode':mode,'family':family,'alpha':a,**({'width':w} if w is not None else {})} for i,(a,w) in enumerate(settings)]
    return result

def fit_alignment(b,e,y):
    b=np.asarray(b,float);e=np.asarray(e,float);safe=np.asarray(y)==0;assert safe.sum()>1
    return {'b_mean':float(b[safe].mean()),'e_mean':float(e[safe].mean()),'b_sd':max(float(b[safe].std()),1e-12),'e_sd':max(float(e[safe].std()),1e-12),'tau':conservative_threshold(b[safe],.05)}

def fuse(b,e,c,h):
    b=np.asarray(b,float);e=np.asarray(e,float)
    if c['mode']=='raw':return e.copy() if c['family']!='B' else b.copy()
    aligned=(e-h['e_mean'])*h['b_sd']/h['e_sd']+h['b_mean']
    gate=np.maximum(0.,1.-np.abs(b-h['tau'])/(c['width']*h['b_sd'])) if c['mode']=='gated' else np.ones_like(b)
    return b+c['alpha']*gate*(aligned-b)

def select(cs):
    winners={mode+'_'+family:max([c for c in cs if c['mode']==mode and c['family']==family],key=lambda c:c['cv_mean'])['name'] for mode in ('global','gated') for family in FAMILIES[1:]}
    allowed=[c for c in cs if c['family']!='random'];best=max(allowed,key=lambda c:c['cv_mean']);b=cs[0];se=best['cv_std']/np.sqrt(10)
    return {'winners':winners,'raw_best':best['name'],'selected':b['name'] if b['cv_mean']>=best['cv_mean']-se else best['name'],'heuristic_se':se,'scope':'meta_train CV only; excludes random; B preferred within heuristic oneSE'}

def contrasts():return {'gated_dynamic_vs_B':('gated_dynamic','B'),'gated_vs_global_dynamic':('gated_dynamic','global_dynamic'),'gated_dynamic_vs_mean':('gated_dynamic','gated_mean'),'gated_dynamic_vs_random':('gated_dynamic','gated_random'),'global_dynamic_vs_B':('global_dynamic','B'),'gated_mean_vs_B':('gated_mean','B'),'gated_dynamic_vs_B_refit':('gated_dynamic','B_refit')}

def sources():
    s=old_sources();s.update({p.relative_to(ROOT).as_posix():digest(p) for p in (ROOT/'configs/anchored').glob('*.py')});return s
def check_sources(s):
    for n,h in s.items():assert digest(ROOT/n)==h,n

def arrays():
    rows=json.loads((ROOT/'configs/anchored/manifest.json').read_text())['samples']
    return rows,np.load(ROOT/'cache/anchored/B.npy',mmap_mode='r'),np.load(ROOT/'cache/anchored/S.npy',mmap_mode='r'),np.load(ROOT/'cache/stability/B.npy',mmap_mode='r'),np.load(ROOT/'cache/stability/S.npy',mmap_mode='r')
def matrices(a,ix):
    rows,b,s,ob,os=a
    v=[ob[rows[i]['stability_index']] if 'stability_index' in rows[i] else b[rows[i]['fresh_index']] for i in ix]
    summaries=[os[rows[i]['stability_index']] if 'stability_index' in rows[i] else s[rows[i]['fresh_index']] for i in ix]
    return np.asarray(v,np.float64).reshape(len(ix),-1)/np.sqrt(32),np.asarray(summaries,np.float16)

def verify_cache(actual=False):
    m=json.loads((ROOT/'configs/anchored/manifest.json').read_text());seal=json.loads((ROOT/'logs/anchored-cache-seal.json').read_text());prep=json.loads((ROOT/'logs/anchored-score-seal.json').read_text())
    assert seal['manifest_sha256']==prep['manifest_sha256']==digest(ROOT/'configs/anchored/manifest.json')
    assert seal['old_cache_seal_sha256']==digest(ROOT/'logs/stability-cache-seal.json')
    assert seal['artifact_lock_sha256']==digest(ROOT/'configs/artifacts-lock.json')
    count=sum('fresh_index' in r for r in m['samples']);assert seal['next_index']==count
    assert seal['pilot_rows']==32 and seal['pilot_maximum_B_difference']==0. and seal['pilot_maximum_summary_difference']==0. and seal['prefix_difference']==0. and seal['peak_allocated_bytes']<22*1024**3
    assert seal['shapes']=={'B.npy':[count,32,4096],'S.npy':[count,4,5,4096]}
    assert prep['old_score_rows_checked']==2200 and prep['maximum_old_score_difference']<1e-7
    for n,h in prep['model_shas'].items():assert digest(ROOT/n)==h,n
    for stage in (seal,prep):
        for n,h in stage['sources'].items():assert hashlib.sha256(subprocess.check_output(['git','show',stage['commit']+':'+n],cwd=ROOT)).hexdigest()==h,n
    if actual:
        old_cache(True);assert digest(ROOT/'data/anchored/records.jsonl')==m['records_sha256']
        for stage in (seal,prep):
            for n,h in stage['files'].items():assert stream_hash(ROOT/n)==h,n
        for n,shape in seal['shapes'].items():
            v=np.load(ROOT/'cache/anchored'/n,mmap_mode='r');assert list(v.shape)==shape and str(v.dtype)=='float16'
        scores=np.load(ROOT/'cache/anchored/base-scores.npy',mmap_mode='r');assert scores.shape==(len(m['samples']),4) and str(scores.dtype)=='float64'
    return seal

def verify_freeze(actual=False):
    out=ROOT/'results/anchored';f=json.loads((out/'freeze.json').read_text());check_sources(f['sources']);verify_cache(actual)
    for n,h in f['dependencies'].items():assert digest(ROOT/n)==h,n
    assert digest(out/'B_refit.joblib')==f['refit_sha256']
    pre=json.loads((out/'pre-evaluation-seal.json').read_text());assert pre['freeze_sha256']==digest(out/'freeze.json')
    if actual:
        expected={**f['sources'],**f['dependencies'],'results/anchored/freeze.json':pre['freeze_sha256'],'results/anchored/B_refit.joblib':f['refit_sha256']}
        for n,h in expected.items():assert hashlib.sha256(subprocess.check_output(['git','show',pre['commit']+':'+n],cwd=ROOT)).hexdigest()==h,n
        subprocess.run(['git','merge-base','--is-ancestor',pre['commit'],'HEAD'],cwd=ROOT,check=True)
    return f
