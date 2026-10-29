"""Matched training perturbations with immutable feature and scoring gates."""
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'configs/heads'))
from head_core import (digest,stream_hash,read_jsonl,write,make_folds,p_auc,bootstrap_metrics,
    conservative_threshold,pad_ids,center_kernel,span_features,fit_kernel,predict_kernel)
from processing_core import summarize,weights,unit,user_indices,ANCHORS
from head_core import sources as previous_sources
SEEDS=list(range(20270201,20270206))
GROUPS={'mean':[0],'dynamic':[1,2],'random':[3,4]}

def draws(rows):
    pool=[i for i,r in enumerate(rows) if r['role']=='pool']
    assert len({rows[i]['group_id'] for i in pool})==len(pool)
    bylabel={y:np.array([i for i in pool if rows[i]['label']==y]) for y in (0,1)}
    assert all(len(v)>=800 for v in bylabel.values())
    return [np.sort(np.r_[np.random.default_rng(seed).choice(bylabel[0],800,replace=False),np.random.default_rng(seed+100).choice(bylabel[1],800,replace=False)]) for seed in SEEDS]

def variants():return [{'name':h+'_'+f,'method':h,'family':f} for h in ('ridge','logistic') for f in ('B','mean','dynamic','random')]
def contrasts():return {h+'_'+label:(h+'_'+a,h+'_'+b) for h in ('ridge','logistic') for label,a,b in [('mean_vs_B','mean','B'),('dynamic_vs_mean','dynamic','mean'),('dynamic_vs_random','dynamic','random')]}
def params(method):return [1e4,1e3,1e2,10.,1.,.1,.01,.001,.0001] if method=='ridge' else [.0001,.001,.01,.1,1.,10.,1e2,1e3,1e4]
def summary_features(s,family):
    if family=='B':return np.zeros((len(s),0))
    x=np.asarray(s[:,:,GROUPS[family],:],np.float64)
    return x.reshape(len(x),-1)/np.sqrt(4*len(GROUPS[family]))

def sources():
    s=previous_sources();s.update({p.relative_to(ROOT).as_posix():digest(p) for p in (ROOT/'configs/stability').glob('*.py')});return s
def check_sources(s):
    for n,h in s.items():assert digest(ROOT/n)==h,n
def arrays():
    rows=json.loads((ROOT/'configs/stability/manifest.json').read_text())['samples']
    return rows,np.load(ROOT/'cache/stability/B.npy',mmap_mode='r'),np.load(ROOT/'cache/stability/S.npy',mmap_mode='r')
def base(a,ix):return np.asarray(a[1][ix],np.float64).reshape(len(ix),-1)/np.sqrt(32)

def verify_cache(actual=False):
    seal=json.loads((ROOT/'logs/stability-cache-seal.json').read_text());m=json.loads((ROOT/'configs/stability/manifest.json').read_text())
    assert seal['manifest_sha256']==digest(ROOT/'configs/stability/manifest.json')
    assert seal['next_index']==len(m['samples']) and seal['prefix_difference']==0. and seal['pilot_maximum_B_difference']<=1e-5 and seal['pilot_maximum_summary_difference']<=1e-5
    assert seal['pilot_rows']==32 and seal['peak_allocated_bytes']<22*1024**3
    assert seal['shapes']=={'B.npy':[len(m['samples']),32,4096],'S.npy':[len(m['samples']),4,5,4096]}
    for n,h in seal['sources'].items():
        assert hashlib.sha256(subprocess.check_output(['git','show',seal['commit']+':'+n],cwd=ROOT)).hexdigest()==h,n
    if actual:
        assert digest(ROOT/'data/stability/records.jsonl')==m['records_sha256']
        for n,h in seal['files'].items():assert stream_hash(ROOT/n)==h,n
        for n,shape in seal['shapes'].items():
            a=np.load(ROOT/'cache/stability'/n,mmap_mode='r');assert list(a.shape)==shape and str(a.dtype)=='float16'
    return seal

def verify_freeze(actual=False):
    out=ROOT/'results/stability';f=json.loads((out/'freeze.json').read_text());check_sources(f['sources']);verify_cache(actual)
    for n,h in f['dependencies'].items():assert digest(ROOT/n)==h,n
    for c in f['models']:assert digest(out/c['file'])==c['sha256']
    pre=json.loads((out/'pre-evaluation-seal.json').read_text());assert pre['freeze_sha256']==digest(out/'freeze.json')
    if actual:
        expected={**f['sources'],**f['dependencies'],'results/stability/freeze.json':pre['freeze_sha256'],**{'results/stability/'+c['file']:c['sha256'] for c in f['models']}}
        for n,h in expected.items():assert hashlib.sha256(subprocess.check_output(['git','show',pre['commit']+':'+n],cwd=ROOT)).hexdigest()==h,n
        subprocess.run(['git','merge-base','--is-ancestor',pre['commit'],'HEAD'],cwd=ROOT,check=True)
    return f
