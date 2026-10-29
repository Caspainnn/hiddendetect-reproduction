"""Full-feature detector heads; exact sample-space computation and frozen gates."""
import hashlib,json,subprocess,sys,time,warnings
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.exceptions import ConvergenceWarning
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'configs/processing'))
from processing_core import digest,stream_hash,read_jsonl,write,make_folds,p_auc,bootstrap_metrics,conservative_threshold,pad_ids
from processing_core import sources as previous_sources

def grid():
    out=[]
    for m in ('ridge','logistic','svm'):
        values=[1e4,1e3,1e2,10.,1.,.1,.01,.001,.0001] if m=='ridge' else [.0001,.001,.01,.1,1.,10.,1e2,1e3,1e4]
        out += [{'name':m+'_'+str(i),'method':m,'value':v} for i,v in enumerate(values)]
    out += [{'name':f'rbf_{i}_{j}','method':'rbf','value':v,'factor':g} for i,v in enumerate([100.,1.,.01]) for j,g in enumerate([.25,1.,4.])]
    return out

def center_kernel(k):
    mu=k.mean(0);grand=float(mu.mean());return k-mu[None,:]-mu[:,None]+grand,mu,grand

def center_cross(k,mu,grand):return k-k.mean(1)[:,None]-mu[None,:]+grand

def squared_distances(k,left,right):return np.maximum(left[:,None]+right[None,:]-2*k,0.)

def span_features(k,cuda=False):
    if cuda:
        import torch
        v,u=torch.linalg.eigh(torch.as_tensor(k,dtype=torch.float64,device='cuda'));v=v.cpu().numpy();u=u.cpu().numpy()
    else:v,u=np.linalg.eigh(k)
    cutoff=max(float(v.max()),1.)*1e-12;keep=v>cutoff;v=v[keep];u=u[:,keep]
    z=u*np.sqrt(v)[None,:];basis=u/np.sqrt(v)[None,:]
    error=float(np.linalg.norm(z@z.T-k)/max(np.linalg.norm(k),1e-12));assert error<1e-9,error
    return z,basis,int(len(v)),error

def span_cross(k,basis):return k@basis

def fit_kernel(k,y,c,d=None,span=None,cuda=False):
    began=time.monotonic();m=c['method'];median=None
    if m=='rbf':
        assert d is not None;median=max(float(np.median(d[np.triu_indices(len(d),1)])),1e-12);raw=np.exp(-c['factor']*d/median)
    else:raw=k
    kc,mu,grand=center_kernel(raw);h={'method':m,'params':c,'kernel_mean':mu,'kernel_grand':grand,'median':median}
    if m in ('ridge','rbf'):
        offset=float((y*2-1).mean());target=y*2-1-offset
        if cuda:
            import torch
            kg=torch.as_tensor(kc.copy(),device='cuda');kg.diagonal().add_(c['value']);beta=torch.linalg.solve(kg,torch.as_tensor(target,dtype=torch.float64,device='cuda')).cpu().numpy()
        else:beta=np.linalg.solve(kc+c['value']*np.eye(len(k)),target)
        h.update(beta=beta,intercept=offset)
    elif m=='logistic':
        z,basis,rank,error=span if span is not None else span_features(kc,cuda)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            clf=LogisticRegression(C=c['value'],penalty='l2',solver='lbfgs',tol=1e-9,max_iter=5000).fit(z,y)
        assert not any(issubclass(w.category,ConvergenceWarning) for w in caught),'Logistic convergence failure'
        h.update(beta=basis@clf.coef_[0],intercept=float(clf.intercept_[0]),rank=rank,gram_reconstruction_error=error,iterations=int(clf.n_iter_[0]))
    else:
        clf=SVC(kernel='precomputed',C=c['value'],tol=1e-7,max_iter=-1,probability=False,cache_size=512).fit(kc,y)
        assert clf.fit_status_==0
        beta=np.zeros(len(k));beta[clf.support_]=clf.dual_coef_[0]
        h.update(beta=beta,intercept=float(clf.intercept_[0]),support_count=len(clf.support_),iterations=np.asarray(getattr(clf,'n_iter_',[])).tolist())
    h['fit_seconds']=time.monotonic()-began;return h

def predict_kernel(h,k,d=None):
    raw=np.exp(-h['params']['factor']*d/h['median']) if h['method']=='rbf' else k
    return center_cross(raw,h['kernel_mean'],h['kernel_grand'])@h['beta']+h['intercept']

def selection(candidates):
    winners={m:max([c for c in candidates if c['method']==m],key=lambda c:c['cv_mean'])['name'] for m in ('ridge','logistic','svm','rbf')}
    byname={c['name']:c for c in candidates};best=max(winners.values(),key=lambda n:byname[n]['cv_mean']);se=byname[best]['cv_std']/np.sqrt(10)
    chosen=winners['ridge'] if byname[winners['ridge']]['cv_mean']>=byname[best]['cv_mean']-se else best
    return {'winners':winners,'selected':chosen,'raw_best':best,'heuristic_se':se,'scope':'trainCV pAUC only; deterministic listed-order ties; tuned ridge preferred within oneSE'}

def arrays():
    m=json.loads((ROOT/'configs/heads/manifest.json').read_text());return m['samples'],np.load(ROOT/'cache/heads/fresh_B.npy',mmap_mode='r'),np.load(ROOT/'cache/routing/fixed.npy',mmap_mode='r')

def matrix(a,ix):
    rows,fresh,old=a;v=[old[rows[i]['baseline_index'],:,3,:] if 'baseline_index' in rows[i] else fresh[rows[i]['fresh_index']] for i in ix]
    return np.asarray(v,np.float64).reshape(len(ix),-1)/np.sqrt(32)

def sources():
    result=previous_sources();result.update({p.relative_to(ROOT).as_posix():digest(p) for p in (ROOT/'configs/heads').glob('*.py')});return result

def check_sources(s):
    for n,h in s.items():assert digest(ROOT/n)==h,n

def verify_cache(actual=True):
    s=json.loads((ROOT/'logs/heads-cache-seal.json').read_text());m=json.loads((ROOT/'configs/heads/manifest.json').read_text());meta=s['metadata'];state=s['state']
    assert s['manifest_sha256']==meta['manifest_sha256']==digest(ROOT/'configs/heads/manifest.json')
    assert state['next_index']==m['fresh_class_count']*2 and state['prefix_difference']==0.
    assert s['pilot']['rows']==32 and s['pilot']['maximum_old_B_difference']<=1e-5 and s['pilot']['prefix_difference']==0.
    assert meta['shape']==[m['fresh_class_count']*2,32,4096] and meta['dtype']=='float16'
    assert meta['artifact_lock_sha256']==digest(ROOT/'configs/artifacts-lock.json') and meta['fixed_length']==512
    for n,h in meta['extraction_sources'].items():assert hashlib.sha256(subprocess.check_output(['git','show',meta['extraction_commit']+':'+n],cwd=ROOT)).hexdigest()==h,n
    old=json.loads((ROOT/'logs/routing-cache-seal.json').read_text());assert s['files']['cache/routing/fixed.npy']==old['files']['cache/routing/fixed.npy']
    if actual:
        assert digest(ROOT/'data/heads/records.jsonl')==m['records_sha256']
        assert json.loads((ROOT/'cache/heads/metadata.json').read_text())==meta
        assert json.loads((ROOT/'cache/heads/checkpoint.json').read_text())==state
        for n,h in s['files'].items():assert stream_hash(ROOT/n)==h,n
    return s

def verify_freeze(actual=True):
    out=ROOT/'results/heads';f=json.loads((out/'freeze.json').read_text());check_sources(f['sources'])
    for n,h in f['dependencies'].items():assert digest(ROOT/n)==h,n
    for c in f['candidates']:assert digest(out/c['file'])==c['sha256']
    pre=json.loads((out/'pre-evaluation-seal.json').read_text());assert pre['freeze_sha256']==digest(out/'freeze.json');verify_cache(actual)
    if actual:
        expected={'results/heads/freeze.json':pre['freeze_sha256'],**f['sources'],**f['dependencies'],**{'results/heads/'+c['file']:c['sha256'] for c in f['candidates']}}
        for n,h in expected.items():assert hashlib.sha256(subprocess.check_output(['git','show',pre['commit']+':'+n],cwd=ROOT)).hexdigest()==h,n
        subprocess.run(['git','merge-base','--is-ancestor',pre['commit'],'HEAD'],cwd=ROOT,check=True)
    return f
