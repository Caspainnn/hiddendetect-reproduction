"""One immutable public benchmark and train-only anchored feature contract."""
import hashlib,json,sys,subprocess
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'configs/fusion'))
from fusion_core import LAYOUT,digest
from provenance import verify_cache_binding,verify_sources
from detectors import make_matrix,columns_for_mask,raw_fit,raw_scores,detector_scores
from ridge import raw_matrix,ridge_scores

def p_auc(y,score):return float(roc_auc_score(y,score,max_fpr=.1))
def bootstrap_metrics(y,score,b0,b1):
    """Exact weighted ROC/AP for stratified resamples; ties share one ROC threshold."""
    y=np.asarray(y);score=np.asarray(score);order=np.argsort(-score,kind='stable');starts=np.r_[0,np.flatnonzero(np.diff(score[order]))+1]
    weights=np.zeros((len(b0),len(y)),dtype=np.float64);row=np.arange(len(b0))[:,None]
    np.add.at(weights,(row,b0),1);np.add.at(weights,(row,b1),1)
    positive=np.add.reduceat((weights*y)[:,order],starts,axis=1);negative=np.add.reduceat((weights*(1-y))[:,order],starts,axis=1)
    tp=positive.cumsum(1);fp=negative.cumsum(1);tpr=np.column_stack([np.zeros(len(b0)),tp/len(b1[0])]);fpr=np.column_stack([np.zeros(len(b0)),fp/len(b0[0])])
    width=np.diff(fpr,axis=1);auc=(width*(tpr[:,1:]+tpr[:,:-1])/2).sum(1)
    precision=np.divide(tp,tp+fp,out=np.zeros_like(tp),where=tp+fp>0)
    ap=(np.diff(tpr,axis=1)*precision).sum(1)
    clipped=np.clip(.1-fpr[:,:-1],0,width)
    right=tpr[:,:-1]+np.divide(clipped,width,out=np.zeros_like(width),where=width>0)*np.diff(tpr,axis=1)
    area=(clipped*(tpr[:,:-1]+right)/2).sum(1);partial=.5*(1+(area-.005)/.095)
    return {'auroc':auc,'average_precision':ap,'partial_auroc':partial}
def deduplicate(rows):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from experiment_core import normalize_prompt
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):
        a,b=find(i),find(j);parent[max(a,b)]=min(a,b)
    texts=[normalize_prompt(x['prompt']) for x in rows];exact={}
    for i,t in enumerate(texts):
        if t in exact:union(i,exact[t])
        else:exact[t]=i
    matrix=TfidfVectorizer(analyzer='char',ngram_range=(3,5),max_features=65536,dtype=np.float32).fit_transform(texts)
    for start in range(0,len(rows),128):
        pairs=(matrix[start:start+128]@matrix.T).tocoo()
        take=(pairs.data>=.92)&(start+pairs.row<pairs.col)
        for i,j in zip(start+pairs.row[take],pairs.col[take]):union(int(i),int(j))
    kept=[];removed=[]
    for i,x in enumerate(rows):
        root=find(i)
        if root==i:kept.append(x)
        else:removed.append({'id':x['id'],'role':x['role'],'duplicate_of':rows[root]['id']})
    return kept,removed
def source_snapshot():
    names=['core.py','search.py','train.py','evaluate.py','verify.py','build_confirmation.py','extract.py','seal_cache.py','check_probes.py']
    paths=['configs/swarm/'+name for name in names]
    paths+=list(json.loads((ROOT/'results/fusion/freeze.json').read_text())['source_sha256'])
    return {name:digest(ROOT/name) for name in sorted(set(paths))}

def benchmark():
    path=ROOT/'configs/benchmarks/v1.json'
    manifests={name:digest(ROOT/'configs'/name/'manifest.json') for name in ('highdim','fusion')}
    rows=json.loads((ROOT/'configs/fusion/manifest.json').read_text())['samples']
    record={'version':'v1','status':'previously observed fixed public comparison benchmark; not untouched confirmation',
        'manifests_sha256':manifests,'samples':[{'id':x['id'],'group_id':x['group_id'],'role':x['role'],'label':x['label'],'index':x['index']} for x in rows if x['role']!='probe_train'],
        'train':'highdim1600; dev600; all future methods keep these exact groups for comparison',
        'thresholds':'Aegis safe400 global; Toxic non-toxic400 source; strict >95% empirical order; never test-retuned'}
    if path.exists():assert json.loads(path.read_text())==record
    else:path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(record,indent=2)+'\n')
    return record

def cache(name):return {n:np.load(ROOT/'cache'/name/(n+'.npy'),mmap_mode='r') for n in LAYOUT}
def standardized(cache,ix):
    refs=np.load(ROOT/'results/highdim/centroids.npy')
    with np.load(ROOT/'results/highdim/scaler.npz') as s: mean=s['mean'];scale=s['scale']
    return (make_matrix(cache,ix,refs,True)-mean)/scale
def joined(base,z,mask,groups,mean,scale):
    cols=columns_for_mask(mask,groups) if np.any(mask) else np.array([],int)
    return np.column_stack([(base-mean)/scale,z[:,cols]]).astype(np.float32)

def verify_freeze(out,actual_caches=False):
    f=json.loads((out/'freeze.json').read_text());verify_sources(f['source_sha256'])
    pre=json.loads((out/'pre-evaluation-seal.json').read_text())
    assert pre['freeze_sha256']==digest(out/'freeze.json')
    assert pre['confirmation_manifest_sha256']==digest(ROOT/'configs/swarm/manifest.json')
    assert pre['confirmation_cache_seal_sha256']==digest(ROOT/'logs/swarm-cache-seal.json')
    assert digest(ROOT/'configs/benchmarks/v1.json')==f['benchmark_sha256']
    contract=json.loads((ROOT/'configs/benchmarks/v1.json').read_text())
    for name,sha in contract['manifests_sha256'].items():assert digest(ROOT/'configs'/name/'manifest.json')==sha
    for name,sha in f['dependencies_sha256'].items():assert digest(ROOT/name)==sha,name
    for c in f['candidates']:
        if 'file' in c:assert digest(out/c['file'])==c['sha256']
    assert digest(out/'search.json')==f['search_sha256']
    if actual_caches:
        committed={'results/swarm/freeze.json':pre['freeze_sha256'],**f['source_sha256'],**f['dependencies_sha256']}
        for c in f['candidates']:
            if 'file' in c:committed['results/swarm/'+c['file']]=c['sha256']
        for name,sha in committed.items():
            assert hashlib.sha256(subprocess.check_output(['git','show',pre['commit']+':'+name],cwd=ROOT)).hexdigest()==sha,name
        subprocess.run(['git','merge-base','--is-ancestor',pre['commit'],'HEAD'],cwd=ROOT,check=True)
        assert digest(ROOT/'cache/swarm/seal.json')==pre['confirmation_cache_seal_sha256']
        for name in ('highdim','fusion','swarm'):
            verify_cache_binding(ROOT/'cache'/name,ROOT/'configs'/name/'manifest.json')
    return f

if __name__=='__main__': print('FIXED_BENCHMARK',len(benchmark()['samples']))
