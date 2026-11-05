import importlib.util,json,os,subprocess,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import joblib,numpy as np
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[2]
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
u=module('uncapped_common',ROOT/'configs/ei2-pca-uncapped/common.py')
s=module('sparse_common',ROOT/'configs/ei2-pca-select/common.py')
digest,write,SEEDS,GRID=u.digest,u.write,u.SEEDS,u.GRID
ORIGIN=u.ORIGIN;PARENT=u.PARENT;BASE=ORIGIN/'cache/ei2-pca-uncapped-worktree'
OUT=ROOT/'results/ei2-pca-joint';CACHE=ROOT/'cache/ei2-pca-joint'
CV,POOL,GATE=['EI-2-3-4-'+str(i) for i in [1,2,3]];FAMILIES=[CV,POOL,GATE];LAMBDAS=[.0001,.001,.01]
CONTROLS={CV:['Q-CV','S-EN'],POOL:['Q-VAR','S-L1'],GATE:['Q-GATE','S-EN']}

def load_data():
 data=u.parent.load_data(check_hashes=True);proof=json.loads((BASE/'results/ei2-pca-uncapped/verification.json').read_text());assert proof['passed']
 for file,key in [('evaluation.json','evaluation_sha256'),('scores.npz','scores_sha256'),('source-lock.json','source_lock_sha256')]:assert digest(BASE/'results/ei2-pca-uncapped'/file)==proof[key]
 manifest=PARENT/'results/ei2-vocabulary/manifest.json';ps=json.loads((PARENT/'results/ei2-vocabulary/source-lock.json').read_text());pp=json.loads((PARENT/'results/ei2-vocabulary/pre-evaluation-seal.json').read_text());bs=json.loads((BASE/'results/ei2-pca-uncapped/source-lock.json').read_text())
 assert digest(manifest)==ps['manifest']==bs['parent_manifest'] and digest(PARENT/'results/ei2-vocabulary/source-lock.json')==pp['source_lock']
 pre=json.loads((BASE/'results/ei2-pca-uncapped/pre-evaluation-seal.json').read_text());assert digest(BASE/'cache/ei2-pca-uncapped/full-basis.joblib')==pre['basis']
 data['joint_basis']=joblib.load(BASE/'cache/ei2-pca-uncapped/full-basis.joblib');data['parent_cv']=json.loads((BASE/'results/ei2-pca-uncapped/cv-selection.json').read_text());return data

def full_indices(ranks):return [np.arange(int(k)) for k in ranks]

def pack(f,a,indices,mean=None,scale=None):
 assert len(indices)==14
 for values,ix in zip(a,indices):assert np.asarray(ix).ndim==1 and len(np.unique(ix))==len(ix) and np.all((np.asarray(ix)>=0)&(np.asarray(ix)<values.shape[1]))
 raw=np.column_stack([f]+[values[:,np.asarray(ix,int)] for values,ix in zip(a,indices)]).astype(np.float32)
 if mean is None:
  mean=raw.mean(0,dtype=np.float64).astype(np.float32);scale=raw.std(0,dtype=np.float64).astype(np.float32);scale[scale==0]=1
 x=(raw-mean)/scale;x[:,:14]/=np.sqrt(14);x[:,14:]/=np.sqrt(56)
 return np.pad(x,((0,0),(0,max(0,512-x.shape[1])))),mean,scale

def rank_directions(weight,ranks):
 result=[];offset=14
 for k in ranks:
  result.append(np.lexsort((np.arange(k),-np.abs(weight[offset:offset+k]))));offset+=k
 assert offset==len(weight);return result

def top_indices(order,counts):return [np.sort(row[:int(k)]) for row,k in zip(order,counts)]
def map_prefix(order,counts):return top_indices(order,counts)

def check_prefix_hardening(g,counts):
 np.testing.assert_array_equal(g>=.5,np.arange(g.shape[1])[None,:]<np.asarray(counts)[:,None])

def mask_indices(mask,ranks):
 result=[];offset=0
 for k in ranks:result.append(np.flatnonzero(mask[offset:offset+k]));offset+=k
 assert offset==len(mask);return result

def map_pool(pool,mask):
 result=[];offset=0
 for row in pool:result.append(np.asarray(row)[np.asarray(mask[offset:offset+len(row)],bool)]);offset+=len(row)
 assert offset==len(mask);return result

def selected_coverage(basis,indices):return [float(b['values'][np.asarray(ix,int)].sum()/b['total_energy']) for b,ix in zip(basis,indices)]

def gate_product_audit(a,p,g,ranks):
 assert a.dtype==p.dtype==g.dtype==np.float32 and a.shape==p.shape==g.shape
 valid=np.arange(a.shape[1])[None,:]<np.asarray(ranks)[:,None];idealp=expit(a.astype(float));eps=np.finfo(np.float32).eps;np.testing.assert_allclose(p,idealp,atol=eps,rtol=eps)
 reference=np.cumprod(p.astype(float),axis=1)*valid;ideal=np.cumprod(idealp,axis=1)*valid;j=np.arange(a.shape[1]);gamma=j*(eps/2)/(1-j*(eps/2));bound=gamma[None,:]*np.abs(reference)+np.finfo(np.float32).tiny*a.shape[1]
 error=np.abs(g-reference);assert np.all(error<=bound+1e-14)
 rounded_diff=(g>=.5)!=(reference>=.5);ideal_diff=(g>=.5)!=(ideal>=.5)
 assert np.all(np.abs(reference[rounded_diff]-.5)<=bound[rounded_diff]+1e-14)
 total_bound=bound+np.abs(reference-ideal);assert np.all(np.abs(ideal[ideal_diff]-.5)<=total_bound[ideal_diff]+1e-14)
 return {'max_rounded_product_error':float(error.max()),'max_ideal_product_error':float(np.max(np.abs(g-ideal))),'minimum_threshold_margin':float(np.min(np.abs(g[valid]-.5))),'rounded_hard_mask_mismatches':int(rounded_diff.sum()),'ideal_hard_mask_mismatches':int(ideal_diff.sum()),'ambiguous_ideal_coordinates':np.argwhere(ideal_diff).tolist()}

def replay_references(data,role,pa):
 pre=json.loads((BASE/'results/ei2-pca-uncapped/pre-evaluation-seal.json').read_text());stored=np.load(BASE/'results/ei2-pca-uncapped/scores.npz');ix=data['roles'][role];scores={};maximum=0.
 for name,key in [('Q-CV','EI-2-3-2-1'),('Q-VAR','EI-2-3-2-2'),('Q-GATE','EI-2-3-2-3'),('ALL','gate-zero-full')]:
  path=BASE/'cache/ei2-pca-uncapped'/(key+'.joblib');assert digest(path)==pre['models'][key];model=joblib.load(path)
  if name=='ALL':assert all(entry['counts']==[len(b['values']) for b in data['joint_basis']] for entry in model['entries'])
  values=np.array([entry['head'].decision_function(u.pack(data['f'][ix],pa,entry['counts'],entry['mean'],entry['scale'])[0]) for entry in model['entries']]);difference=float(np.max(np.abs(values-stored[role+'__'+key])));assert difference<1e-7;maximum=max(maximum,difference);scores[name]=values
 old,difference=u.parent.references(data,role,u.parent.project(data,ix));scores['B-F']=old['B-F'];return scores,max(maximum,difference)
