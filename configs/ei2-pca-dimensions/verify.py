"""Reload heads and transforms; verify point metrics and all weighted-AUC intervals."""
import importlib.util,json,sys,time
import joblib,numpy as np
from scipy.stats import rankdata
from run import ROOT,ORIGIN,PARENT,OUT,CACHE,NAMES,SEEDS,GRID,digest,write

def weighted_auc(y,s,counts):
 order=np.argsort(s,kind='stable');starts=np.r_[0,np.flatnonzero(np.diff(s[order]))+1]
 pos=np.add.reduceat(counts[:,order]*y[order],starts,axis=1);neg=np.add.reduceat(counts[:,order]*(1-y[order]),starts,axis=1)
 return (pos*(np.cumsum(neg,axis=1)-neg/2)).sum(1)/(pos.sum(1)*neg.sum(1))

def main():
 started=time.monotonic();source=json.loads((OUT/'source-lock.json').read_text());pre=json.loads((OUT/'pre-evaluation-seal.json').read_text());done=json.loads((OUT/'completion.json').read_text());e=json.loads((OUT/'evaluation.json').read_text())
 assert done['fits']==63 and digest(OUT/'evaluation.json')==done['evaluation_sha256'] and digest(OUT/'scores.npz')==done['scores_sha256']
 assert digest(OUT/'source-lock.json')==pre['source_lock']
 for name,sha in source['sources'].items():assert digest(ROOT/name)==sha
 assert digest(PARENT/'results/ei2-vocabulary/verification.json')==source['parent_verification']
 assert digest(PARENT/'results/ei2-vocabulary/manifest.json')==source['parent_manifest']
 spec=importlib.util.spec_from_file_location('point_verifier',ROOT/'configs/ei2-vocabulary/verify.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 rows=json.loads((OUT/'manifest.json').read_text())['samples'];z=np.load(PARENT/'cache/ei2-vocabulary/Z.npy',mmap_mode='r');f=np.load(PARENT/'cache/ei2-vocabulary/F.npy',mmap_mode='r')
 scores=np.load(OUT/'scores.npz');parent=np.load(PARENT/'results/ei2-vocabulary/scores.npz');basis=joblib.load(CACHE/'transforms.joblib');assert digest(CACHE/'transforms.joblib')==pre['transform']
 models={n:joblib.load(CACHE/(n+'.joblib')) for n in pre['models']}
 coords=json.loads((ORIGIN/'cache/oc4c-worktree/results/oc23/token-audit.json').read_text())['token_ids'];rhat=np.zeros(z.shape[-1],np.float32);rhat[coords]=1/np.sqrt(len(coords))
 search=json.loads((OUT/'search.json').read_text());maximum=0.;capacity={}
 for n,m in models.items():
  assert digest(CACHE/(n+'.joblib'))==pre['models'][n]
  records=[r for r in search if r['name']==n];assert len(records)==9 and all(max(r['n_iter'])<5000 for r in records)
  selected=max(GRID,key=lambda c:(np.mean([r['dev_auroc'] for r in records if r['C']==c]),-GRID.index(c)));assert m['C']==selected==pre['selected_C'][n]
  capacity[n]={'raw_dimension':len(m['mean']),'interface_dimension':512,'parameters_including_intercept':513,'rank_train':m['rank_train']}
 for role in ['train','dev','cal','C','RV','OR_cal','OR_test']:
  ix=np.array([i for i,r in enumerate(rows) if r['role']==role]);y=np.array([rows[i]['label'] for i in ix]);parts=[]
  for l,t in enumerate(basis):
   raw=np.asarray(z[ix,l],np.float32);u=(raw/np.maximum(np.linalg.norm(raw.astype(np.float64),axis=1),1e-12)[:,None]).astype(np.float32);res=u-f[ix,l,None].astype(np.float32)*rhat
   if role=='train':
    np.testing.assert_allclose(t.mean_,res.mean(0),atol=2e-8);np.testing.assert_allclose(t.components_@t.components_.T,np.eye(8),atol=2e-5)
   parts.append(t.transform(res))
  a=np.stack(parts,axis=1).astype(np.float32)
  for n,m in models.items():
   if n.startswith('Noise-'):
    k=int(n[6:]);extra=np.vstack([np.random.default_rng(np.random.SeedSequence([20261105,k,int(i)])).normal(size=14*k) for i in ix]).astype(np.float32)
   else:extra=a[:,:,:NAMES[n]].reshape(len(ix),-1)
   x=np.column_stack([f[ix],extra]).astype(np.float32)
   if role=='train':
    mean=x.mean(0,dtype=np.float64).astype(np.float32);scale=x.std(0,dtype=np.float64).astype(np.float32);scale[scale==0]=1
    np.testing.assert_array_equal(mean,m['mean']);np.testing.assert_array_equal(scale,m['scale'])
   x=(x-m['mean'])/m['scale'];x[:,:14]/=np.sqrt(14);x[:,14:]/=np.sqrt(56)
   for j,h in enumerate(m['models']):
    prediction=x@h.coef_[0,:x.shape[1]]+h.intercept_[0]
    if role=='train':continue
    if role=='dev':
     expected=next(r['dev_auroc'] for r in search if r['name']==n and r['C']==m['C'] and r['seed']==SEEDS[j]);assert abs(old.auc(y,prediction)-expected)<1e-12;continue
    delta=float(np.max(np.abs(prediction-scores[role+'__'+n][j])));maximum=max(maximum,delta);assert delta<1e-7,(role,n,delta)
  if role in ['train','dev']:continue
  for name,parent_name in [('old-K4','EI-2-3'),('B-F','B-F')]:np.testing.assert_array_equal(scores[role+'__'+name],parent[role+'__'+parent_name])
  np.testing.assert_array_equal(y,scores[role+'__y'])
  if role in ['cal','OR_cal']:continue
  for n,stats in e['cohorts'][role].items():
   if n=='comparisons':continue
   for j,m in enumerate(stats):
    s=scores[role+'__'+n][j];cut=float(np.partition(scores['cal__'+n][j],380)[380]);assert cut==m['threshold']==e['thresholds']['cal'][n][j]
    assert abs(old.auc(y,s)-m['auroc'])<1e-12 and abs(old.pauc(y,s)-m['pauc'])<1e-12
    assert m['fp']==int((s[y==0]>cut).sum()) and m['fpr']==float((s[y==0]>cut).mean()) and m['tpr']==float((s[y==1]>cut).mean())
    if role=='OR_test':
     cut=float(np.partition(scores['OR_cal__'+n][j],380)[380]);v=m['source_calibrated'];assert cut==v['threshold']==e['thresholds']['OR_cal'][n][j]
     assert v['fp']==int((s[y==0]>cut).sum()) and v['fpr']==float((s[y==0]>cut).mean()) and v['tpr']==float((s[y==1]>cut).mean())
  names=['K4','EI-2-3-1','EI-2-3-2','EI-2-3-3'];groups=[np.flatnonzero(y==j) for j in [0,1]];rng=np.random.default_rng(20261105);dist=np.empty((10000,4))
  for start in range(0,10000,100):
   draws=np.array([np.r_[rng.choice(groups[0],len(groups[0]),replace=True),rng.choice(groups[1],len(groups[1]),replace=True)] for _ in range(100)])
   counts=np.zeros((100,len(y)));np.add.at(counts,(np.arange(100)[:,None],draws),1)
   for j,n in enumerate(names):dist[start:start+100,j]=weighted_auc(y,scores[role+'__'+n][0],counts)
  for j,n in enumerate(names[1:],1):
   c=e['cohorts'][role]['comparisons'][n];actual=np.quantile(dist[:,j]-dist[:,0],[.05/6,1-.05/6]);np.testing.assert_allclose(actual,c['ci983333'],atol=1e-12,rtol=0)
   assert abs(c['delta_auroc']-(e['cohorts'][role][n][0]['auroc']-e['cohorts'][role]['K4'][0]['auroc']))<1e-12
  print('VERIFIED_METRICS_AND_ALL_INTERVALS',role,flush=True)
 write(OUT/'verification.json',{'passed':True,'evaluation_sha256':digest(OUT/'evaluation.json'),'scores_sha256':digest(OUT/'scores.npz'),'source_lock_sha256':digest(OUT/'source-lock.json'),'verify_code_sha256':digest(ROOT/'configs/ei2-pca-dimensions/verify.py'),'rows':len(rows),'fits':len(search),'prediction_max_difference':maximum,'old_parent_score_reproduction_max_difference':e['old_K4_reproduction_max_difference'],'all_primary_intervals_independently_recomputed':True,'capacity':capacity,'seconds':time.monotonic()-started});print('EI_PCA1_VERIFICATION_PASS',flush=True)

if __name__=='__main__':main()
