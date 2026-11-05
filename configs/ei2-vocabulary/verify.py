"""Reload frozen transforms/heads; independently rebuild inputs and all point metrics."""
import hashlib,json,platform,subprocess,time
import joblib,numpy as np
from scipy.stats import rankdata
from core import ROOT,ORIGIN,OUT,CACHE,SEEDS,GRID,digest,write

def auc(y,s):
 n=int(y.sum());m=len(y)-n
 return float((rankdata(s)[y==1].sum()-n*(n+1)/2)/(n*m))

def pauc(y,s):
 order=np.argsort(-s,kind='stable');ys=y[order];ss=s[order]
 ends=np.r_[np.flatnonzero(np.diff(ss)),len(ss)-1]
 tp=np.r_[0,np.cumsum(ys)[ends]]/y.sum();fp=np.r_[0,np.cumsum(1-ys)[ends]]/(len(y)-y.sum())
 stop=np.searchsorted(fp,.1,side='right');xp=np.r_[fp[:stop],.1];yp=np.r_[tp[:stop],np.interp(.1,fp,tp)]
 area=np.trapz(yp,xp)
 return float(.5*(1+(area-.005)/(.1-.005)))

def main():
 began=time.monotonic();m=json.loads((OUT/'manifest.json').read_text());rows=m['samples']
 source=json.loads((OUT/'source-lock.json').read_text());pre=json.loads((OUT/'pre-evaluation-seal.json').read_text())
 assert digest(OUT/'source-lock.json')==pre['source_lock'] and digest(OUT/'manifest.json')==source['manifest']
 for name,sha in source['sources'].items():assert digest(ROOT/name)==sha
 assert digest(ROOT/'docs/ei2-vocabulary-protocol-2026-11-04.md')==source['protocol']
 assert digest(CACHE/'transforms.joblib')==pre['transform']
 completion=json.loads((OUT/'completion.json').read_text());assert completion['new_training_runs']==108 and completion['full_endpoints_executed']
 for key,path in [('evaluation_sha256','evaluation.json'),('score_sha256','scores.npz')]:assert completion[key]==digest(OUT/path)
 z=np.load(CACHE/'Z.npy',mmap_mode='r');f=np.load(CACHE/'F.npy',mmap_mode='r')
 assert np.isfinite(f).all() and len(json.loads((CACHE/'checkpoint.json').read_text())['completed'])==len(rows)
 # Compare every original F (all cohorts) against independently existing caches.
 h1=np.load(ORIGIN/'cache/oc4-worktree/cache/oc-heads/features.npz');h2=np.load(ORIGIN/'cache/oc4c-worktree/cache/oc-confirmation/features.npz')
 differences={}
 for role in sorted({r['role'] for r in rows}):
  ix=[i for i,r in enumerate(rows) if r['role']==role];old=(h2[role[3:]+'_X'] if role.startswith('OR_') else h1[role+'_X'])
  delta=float(np.max(np.abs(f[ix]-old)));assert delta<=1e-7,(role,delta);differences[role]=delta
 transforms=joblib.load(CACHE/'transforms.joblib');seal=json.loads((OUT/'transform-seal.json').read_text())
 assert seal['fitted_on']=='train only'
 train=[i for i,r in enumerate(rows) if r['role']=='train']
 assert seal['train_ids_sha256']==hashlib.sha256('\n'.join(rows[i]['id'] for i in train).encode()).hexdigest()
 # Token IDs come from the original refusal audit, not a new semantic list.
 coords=json.loads((ORIGIN/'cache/oc4c-worktree/results/oc23/token-audit.json').read_text())['token_ids'];coords=sorted(set(coords))
 assert len(coords)==json.loads((OUT/'identity.json').read_text())['refusal_coordinate_count']
 rhat=np.zeros(z.shape[-1],np.float32);rhat[coords]=1/np.sqrt(len(coords))
 models={n:joblib.load(CACHE/(n+'.joblib')) for n in pre['models']}
 def check_scaling(values,model,sl):
  mean=values.mean(0,dtype=np.float64).astype(np.float32);scale=values.std(0,dtype=np.float64).astype(np.float32);scale[scale==0]=1
  np.testing.assert_allclose(model['mean'][sl],mean,atol=1e-7,rtol=1e-6)
  np.testing.assert_allclose(model['scale'][sl],scale,atol=1e-7,rtol=1e-6)
 for model in models.values():check_scaling(f[train].astype(np.float32),model,slice(0,14))
 for l,t in enumerate(transforms):
  assert t['coordinates'].tolist()==seal['coordinates'][l] and not set(coords)&set(t['coordinates'])
  arr=np.asarray(z[train,l],np.float32);u=(arr/np.linalg.norm(arr.astype(np.float64),axis=1)[:,None]).astype(np.float32)
  variance=u.var(0,dtype=np.float64);variance[coords]=-np.inf
  assert np.array_equal(t['coordinates'],np.lexsort((np.arange(z.shape[-1]),-variance))[:32])
  residual=u-f[train,l,None].astype(np.float32)*rhat
  np.testing.assert_allclose(t['pca'].mean_,residual.mean(0),atol=2e-8)
  values=[u[:,coords],u[:,t['coordinates']],t['pca'].transform(residual),np.log(np.linalg.norm(arr.astype(np.float64),axis=1)).astype(np.float32)[:,None],u,arr]
  for j,block in enumerate(values):check_scaling(block,models['EI-2-'+str(j+1)],slice(14+l*block.shape[1],14+(l+1)*block.shape[1]))
 for n,model in models.items():
  assert digest(CACHE/(n+'.joblib'))==pre['models'][n]
  search=[r for r in json.loads((OUT/'search.json').read_text()) if r['name']==n]
  assert len(search)==9 and set(r['seed'] for r in search)==set(SEEDS)
  best=max(GRID,key=lambda c:(np.mean([r['dev_auroc'] for r in search if r['C']==c]),-GRID.index(c)))
  assert best==model['C']==pre['selected_C'][n]
 scores=np.load(OUT/'scores.npz');evaluation=json.loads((OUT/'evaluation.json').read_text());maximum=0.;pca_batch_difference=0.
 for role in ['cal','C','RV','OR_cal','OR_test']:
  indices=np.array([i for i,r in enumerate(rows) if r['role']==role]);y=np.array([rows[i]['label'] for i in indices]);assert np.array_equal(y,scores[role+'__y'])
  # Match the frozen evaluator's cohort-size float32 PCA GEMM. Chunking the
  # feature rows changes BLAS rounding even with unchanged components/weights.
  pca_parts=[]
  for l,t in enumerate(transforms):
   raw=np.asarray(z[indices,l],np.float32);u=(raw/np.linalg.norm(raw.astype(np.float64),axis=1)[:,None]).astype(np.float32)
   pca_parts.append(t['pca'].transform(u-f[indices,l,None].astype(np.float32)*rhat))
  pca_full=np.column_stack(pca_parts)
  for start in range(0,len(indices),128):
   ix=indices[start:start+128];parts=[[] for _ in range(6)]
   for l,t in enumerate(transforms):
    raw=np.asarray(z[ix,l],np.float32);norm=np.linalg.norm(raw.astype(np.float64),axis=1);u=(raw/norm[:,None]).astype(np.float32)
    parts[0].append(u[:,coords]);parts[1].append(u[:,t['coordinates']]);parts[2].append(t['pca'].transform(u-f[ix,l,None].astype(np.float32)*rhat));parts[3].append(np.log(norm).astype(np.float32)[:,None]);parts[4].append(u);parts[5].append(raw)
   extra={'B-F':np.empty((len(ix),0),np.float32),'Fcopy':np.tile(f[ix],(1,20)).astype(np.float32)}
   for j in range(6):extra['EI-2-'+str(j+1)]=np.column_stack(parts[j]).astype(np.float32)
   pca_batch_difference=max(pca_batch_difference,float(np.max(np.abs(extra['EI-2-3']-pca_full[start:start+len(ix)]))))
   extra['EI-2-3']=pca_full[start:start+len(ix)]
   for j,q in enumerate([14*len(coords),448,56,14]):extra['Noise-'+str(j+1)]=np.vstack([np.random.default_rng(np.random.SeedSequence([20261104,j,int(i)])).normal(size=q) for i in ix]).astype(np.float32)
   for n,model in models.items():
    x=np.column_stack([f[ix],extra[n]]).astype(np.float32);x=(x-model['mean'])/model['scale'];x[:,:14]/=np.sqrt(14)
    if extra[n].shape[1]:x[:,14:]/=np.sqrt(extra[n].shape[1])
    for seed,head in enumerate(model['models']):
     predicted=x@head.coef_[0,:x.shape[1]]+head.intercept_[0];stored=scores[role+'__'+n][seed,start:start+len(ix)]
     delta=float(np.max(np.abs(predicted-stored)));maximum=max(maximum,delta);assert delta<1e-7,(role,n,delta)
   old=(f[ix,0]+f[ix,-1])/2+f[ix,1:-1].sum(1)
   np.testing.assert_allclose(old,scores[role+'__B-INT'][start:start+len(ix)],atol=1e-12)
  if role in ['cal','OR_cal']:continue
  for n,ms in evaluation['cohorts'][role].items():
   if n not in models and n!='B-INT':continue
   s=scores[role+'__'+n]
   if n=='B-INT':s=np.tile(s,(3,1))
   for j,metrics in enumerate(ms):
    cuts=scores['cal__'+n];cut=np.partition(cuts if n=='B-INT' else cuts[j],380)[380]
    assert cut==metrics['threshold']
    assert abs(auc(y,s[j])-metrics['auroc'])<1e-12
    assert abs(pauc(y,s[j])-metrics['pauc'])<1e-12
    assert int((s[j,y==0]>cut).sum())==metrics['fp']
    assert (s[j,y==0]>cut).mean()==metrics['fpr'] and (s[j,y==1]>cut).mean()==metrics['tpr']
    assert evaluation['thresholds'][n][j]==cut
    if role=='OR_test':
     calibration=scores['OR_cal__'+n];ocut=np.partition(calibration if n=='B-INT' else calibration[j],380)[380]
     source_point=metrics['source_calibrated']
     assert evaluation['OR_thresholds'][n][j]==source_point['threshold']==ocut
     assert source_point['fp']==int((s[j,y==0]>ocut).sum())
     assert source_point['fpr']==float((s[j,y==0]>ocut).mean()) and source_point['tpr']==float((s[j,y==1]>ocut).mean())
  print('VERIFIED',role,flush=True)
 capacity={}
 for name,model in models.items():
  capacity[name]={'weight_count_including_intercept':model['interface_dimension']+1,'nonconstant_columns':model['nonconstant_columns'],'train_rank':model['rank_train'],'rank_method':'float64 SVD of float32 design'}
  if model['rank_train'] is not None:continue
  start=time.monotonic();base=f[train].astype(np.float32);base=(base-model['mean'][:14])/model['scale'][:14]/np.sqrt(14);base=base.astype(np.float64);gram=base@base.T
  for l in range(14):
   block=np.asarray(z[train,l],np.float32)
   if name=='EI-2-5':block=(block/np.linalg.norm(block.astype(np.float64),axis=1)[:,None]).astype(np.float32)
   sl=slice(14+l*z.shape[-1],14+(l+1)*z.shape[-1])
   block=((block-model['mean'][sl])/model['scale'][sl]/np.sqrt(14*z.shape[-1])).astype(np.float64)
   gram+=block@block.T
  eig=np.linalg.eigvalsh(gram);tolerance=float(eig[-1]*len(train)*np.finfo(np.float64).eps)
  capacity[name].update(train_rank=int(np.count_nonzero(eig>tolerance)),rank_method='float64 train Gram eigenvalues',gram_eigenvalue_tolerance=tolerance,rank_seconds=time.monotonic()-start)
  print('CAPACITY_CHECKED',name,capacity[name],flush=True)
 result={'passed':True,'evaluation_sha256':digest(OUT/'evaluation.json'),'rows':len(rows),'all_F_cache_max_differences':differences,'reloaded_prediction_max_difference':maximum,'pca_chunk_vs_frozen_cohort_max_difference':pca_batch_difference,'independent_auc_pauc_and_working_points':True,'source_and_train_only_transform_checks':True,'capacity':capacity,'sha256':{str(p.relative_to(ROOT)):digest(p) for p in [CACHE/'F.npy',CACHE/'Z.npy',OUT/'scores.npz',ROOT/'configs/ei2-vocabulary/verify.py']},'seconds':time.monotonic()-began,'verification_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()}
 write(OUT/'verification.json',result);print('INDEPENDENT_VERIFICATION_PASS',result,flush=True)

if __name__=='__main__':main()
