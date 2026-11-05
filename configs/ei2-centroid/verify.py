"""Read-only replay, independent centroid/metric construction and trial binding."""
from core import *

def main():
 from scipy.stats import rankdata
 rows,roles,f,y=data();train,dev=roles['train'],roles['dev'];source=json.loads((OUT/'source-lock.json').read_text());pre=json.loads((OUT/'pre-evaluation-seal.json').read_text());e=json.loads((OUT/'evaluation.json').read_text());done=json.loads((OUT/'completion.json').read_text())
 for name,sha in source['sources'].items():assert digest(ROOT/name)==sha
 for name,key in [('evaluation.json','evaluation_sha256'),('scores.npz','scores_sha256')]:assert digest(OUT/name)==done[key]
 assert digest(OUT/'source-lock.json')==pre['source_lock'] and digest(OUT/'search.json')==pre['search'];np.testing.assert_array_equal(pre['fit_indices'],train)
 assert source['parent_manifest']==digest(PARENT/'results/ei2-vocabulary/manifest.json') and source['F']==digest(PARENT/'cache/ei2-vocabulary/F.npy');assert json.loads((OUT/'manifest.json').read_text())['samples']==rows
 identity=json.loads((OUT/'extraction.json').read_text());h=np.load(CACHE/'H.npy',mmap_mode='r');fresh=np.load(CACHE/'forward-F.npy',mmap_mode='r');assert h.shape==(6582,14,4096) and h.dtype==np.float16 and np.isfinite(h).all();assert digest(CACHE/'H.npy')==identity['H_sha256'] and digest(CACHE/'forward-F.npy')==identity['forward_F_sha256'];assert sorted(json.loads((CACHE/'checkpoint.json').read_text())['completed'])==list(range(6582));assert float(np.max(np.abs(fresh-f)))==identity['F_max_difference']<=1e-7
 raw=h[train].astype(np.float32);normalized=raw/np.maximum(np.sqrt((raw*raw).sum(-1,keepdims=True)),1e-12);centers=np.stack([normalized[y[train]==label].mean(0) for label in [0,1]]);centers/=np.maximum(np.sqrt((centers*centers).sum(-1,keepdims=True)),1e-12);refs=np.load(CACHE/'centroids.npy');np.testing.assert_array_equal(centers,refs);assert digest(CACHE/'centroids.npy')==pre['centroids']
 def independent_extra(name,ix):
  if name=='B-F':return np.empty((len(ix),0),np.float32)
  if name=='Noise':return np.array([np.random.default_rng(np.random.SeedSequence([20261104,7,int(i)])).normal(size=14) for i in ix],np.float32)
  raw=h[ix].astype(np.float32);raw/=np.maximum(np.sqrt((raw*raw).sum(-1,keepdims=True)),1e-12);return (raw*(refs[1]-refs[0])[None,:,:]).sum(-1)
 def matrix(name,ix,mean,scale):
  z=np.column_stack([f[ix],independent_extra(name,ix)]).astype(np.float32);z=(z-mean)/scale;z[:,:14]/=np.sqrt(14)
  if name!='B-F':z[:,14:]/=np.sqrt(14)
  return np.pad(z,((0,0),(0,512-z.shape[1])))
 point=module('independent_point_metrics',ROOT/'configs/ei2-vocabulary/verify.py');search=json.loads((OUT/'search.json').read_text());assert len(search)==27;models={};trialaucs={};maximum=0.
 for record in search:
  p=ROOT/record['path'];assert digest(p)==record['sha256'];entry=joblib.load(p);name=record['name'];raw=np.column_stack([f[train],independent_extra(name,train)]).astype(np.float32);mean=raw.mean(0,dtype=np.float64).astype(np.float32);scale=raw.std(0,dtype=np.float64).astype(np.float32);scale[scale==0]=1
  # Different summation of cosine features is independently allowed only within
  # a small float32 bound; the exact sealed scaler is checked using original pack.
  exactmean=pack(f[train],extra(name,h,refs,train))[1];exactscale=pack(f[train],extra(name,h,refs,train))[2];np.testing.assert_array_equal(entry['mean'],exactmean);np.testing.assert_array_equal(entry['scale'],exactscale);np.testing.assert_allclose(mean,entry['mean'],atol=2e-7,rtol=2e-5);np.testing.assert_allclose(scale,entry['scale'],atol=2e-7,rtol=2e-5)
  exact=pack(f[dev],extra(name,h,refs,dev),entry['mean'],entry['scale'])[0];auc=point.auc(y[dev],entry['head'].decision_function(exact));assert abs(auc-record['dev_auroc'])<1e-12;assert entry['head'].C==record['C'] and entry['head'].solver=='liblinear' and max(entry['head'].n_iter_)<5000;trialaucs[(name,record['C'],record['seed'])]=auc
 for name in NAMES:
  path=CACHE/(name+'.joblib');assert digest(path)==pre['models'][name];m=joblib.load(path);winner=max(enumerate(GRID),key=lambda j:(np.mean([trialaucs[(name,j[1],s)] for s in SEEDS]),-j[0]))[1];assert m['C']==winner==e['selected'][name]['C'];assert abs(m['dev_auroc_mean']-np.mean([trialaucs[(name,winner,s)] for s in SEEDS]))<1e-12;assert m['dev_auroc_mean']==e['selected'][name]['dev_auroc_mean'];assert len(m['entries'])==3
  for j,entry in enumerate(m['entries']):
   record=next(r for r in search if (r['name'],r['C'],r['seed'])==(name,winner,SEEDS[j]));trial=joblib.load(ROOT/record['path'])
   for key in ['mean','scale']:assert entry[key].dtype==trial[key].dtype;np.testing.assert_array_equal(entry[key],trial[key])
   for key in ['name','C','seed']:assert entry[key]==trial[key]
   a,b=entry['head'],trial['head'];assert type(a) is type(b) and a.get_params()==b.get_params() and a.n_features_in_==b.n_features_in_
   for key in ['coef_','intercept_','classes_','n_iter_']:np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
  models[name]=m
 scores=np.load(OUT/'scores.npz');parentproof=json.loads((PARENT/'results/ei2-vocabulary/verification.json').read_text());assert digest(PARENT/'results/ei2-vocabulary/scores.npz')==parentproof['sha256']['results/ei2-vocabulary/scores.npz'];old=np.load(PARENT/'results/ei2-vocabulary/scores.npz');intervals=0;independent_difference=0.
 for role in ['cal','C','RV','OR_cal','OR_test']:
  ix=roles[role];yy=y[ix];np.testing.assert_array_equal(yy,scores[role+'__y']);np.testing.assert_allclose(old[role+'__B-F'],scores[role+'__B-F'],atol=1e-7,rtol=0)
  for name,m in models.items():
   for j,entry in enumerate(m['entries']):
    exact=pack(f[ix],extra(name,h,refs,ix),entry['mean'],entry['scale'])[0];ss=entry['head'].decision_function(exact);stored=scores[role+'__'+name][j];delta=float(np.max(np.abs(ss-stored)));assert delta<1e-7;maximum=max(maximum,delta)
    independent=entry['head'].decision_function(matrix(name,ix,entry['mean'],entry['scale']));diff=float(np.max(np.abs(independent-stored)));assert diff<1e-4;independent_difference=max(independent_difference,diff)
    if role not in ['C','RV','OR_test']:continue
    metric=e['cohorts'][role][name][j];cut=float(np.partition(scores['cal__'+name][j],380)[380]);assert cut==metric['threshold']==e['thresholds']['cal'][name][j];assert abs(point.auc(yy,ss)-metric['auroc'])<1e-12 and abs(point.pauc(yy,ss)-metric['pauc'])<1e-12;assert metric['tpr']==float((ss[yy==1]>cut).mean()) and metric['fpr']==float((ss[yy==0]>cut).mean()) and metric['fp']==int((ss[yy==0]>cut).sum())
    if role=='OR_test':
     cut=float(np.partition(scores['OR_cal__'+name][j],380)[380]);q=metric['source_calibrated'];assert q['threshold']==cut==e['thresholds']['OR_cal'][name][j] and q['tpr']==float((ss[yy==1]>cut).mean()) and q['fpr']==float((ss[yy==0]>cut).mean()) and q['fp']==int((ss[yy==0]>cut).sum())
  if role not in ['C','RV','OR_test']:continue
  groups=[np.flatnonzero(yy==label) for label in [0,1]];rng=np.random.default_rng(20261105);dist=[]
  for _ in range(10000):
   draw=np.r_[rng.choice(groups[0],len(groups[0]),replace=True),rng.choice(groups[1],len(groups[1]),replace=True)];dist.append(point.auc(yy[draw],scores[role+'__'+ARM][0,draw])-point.auc(yy[draw],scores[role+'__B-F'][0,draw]))
  np.testing.assert_allclose(np.quantile(dist,[.05/6,1-.05/6]),e['cohorts'][role]['comparison']['ci983333'],atol=1e-12,rtol=0);assert abs(e['cohorts'][role]['comparison']['delta_auroc']-(e['cohorts'][role][ARM][0]['auroc']-e['cohorts'][role]['B-F'][0]['auroc']))<1e-12;intervals+=1;print('VERIFIED_CENTROID',role,flush=True)
 write(OUT/'verification.json',{'passed':True,'run_id':'EI-C1','evaluation_sha256':digest(OUT/'evaluation.json'),'scores_sha256':digest(OUT/'scores.npz'),'source_lock_sha256':digest(OUT/'source-lock.json'),'verify_code_sha256':digest(ROOT/'configs/ei2-centroid/verify.py'),'trial_heads_reloaded':27,'selected_entries_bound':9,'centroids_train_only_recomputed':True,'primary_intervals_recomputed':intervals,'prediction_max_difference':maximum,'independent_cosine_prediction_max_difference':independent_difference,'all_6582_F_verified':True});print('EI_C1_VERIFICATION_PASS',flush=True)

if __name__=='__main__':main()
