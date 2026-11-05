"""Independent train-side reconstruction, model binding, points and cluster CIs."""
from core import *

def main():
 from sklearn.decomposition import PCA
 rows,private,roles,y=load();source=json.loads((OUT/'source-lock.json').read_text());pre=json.loads((OUT/'pre-evaluation-seal.json').read_text());complete=json.loads((OUT/'completion.json').read_text());e=json.loads((OUT/'evaluation.json').read_text());extraction=json.loads((OUT/'extraction.json').read_text());assert e['all_five_executed'] and e['lr_fits']==126
 for name,sha in source['sources'].items():assert digest(ROOT/name)==sha,name
 for name,key in [('evaluation.json','evaluation_sha256'),('scores.npz','scores_sha256')]:assert digest(OUT/name)==complete[key]
 for name,key in [('manifest.json','manifest_sha256'),('grouping.json','grouping_sha256')]:assert digest(OUT/name)==source[key]
 for path,key in [(ORIGIN/'cache/ei2-image-acquisition/acquisition.json','acquisition_sha256'),(ORIGIN/'configs/artifacts-lock.json','model_lock_sha256'),(ORIGIN/'cache/ei2-image-ocr/lock.json','ocr_lock_sha256'),(OUT/'preflight.json','preflight_sha256')]:assert digest(path)==source[key]
 assert digest(OUT/'source-lock.json')==pre['source_lock_sha256'] and digest(OUT/'search.json')==pre['search_sha256'];assert digest(CACHE/'transforms.joblib')==pre['transforms_sha256'];assert extraction['completed']==len(rows)==1270 and extraction['all_vision_tokens_validated'] and extraction['all_neutral_description_lengths_equal'];assert digest(CACHE/'checkpoint.json')==extraction['generated_descriptions_sha256'];checkpoint=json.loads((CACHE/'checkpoint.json').read_text());assert checkpoint['completed']==list(range(len(rows)))
 a=arrays(len(rows));allowed=sorted(np.r_[roles['train'],roles['dev']].tolist());assert pre['train_dev_indices']==allowed and pre['before_heldout_feature_access'];assert extraction['source_lock_sha256']==digest(OUT/'source-lock.json')
 for name,value in a.items():
  assert digest(CACHE/(name+'.npy'))==extraction['arrays_sha256'][name] and np.isfinite(value).all();assert hashlib.sha256(np.asarray(value[allowed]).tobytes()).hexdigest()==pre['train_dev_feature_sha256'][name]
 image_roles={}
 for row in rows:
  key=row['image_sha256'];assert key not in image_roles or image_roles[key]==row['role'];image_roles[key]=row['role']
  metadata=checkpoint['metadata'][str(rows.index(row))];description=checkpoint['descriptions'][str(rows.index(row))];assert metadata['id']==row['id'] and metadata['valid_image_tokens']>=576 and metadata['expanded_tokens']<=4096 and metadata['neutral_description_length_equal'];assert hashlib.sha256(description['caption'].encode()).hexdigest()==metadata['caption_sha256'] and hashlib.sha256(description['ocr'].encode()).hexdigest()==metadata['ocr_sha256']
 tr,dev=roles['train'],roles['dev'];t=joblib.load(CACHE/'transforms.joblib');np.testing.assert_array_equal(t['fit_indices'],tr);pca=PCA(n_components=14,svd_solver='full').fit(unit(a['vision'][tr]));np.testing.assert_allclose(pca.components_,t['pca'].components_,atol=1e-6,rtol=1e-6);np.testing.assert_array_equal(pca.mean_,t['pca'].mean_)
 def normalize(raw):
  z=np.asarray(raw,np.float32);return (z/np.maximum(np.sqrt(np.sum(z.astype(float)**2,axis=-1,keepdims=True)),1e-12)).astype(np.float32)
 for kind in ['roi','uniform','random']:
  normalized=normalize(a[kind][tr]);centers=np.stack([normalized[y[tr]==label].mean(0) for label in [0,1]]);centers=normalize(centers);np.testing.assert_array_equal(centers,t[kind])
 point=module('point_metrics',ROOT/'configs/ei2-vocabulary/verify.py');search=json.loads((OUT/'search.json').read_text());assert len(search)==126;aucs={};models={}
 for record in search:
  path=ROOT/record['path'];assert digest(path)==record['sha256'];entry=joblib.load(path);name=record['name'];f,extra=feature(name,tr,a,t);raw=np.column_stack([f,extra]).astype(np.float32);mean=raw.mean(0,dtype=np.float64).astype(np.float32);scale=raw.std(0,dtype=np.float64).astype(np.float32);scale[scale==0]=1;np.testing.assert_array_equal(mean,entry['mean']);np.testing.assert_array_equal(scale,entry['scale']);f,extra=feature(name,dev,a,t);x=pack(f,extra,mean,scale)[0];auc=point.auc(y[dev],entry['head'].decision_function(x));assert abs(auc-record['dev_auroc'])<1e-12 and max(entry['head'].n_iter_)<5000;assert entry['head'].solver=='liblinear' and entry['head'].C==record['C'];aucs[(name,record['C'],record['seed'])]=auc
 bound=0
 for name in NAMES:
  path=CACHE/(name+'.joblib');assert digest(path)==pre['models'][name];model=joblib.load(path);winner=max(enumerate(GRID),key=lambda p:(np.mean([aucs[(name,p[1],s)] for s in SEEDS]),-p[0]))[1];assert winner==model['C']==e['selected'][name]['C'];assert abs(np.mean([aucs[(name,winner,s)] for s in SEEDS])-model['dev_auroc_mean'])<1e-12 and model['dev_auroc_mean']==e['selected'][name]['dev_auroc_mean'];assert len(model['entries'])==3
  for j,entry in enumerate(model['entries']):
   record=next(r for r in search if (r['name'],r['C'],r['seed'])==(name,winner,SEEDS[j]));trial=joblib.load(ROOT/record['path'])
   for key in ['name','C','seed']:assert entry[key]==trial[key]
   for key in ['mean','scale']:assert entry[key].dtype==trial[key].dtype;np.testing.assert_array_equal(entry[key],trial[key])
   first,second=entry['head'],trial['head'];assert type(first) is type(second) and first.get_params()==second.get_params() and first.n_features_in_==second.n_features_in_
   for key in ['coef_','intercept_','classes_','n_iter_']:np.testing.assert_array_equal(getattr(first,key),getattr(second,key))
   bound+=1
  models[name]=model
 assert bound==42;scores=np.load(OUT/'scores.npz');maximum=0.;intervals=0
 for role in ['cal','test','task_test']:
  ix=roles[role];yy=y[ix];np.testing.assert_array_equal(scores[role+'__y'],yy)
  for name,model in models.items():
   f,extra=feature(name,ix,a,t)
   for j,entry in enumerate(model['entries']):
    x=pack(f,extra,entry['mean'],entry['scale'])[0];ss=entry['head'].decision_function(x);delta=float(np.max(np.abs(ss-scores[role+'__'+name][j])));assert delta<1e-7;maximum=max(maximum,delta)
    if role=='cal':continue
    cut=float(np.partition(scores['cal__'+name][j],77)[77]);m=e['cohorts'][role][name][j];assert cut==m['threshold']==e['thresholds'][name][j];assert abs(point.auc(yy,ss)-m['auroc'])<1e-12 and abs(point.pauc(yy,ss)-m['pauc'])<1e-12;assert m['tpr']==float((ss[yy==1]>cut).mean()) and m['fpr']==float((ss[yy==0]>cut).mean()) and m['fp']==int((ss[yy==0]>cut).sum())
  if role=='cal':continue
  groups=sorted({rows[int(i)]['group_id'] for i in ix});members=[np.flatnonzero(np.array([rows[int(i)]['group_id'] for i in ix])==g) for g in groups];rng=np.random.default_rng(20261105);dist=np.empty((10000,5))
  for b in range(10000):
   draw=np.concatenate([members[j] for j in rng.integers(len(groups),size=len(groups))]);baseline=point.auc(yy[draw],scores[role+'__'+BASE][0,draw])
   for j,name in enumerate(ARMS):dist[b,j]=point.auc(yy[draw],scores[role+'__'+name][0,draw])-baseline
  tail=.005 if role=='test' else .025
  for j,name in enumerate(ARMS):
   c=e['cohorts'][role]['comparisons'][name];np.testing.assert_allclose(np.quantile(dist[:,j],[tail,1-tail]),c['ci'],atol=1e-12,rtol=0);assert c['clusters']==len(groups) and c['primary']==(role=='test');assert abs(c['delta_auroc']-(e['cohorts'][role][name][0]['auroc']-e['cohorts'][role][BASE][0]['auroc']))<1e-12;intervals+=1
  print('IMAGE_POINTS_AND_CLUSTER_INTERVALS_VERIFIED',role,flush=True)
 from extract import Extractor
 x=Extractor();x.full_projection=json.loads((OUT/'preflight.json').read_text())['use_full_projection'];replays=[]
 for i in np.r_[roles['train'][:2],roles['test'][:2],roles['task_test'][:2]]:
  i=int(i);x.ocr_cache.pop(private[i]['image_sha256'],None);s,v,pools,meta,description=x.row(private[i]);difference=float(np.max(np.abs(s-a['signals'][i])));assert difference<=1e-7;np.testing.assert_allclose(v,a['vision'][i],atol=1e-6,rtol=1e-6)
  for j,kind in enumerate(['roi','uniform','random']):np.testing.assert_allclose(pools[:,j],a[kind][i],atol=1e-5,rtol=1e-5)
  old=checkpoint['metadata'][str(i)]
  for key in ['caption_sha256','ocr_sha256','input_ids_sha256','pixel_values_sha256','valid_image_tokens']:assert meta[key]==old[key],(i,key)
  replays.append({'id':rows[i]['id'],'role':rows[i]['role'],'all_nine_F_branches_max_difference':difference,'ocr_rerun':True,'caption_regenerated':True});print('REAL_IMAGE_REPLAY_VERIFIED',rows[i]['id'],flush=True)
 x.close();write(OUT/'verification.json',{'passed':True,'run_id':'EI-IM1','evaluation_sha256':digest(OUT/'evaluation.json'),'scores_sha256':digest(OUT/'scores.npz'),'source_lock_sha256':digest(OUT/'source-lock.json'),'verify_code_sha256':digest(ROOT/'configs/ei2-images/verify.py'),'trial_heads_reloaded':126,'selected_entries_bound':bound,'cluster_intervals_recomputed':intervals,'prediction_max_difference':maximum,'train_only_pca_and_centroids_recomputed':True,'real_image_replays':replays,'main_test_clusters':len({r['group_id'] for r in rows if r['role']=='test'}),'task_test_clusters':len({r['group_id'] for r in rows if r['role']=='task_test'})});print('EI_IM1_VERIFICATION_PASS',flush=True)
if __name__=='__main__':main()
