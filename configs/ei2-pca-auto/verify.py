import importlib.util,json,time
import joblib,numpy as np
from common import *

def weighted_auc(y,s,counts):
 order=np.argsort(s,kind='stable');starts=np.r_[0,np.flatnonzero(np.diff(s[order]))+1]
 pos=np.add.reduceat(counts[:,order]*y[order],starts,axis=1);neg=np.add.reduceat(counts[:,order]*(1-y[order]),starts,axis=1)
 return (pos*(np.cumsum(neg,axis=1)-neg/2)).sum(1)/(pos.sum(1)*neg.sum(1))

def scaling_matches(f,a,counts,artifact):
 indices=np.concatenate([np.arange(l*32,l*32+k) for l,k in enumerate(counts)])
 x=np.column_stack([f,a.reshape(len(a),448)[:,indices]]).astype(np.float32)
 mean=x.mean(0,dtype=np.float64).astype(np.float32);scale=x.std(0,dtype=np.float64).astype(np.float32);scale[scale==0]=1
 np.testing.assert_array_equal(mean,artifact['mean']);np.testing.assert_array_equal(scale,artifact['scale'])

def main():
 started=time.monotonic();data=load_data();rows=data['rows'];roles=data['roles'];train=roles['train'];dev=roles['dev'];y=np.array([r['label'] for r in rows])
 source=json.loads((OUT/'source-lock.json').read_text());pre=json.loads((OUT/'pre-evaluation-seal.json').read_text());done=json.loads((OUT/'completion.json').read_text());e=json.loads((OUT/'evaluation.json').read_text())
 assert done['lr_fits']==360 and done['gate_fits']==12
 assert digest(OUT/'evaluation.json')==done['evaluation_sha256'] and digest(OUT/'scores.npz')==done['scores_sha256'] and digest(OUT/'source-lock.json')==pre['source_lock']
 for name,sha in source['sources'].items():assert digest(ROOT/name)==sha
 assert digest(BASE/'results/ei2-pca-extended/verification.json')==source['parent_verification']
 assert digest(BASE/'cache/ei2-pca-extended/transforms.joblib')==source['parent_basis']==pre['parent_basis']
 assert digest(OUT/'cv-selection.json')==pre['cv_selection'] and digest(OUT/'gate-training.json')==pre['gate_training']
 spec=importlib.util.spec_from_file_location('point_verifier',ROOT/'configs/ei2-vocabulary/verify.py');point=importlib.util.module_from_spec(spec);spec.loader.exec_module(point)
 cv=json.loads((OUT/'cv-selection.json').read_text());cv_array=np.zeros((3,32,3));seen=[]
 from sklearn.model_selection import StratifiedKFold
 expected_folds=list(StratifiedKFold(3,shuffle=True,random_state=SEEDS[0]).split(train,y[train]))
 for fold in cv['folds']:
  j=fold['fold'];ti=np.array(fold['train_indices']);vi=np.array(fold['validation_indices']);et,ev=expected_folds[j]
  np.testing.assert_array_equal(ti,train[et]);np.testing.assert_array_equal(vi,train[ev]);assert not set(ti)&set(vi) and set(ti)|set(vi)==set(train);seen.extend(vi.tolist())
  path=CACHE/f'fold-{j}-basis.joblib';assert digest(path)==fold['basis_sha256'];basis=joblib.load(path)
  for l,pca in enumerate(basis):
   raw=np.asarray(data['z'][ti,l],np.float32);u=(raw/np.maximum(np.linalg.norm(raw.astype(np.float64),axis=1),1e-12)[:,None]).astype(np.float32)
   res=u-data['f'][ti,l,None].astype(np.float32)*data['rhat'];np.testing.assert_allclose(pca.mean_,res.mean(0),atol=2e-8);np.testing.assert_allclose(pca.components_@pca.components_.T,np.eye(32),atol=2e-5)
  ta,va=project(data,ti,basis),project(data,vi,basis)
  for record in [r for r in cv['records'] if r['fold']==j]:
   path=ROOT/record['path'];assert digest(path)==record['sha256'];artifact=joblib.load(path);assert max(record['n_iter'])<5000
   scaling_matches(data['f'][ti],ta,artifact['counts'],artifact);x,_,_=pack(data['f'][vi],va,artifact['counts'],artifact['mean'],artifact['scale'])
   auc=point.auc(y[vi],artifact['head'].decision_function(x));assert abs(auc-record['auroc'])<1e-12
   cv_array[j,record['k']-1,GRID.index(record['C'])]=auc
  print('VERIFIED_CV_FOLD',j,flush=True)
 assert sorted(seen)==sorted(train.tolist()) and len(cv['records'])==288
 np.testing.assert_allclose(cv_array,cv['scores'],atol=1e-12,rtol=0)
 # Independently apply the frozen one-standard-error preference.
 means=cv_array.mean(0);best=sorted(np.ndindex(means.shape),key=lambda p:(-means[p],p[0],p[1]))[0];se=float(np.std(cv_array[:,best[0],best[1]],ddof=1)/np.sqrt(3));threshold=float(means[best]-se)
 allowed=[p for p in np.ndindex(means.shape) if means[p]>=threshold-1e-12];chosen=sorted(allowed,key=lambda p:(p[0],-means[p],p[1]))[0]
 assert cv['selection']['k']==chosen[0]+1 and cv['selection']['C']==GRID[chosen[1]] and abs(cv['selection']['threshold']-threshold)<1e-12
 models={n:joblib.load(CACHE/(n+'.joblib')) for n in pre['models']};search=json.loads((OUT/'final-search.json').read_text());assert len(search)==72
 for record in search:assert digest(ROOT/record['path'])==record['sha256'] and max(record['n_iter'])<5000
 gates=json.loads((OUT/'gate-training.json').read_text());assert len(gates)==12
 for record in gates:
  path=ROOT/record['path'];assert digest(path)==record['sha256'];gate=joblib.load(path);g=prefix_gates(gate['a']);np.testing.assert_allclose(g,gate['soft_gates'],atol=2e-6,rtol=0)
  assert gate['steps']==STEPS and gate['threshold']==.5 and (g>=.5).sum(1).tolist()==gate['counts']==record['counts']
  assert np.all(np.diff(g,axis=1)<=1e-12) and np.isfinite(gate['loss_start_end']).all()
 for family,m in models.items():
  assert digest(CACHE/(family+'.joblib'))==pre['models'][family]
  candidates={}
  for record in [r for r in search if r['family']==family]:candidates.setdefault((record['variant'],record['C']),[]).append(record)
  choices=[]
  for (variant,c),records in candidates.items():
   assert sorted(r['seed'] for r in records)==SEEDS
   choices.append((float(np.mean([r['dev_auroc'] for r in records])),float(np.mean([sum(r['counts']) for r in records])),variant,c))
  best=max(range(len(choices)),key=lambda j:(choices[j][0],-choices[j][1],-j));expected=choices[best]
  assert expected[2]==m['variant']['key'] and expected[3]==m['C'] and abs(expected[0]-m['dev_auroc_mean'])<1e-12
  assert e['selected'][family]['counts_by_seed']==[entry['counts'] for entry in m['entries']]
  for j,entry in enumerate(m['entries']):
   if family==CV:assert entry['counts']==[cv['selection']['k']]*14
   elif family==VAR:
    counts=[];coverage=[];capped=[]
    for pca in data['basis']:
     cumulative=np.cumsum(pca.explained_variance_ratio_,dtype=np.float64);k=next((i+1 for i,value in enumerate(cumulative) if value>=m['variant']['tau']),32);counts.append(k);coverage.append(float(cumulative[k-1]));capped.append(bool(cumulative[-1]<m['variant']['tau']))
    assert counts==entry['counts']==m['variant']['counts'][j] and capped==m['variant']['capped'];np.testing.assert_allclose(coverage,m['variant']['coverage'],atol=1e-12)
   else:
    path=ROOT/m['variant']['gates'][j]['path'];assert digest(path)==m['variant']['gates'][j]['sha256'];assert joblib.load(path)['counts']==entry['counts']
 scores=np.load(OUT/'scores.npz');maximum=0.;refmax=0.;a_train=None;capacity={}
 for role in ['train','dev','cal','C','RV','OR_cal','OR_test']:
  ix=roles[role];pa=project(data,ix)
  if role=='train':a_train=pa
  if role not in ['train','dev']:
   values,difference=references(data,role,pa);refmax=max(refmax,difference)
   for name,ss in values.items():np.testing.assert_allclose(ss,scores[role+'__'+name],atol=1e-7,rtol=0)
   np.testing.assert_array_equal(y[ix],scores[role+'__y'])
  for family,m in models.items():
   for j,entry in enumerate(m['entries']):
    if role=='train':scaling_matches(data['f'][ix],pa,entry['counts'],entry)
    x,_,_=pack(data['f'][ix],pa,entry['counts'],entry['mean'],entry['scale']);prediction=entry['head'].decision_function(x)
    if role=='train':
     rank=int(np.linalg.matrix_rank(x.astype(np.float64)));assert rank==entry['rank_train']
     capacity.setdefault(family,[]).append({'seed':entry['seed'],'raw_dimension':14+sum(entry['counts']),'interface_dimension':512,'rank_train':rank,'parameters_including_intercept':513});continue
    if role=='dev':
     assert abs(point.auc(y[ix],prediction)-next(r['dev_auroc'] for r in search if r['family']==family and r['variant']==entry['variant'] and r['C']==entry['C'] and r['seed']==entry['seed']))<1e-12;continue
    delta=float(np.max(np.abs(prediction-scores[role+'__'+family][j])));maximum=max(maximum,delta);assert delta<1e-7
  if role not in ['C','RV','OR_test']:continue
  yy=y[ix]
  for name,stats in e['cohorts'][role].items():
   if name=='comparisons':continue
   for j,m in enumerate(stats):
    s=scores[role+'__'+name][j];cut=float(np.partition(scores['cal__'+name][j],380)[380]);assert cut==m['threshold']==e['thresholds']['cal'][name][j]
    assert abs(point.auc(yy,s)-m['auroc'])<1e-12 and abs(point.pauc(yy,s)-m['pauc'])<1e-12
    assert m['tpr']==float((s[yy==1]>cut).mean()) and m['fpr']==float((s[yy==0]>cut).mean()) and m['fp']==int((s[yy==0]>cut).sum())
    if role=='OR_test':
     cut=float(np.partition(scores['OR_cal__'+name][j],380)[380]);p=m['source_calibrated'];assert cut==p['threshold']==e['thresholds']['OR_cal'][name][j]
     assert p['tpr']==float((s[yy==1]>cut).mean()) and p['fpr']==float((s[yy==0]>cut).mean()) and p['fp']==int((s[yy==0]>cut).sum())
  groups=[np.flatnonzero(yy==j) for j in [0,1]];rng=np.random.default_rng(20261105);dist=np.empty((10000,4));names=['K8',CV,VAR,GATE]
  for start in range(0,10000,100):
   draws=np.array([np.r_[rng.choice(groups[0],len(groups[0]),replace=True),rng.choice(groups[1],len(groups[1]),replace=True)] for _ in range(100)]);counts=np.zeros((100,len(yy)));np.add.at(counts,(np.arange(100)[:,None],draws),1)
   for j,name in enumerate(names):dist[start:start+100,j]=weighted_auc(yy,scores[role+'__'+name][0],counts)
  for j,name in enumerate(names[1:],1):
   c=e['cohorts'][role]['comparisons'][name];np.testing.assert_allclose(np.quantile(dist[:,j]-dist[:,0],[.05/6,1-.05/6]),c['ci983333'],atol=1e-12,rtol=0)
   assert abs(c['delta_auroc']-(e['cohorts'][role][name][0]['auroc']-e['cohorts'][role]['K8'][0]['auroc']))<1e-12
  print('VERIFIED_ALL_METRICS_AND_INTERVALS',role,flush=True)
 assert maximum<1e-7 and refmax<1e-7
 write(OUT/'verification.json',{'passed':True,'evaluation_sha256':digest(OUT/'evaluation.json'),'scores_sha256':digest(OUT/'scores.npz'),'verify_code_sha256':digest(ROOT/'configs/ei2-pca-auto/verify.py'),'source_lock_sha256':digest(OUT/'source-lock.json'),'cv_fold_fits_reloaded':288,'selected_final_heads_reloaded':12,'gate_states_verified':12,'prediction_max_difference':maximum,'reference_reproduction_max_difference':refmax,'all_primary_intervals_independently_recomputed':True,'capacity':capacity,'seconds':time.monotonic()-started});print('EI_PA1_VERIFICATION_PASS',flush=True)

if __name__=='__main__':main()
