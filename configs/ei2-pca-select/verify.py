from common import *

def main():
 started=time.monotonic();data=parent.load_data(check_hashes=True);roles=data['roles'];train,dev=roles['train'],roles['dev'];y=np.array([r['label'] for r in data['rows']])
 source=json.loads((OUT/'source-lock.json').read_text());pre=json.loads((OUT/'pre-evaluation-seal.json').read_text());done=json.loads((OUT/'completion.json').read_text());e=json.loads((OUT/'evaluation.json').read_text())
 assert done['lr_fits']==90 and done['selector_fits']==22 and e['numbering_version']==3
 assert done['evaluation_sha256']==digest(OUT/'evaluation.json') and done['scores_sha256']==digest(OUT/'scores.npz') and pre['source_lock']==digest(OUT/'source-lock.json')
 for name,sha in source['sources'].items():assert digest(ROOT/name)==sha
 assert source['parent_manifest']==digest(PARENT/'results/ei2-vocabulary/manifest.json')
 assert json.loads((OUT/'manifest.json').read_text())['samples']==data['rows']
 assert source['parent_verification']==digest(BASE/'results/ei2-pca-extended/verification.json') and source['parent_basis']==digest(BASE/'cache/ei2-pca-extended/transforms.joblib')
 for file,key in [('final-search.json','final_search'),('selector-training.json','selectors'),('stability.json','stability')]:assert digest(OUT/file)==pre[key]
 spec=importlib.util.spec_from_file_location('point_verifier',ROOT/'configs/ei2-vocabulary/verify.py');point=importlib.util.module_from_spec(spec);spec.loader.exec_module(point)
 spec=importlib.util.spec_from_file_location('weighted_verifier',ROOT/'configs/ei2-pca-auto/verify.py');weighted=importlib.util.module_from_spec(spec);spec.loader.exec_module(weighted)
 a={r:parent.project(data,roles[r]) for r in ['train','dev']};sx=pack(data['f'][train],a['train'],np.ones((14,32),bool))[0][:,:462].astype(np.float64)
 selectors=json.loads((OUT/'selector-training.json').read_text());stability=json.loads((OUT/'stability.json').read_text());assert len(selectors)==22 and len(stability)==15
 rng=np.random.default_rng(20261106);classes=[np.flatnonzero(y[train]==j) for j in [0,1]];draws=[np.r_[rng.choice(classes[0],len(classes[0]),replace=True),rng.choice(classes[1],len(classes[1]),replace=True)] for _ in range(5)]
 maximum_kkt=0.;maximum_map=0.;covdiff=0.;ard_limited=[]
 states={}
 for record in selectors:
  path=ROOT/record['path'];assert digest(path)==record['sha256'];state=joblib.load(path);states[record['key']]=state
  b=record['bootstrap_index'];ix=np.arange(len(train)) if b is None else draws[b];x=sx[ix];yy=y[train][ix]
  if 'rho' in state:
   w=state['weight'];n=len(yy);gradient=x.T@(expit(x@w+state['bias'])-yy)/n+w/(10*n);gradient[14:]+=state['lambda']*(1-state['rho'])*w[14:];nz=np.abs(w[14:])>1e-8
   tail=np.where(nz,np.abs(gradient[14:]+state['lambda']*state['rho']*np.sign(w[14:])),np.maximum(np.abs(gradient[14:])-state['lambda']*state['rho'],0))
   kkt=max(np.max(np.abs(gradient[:14])),abs(np.mean(expit(x@w+state['bias'])-yy)),np.max(tail,initial=0));assert kkt<=2e-6 and abs(kkt-state['kkt'])<1e-12;maximum_kkt=max(maximum_kkt,kkt)
   np.testing.assert_array_equal(state['mask'],np.abs(w[14:])>1e-8)
  else:
   active=np.r_[state['active'],True];precision=np.r_[state['alpha'],1e-6][active];xx=np.column_stack([x,np.ones(len(x))])[:,active];w=np.r_[state['weight'],state['bias']][active];probability=expit(xx@w)
   gradient=xx.T@(probability-yy)+precision*w;err=float(np.max(np.abs(gradient)));assert err<=2e-5 and abs(err-state['map_gradient'])<1e-10;maximum_map=max(maximum_map,err)
   h=xx.T@((probability*(1-probability))[:,None]*xx)+np.diag(precision);diag=np.diag(np.linalg.inv(h));error=float(np.max(np.abs(diag-state['covariance_diagonal'][active])));assert error<1e-5;covdiff=max(covdiff,error)
   assert np.all(state['alpha'][:14]==.1) and state['active'][:14].all()
   if not state['outer_converged']:ard_limited.append(record['key'])
  print('VERIFIED_SELECTOR',record['key'],flush=True)
 models={n:joblib.load(CACHE/(n+'.joblib')) for n in pre['models']};search=json.loads((OUT/'final-search.json').read_text());assert len(search)==90
 maximum=0.;trial_scores={}
 for record in search:
  path=ROOT/record['path'];assert digest(path)==record['sha256'];entry=joblib.load(path);assert max(record['n_iter'])<5000
  _,mean,scale=pack(data['f'][train],a['train'],entry['mask']);np.testing.assert_array_equal(mean,entry['mean']);np.testing.assert_array_equal(scale,entry['scale'])
  d=pack(data['f'][dev],a['dev'],entry['mask'],entry['mean'],entry['scale'])[0];auc=point.auc(y[dev],entry['head'].decision_function(d));assert abs(auc-record['dev_auroc'])<1e-12
  trial_scores.setdefault(record['family'],{}).setdefault((record['variant'],record['C']),[]).append(record)
 for name,model in models.items():
  assert digest(CACHE/(name+'.joblib'))==pre['models'][name];mask=model['variant']['mask'];summary=e['selected'][name];assert summary['counts']==mask.sum(1).tolist() and summary['directions_mean']==int(mask.sum())
  assert summary['direction_indices_1based_by_layer']==[(np.flatnonzero(row)+1).tolist() for row in mask] and summary['is_prefix']==bool(np.array_equal(mask,prefix(mask)))
  if name in FAMILIES:
   choices=[]
   for (variant,c),records in trial_scores[name].items():
    assert sorted(r['seed'] for r in records)==SEEDS;choices.append((float(np.mean([r['dev_auroc'] for r in records])),records[0]['directions'],variant,c))
   winner=max(enumerate(choices),key=lambda j:(j[1][0],-j[1][1],-j[0]))[1];assert winner[2]==model['variant']['key'] and winner[3]==model['C'] and abs(winner[0]-model['dev_auroc_mean'])<1e-12
   state=states[model['variant']['selector']['key']];expected=state['mask'] if 'rho' in state else (state['active'][14:] & (state['alpha'][14:]<=model['variant']['alpha_cut']));np.testing.assert_array_equal(mask.ravel(),expected)
   matched=models[name+'-prefix'];assert matched['C']==model['C'];np.testing.assert_array_equal(matched['variant']['mask'],prefix(mask))
   for record in [r for r in stability if r['family']==name]:
    b=record['bootstrap'];np.testing.assert_array_equal(record['indices'],draws[b]);state=states[record['selector']['key']];actual=state['mask'] if 'rho' in state else (state['active'][14:] & (state['alpha'][14:]<=model['variant']['alpha_cut']));actual=actual.reshape(14,32)
    np.testing.assert_array_equal(actual,record['mask']);assert record['counts']==actual.sum(1).tolist() and record['directions']==actual.sum()
    union=np.logical_or(actual,mask).sum();jac=float(np.logical_and(actual,mask).sum()/union) if union else 1.;assert jac==record['jaccard']
  for entry in model['entries']:
   np.testing.assert_array_equal(entry['mask'],mask);assert entry['C']==model['C'];x=pack(data['f'][train],a['train'],mask,entry['mean'],entry['scale'])[0];assert int(np.linalg.matrix_rank(x.astype(np.float64)))==entry['rank_train']
 scores=np.load(OUT/'scores.npz');refmax=0.
 for role in ['cal','C','RV','OR_cal','OR_test']:
  ix=roles[role];pa=parent.project(data,ix);refs,delta=parent.references(data,role,pa);refmax=max(refmax,delta);np.testing.assert_array_equal(scores[role+'__y'],y[ix])
  for name,ss in refs.items():np.testing.assert_allclose(ss,scores[role+'__'+name],atol=1e-7,rtol=0)
  for name,model in models.items():
   for j,entry in enumerate(model['entries']):
    x=pack(data['f'][ix],pa,entry['mask'],entry['mean'],entry['scale'])[0];delta=float(np.max(np.abs(entry['head'].decision_function(x)-scores[role+'__'+name][j])));assert delta<1e-7;maximum=max(maximum,delta)
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
  names=['K32']+FAMILIES+[n+'-prefix' for n in FAMILIES];groups=[np.flatnonzero(yy==j) for j in [0,1]];rng=np.random.default_rng(20261105);dist=np.empty((10000,7))
  for start in range(0,10000,100):
   draws=np.array([np.r_[rng.choice(groups[0],len(groups[0]),replace=True),rng.choice(groups[1],len(groups[1]),replace=True)] for _ in range(100)]);counts=np.zeros((100,len(yy)));np.add.at(counts,(np.arange(100)[:,None],draws),1)
   for j,name in enumerate(names):dist[start:start+100,j]=weighted.weighted_auc(yy,scores[role+'__'+name][0],counts)
  for j,name in enumerate(FAMILIES):
   for control,index in [('K32',0),(name+'-prefix',4+j)]:
    comparison=e['cohorts'][role]['comparisons'][name][control];np.testing.assert_allclose(comparison['ci991667'],np.quantile(dist[:,1+j]-dist[:,index],[.05/12,1-.05/12]),atol=1e-12,rtol=0)
    assert abs(comparison['delta_auroc']-(e['cohorts'][role][name][0]['auroc']-e['cohorts'][role][control][0]['auroc']))<1e-12
  print('VERIFIED_METRICS_AND_SIX_INTERVALS',role,flush=True)
 write(OUT/'verification.json',{'passed':True,'evaluation_sha256':digest(OUT/'evaluation.json'),'scores_sha256':digest(OUT/'scores.npz'),'source_lock_sha256':digest(OUT/'source-lock.json'),'verify_code_sha256':digest(ROOT/'configs/ei2-pca-select/verify.py'),'selector_states_verified':22,'all_final_trial_heads_reloaded':90,'selected_final_heads_reloaded':18,'primary_intervals_recomputed':18,'conditional_stability_masks_verified':15,'maximum_kkt':maximum_kkt,'maximum_map_gradient':maximum_map,'maximum_covariance_difference':covdiff,'ard_outer_iteration_limit':ard_limited,'prediction_max_difference':maximum,'reference_reproduction_max_difference':refmax,'seconds':time.monotonic()-started});print('EI_PS1_VERIFICATION_PASS',flush=True)

if __name__=='__main__':main()
