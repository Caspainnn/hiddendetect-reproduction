from common import *

def main():
 from sklearn.model_selection import StratifiedKFold
 import torch
 started=time.monotonic();data=load_data();roles=data['roles'];train,dev=roles['train'],roles['dev'];y=np.array([r['label'] for r in data['rows']]);basis=data['joint_basis'];ranks=[len(b['values']) for b in basis]
 source=json.loads((OUT/'source-lock.json').read_text());pre=json.loads((OUT/'pre-evaluation-seal.json').read_text());done=json.loads((OUT/'completion.json').read_text());e=json.loads((OUT/'evaluation.json').read_text());assert done['lr_fits']==588 and done['selector_fits']==18 and done['gate_fits']==12
 assert digest(OUT/'evaluation.json')==done['evaluation_sha256'] and digest(OUT/'scores.npz')==done['scores_sha256'] and digest(OUT/'source-lock.json')==pre['source_lock']
 for name,sha in source['sources'].items():assert digest(ROOT/name)==sha
 assert digest(BASE/'results/ei2-pca-uncapped/verification.json')==source['parent_verification'] and digest(BASE/'results/ei2-pca-uncapped/pre-evaluation-seal.json')==source['parent_preseal'] and digest(PARENT/'results/ei2-vocabulary/manifest.json')==source['parent_manifest']
 assert json.loads((OUT/'manifest.json').read_text())['samples']==data['rows']
 for file,key in [('cv-selection.json','cv_selection'),('selector-training.json','selectors'),('gate-training.json','gates'),('final-search.json','search')]:assert digest(OUT/file)==pre[key]
 point=module('point_check',ROOT/'configs/ei2-vocabulary/verify.py');weighted=module('weighted_check',ROOT/'configs/ei2-pca-auto/verify.py');records=json.loads((OUT/'selector-training.json').read_text());assert len(records)==18;states={};maximum_kkt=0.
 def equal_indices(a,b):
  assert len(a)==len(b)==14
  for x,z in zip(a,b):np.testing.assert_array_equal(x,z)
 def standardization(f,a,indices,artifact):
  raw=np.column_stack([f]+[values[:,np.asarray(ix,int)] for values,ix in zip(a,indices)]).astype(np.float32);mean=raw.mean(0,dtype=np.float64).astype(np.float32);scale=raw.std(0,dtype=np.float64).astype(np.float32);scale[scale==0]=1
  np.testing.assert_array_equal(mean,artifact['mean']);np.testing.assert_array_equal(scale,artifact['scale'])
 def check_selector(record,a,fitix):
  nonlocal maximum_kkt
  path=ROOT/record['path'];assert digest(path)==record['sha256'];artifact=joblib.load(path);state=artifact['state'];np.testing.assert_array_equal(artifact['fit_indices'],fitix);np.testing.assert_array_equal(record['fit_indices'],fitix)
  standardization(data['f'][fitix],a,artifact['indices'],artifact);x=pack(data['f'][fitix],a,artifact['indices'],artifact['mean'],artifact['scale'])[0][:,:len(state['weight'])].astype(float);yy=y[fitix];w=state['weight'];p=expit(x@w+state['bias']);g=x.T@(p-yy)/len(yy)+w/(10*len(yy));g[14:]+=state['lambda']*(1-state['rho'])*w[14:];nz=np.abs(w[14:])>1e-8;tail=np.where(nz,np.abs(g[14:]+state['lambda']*state['rho']*np.sign(w[14:])),np.maximum(np.abs(g[14:])-state['lambda']*state['rho'],0));error=float(max(np.max(np.abs(g[:14])),abs(np.mean(p-yy)),np.max(tail,initial=0)));assert error<=2e-6 and abs(error-state['kkt'])<1e-12;maximum_kkt=max(maximum_kkt,error)
  assert state['lambda']==record['lambda'] and state['rho']==record['rho'];np.testing.assert_array_equal(state['mask'],nz);states[record['key']]=artifact;return artifact
 def independent_order(w,rr):
  result=[];offset=14
  for rank in rr:
   result.append(np.array(sorted(range(rank),key=lambda i:(-abs(w[offset+i]),i))));offset+=rank
  assert offset==len(w);return result
 cv=json.loads((OUT/'cv-selection.json').read_text());array=np.zeros((3,44,3));counts=np.zeros((3,44),int);expected=list(StratifiedKFold(3,shuffle=True,random_state=SEEDS[0]).split(train,y[train]));seen=[]
 assert cv['tags']==u.TAGS and len(cv['records'])==396
 for fold in cv['folds']:
  j=fold['fold'];ti,vi=train[expected[j][0]],train[expected[j][1]];np.testing.assert_array_equal(ti,fold['train_indices']);np.testing.assert_array_equal(vi,fold['validation_indices']);seen.extend(vi.tolist());path=ORIGIN/fold['basis_path'];assert digest(path)==fold['basis_sha256'];fb=joblib.load(path);fr=[len(b['values']) for b in fb]
  for b in fb:np.testing.assert_array_equal(b['train_indices'],ti)
  ta,va=u.project(data,ti,fb),u.project(data,vi,fb);record=next(r for r in records if r['key']==f'fold-{j}-EN');assert record==fold['selector'] and record['lambda']==.0001 and record['rho']==.5;artifact=check_selector(record,ta,ti);equal_indices(artifact['indices'],full_indices(fr));order=independent_order(artifact['state']['weight'],fr);equal_indices(order,[np.array(row) for row in fold['order']])
  for record in [r for r in cv['records'] if r['fold']==j]:
   path=ROOT/record['path'];assert digest(path)==record['sha256'] and max(record['n_iter'])<5000;head=joblib.load(path);tagindex=u.TAGS.index(record['tag']);k=int(np.ceil(min(fr)*int(record['tag'].split('-')[1])/100)) if record['tag'].startswith('rank-') else int(record['tag']);indices=[np.sort(row[:k]) for row in order];equal_indices(head['indices'],indices);assert record['k']==k
   standardization(data['f'][ti],ta,indices,head);x=pack(data['f'][vi],va,indices,head['mean'],head['scale'])[0];auc=point.auc(y[vi],head['head'].decision_function(x));assert abs(auc-record['auroc'])<1e-12 and head['head'].C==record['C'];array[j,tagindex,GRID.index(record['C'])]=auc;counts[j,tagindex]=k
  print('VERIFIED_SORTED_CV_FOLD',j,flush=True)
 assert sorted(seen)==sorted(train.tolist());np.testing.assert_array_equal(counts,cv['counts']);np.testing.assert_allclose(array,cv['scores'],atol=1e-12,rtol=0)
 means=array.mean(0);best=sorted(np.ndindex(means.shape),key=lambda p:(-means[p],float(counts[:,p[0]].mean()),p[0],p[1]))[0];threshold=means[best]-array[:,best[0],best[1]].std(ddof=1)/np.sqrt(3);chosen=sorted([p for p in np.ndindex(means.shape) if means[p]>=threshold-1e-12],key=lambda p:(float(counts[:,p[0]].mean()),-means[p],p[0],p[1]))[0];assert cv['selection']['tag']==u.TAGS[chosen[0]] and cv['selection']['C']==GRID[chosen[1]] and abs(threshold-cv['selection']['threshold'])<1e-12
 a={role:u.project(data,roles[role],basis) for role in ['train','dev']}
 for record in records:
  if record['fold'] is not None:continue
  artifact=check_selector(record,a['train'],train)
  if record['key'].startswith('pool-'):
   tau=float(record['key'].split('-')[1]);ks=[]
   for b in basis:ks.append(next(i+1 for i,v in enumerate(np.cumsum(b['values'])/b['total_energy']) if v>=tau))
   equal_indices(artifact['indices'],full_indices(ks));assert record['rho']==1.
  else:equal_indices(artifact['indices'],full_indices(ranks))
 assert len(states)==18;ranking=independent_order(states['S-EN-0.0001']['state']['weight'],ranks)
 gates=json.loads((OUT/'gate-training.json').read_text());assert len(gates)==12 and {(r['lambda'],r['seed']) for r in gates}=={(lam,seed) for lam in [0.,.01,.1,1.] for seed in SEEDS};audits=[]
 for record in gates:
  path=ROOT/record['path'];assert digest(path)==record['sha256'];state=joblib.load(path);assert state['ranks']==record['ranks']==ranks and state['seed']==record['seed'] and state['lambda']==record['lambda']
  torch.use_deterministic_algorithms(True);tensor=torch.tensor(state['a'],device='cuda');p=torch.sigmoid(tensor);valid=torch.tensor(np.arange(max(ranks))[None,:]<np.asarray(ranks)[:,None],device='cuda');g=(u.scan_prefix(p,torch)*valid).cpu().numpy();np.testing.assert_array_equal(g,state['soft_gates']);audit=gate_product_audit(state['a'],p.cpu().numpy(),g,ranks);assert (g>=.5).sum(1).tolist()==state['counts'] and state['steps']==800
  np.testing.assert_array_equal(g>=.5,np.arange(max(ranks))[None,:]<np.array(state['counts'])[:,None]);audits.append({'path':record['path'],'state_sha256':record['sha256'],'gpu_replay_bitwise_identical':True,**audit})
 write(OUT/'gate-numerics-audit.json',{'passed':True,'evaluation_sha256':digest(OUT/'evaluation.json'),'records':audits})
 search=json.loads((OUT/'final-search.json').read_text());assert len(search)==192;trials={}
 for record in search:
  path=ROOT/record['path'];assert digest(path)==record['sha256'] and max(record['n_iter'])<5000;entry=joblib.load(path);standardization(data['f'][train],a['train'],entry['indices'],entry);x=pack(data['f'][dev],a['dev'],entry['indices'],entry['mean'],entry['scale'])[0];auc=point.auc(y[dev],entry['head'].decision_function(x));assert abs(auc-record['dev_auroc'])<1e-12 and entry['head'].C==record['C'] and entry['head'].solver=='liblinear';assert record['counts']==[len(ix) for ix in entry['indices']];trials.setdefault(record['family'],{}).setdefault((record['variant'],record['C']),[]).append(record)
 models={n:joblib.load(CACHE/(n+'.joblib')) for n in pre['models']}
 for name,model in models.items():
  assert digest(CACHE/(name+'.joblib'))==pre['models'][name];summary=e['selected'][name];assert summary['counts_by_seed']==[[len(ix) for ix in entry['indices']] for entry in model['entries']]
  if name not in [f+'-prefix' for f in FAMILIES]+['same-tau-pool']:
   choices=[]
   for (variant,c),rows in trials[name].items():
    assert sorted(r['seed'] for r in rows)==SEEDS;choices.append((float(np.mean([r['dev_auroc'] for r in rows])),float(np.mean([sum(r['counts']) for r in rows])),variant,c))
   winner=max(enumerate(choices),key=lambda j:(j[1][0],-j[1][1],-j[0]))[1];assert winner[2]==model['variant']['key'] and winner[3]==model['C'] and abs(winner[0]-model['dev_auroc_mean'])<1e-12
  for j,entry in enumerate(model['entries']):
   variant=model['variant'];indices=entry['indices'];assert entry['C']==model['C'] and entry['seed']==SEEDS[j];assert summary['indices_1based_by_seed'][j]==[[int(k)+1 for k in ix] for ix in indices]
   for ix,rank in zip(indices,ranks):assert len(ix)==len(np.unique(ix)) and np.all((ix>=0)&(ix<rank)) and np.all(np.diff(ix)>0)
   if name==CV:
    k=u.count_for(cv['selection']['tag'],min(ranks));expected_ix=[np.sort(row[:k]) for row in ranking]
   elif name in ['S-EN','S-L1',POOL]:
    record=variant['selector'];assert record in records;artifact=states[record['key']];mask=np.abs(artifact['state']['weight'][14:])>1e-8;expected_ix=[];offset=0
    for pool in artifact['indices']:expected_ix.append(np.asarray(pool)[mask[offset:offset+len(pool)]]);offset+=len(pool)
    if name==POOL:
     coverage=[float(b['values'][ix].sum()/b['total_energy']) for b,ix in zip(basis,expected_ix)];np.testing.assert_allclose(coverage,variant['selected_coverage'],atol=1e-12);assert variant['pool_counts']==[len(ix) for ix in artifact['indices']]
   elif name in [GATE,'ranked-gate-zero']:
    record=variant['gates'][j];assert record in gates and record['lambda']==variant['lambda'] and record['seed']==entry['seed'];state=joblib.load(ROOT/record['path']);expected_ix=[np.sort(row[:k]) for row,k in zip(ranking,state['counts'])]
   elif name.endswith('-prefix'):
    original=models[variant['matched_to']];assert original['C']==model['C'];expected_ix=[np.arange(len(ix)) for ix in original['entries'][j]['indices']]
   else:
    assert name=='same-tau-pool';original=models[POOL];assert model['C']==original['C'] and variant['tau']==original['variant']['tau'];expected_ix=full_indices(original['variant']['pool_counts'])
   equal_indices(indices,expected_ix)
 scores=np.load(OUT/'scores.npz');maximum=0.;refmax=0.;names=['Q-CV','Q-VAR','Q-GATE','S-EN','S-L1']+FAMILIES
 for role in ['cal','C','RV','OR_cal','OR_test']:
  ix=roles[role];pa=u.project(data,ix,basis);refs,difference=replay_references(data,role,pa);refmax=max(refmax,difference);np.testing.assert_array_equal(y[ix],scores[role+'__y'])
  for name,values in refs.items():np.testing.assert_allclose(values,scores[role+'__'+name],atol=1e-7,rtol=0)
  for name,model in models.items():
   for j,entry in enumerate(model['entries']):
    x=pack(data['f'][ix],pa,entry['indices'],entry['mean'],entry['scale'])[0];difference=float(np.max(np.abs(entry['head'].decision_function(x)-scores[role+'__'+name][j])));assert difference<1e-7;maximum=max(maximum,difference)
  if role not in ['C','RV','OR_test']:continue
  yy=y[ix]
  for name,stats in e['cohorts'][role].items():
   if name=='comparisons':continue
   for j,m in enumerate(stats):
    ss=scores[role+'__'+name][j];cut=float(np.partition(scores['cal__'+name][j],380)[380]);assert cut==m['threshold']==e['thresholds']['cal'][name][j] and abs(point.auc(yy,ss)-m['auroc'])<1e-12 and abs(point.pauc(yy,ss)-m['pauc'])<1e-12
    assert m['tpr']==float((ss[yy==1]>cut).mean()) and m['fpr']==float((ss[yy==0]>cut).mean()) and m['fp']==int((ss[yy==0]>cut).sum())
    if role=='OR_test':
     cut=float(np.partition(scores['OR_cal__'+name][j],380)[380]);m=m['source_calibrated'];assert cut==m['threshold']==e['thresholds']['OR_cal'][name][j] and m['tpr']==float((ss[yy==1]>cut).mean()) and m['fpr']==float((ss[yy==0]>cut).mean()) and m['fp']==int((ss[yy==0]>cut).sum())
  groups=[np.flatnonzero(yy==j) for j in [0,1]];rng=np.random.default_rng(20261105);dist=np.empty((10000,8))
  for start in range(0,10000,100):
   draws=np.array([np.r_[rng.choice(groups[0],len(groups[0]),replace=True),rng.choice(groups[1],len(groups[1]),replace=True)] for _ in range(100)]);counts=np.zeros((100,len(yy)));np.add.at(counts,(np.arange(100)[:,None],draws),1)
   for j,name in enumerate(names):dist[start:start+100,j]=weighted.weighted_auc(yy,scores[role+'__'+name][0],counts)
  for name in FAMILIES:
   for control in CONTROLS[name]:
    comparison=e['cohorts'][role]['comparisons'][name][control];np.testing.assert_allclose(comparison['ci991667'],np.quantile(dist[:,names.index(name)]-dist[:,names.index(control)],[.05/12,1-.05/12]),atol=1e-12,rtol=0);assert abs(comparison['delta_auroc']-(e['cohorts'][role][name][0]['auroc']-e['cohorts'][role][control][0]['auroc']))<1e-12
  print('VERIFIED_FOUR_METRICS_AND_SIX_INTERVALS',role,flush=True)
 write(OUT/'verification.json',{'passed':True,'evaluation_sha256':digest(OUT/'evaluation.json'),'scores_sha256':digest(OUT/'scores.npz'),'source_lock_sha256':digest(OUT/'source-lock.json'),'verify_code_sha256':digest(ROOT/'configs/ei2-pca-joint/verify.py'),'gate_numerics_audit_sha256':digest(OUT/'gate-numerics-audit.json'),'cv_heads_reloaded':396,'final_trial_heads_reloaded':192,'selector_states_verified':18,'gate_states_verified':12,'primary_intervals_recomputed':18,'maximum_kkt':maximum_kkt,'prediction_max_difference':maximum,'reference_reproduction_max_difference':refmax,'ideal_gate_boundary_disagreements':sum(r['ideal_hard_mask_mismatches'] for r in audits),'seconds':time.monotonic()-started});print('EI_PJ1_VERIFICATION_PASS',flush=True)

if __name__=='__main__':main()
