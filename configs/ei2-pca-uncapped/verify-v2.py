from common import *

def check_gate_ranges(records,ranks):
 expected={(scope,lam,seed) for scope in ['full','cap32'] for lam in [0.,.01,.1,1.] for seed in SEEDS}
 assert len(records)==24 and {(r['scope'],r['lambda'],r['seed']) for r in records}==expected
 for record in records:assert record['ranks']==(ranks if record['scope']=='full' else [min(32,k) for k in ranks])

def main():
 from scipy.special import expit
 from sklearn.model_selection import StratifiedKFold
 for name in ['verify-v2.py','gate_audit.py','test_gate_numerics.py']:
  path=ROOT/'configs/ei2-pca-uncapped'/name;assert path.read_bytes()==subprocess.check_output(['git','-C',str(ROOT),'show','HEAD:'+path.relative_to(ROOT).as_posix()])
 started=time.monotonic();data=parent.load_data(check_hashes=True);roles=data['roles'];train,dev=roles['train'],roles['dev'];y=np.array([r['label'] for r in data['rows']])
 source=json.loads((OUT/'source-lock.json').read_text());pre=json.loads((OUT/'pre-evaluation-seal.json').read_text());e=json.loads((OUT/'evaluation.json').read_text());done=json.loads((OUT/'completion.json').read_text());assert done['lr_fits']==549 and done['gate_fits']==24
 assert done['evaluation_sha256']==digest(OUT/'evaluation.json') and done['scores_sha256']==digest(OUT/'scores.npz') and pre['source_lock']==digest(OUT/'source-lock.json')
 for name,sha in source['sources'].items():assert digest(ROOT/name)==sha
 assert source['parent_manifest']==digest(PARENT/'results/ei2-vocabulary/manifest.json') and source['parent_verification']==digest(BASE/'results/ei2-pca-extended/verification.json')
 assert json.loads((OUT/'manifest.json').read_text())['samples']==data['rows']
 for name,key in [('cv-selection.json','cv_selection'),('gate-training.json','gates'),('final-search.json','search')]:assert digest(OUT/name)==pre[key]
 spec=importlib.util.spec_from_file_location('point',ROOT/'configs/ei2-vocabulary/verify.py');point=importlib.util.module_from_spec(spec);spec.loader.exec_module(point)
 spec=importlib.util.spec_from_file_location('weighted',ROOT/'configs/ei2-pca-auto/verify.py');weighted=importlib.util.module_from_spec(spec);spec.loader.exec_module(weighted)
 def check_basis(basis,indices):
  for l,b in enumerate(basis):
   np.testing.assert_array_equal(b['train_indices'],indices);raw=residual(data,indices,l).astype(np.float64);mean=raw.mean(0);np.testing.assert_array_equal(mean,b['mean']);centered=raw-mean;gram=matmul(centered,centered.T);vectors=b['dual']*np.sqrt(b['values']);total=float(np.sum(centered*centered))
   assert abs(total-b['total_energy'])<1e-10 and len(b['values'])<=len(indices)-1 and len(b['values'])>32
   np.testing.assert_allclose(vectors.T@vectors,np.eye(len(b['values'])),atol=1e-8,rtol=0)
   assert np.max(np.abs(gram@vectors-vectors*b['values']))/max(1.,b['values'][0])<1e-8
   assert b['retained_energy']>.999999 and abs(b['values'].sum()/total-b['retained_energy'])<1e-12
   tolerance=b['values'][0]*max(raw.shape)*np.finfo(float).eps;assert abs(tolerance-b['rank_tolerance'])<1e-12 and min(b['values'])>tolerance
 cv=json.loads((OUT/'cv-selection.json').read_text());array=np.zeros((3,len(TAGS),3));counts=np.zeros((3,len(TAGS)),int);folds=list(StratifiedKFold(3,shuffle=True,random_state=SEEDS[0]).split(train,y[train]));seen=[]
 assert cv['tags']==TAGS and len(cv['records'])==396
 for fold in cv['folds']:
  j=fold['fold'];ti,vi=train[folds[j][0]],train[folds[j][1]];np.testing.assert_array_equal(ti,fold['train_indices']);np.testing.assert_array_equal(vi,fold['validation_indices']);seen.extend(vi.tolist());path=ROOT/fold['basis_path'];assert digest(path)==fold['basis_sha256'];basis=joblib.load(path);check_basis(basis,ti);rank=min(len(b['values']) for b in basis);ta,va=project(data,ti,basis),project(data,vi,basis)
  for record in [r for r in cv['records'] if r['fold']==j]:
   path=ROOT/record['path'];assert digest(path)==record['sha256'] and max(record['n_iter'])<5000;entry=joblib.load(path);k=count_for(record['tag'],rank);assert entry['counts']==[k]*14 and record['k']==k
   _,mean,scale=pack(data['f'][ti],ta,entry['counts']);np.testing.assert_array_equal(mean,entry['mean']);np.testing.assert_array_equal(scale,entry['scale']);x=pack(data['f'][vi],va,entry['counts'],mean,scale)[0];auc=point.auc(y[vi],entry['head'].decision_function(x));assert abs(auc-record['dev_auroc'])<1e-12;tag=TAGS.index(record['tag']);array[j,tag,GRID.index(record['C'])]=auc;counts[j,tag]=k
  print('VERIFIED_CV_AND_BASIS',j,flush=True)
 assert sorted(seen)==sorted(train.tolist());np.testing.assert_array_equal(counts,cv['counts']);np.testing.assert_allclose(array,cv['scores'],atol=1e-12,rtol=0)
 # Independent one-SE rule, including relative-rank candidates.
 for scope,stop in [('full',len(TAGS)),('cap32',32)]:
  values=array[:,:stop];means=values.mean(0);best=sorted(np.ndindex(means.shape),key=lambda p:(-means[p],float(counts[:,p[0]].mean()),p[0],p[1]))[0];threshold=means[best]-values[:,best[0],best[1]].std(ddof=1)/np.sqrt(3)
  chosen=sorted([p for p in np.ndindex(means.shape) if means[p]>=threshold-1e-12],key=lambda p:(float(counts[:,p[0]].mean()),-means[p],p[0],p[1]))[0];assert cv['selection'][scope]['tag']==TAGS[chosen[0]] and cv['selection'][scope]['C']==GRID[chosen[1]]
 assert digest(CACHE/'full-basis.joblib')==pre['basis'];basis=joblib.load(CACHE/'full-basis.joblib');check_basis(basis,train);ranks=[len(b['values']) for b in basis];assert ranks==e['ranks'];a={r:project(data,roles[r],basis) for r in ['train','dev']}
 gates=json.loads((OUT/'gate-training.json').read_text());check_gate_ranges(gates,ranks);precision_audits=[]
 for record in gates:
  path=ROOT/record['path'];assert digest(path)==record['sha256'];state=joblib.load(path);assert state['ranks']==record['ranks'] and state['seed']==record['seed'] and state['lambda']==record['lambda'];maximum=max(state['ranks'])
  import torch
  from gate_audit import check_gate_float32
  torch.use_deterministic_algorithms(True);tensor=torch.tensor(state['a'],device='cuda');probability=torch.sigmoid(tensor);valid=torch.tensor(np.arange(maximum)[None,:]<np.asarray(state['ranks'])[:,None],device='cuda')
  replay=(scan_prefix(probability,torch)*valid).cpu().numpy();np.testing.assert_array_equal(replay,state['soft_gates'])
  audit=check_gate_float32(state['a'],probability.cpu().numpy(),state['soft_gates'],state['ranks']);precision_audits.append({'path':record['path'],'state_sha256':record['sha256'],'gpu_replay_bitwise_identical':True,**audit})
  g=state['soft_gates']
  assert (g>=.5).sum(1).tolist()==state['counts']==record['counts'] and state['steps']==800 and np.isfinite(state['loss_start_end']).all()
 write(OUT/'gate-numerics-audit.json',{'passed':True,'evaluation_sha256':digest(OUT/'evaluation.json'),'all24_gpu_replays_and_independent_hard_masks_identical':True,'records':precision_audits})
 models={n:joblib.load(CACHE/(n+'.joblib')) for n in pre['models']};search=json.loads((OUT/'final-search.json').read_text());assert len(search)==153;trials={}
 for record in search:
  path=ROOT/record['path'];assert digest(path)==record['sha256'] and max(record['n_iter'])<5000;entry=joblib.load(path);_,mean,scale=pack(data['f'][train],a['train'],entry['counts']);np.testing.assert_array_equal(mean,entry['mean']);np.testing.assert_array_equal(scale,entry['scale']);x=pack(data['f'][dev],a['dev'],entry['counts'],mean,scale)[0];auc=point.auc(y[dev],entry['head'].decision_function(x));assert abs(auc-record['dev_auroc'])<1e-12;trials.setdefault(record['family'],{}).setdefault((record['variant'],record['C']),[]).append(record)
 for name,model in models.items():
  assert digest(CACHE/(name+'.joblib'))==pre['models'][name];choices=[(np.mean([r['dev_auroc'] for r in rows]),np.mean([sum(r['counts']) for r in rows]),variant,c) for (variant,c),rows in trials[name].items()];winner=max(enumerate(choices),key=lambda j:(j[1][0],-j[1][1],-j[0]))[1];assert winner[2]==model['variant']['key'] and winner[3]==model['C'];assert e['selected'][name]['counts_by_seed']==[entry['counts'] for entry in model['entries']]
  scope='cap32' if name.endswith('cap32') else 'full'
  for j,entry in enumerate(model['entries']):
   if name.startswith(FAMILIES[0]):assert entry['counts']==[count_for(cv['selection'][scope]['tag'],min(ranks))]*14
   elif name.startswith(FAMILIES[1]):
    expected=[];coverage=[];capped=[]
    for b in basis:
     limit=min(32,len(b['values'])) if scope=='cap32' else len(b['values']);cumulative=np.cumsum(b['values'][:limit])/b['total_energy'];k=next((i+1 for i,v in enumerate(cumulative) if v>=model['variant']['tau']),limit);expected.append(k);coverage.append(float(cumulative[k-1]));capped.append(bool(cumulative[-1]<model['variant']['tau']))
    assert expected==entry['counts'] and capped==model['variant']['capped'];np.testing.assert_allclose(coverage,model['variant']['coverage'],atol=1e-12)
   elif name!='NEW-K32':
    reference=model['variant']['gates'][j];assert reference in gates and reference['scope']==scope and reference['seed']==entry['seed'] and reference['lambda']==model['variant']['lambda'];assert digest(ROOT/reference['path'])==reference['sha256'];assert joblib.load(ROOT/reference['path'])['counts']==entry['counts']
   else:assert entry['counts']==[32]*14
 scores=np.load(OUT/'scores.npz');maximum=0.;refmax=0.
 for role in ['cal','C','RV','OR_cal','OR_test']:
  ix=roles[role];pa=project(data,ix,basis);refs,delta=parent.references(data,role,parent.project(data,ix));refmax=max(refmax,delta);np.testing.assert_array_equal(y[ix],scores[role+'__y'])
  for name,ss in refs.items():np.testing.assert_allclose(ss,scores[role+'__'+name],atol=1e-7,rtol=0)
  for name,model in models.items():
   for j,entry in enumerate(model['entries']):
    x=pack(data['f'][ix],pa,entry['counts'],entry['mean'],entry['scale'])[0];delta=float(np.max(np.abs(entry['head'].decision_function(x)-scores[role+'__'+name][j])));assert delta<1e-7;maximum=max(maximum,delta)
  if role not in ['C','RV','OR_test']:continue
  yy=y[ix]
  for name,stats in e['cohorts'][role].items():
   if name=='comparisons':continue
   for j,m in enumerate(stats):
    s=scores[role+'__'+name][j];cut=float(np.partition(scores['cal__'+name][j],380)[380]);assert cut==m['threshold']==e['thresholds']['cal'][name][j] and abs(point.auc(yy,s)-m['auroc'])<1e-12 and abs(point.pauc(yy,s)-m['pauc'])<1e-12
    assert m['tpr']==float((s[yy==1]>cut).mean()) and m['fpr']==float((s[yy==0]>cut).mean()) and m['fp']==int((s[yy==0]>cut).sum())
    if role=='OR_test':
     cut=float(np.partition(scores['OR_cal__'+name][j],380)[380]);p=m['source_calibrated'];assert cut==p['threshold']==e['thresholds']['OR_cal'][name][j] and p['tpr']==float((s[yy==1]>cut).mean()) and p['fpr']==float((s[yy==0]>cut).mean()) and p['fp']==int((s[yy==0]>cut).sum())
  groups=[np.flatnonzero(yy==j) for j in [0,1]];rng=np.random.default_rng(20261105);dist=np.empty((10000,6));names=FAMILIES+[n+'-cap32' for n in FAMILIES]
  for start in range(0,10000,100):
   draws=np.array([np.r_[rng.choice(groups[0],len(groups[0]),replace=True),rng.choice(groups[1],len(groups[1]),replace=True)] for _ in range(100)]);counts=np.zeros((100,len(yy)));np.add.at(counts,(np.arange(100)[:,None],draws),1)
   for j,name in enumerate(names):dist[start:start+100,j]=weighted.weighted_auc(yy,scores[role+'__'+name][0],counts)
  for j,name in enumerate(FAMILIES):
   comparison=e['cohorts'][role]['comparisons'][name];np.testing.assert_allclose(comparison['ci983333'],np.quantile(dist[:,j]-dist[:,j+3],[.05/6,1-.05/6]),atol=1e-12,rtol=0);assert abs(comparison['delta_auroc']-(e['cohorts'][role][name][0]['auroc']-e['cohorts'][role][name+'-cap32'][0]['auroc']))<1e-12
  print('VERIFIED_METRICS_AND_INTERVALS',role,flush=True)
 write(OUT/'verification.json',{'passed':True,'evaluation_sha256':digest(OUT/'evaluation.json'),'scores_sha256':digest(OUT/'scores.npz'),'verify_code_sha256':digest(ROOT/'configs/ei2-pca-uncapped/verify-v2.py'),'verification_revision':'v2-float32-product-audit','gate_numerics_audit_sha256':digest(OUT/'gate-numerics-audit.json'),'gate_audit_code_sha256':digest(ROOT/'configs/ei2-pca-uncapped/gate_audit.py'),'verification_source_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),'source_lock_sha256':digest(OUT/'source-lock.json'),'cv_heads_reloaded':396,'final_trial_heads_reloaded':153,'gate_states_verified':24,'full_rank_bases_checked':4,'primary_intervals_recomputed':9,'prediction_max_difference':maximum,'reference_reproduction_max_difference':refmax,'seconds':time.monotonic()-started});print('EI_PA2_VERIFICATION_PASS',flush=True)

if __name__=='__main__':main()
