from common import *

def main():
 import fcntl
 from sklearn.linear_model import LogisticRegression
 from sklearn.metrics import roc_auc_score
 OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
 lock=(CACHE/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'source-lock.json').exists(),'Inspect existing attempt'
 data=parent.load_data(check_hashes=True);rows=data['rows'];roles=data['roles'];train,dev=roles['train'],roles['dev'];y=np.array([r['label'] for r in rows])
 manifest=PARENT/'results/ei2-vocabulary/manifest.json';ps=json.loads((PARENT/'results/ei2-vocabulary/source-lock.json').read_text());bs=json.loads((BASE/'results/ei2-pca-extended/source-lock.json').read_text())
 assert digest(manifest)==ps['manifest']==bs['parent_manifest']
 sources=list((ROOT/'configs/ei2-pca-select').glob('*.py'))+[ROOT/'configs/ei2-pca-auto/common.py',ROOT/'configs/ei2-pca-auto/verify.py',ROOT/'configs/ei2-vocabulary/core.py',ROOT/'configs/ei2-vocabulary/verify.py',ROOT/'docs/ei2-pca-select-protocol-2026-11-05.md']
 for p in sources:assert p.read_bytes()==subprocess.check_output(['git','-C',str(ROOT),'show','HEAD:'+p.relative_to(ROOT).as_posix()]),str(p)
 write(OUT/'source-lock.json',{'run_id':'EI-PS1','numbering_version':3,'source_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),'sources':{p.relative_to(ROOT).as_posix():digest(p) for p in sources},'parent_manifest':digest(manifest),'parent_verification':digest(BASE/'results/ei2-pca-extended/verification.json'),'parent_basis':digest(BASE/'cache/ei2-pca-extended/transforms.joblib'),'history_only':True})
 write(OUT/'manifest.json',{'run_id':'EI-PS1','numbering_version':3,'samples':rows,'history_only':True})
 a={r:parent.project(data,roles[r]) for r in ['train','dev']};full,full_mean,full_scale=pack(data['f'][train],a['train'],np.ones((14,32),bool));sx=full[:,:462].astype(np.float64)
 states={};variants={n:[] for n in FAMILIES};selector_records=[]
 def save_selector(state,key,kind,bootstrap_index=None):
  path=CACHE/'selectors'/(key+'.joblib');path.parent.mkdir(exist_ok=True);joblib.dump(state,path)
  record={'key':key,'family':kind,'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path),'bootstrap_index':bootstrap_index,'seconds':state['seconds']}
  for name in ['kkt','iterations','optimizer_success','optimizer_message','map_gradient','outer_converged','fixed_point_log_residual']: 
   if name in state:record[name]=state[name]
  if 'history' in state:record['outer_iterations']=len(state['history'])
  selector_records.append(record);print('SELECTOR_FIT',record,flush=True);return record
 for j,rho in enumerate([1.,.5]):
  family=FAMILIES[j]
  for lam in LAMBDAS:
   state=fit_sparse(sx,y[train],lam,rho);key=f'{family}-lambda-{lam}';record=save_selector(state,key,family);mask=state['mask'].reshape(14,32)
   variants[family].append({'key':key,'lambda':lam,'rho':rho,'mask':mask,'selector':record});states[key]=state
 state=fit_ard(sx,y[train]);record=save_selector(state,'ard-posterior',FAMILIES[2]);states['ard-posterior']=state
 for cut in CUTS:variants[FAMILIES[2]].append({'key':f'alpha-{cut}','alpha_cut':cut,'mask':ard_mask(state,cut).reshape(14,32),'selector':record})
 search=[];models={}
 def fit_heads(family,variant,c):
  mask=variant['mask'];x,mean,scale=pack(data['f'][train],a['train'],mask);d,_,_=pack(data['f'][dev],a['dev'],mask,mean,scale);entries=[];aucs=[]
  for seed in SEEDS:
   start=time.monotonic();head=LogisticRegression(C=c,solver='liblinear',max_iter=5000,random_state=seed).fit(x,y[train]);assert head.n_iter_.max()<5000
   auc=float(roc_auc_score(y[dev],head.decision_function(d)));aucs.append(auc);entry={'head':head,'mean':mean,'scale':scale,'mask':mask,'seed':seed,'C':c,'variant':variant['key']}
   path=CACHE/'trials'/f'{family}-{variant["key"]}-C{c}-S{seed}.joblib';path.parent.mkdir(exist_ok=True);joblib.dump(entry,path)
   search.append({'family':family,'variant':variant['key'],'C':c,'seed':seed,'dev_auroc':auc,'directions':int(mask.sum()),'n_iter':head.n_iter_.tolist(),'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path),'seconds':time.monotonic()-start});entries.append(entry)
  return {'entries':entries,'variant':variant,'C':c,'dev_auroc_mean':float(np.mean(aucs)),'directions_mean':int(mask.sum())}
 for family,candidates in variants.items():
  choices=[fit_heads(family,variant,c) for variant in candidates for c in GRID];chosen=max(enumerate(choices),key=lambda j:(j[1]['dev_auroc_mean'],-j[1]['directions_mean'],-j[0]))[1];models[family]=chosen
  models[family+'-prefix']=fit_heads(family+'-prefix',{'key':'matched-prefix','mask':prefix(chosen['variant']['mask']),'matched_to':family},chosen['C'])
  print('FINAL_SELECTED',family,chosen['variant']['key'],chosen['C'],chosen['directions_mean'],chosen['dev_auroc_mean'],flush=True)
 for name,model in models.items():
  for entry in model['entries']:
   x,_,_=pack(data['f'][train],a['train'],entry['mask'],entry['mean'],entry['scale']);entry['rank_train']=int(np.linalg.matrix_rank(x.astype(np.float64)))
  joblib.dump(model,CACHE/(name+'.joblib'))
 write(OUT/'final-search.json',search)
 # Conditional mask stability, with frozen PCA/scaler and frozen winning hyperparameters.
 rng=np.random.default_rng(20261106);classes=[np.flatnonzero(y[train]==j) for j in [0,1]];stability=[]
 for b in range(5):
  ix=np.r_[rng.choice(classes[0],len(classes[0]),replace=True),rng.choice(classes[1],len(classes[1]),replace=True)]
  for j,family in enumerate(FAMILIES):
   variant=models[family]['variant'];state=fit_sparse(sx[ix],y[train][ix],variant['lambda'],variant['rho']) if j<2 else fit_ard(sx[ix],y[train][ix]);mask=state['mask'].reshape(14,32) if j<2 else ard_mask(state,variant['alpha_cut']).reshape(14,32)
   record=save_selector(state,f'{family}-bootstrap-{b}',family,b);stability.append({'family':family,'bootstrap':b,'indices':ix.tolist(),'selector':record,'counts':mask.sum(1).tolist(),'directions':int(mask.sum()),'jaccard':jaccard(mask,variant['mask']),'mask':mask.astype(int).tolist()})
 write(OUT/'selector-training.json',selector_records);write(OUT/'stability.json',stability)
 write(OUT/'pre-evaluation-seal.json',{'source_lock':digest(OUT/'source-lock.json'),'models':{n:digest(CACHE/(n+'.joblib')) for n in models},'final_search':digest(OUT/'final-search.json'),'selectors':digest(OUT/'selector-training.json'),'stability':digest(OUT/'stability.json'),'before_evaluation_projection':True})
 scores={};labels={};maximum=0.
 for role in ['cal','C','RV','OR_cal','OR_test']:
  ix=roles[role];pa=parent.project(data,ix);labels[role]=y[ix];scores[role],difference=parent.references(data,role,pa);maximum=max(maximum,difference)
  for name,model in models.items():scores[role][name]=np.array([entry['head'].decision_function(pack(data['f'][ix],pa,entry['mask'],entry['mean'],entry['scale'])[0]) for entry in model['entries']])
  print('EVALUATED',role,flush=True)
 cuts={r:{n:[float(np.sort(s[j])[380]) for j in range(3)] for n,s in scores[r].items()} for r in ['cal','OR_cal']};cohorts={}
 for role in ['C','RV','OR_test']:
  yy=labels[role];cohorts[role]={}
  for name,ss in scores[role].items():
   stats=[]
   for j,seed in enumerate(SEEDS):
    cut=cuts['cal'][name][j];s=ss[j];m={'seed':seed,'auroc':float(roc_auc_score(yy,s)),'pauc':float(roc_auc_score(yy,s,max_fpr=.1)),'threshold':cut,'tpr':float((s[yy==1]>cut).mean()),'fpr':float((s[yy==0]>cut).mean()),'fp':int((s[yy==0]>cut).sum())}
    if role=='OR_test':
     cut=cuts['OR_cal'][name][j];m['source_calibrated']={'threshold':cut,'tpr':float((s[yy==1]>cut).mean()),'fpr':float((s[yy==0]>cut).mean()),'fp':int((s[yy==0]>cut).sum())}
    stats.append(m)
   cohorts[role][name]=stats
  comparisons=bootstrap(scores[role],yy)
  for name,controls in comparisons.items():
   for control,m in controls.items():m['delta_auroc']=cohorts[role][name][0]['auroc']-cohorts[role][control][0]['auroc']
  cohorts[role]['comparisons']=comparisons;print('RESULT',role,{n:cohorts[role][n][0] for n in ['K32']+FAMILIES},flush=True)
 np.savez_compressed(OUT/'scores.npz',**{r+'__'+n:s for r,d in scores.items() for n,s in d.items()},**{r+'__y':yy for r,yy in labels.items()})
 summary={}
 for name,model in models.items():
  v=model['variant'];mask=v['mask'];details={k:value for k,value in v.items() if k!='mask'}
  summary[name]={'variant':details,'C':model['C'],'dev_auroc_mean':model['dev_auroc_mean'],'directions_mean':int(mask.sum()),'counts':mask.sum(1).tolist(),'direction_indices_1based_by_layer':[np.flatnonzero(row).astype(int).__add__(1).tolist() for row in mask],'is_prefix':bool(np.array_equal(mask,prefix(mask))),'rank_train_by_seed':[entry['rank_train'] for entry in model['entries']]}
  if name in FAMILIES:
   records=[r for r in stability if r['family']==name];summary[name]['stability']={'jaccard':[r['jaccard'] for r in records],'directions':[r['directions'] for r in records]}
 write(OUT/'evaluation.json',{'run_id':'EI-PS1','numbering_version':3,'history_only':True,'primary':'AUROC','primary_control':'K32 same PCA32 basis','selected':summary,'cohorts':cohorts,'thresholds':cuts,'reference_reproduction_max_difference':maximum,'final_lr_fits':len(search),'selector_fits':len(selector_records)})
 assert len(search)==90 and len(selector_records)==22
 write(OUT/'completion.json',{'lr_fits':90,'selector_fits':22,'evaluation_sha256':digest(OUT/'evaluation.json'),'scores_sha256':digest(OUT/'scores.npz'),'independent_confirmation':False});print('EI_PS1_COMPLETE',flush=True)

if __name__=='__main__':main()
