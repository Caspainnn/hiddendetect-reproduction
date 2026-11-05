from common import *

def main():
 import fcntl
 from sklearn.model_selection import StratifiedKFold
 from sklearn.linear_model import LogisticRegression
 from sklearn.metrics import roc_auc_score
 OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
 lock=(CACHE/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'source-lock.json').exists(),'Inspect existing attempt before relaunching'
 data=parent.load_data(check_hashes=True);rows=data['rows'];roles=data['roles'];train,dev=roles['train'],roles['dev'];y=np.array([r['label'] for r in rows])
 manifest=PARENT/'results/ei2-vocabulary/manifest.json';ps=json.loads((PARENT/'results/ei2-vocabulary/source-lock.json').read_text());bs=json.loads((BASE/'results/ei2-pca-extended/source-lock.json').read_text());assert digest(manifest)==ps['manifest']==bs['parent_manifest']
 sources=list((ROOT/'configs/ei2-pca-uncapped').glob('*.py'))+[ROOT/'docs/ei2-pca-uncapped-protocol-2026-11-05.md',ROOT/'configs/ei2-pca-auto/common.py',ROOT/'configs/ei2-pca-auto/verify.py',ROOT/'configs/ei2-vocabulary/core.py',ROOT/'configs/ei2-vocabulary/verify.py']
 for p in sources:assert p.read_bytes()==subprocess.check_output(['git','-C',str(ROOT),'show','HEAD:'+p.relative_to(ROOT).as_posix()]),str(p)
 write(OUT/'source-lock.json',{'run_id':'EI-PA2','numbering_version':3,'source_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),'sources':{p.relative_to(ROOT).as_posix():digest(p) for p in sources},'parent_manifest':digest(manifest),'parent_verification':digest(BASE/'results/ei2-pca-extended/verification.json'),'history_only':True})
 write(OUT/'manifest.json',{'run_id':'EI-PA2','samples':rows,'numbering_version':3,'history_only':True})
 folds=[];records=[];cvscores=np.full((3,len(TAGS),3),np.nan);cvcounts=np.zeros((3,len(TAGS)),int)
 for fold,(ft,fv) in enumerate(StratifiedKFold(3,shuffle=True,random_state=SEEDS[0]).split(train,y[train])):
  ti,vi=train[ft],train[fv];basis,path=fit_basis(data,ti,f'fold-{fold}');rank=min(len(b['values']) for b in basis);ta,va=project(data,ti,basis),project(data,vi,basis)
  assert rank>=1024,'Protocol grid requires revisiting if natural fold rank is below1024'
  folds.append({'fold':fold,'train_indices':ti.tolist(),'validation_indices':vi.tolist(),'basis_path':path.relative_to(ROOT).as_posix(),'basis_sha256':digest(path),'ranks':[len(b['values']) for b in basis]})
  for j,tag in enumerate(TAGS):
   k=count_for(tag,rank);cvcounts[fold,j]=k;x,mean,scale=pack(data['f'][ti],ta,[k]*14);d,_,_=pack(data['f'][vi],va,[k]*14,mean,scale)
   for cindex,c in enumerate(GRID):
    start=time.monotonic();head=LogisticRegression(C=c,solver='liblinear',max_iter=5000,random_state=SEEDS[0]).fit(x,y[ti]);assert head.n_iter_.max()<5000
    auc=float(roc_auc_score(y[vi],head.decision_function(d)));cvscores[fold,j,cindex]=auc;artifact={'head':head,'mean':mean,'scale':scale,'counts':[k]*14}
    path=CACHE/'cv'/f'f{fold}-{tag}-c{cindex}.joblib';path.parent.mkdir(exist_ok=True);joblib.dump(artifact,path);records.append({'fold':fold,'tag':tag,'k':k,'C':c,'dev_auroc':auc,'n_iter':head.n_iter_.tolist(),'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path),'seconds':time.monotonic()-start})
   print('CV_GRID_COMPLETE',fold,tag,k,flush=True)
 assert np.isfinite(cvscores).all()
 cvselection={'full':choose_cv(cvscores,TAGS,cvcounts),'cap32':choose_cv(cvscores[:,:32],TAGS[:32],cvcounts[:,:32])}
 write(OUT/'cv-selection.json',{'selection':cvselection,'folds':folds,'records':records,'scores':cvscores.tolist(),'counts':cvcounts.tolist(),'tags':TAGS})
 basis,path=fit_basis(data,train,'full');basis_sha=digest(path);a={r:project(data,roles[r],basis) for r in ['train','dev']};ranks=[len(b['values']) for b in basis];variants={};gates=[]
 for scope in ['full','cap32']:
  key=lambda family:family if scope=='full' else family+'-cap32'
  k=count_for(cvselection[scope]['tag'],min(ranks));variants[key(FAMILIES[0])]=[{'key':'cv-'+cvselection[scope]['tag'],'counts_by_seed':[[k]*14]*3,'cv_selection':cvselection[scope]}]
  variants[key(FAMILIES[1])]=[];variants[key(FAMILIES[2])]=[];variants['gate-zero-'+scope]=[]
  for tau in [.8,.9,.95]:
   counts,coverage,capped=variance_counts(basis,tau,None if scope=='full' else 32)
   if scope=='full':assert not any(capped),'Full numerical spectrum failed target'
   variants[key(FAMILIES[1])].append({'key':f'tau-{tau}','tau':tau,'counts_by_seed':[counts]*3,'coverage':coverage,'capped':capped})
  available=ranks if scope=='full' else [min(32,r) for r in ranks];gx,_,_=pack(data['f'][train],a['train'],available)
  for lam in [0.,.01,.1,1.]:
   counts=[];paths=[]
   for seed in SEEDS:
    state=fit_gate(gx,y[train],available,lam,seed);path=CACHE/f'gate-{scope}-{lam}-{seed}.joblib';joblib.dump(state,path);counts.append(state['counts']);record={k:state[k] for k in ['seed','lambda','ranks','counts','steps','seconds','loss_start_end','device']}|{'scope':scope,'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path)};gates.append(record);paths.append(record);print('GATE_COMPLETE',scope,lam,seed,state['counts'],flush=True)
   family=key(FAMILIES[2]) if lam else 'gate-zero-'+scope;variants[family].append({'key':f'lambda-{lam}','lambda':lam,'counts_by_seed':counts,'gates':paths})
 # New-basis32 control separates PCA implementation from removal of the bound.
 variants['NEW-K32']=[{'key':'fixed32','counts_by_seed':[[32]*14]*3}]
 models={};search=[]
 for family,candidates in variants.items():
  choices=[]
  for variant in candidates:
   for c in GRID:
    entries=[];aucs=[]
    for j,seed in enumerate(SEEDS):
     counts=variant['counts_by_seed'][j];x,mean,scale=pack(data['f'][train],a['train'],counts);d,_,_=pack(data['f'][dev],a['dev'],counts,mean,scale);start=time.monotonic();head=LogisticRegression(C=c,solver='liblinear',max_iter=5000,random_state=seed).fit(x,y[train]);assert head.n_iter_.max()<5000
     auc=float(roc_auc_score(y[dev],head.decision_function(d)));aucs.append(auc);entry={'head':head,'mean':mean,'scale':scale,'counts':counts,'seed':seed,'C':c,'variant':variant['key']};path=CACHE/'trials'/f'{family}-{variant["key"]}-C{c}-S{seed}.joblib';path.parent.mkdir(exist_ok=True);joblib.dump(entry,path)
     search.append({'family':family,'variant':variant['key'],'C':c,'seed':seed,'counts':counts,'dev_auroc':auc,'effective_dimension':14+sum(counts),'interface_dimension':x.shape[1],'n_iter':head.n_iter_.tolist(),'seconds':time.monotonic()-start,'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path)});entries.append(entry)
    choices.append({'variant':variant,'C':c,'dev_auroc_mean':float(np.mean(aucs)),'directions_mean':float(np.mean([sum(k) for k in variant['counts_by_seed']])),'entries':entries})
  chosen=max(enumerate(choices),key=lambda j:(j[1]['dev_auroc_mean'],-j[1]['directions_mean'],-j[0]))[1];models[family]=chosen;joblib.dump(chosen,CACHE/(family+'.joblib'));print('FINAL_SELECTED',family,chosen['variant']['key'],chosen['C'],chosen['directions_mean'],flush=True)
 write(OUT/'gate-training.json',gates);write(OUT/'final-search.json',search)
 write(OUT/'pre-evaluation-seal.json',{'source_lock':digest(OUT/'source-lock.json'),'basis':basis_sha,'cv_selection':digest(OUT/'cv-selection.json'),'gates':digest(OUT/'gate-training.json'),'search':digest(OUT/'final-search.json'),'models':{n:digest(CACHE/(n+'.joblib')) for n in models},'before_evaluation_projection':True})
 scores={};labels={};reference_maximum=0.
 for role in ['cal','C','RV','OR_cal','OR_test']:
  ix=roles[role];pa=project(data,ix,basis);oldpa=parent.project(data,ix);scores[role],difference=parent.references(data,role,oldpa);reference_maximum=max(reference_maximum,difference);labels[role]=y[ix]
  for name,model in models.items():scores[role][name]=np.array([entry['head'].decision_function(pack(data['f'][ix],pa,entry['counts'],entry['mean'],entry['scale'])[0]) for entry in model['entries']])
  print('EVALUATED',role,flush=True)
 thresholds={r:{n:[float(np.sort(s[j])[380]) for j in range(3)] for n,s in scores[r].items()} for r in ['cal','OR_cal']};cohorts={}
 from scipy.stats import rankdata
 for role in ['C','RV','OR_test']:
  yy=labels[role];cohorts[role]={}
  for name,ss in scores[role].items():
   stats=[]
   for j,seed in enumerate(SEEDS):
    cut=thresholds['cal'][name][j];s=ss[j];m={'seed':seed,'auroc':float(roc_auc_score(yy,s)),'pauc':float(roc_auc_score(yy,s,max_fpr=.1)),'threshold':cut,'tpr':float((s[yy==1]>cut).mean()),'fpr':float((s[yy==0]>cut).mean()),'fp':int((s[yy==0]>cut).sum())}
    if role=='OR_test':
     cut=thresholds['OR_cal'][name][j];m['source_calibrated']={'threshold':cut,'tpr':float((s[yy==1]>cut).mean()),'fpr':float((s[yy==0]>cut).mean()),'fp':int((s[yy==0]>cut).sum())}
    stats.append(m)
   cohorts[role][name]=stats
  names=FAMILIES+[n+'-cap32' for n in FAMILIES];matrix=np.column_stack([scores[role][n][0] for n in names]);rng=np.random.default_rng(20261105);groups=[np.flatnonzero(yy==j) for j in [0,1]];n0,n1=map(len,groups);dist=np.empty((10000,6))
  for b in range(10000):
   ix=np.r_[rng.choice(groups[0],n0,replace=True),rng.choice(groups[1],n1,replace=True)];ranks_=rankdata(matrix[ix],axis=0);dist[b]=(ranks_[n0:].sum(0)-n1*(n1+1)/2)/(n0*n1)
  cohorts[role]['comparisons']={n:{'control':n+'-cap32','delta_auroc':cohorts[role][n][0]['auroc']-cohorts[role][n+'-cap32'][0]['auroc'],'ci983333':np.quantile(dist[:,j]-dist[:,j+3],[.05/6,1-.05/6]).tolist()} for j,n in enumerate(FAMILIES)}
 np.savez_compressed(OUT/'scores.npz',**{r+'__'+n:s for r,d in scores.items() for n,s in d.items()},**{r+'__y':yy for r,yy in labels.items()})
 summary={n:{k:model[k] for k in ['variant','C','dev_auroc_mean','directions_mean']}|{'counts_by_seed':[entry['counts'] for entry in model['entries']]} for n,model in models.items()}
 write(OUT/'evaluation.json',{'run_id':'EI-PA2','numbering_version':3,'history_only':True,'primary':'AUROC','primary_control':'Each full-range candidate vs same-method same-basis cap32','selected':summary,'ranks':ranks,'energy_retained':[b['retained_energy'] for b in basis],'cohorts':cohorts,'thresholds':thresholds,'cv_lr_fits':len(records),'final_lr_fits':len(search),'gate_fits':len(gates),'reference_reproduction_max_difference':reference_maximum})
 assert len(records)==396 and len(search)==153 and len(gates)==24
 write(OUT/'completion.json',{'lr_fits':len(records)+len(search),'gate_fits':24,'evaluation_sha256':digest(OUT/'evaluation.json'),'scores_sha256':digest(OUT/'scores.npz'),'independent_confirmation':False});print('EI_PA2_COMPLETE',flush=True)

if __name__=='__main__':main()
