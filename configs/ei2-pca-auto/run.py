import json,subprocess,time
import joblib,numpy as np
from common import *

def main():
 import fcntl
 from sklearn.decomposition import PCA
 from sklearn.model_selection import StratifiedKFold
 from sklearn.linear_model import LogisticRegression
 from sklearn.metrics import roc_auc_score
 from scipy.stats import rankdata
 OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
 lock=(CACHE/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (OUT/'source-lock.json').exists(),'Inspect old attempt before relaunching'
 data=load_data(check_hashes=True);rows=data['rows'];roles=data['roles'];train=roles['train'];dev=roles['dev'];y=np.array([r['label'] for r in rows])
 sources=list((ROOT/'configs/ei2-pca-auto').glob('*.py'))+[ROOT/'docs/ei2-pca-auto-protocol-2026-11-05.md',ROOT/'configs/ei2-vocabulary/core.py',ROOT/'configs/ei2-vocabulary/verify.py']
 for p in sources:assert p.read_bytes()==subprocess.check_output(['git','-C',str(ROOT),'show','HEAD:'+p.relative_to(ROOT).as_posix()])
 write(OUT/'source-lock.json',{'run_id':'EI-PA1','numbering_version':3,'source_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),'sources':{p.relative_to(ROOT).as_posix():digest(p) for p in sources},'parent_verification':digest(BASE/'results/ei2-pca-extended/verification.json'),'parent_basis':digest(BASE/'cache/ei2-pca-extended/transforms.joblib'),'history_only':True})
 write(OUT/'manifest.json',{'run_id':'EI-PA1','samples':rows,'history_only':True,'numbering_version':3})
 cvscores=np.zeros((3,32,3));cvrecords=[];folds=[]
 for fold,(ft,fv) in enumerate(StratifiedKFold(3,shuffle=True,random_state=SEEDS[0]).split(train,y[train])):
  ti,vi=train[ft],train[fv];basis=[]
  for l in range(14):
   u,_=unit(data['z'][ti,l]);res=u-data['f'][ti,l,None].astype(np.float32)*data['rhat'];basis.append(PCA(n_components=32,svd_solver='randomized',iterated_power=4,random_state=SEEDS[0]).fit(res))
  path=CACHE/f'fold-{fold}-basis.joblib';joblib.dump(basis,path)
  folds.append({'fold':fold,'train_indices':ti.tolist(),'validation_indices':vi.tolist(),'basis_sha256':digest(path)})
  ta,va=project(data,ti,basis),project(data,vi,basis)
  for k in range(1,33):
   x,mean,scale=pack(data['f'][ti],ta,[k]*14);d,_,_=pack(data['f'][vi],va,[k]*14,mean,scale)
   for j,c in enumerate(GRID):
    started=time.monotonic();head=LogisticRegression(C=c,solver='liblinear',max_iter=5000,random_state=SEEDS[0]).fit(x,y[ti]);assert head.n_iter_.max()<5000
    auc=float(roc_auc_score(y[vi],head.decision_function(d)));cvscores[fold,k-1,j]=auc
    path=CACHE/'cv'/f'f{fold}-k{k}-c{j}.joblib';path.parent.mkdir(exist_ok=True);joblib.dump({'head':head,'mean':mean,'scale':scale,'counts':[k]*14},path)
    cvrecords.append({'fold':fold,'k':k,'C':c,'seed':SEEDS[0],'auroc':auc,'n_iter':head.n_iter_.tolist(),'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path),'seconds':time.monotonic()-started})
  print('CV_FOLD_COMPLETE',fold,flush=True)
 selection=choose_cv(cvscores);write(OUT/'cv-selection.json',{'selection':selection,'folds':folds,'scores':cvscores.tolist(),'records':cvrecords});print('CV_SELECTED',selection,flush=True)
 a={role:project(data,roles[role]) for role in ['train','dev']}
 variants={CV:[{'key':'cv-one-se','counts':[[selection['k']]*14]*3,'selection':selection}],VAR:[],GATE:[],'gate-no-penalty':[]}
 for tau in TAUS:
  counts,coverage,capped=variance_counts(data['basis'],tau);variants[VAR].append({'key':f'tau-{tau}','tau':tau,'counts':[counts]*3,'coverage':coverage,'capped':capped})
 full,_,_=pack(data['f'][train],a['train'],[32]*14)
 gaterecords=[]
 for lam in [0.]+LAMBDAS:
  counts=[];paths=[]
  for seed in SEEDS:
   gate=fit_gate(full,y[train],lam,seed);path=CACHE/f'gate-{lam}-{seed}.joblib';joblib.dump(gate,path);counts.append(gate['counts']);paths.append({'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path)})
   gaterecords.append({k:gate[k] for k in ['seed','lambda','device','steps','seconds','loss_start_end','counts','minimum_threshold_margin']}|paths[-1]);print('GATE_TRAINED',lam,seed,gate['counts'],flush=True)
  family='gate-no-penalty' if lam==0 else GATE;variants[family].append({'key':f'lambda-{lam}','lambda':lam,'counts':counts,'gates':paths})
 write(OUT/'gate-training.json',gaterecords);models={};search=[]
 for family,candidates in variants.items():
  choices=[]
  for variant in candidates:
   for c in GRID:
    entries=[];aucs=[]
    for j,seed in enumerate(SEEDS):
     counts=variant['counts'][j];x,mean,scale=pack(data['f'][train],a['train'],counts);d,_,_=pack(data['f'][dev],a['dev'],counts,mean,scale)
     started=time.monotonic();head=LogisticRegression(C=c,solver='liblinear',max_iter=5000,random_state=seed).fit(x,y[train]);assert head.n_iter_.max()<5000
     auc=float(roc_auc_score(y[dev],head.decision_function(d)));aucs.append(auc)
     artifact={'head':head,'mean':mean,'scale':scale,'counts':counts,'seed':seed,'C':c,'variant':variant['key']};path=CACHE/'final-trials'/f'{family}-{variant["key"]}-C{c}-S{seed}.joblib';path.parent.mkdir(exist_ok=True);joblib.dump(artifact,path)
     entries.append(artifact);search.append({'family':family,'variant':variant['key'],'C':c,'seed':seed,'counts':counts,'dev_auroc':auc,'n_iter':head.n_iter_.tolist(),'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path),'seconds':time.monotonic()-started})
    choices.append({'dev_auroc_mean':float(np.mean(aucs)),'entries':entries,'variant':variant,'C':c,'directions_mean':float(np.mean([sum(s) for s in variant['counts']]))})
  best=max(range(len(choices)),key=lambda j:(choices[j]['dev_auroc_mean'],-choices[j]['directions_mean'],-j));chosen=choices[best]
  for entry in chosen['entries']:
   x,_,_=pack(data['f'][train],a['train'],entry['counts'],entry['mean'],entry['scale']);entry['rank_train']=int(np.linalg.matrix_rank(x.astype(np.float64)))
  joblib.dump(chosen,CACHE/(family+'.joblib'));models[family]=chosen;print('FINAL_SELECTED',family,chosen['variant'],chosen['C'],chosen['dev_auroc_mean'],flush=True)
 write(OUT/'final-search.json',search)
 write(OUT/'pre-evaluation-seal.json',{'source_lock':digest(OUT/'source-lock.json'),'parent_basis':digest(BASE/'cache/ei2-pca-extended/transforms.joblib'),'models':{n:digest(CACHE/(n+'.joblib')) for n in models},'cv_selection':digest(OUT/'cv-selection.json'),'gate_training':digest(OUT/'gate-training.json'),'before_evaluation_projection':True})
 scores={};labels={};maximum=0.
 for role in ['cal','C','RV','OR_cal','OR_test']:
  ix=roles[role];pa=project(data,ix);labels[role]=y[ix];scores[role],difference=references(data,role,pa);maximum=max(maximum,difference)
  for family,m in models.items():
   values=[]
   for entry in m['entries']:
    x,_,_=pack(data['f'][ix],pa,entry['counts'],entry['mean'],entry['scale']);values.append(entry['head'].decision_function(x))
   scores[role][family]=np.array(values)
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
  names=['K8',CV,VAR,GATE];matrix=np.column_stack([scores[role][n][0] for n in names]);rng=np.random.default_rng(20261105);classes=[np.flatnonzero(yy==j) for j in [0,1]];n0,n1=map(len,classes);dist=np.empty((10000,4))
  for b in range(10000):
   ix=np.r_[rng.choice(classes[0],n0,replace=True),rng.choice(classes[1],n1,replace=True)];rank=rankdata(matrix[ix],axis=0);dist[b]=(rank[n0:].sum(0)-n1*(n1+1)/2)/(n0*n1)
  cohorts[role]['comparisons']={n:{'delta_auroc':cohorts[role][n][0]['auroc']-cohorts[role]['K8'][0]['auroc'],'ci983333':np.quantile(dist[:,j]-dist[:,0],[.05/6,1-.05/6]).tolist()} for j,n in enumerate(names) if j}
  print('RESULT',role,{n:cohorts[role][n][0] for n in names},flush=True)
 np.savez_compressed(OUT/'scores.npz',**{r+'__'+n:s for r,d in scores.items() for n,s in d.items()},**{r+'__y':yy for r,yy in labels.items()})
 summary={n:{k:m[k] for k in ['variant','C','dev_auroc_mean','directions_mean']}|{'counts_by_seed':[entry['counts'] for entry in m['entries']],'rank_train_by_seed':[entry['rank_train'] for entry in m['entries']]} for n,m in models.items()}
 write(OUT/'evaluation.json',{'run_id':'EI-PA1','numbering_version':3,'history_only':True,'primary':'AUROC','primary_control':'K8 same PCA32 basis','selected':summary,'cohorts':cohorts,'thresholds':cuts,'reference_reproduction_max_difference':maximum,'cv_lr_fits':len(cvrecords),'final_lr_fits':len(search),'gate_training_runs':len(gaterecords)})
 assert len(cvrecords)==288 and len(search)==72 and len(gaterecords)==12
 write(OUT/'completion.json',{'lr_fits':360,'gate_fits':12,'evaluation_sha256':digest(OUT/'evaluation.json'),'scores_sha256':digest(OUT/'scores.npz'),'independent_confirmation':False});print('EI_PA1_COMPLETE',flush=True)

if __name__=='__main__':main()
