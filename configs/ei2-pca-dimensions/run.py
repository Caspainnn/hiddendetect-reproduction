import hashlib,json,os,subprocess,sys,time
from pathlib import Path
import joblib,numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'configs/ei2-vocabulary'))
from core import SEEDS,GRID,digest,write,unit
ORIGIN=Path(os.environ.get('EI_ORIGIN',str(ROOT)));PARENT=ORIGIN/'cache/ei2-worktree'
OUT=ROOT/'results/ei2-pca-dimensions';CACHE=ROOT/'cache/ei2-pca-dimensions'
NAMES={'EI-2-3-1':1,'EI-2-3-2':2,'K4':4,'EI-2-3-3':8}

def pack_fixed(f,extra,mean=None,scale=None):
 x=np.column_stack([f,extra]).astype(np.float32)
 if mean is None:
  mean=x.mean(0,dtype=np.float64).astype(np.float32);scale=x.std(0,dtype=np.float64).astype(np.float32);scale[scale==0]=1
 x=(x-mean)/scale;x[:,:14]/=np.sqrt(14);x[:,14:]/=np.sqrt(56)
 assert x.shape[1]<=512
 return np.pad(x,((0,0),(0,512-x.shape[1]))),mean,scale

def main():
 import fcntl
 from sklearn.decomposition import PCA
 from sklearn.linear_model import LogisticRegression
 from sklearn.metrics import roc_auc_score
 from scipy.stats import rankdata
 OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
 lock=(CACHE/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (OUT/'evaluation.json').exists(),'Completed run must not be overwritten'
 pv=json.loads((PARENT/'results/ei2-vocabulary/verification.json').read_text());assert pv['passed']
 for name in ['F.npy','Z.npy']:assert digest(PARENT/'cache/ei2-vocabulary'/name)==pv['sha256']['cache/ei2-vocabulary/'+name]
 assert digest(PARENT/'results/ei2-vocabulary/evaluation.json')==pv['evaluation_sha256']
 assert digest(PARENT/'results/ei2-vocabulary/scores.npz')==pv['sha256']['results/ei2-vocabulary/scores.npz']
 rows=json.loads((PARENT/'results/ei2-vocabulary/manifest.json').read_text())['samples'];assert len(rows)==6582
 for key in ['id','group_id','prompt_sha256','input_ids_sha256']:assert len({r[key] for r in rows})==len(rows)
 paths=list((ROOT/'configs/ei2-pca-dimensions').glob('*.py'))+[ROOT/'docs/ei2-pca-dimensions-protocol-2026-11-05.md',ROOT/'configs/ei2-vocabulary/core.py',ROOT/'configs/ei2-vocabulary/verify.py']
 for p in paths:assert p.read_bytes()==subprocess.check_output(['git','-C',str(ROOT),'show','HEAD:'+p.relative_to(ROOT).as_posix()])
 write(OUT/'source-lock.json',{'run_id':'EI-PCA1','source_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),'sources':{p.relative_to(ROOT).as_posix():digest(p) for p in paths},'parent_verification':digest(PARENT/'results/ei2-vocabulary/verification.json'),'parent_manifest':digest(PARENT/'results/ei2-vocabulary/manifest.json'),'history_only':True,'names':NAMES})
 write(OUT/'manifest.json',{'run_id':'EI-PCA1','samples':rows,'parent':'EI-V1','history_only':True})
 z=np.load(PARENT/'cache/ei2-vocabulary/Z.npy',mmap_mode='r');f=np.load(PARENT/'cache/ei2-vocabulary/F.npy',mmap_mode='r')
 roles={r:np.array([i for i,s in enumerate(rows) if s['role']==r]) for r in ['train','dev','cal','C','RV','OR_cal','OR_test']}
 coords=json.loads((ORIGIN/'cache/oc4c-worktree/results/oc23/token-audit.json').read_text())['token_ids'];rhat=np.zeros(z.shape[-1],np.float32);rhat[coords]=1/np.sqrt(len(coords))
 transforms=[]
 for l in range(14):
  u,_=unit(z[roles['train'],l]);res=u-f[roles['train'],l,None].astype(np.float32)*rhat
  pca=PCA(n_components=8,svd_solver='randomized',iterated_power=4,random_state=SEEDS[0]).fit(res);transforms.append(pca)
  print('TRAIN_PCA8',l,flush=True)
 joblib.dump(transforms,CACHE/'transforms.joblib')
 def project(ix,basis):
  return np.stack([t.transform(unit(z[ix,l])[0]-f[ix,l,None].astype(np.float32)*rhat) for l,t in enumerate(basis)],axis=1).astype(np.float32)
 a={role:project(roles[role],transforms) for role in ['train','dev']}
 def extra(role,name):
  if name.startswith('Noise-'):
   k=int(name[6:]);return np.vstack([np.random.default_rng(np.random.SeedSequence([20261105,k,int(i)])).normal(size=14*k) for i in roles[role]]).astype(np.float32)
  return a[role][:,:,:NAMES[name]].reshape(len(roles[role]),-1)
 models={};search=[]
 for name in list(NAMES)+['Noise-'+str(k) for k in [1,2,8]]:
  x,mean,scale=pack_fixed(f[roles['train']],extra('train',name));d,_,_=pack_fixed(f[roles['dev']],extra('dev',name),mean,scale);choices=[]
  for c in GRID:
   fits=[];aucs=[]
   for seed in SEEDS:
    started=time.monotonic();head=LogisticRegression(C=c,solver='liblinear',max_iter=5000,random_state=seed).fit(x,[rows[i]['label'] for i in roles['train']]);assert head.n_iter_.max()<5000
    auc=float(roc_auc_score([rows[i]['label'] for i in roles['dev']],head.decision_function(d)));aucs.append(auc);fits.append(head)
    search.append({'name':name,'C':c,'seed':seed,'dev_auroc':auc,'seconds':time.monotonic()-started,'n_iter':head.n_iter_.tolist()})
   choices.append((np.mean(aucs),fits,c))
  best=max(range(3),key=lambda j:(choices[j][0],-j));_,fits,c=choices[best]
  model={'models':fits,'mean':mean,'scale':scale,'C':c,'raw_dimension':len(mean),'rank_train':int(np.linalg.matrix_rank(x.astype(np.float64)))}
  models[name]=model;joblib.dump(model,CACHE/(name+'.joblib'));print('SELECTED',name,c,float(choices[best][0]),flush=True)
 write(OUT/'search.json',search)
 write(OUT/'pre-evaluation-seal.json',{'source_lock':digest(OUT/'source-lock.json'),'transform':digest(CACHE/'transforms.joblib'),'models':{n:digest(CACHE/(n+'.joblib')) for n in models},'selected_C':{n:m['C'] for n,m in models.items()},'before_new_evaluation_projection':True})
 old_scores=np.load(PARENT/'results/ei2-vocabulary/scores.npz');old_basis=joblib.load(PARENT/'cache/ei2-vocabulary/transforms.joblib');old_model=joblib.load(PARENT/'cache/ei2-vocabulary/EI-2-3.joblib')
 scores={};labels={};maximum=0.
 for role in ['cal','C','RV','OR_cal','OR_test']:
  ix=roles[role];a[role]=project(ix,transforms);labels[role]=np.array([rows[i]['label'] for i in ix]);scores[role]={}
  for name,m in models.items():
   x,_,_=pack_fixed(f[ix],extra(role,name),m['mean'],m['scale']);scores[role][name]=np.array([h.decision_function(x) for h in m['models']])
  old_extra=project(ix,[t['pca'] for t in old_basis]).reshape(len(ix),-1);x,_,_=pack_fixed(f[ix],old_extra,old_model['mean'],old_model['scale'])
  reproduced=np.array([h.decision_function(x) for h in old_model['models']]);maximum=max(maximum,float(np.max(np.abs(reproduced-old_scores[role+'__EI-2-3']))));assert maximum<1e-7
  for name,key in [('old-K4','EI-2-3'),('B-F','B-F')]:scores[role][name]=old_scores[role+'__'+key]
  del a[role];print('PROJECTED',role,flush=True)
 cuts={r:{n:[float(np.sort(s[j])[380]) for j in range(3)] for n,s in scores[r].items()} for r in ['cal','OR_cal']}
 cohorts={}
 for role in ['C','RV','OR_test']:
  y=labels[role];cohorts[role]={}
  for name,s in scores[role].items():
   stats=[]
   for j,seed in enumerate(SEEDS):
    cut=cuts['cal'][name][j];m={'seed':seed,'auroc':float(roc_auc_score(y,s[j])),'pauc':float(roc_auc_score(y,s[j],max_fpr=.1)),'threshold':cut,'tpr':float((s[j,y==1]>cut).mean()),'fpr':float((s[j,y==0]>cut).mean()),'fp':int((s[j,y==0]>cut).sum())}
    if role=='OR_test':
     cut=cuts['OR_cal'][name][j];m['source_calibrated']={'threshold':cut,'tpr':float((s[j,y==1]>cut).mean()),'fpr':float((s[j,y==0]>cut).mean()),'fp':int((s[j,y==0]>cut).sum())}
    stats.append(m)
   cohorts[role][name]=stats
  names=['K4','EI-2-3-1','EI-2-3-2','EI-2-3-3'];matrix=np.column_stack([scores[role][n][0] for n in names]);rng=np.random.default_rng(20261105);classes=[np.flatnonzero(y==j) for j in [0,1]];n0,n1=map(len,classes);dist=np.empty((10000,4))
  for b in range(10000):
   ix=np.r_[rng.choice(classes[0],n0,replace=True),rng.choice(classes[1],n1,replace=True)];ranks=rankdata(matrix[ix],axis=0);dist[b]=(ranks[n0:].sum(0)-n1*(n1+1)/2)/(n0*n1)
  cohorts[role]['comparisons']={n:{'delta_auroc':cohorts[role][n][0]['auroc']-cohorts[role]['K4'][0]['auroc'],'ci983333':np.quantile(dist[:,j]-dist[:,0],[.05/6,1-.05/6]).tolist()} for j,n in enumerate(names) if j}
  print('RESULT',role,{n:cohorts[role][n][0] for n in names},flush=True)
 np.savez_compressed(OUT/'scores.npz',**{r+'__'+n:s for r,d in scores.items() for n,s in d.items()},**{r+'__y':y for r,y in labels.items()})
 write(OUT/'evaluation.json',{'run_id':'EI-PCA1','history_only':True,'primary':'AUROC','primary_control':'K4 common PCA8 basis, first4','mapping':NAMES,'cohorts':cohorts,'thresholds':cuts,'old_K4_reproduction_max_difference':maximum,'dimensions':{n:m['raw_dimension'] for n,m in models.items()}})
 write(OUT/'completion.json',{'fits':len(search),'evaluation_sha256':digest(OUT/'evaluation.json'),'scores_sha256':digest(OUT/'scores.npz'),'independent_confirmation':False});print('EI_PCA1_COMPLETE',flush=True)

if __name__=='__main__':main()
