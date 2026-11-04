import importlib.util,json,os,subprocess,sys,time,warnings
from collections import Counter
import numpy as np
import joblib
from core import ROOT,ORIGIN,OUT,CACHE,SEEDS,GRID,digest,read,write,unit,pack

def main():
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
 OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
 import fcntl
 lock=(CACHE/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 h1=ORIGIN/'cache/oc4-worktree';h2=ORIGIN/'cache/oc4c-worktree'
 hm=json.loads((h1/'results/oc-heads/manifest.json').read_text());om=json.loads((h2/'results/oc-confirmation/manifest.json').read_text())
 assert (hm['train_label_budget'],hm['selection_label_budget'])==(1600,600)
 fx=np.load(h1/'cache/oc-heads/features.npz');counts=Counter();rows=[];expected={}
 for r in hm['samples']:
  role=r['cohort'];i=counts[role];counts[role]+=1
  expected[r['id']]=fx[role+'_X'][i].tolist();rows.append({k:r[k] for k in ['id','label','group_id','prompt_sha256','input_ids_sha256']}|{'role':role})
 for r in om['samples']:rows.append({k:r[k] for k in ['id','label','group_id','prompt_sha256','input_ids_sha256']}|{'role':'OR_'+r['role']})
 assert len(rows)==6582
 for key in ['id','group_id','prompt_sha256','input_ids_sha256']:assert len({r[key] for r in rows})==len(rows),key
 worktrees=subprocess.check_output(['git','-C',str(ORIGIN),'worktree','list','--porcelain'],text=True)
 sources={};texts={};wanted={r['id'] for r in rows}
 for line in worktrees.splitlines():
  if not line.startswith('worktree '):continue
  p=__import__('pathlib').Path(line[9:])/'data'
  for path in sorted(p.glob('**/*.jsonl')):
   if path.name not in ['records.jsonl','probes.jsonl']:continue
   sources[str(path.relative_to(ORIGIN))]=digest(path)
   for r in read(path):
    if r['id'] in wanted and 'prompt' in r:
     if r['id'] in texts:assert texts[r['id']]==r['prompt']
     texts[r['id']]=r['prompt']
 assert set(texts)==wanted
 for r in rows:assert __import__('hashlib').sha256(texts[r['id']].encode()).hexdigest()==r['prompt_sha256']
 prefix,suffix=om['prefix'],om['suffix']
 manifest={'run_id':'EI-V1','samples':rows,'counts':dict(Counter(r['role']+':'+str(r['label']) for r in rows)),'history_only':True,'sources':sources,'h1_manifest_sha256':digest(h1/'results/oc-heads/manifest.json'),'h2_manifest_sha256':digest(h2/'results/oc-confirmation/manifest.json')}
 mp=OUT/'manifest.json'
 if mp.exists():assert json.loads(mp.read_text())==manifest
 else:write(mp,manifest)
 sealed={'sources':{str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'configs/ei2-vocabulary').glob('*.py')},'protocol':digest(ROOT/'docs/ei2-vocabulary-protocol-2026-11-04.md'),'manifest':digest(mp),'source_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()}
 for name in list(sealed['sources'])+['docs/ei2-vocabulary-protocol-2026-11-04.md']:
  assert (ROOT/name).read_bytes()==subprocess.check_output(['git','-C',str(ROOT),'show','HEAD:'+name]),name
 sp=OUT/'source-lock.json'
 if sp.exists():assert json.loads(sp.read_text())==sealed
 else:write(sp,sealed)
 print('DATA_FROZEN',manifest['counts'],'ALL_EVALUATION_HISTORICAL',flush=True)
 # The original vision-tower identifier is a relative local alias in ORIGIN.
 # Keep absolute ROOT for outputs, but resolve the unmodified model config there.
 os.chdir(ORIGIN)
 import torch
 import torch.nn.functional as nnf
 torch.set_num_threads(4)
 from llava.model.builder import load_pretrained_model
 torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
 for item in json.loads((ORIGIN/'configs/artifacts-lock.json').read_text())['files']:
  p=ORIGIN/item['target'];assert p.stat().st_size==item['bytes'] and digest(p)==item['sha256']
 author=ORIGIN/'source/HiddenDetect';assert not subprocess.check_output(['git','-C',str(author),'status','--porcelain','--untracked-files=no'],text=True).strip()
 spec=importlib.util.spec_from_file_location('author',author/'code/test.py');mod=importlib.util.module_from_spec(spec);sys.path.insert(0,str(author/'code'));spec.loader.exec_module(mod)
 tok,model,_,_=load_pretrained_model(str(ORIGIN/'models/llava-v1.6-vicuna-7b'),None,'llava-v1.6-vicuna-7b',device_map='auto',torch_dtype=torch.float16)
 V=model.config.vocab_size;coords=sorted({tok.encode(s,add_special_tokens=False)[0] for s in mod.refusal_lst});assert 14+14*len(coords)<=512
 ref=torch.zeros(V,device=model.device,dtype=torch.float32);ref[coords]=1
 rhat=np.zeros(V,np.float32);rhat[coords]=1/np.sqrt(len(coords))
 zpath=CACHE/'Z.npy';fpath=CACHE/'F.npy';exists=zpath.exists()
 z=np.lib.format.open_memmap(zpath,mode='r+' if exists else 'w+',dtype=np.float16,shape=(len(rows),14,V))
 f=np.lib.format.open_memmap(fpath,mode='r+' if exists else 'w+',dtype=np.float64,shape=(len(rows),14))
 done=json.loads((CACHE/'checkpoint.json').read_text()) if (CACHE/'checkpoint.json').exists() else {'completed':[]}
 completed=set(done['completed'])
 def extract(i):
  ids=tok.encode(prefix+texts[rows[i]['id']]+suffix)
  assert len(ids)<=512 and __import__('hashlib').sha256(np.asarray([ids],dtype=np.int64).tobytes()).hexdigest()==rows[i]['input_ids_sha256']
  with torch.no_grad():
   output=model(torch.tensor([ids],device=model.device),images=None,image_sizes=None,output_hidden_states=True)
   zs=[];fs=[]
   for state in output.hidden_states[17:31]:
    logits=model.lm_head(model.model.norm(state))[:,-1,:]
    fs.append(float(nnf.cosine_similarity(logits,ref).item()));zs.append(logits[0].half().cpu().numpy())
  return np.array(zs),np.array(fs)
 probes=[i for i,r in enumerate(rows) if r['role']=='train'][:8];deltas=[]
 for i in probes:
  zi,fi=extract(i);deltas.append(float(np.max(np.abs(fi-np.asarray(expected[rows[i]['id']])))))
 assert max(deltas)<=1e-7,('Original F identity failed',deltas)
 write(OUT/'identity.json',{'rows':8,'maximum_difference':max(deltas),'tolerance':1e-7,'V':V,'refusal_coordinate_count':len(coords)})
 def collect(indices):
  for j,i in enumerate(indices):
   if i in completed:continue
   z[i],f[i]=extract(i);completed.add(int(i))
   if len(completed)%20==0 or j+1==len(indices):
    z.flush();f.flush();write(CACHE/'checkpoint.json',{'completed':sorted(completed)})
   if len(completed)%100==0:print('EXTRACTED',len(completed),'/',len(rows),flush=True)
 train=np.array([i for i,r in enumerate(rows) if r['role']=='train']);dev=np.array([i for i,r in enumerate(rows) if r['role']=='dev'])
 collect(np.r_[train,dev])
 from sklearn.decomposition import PCA
 transforms=[];tp=CACHE/'transforms.joblib'
 if tp.exists():transforms=joblib.load(tp)
 else:
  for l in range(14):
   u,_=unit(z[train,l]);variance=u.var(0,dtype=np.float64);variance[coords]=-np.inf
   picks=np.lexsort((np.arange(V),-variance))[:32];res=u-np.asarray(f[train,l],np.float32)[:,None]*rhat
   pca=PCA(n_components=4,svd_solver='randomized',iterated_power=4,random_state=SEEDS[0]).fit(res)
   transforms.append({'coordinates':picks,'pca':pca});print('TRANSFORM_TRAIN_ONLY',l,flush=True)
  joblib.dump(transforms,tp)
 write(OUT/'transform-seal.json',{'file_sha256':digest(tp),'train_ids_sha256':__import__('hashlib').sha256('\n'.join(rows[i]['id'] for i in train).encode()).hexdigest(),'coordinates':[t['coordinates'].tolist() for t in transforms],'pca_dimension':4,'fitted_on':'train only'})
 def features(ix):
  a=[[],[],[],[],[],[]]
  for l,t in enumerate(transforms):
   u,n=unit(z[ix,l]);a[0].append(u[:,coords]);a[1].append(u[:,t['coordinates']]);a[2].append(t['pca'].transform(u-np.asarray(f[ix,l],np.float32)[:,None]*rhat));a[3].append(n[:,None]);a[4].append(u);a[5].append(np.asarray(z[ix,l],np.float32))
  result={'B-F':np.empty((len(ix),0),np.float32)}
  for j in range(6):result['EI-2-'+str(j+1)]=np.column_stack(a[j]).astype(np.float32)
  result['Fcopy']=np.tile(f[ix],(1,20)).astype(np.float32)
  for j,q in enumerate([14*len(coords),448,56,14]):
   # Independent per-row stream; cohort/global row positions are never model inputs.
   result['Noise-'+str(j+1)]=np.vstack([np.random.default_rng(np.random.SeedSequence([20261104,j,int(i)])).normal(size=q) for i in ix]).astype(np.float32)
  return result
 tx=features(train);dx=features(dev);yt=np.array([rows[i]['label'] for i in train]);yd=np.array([rows[i]['label'] for i in dev])
 from sklearn.linear_model import LogisticRegression
 from sklearn.metrics import roc_auc_score
 models={};search=[]
 for name in tx:
  pad=name not in ['EI-2-5','EI-2-6']
  x,mean,scale=pack(f[train],tx[name],pad=pad);d,_,_=pack(f[dev],dx[name],mean,scale,pad=pad)
  choices=[]
  for c in GRID:
   fits=[];aucs=[]
   for seed in SEEDS:
    started=time.monotonic();head=LogisticRegression(C=c,solver='liblinear',max_iter=5000,random_state=seed).fit(x,yt)
    assert head.n_iter_.max()<5000,('Nonconvergence',name,c,seed)
    auc=float(roc_auc_score(yd,head.decision_function(d)));aucs.append(auc);fits.append(head)
    search.append({'name':name,'C':c,'seed':seed,'dev_auroc':auc,'seconds':time.monotonic()-started})
   choices.append((float(np.mean(aucs)),fits,c))
  best=max(range(3),key=lambda i:(choices[i][0],-i));_,fits,c=choices[best]
  artifact={'mean':mean,'scale':scale,'models':fits,'C':c,'raw_dimension':len(mean),'interface_dimension':x.shape[1],'pad':pad,'nonconstant_columns':int(np.count_nonzero(np.std(x,axis=0))),'rank_train':int(np.linalg.matrix_rank(x.astype(np.float64))) if pad else None}
  joblib.dump(artifact,CACHE/(name+'.joblib'));models[name]=artifact;print('SELECTED',name,c,choices[best][0],flush=True)
  del x,d
 write(OUT/'search.json',search)
 write(OUT/'pre-evaluation-seal.json',{'source_lock':digest(sp),'transform':digest(tp),'models':{n:digest(CACHE/(n+'.joblib')) for n in models},'selected_C':{n:m['C'] for n,m in models.items()},'before_evaluation_feature_access':True})
 del tx,dx
 collect(np.array([i for i,r in enumerate(rows) if r['role'] not in ['train','dev']]))
 # Release GPU weights before CPU evaluation.
 del model;torch.cuda.empty_cache()
 scores={};labels={}
 for role in ['cal','C','RV','OR_cal','OR_test']:
  ix=np.array([i for i,r in enumerate(rows) if r['role']==role]);extra=features(ix);labels[role]=np.array([rows[i]['label'] for i in ix])
  scores[role]={'B-INT':np.trapz(np.asarray(f[ix]),axis=1)}
  for n,m in models.items():
   x,_,_=pack(f[ix],extra[n],m['mean'],m['scale'],pad=m['pad']);scores[role][n]=np.array([h.decision_function(x) for h in m['models']])
  del extra,x
  print('EVALUATED',role,flush=True)
 thresholds={n:[float(np.sort(scores['cal'][n][s])[380]) for s in range(3)] for n in models}
 thresholds['B-INT']=[float(np.sort(scores['cal']['B-INT'])[380])]*3
 or_thresholds={n:[float(np.sort(scores['OR_cal'][n][s])[380]) for s in range(3)] for n in models}
 or_thresholds['B-INT']=[float(np.sort(scores['OR_cal']['B-INT'])[380])]*3
 summary={}
 for role in ['C','RV','OR_test']:
  y=labels[role];summary[role]={}
  for n,s in scores[role].items():
   if n=='B-INT':s=np.tile(s,(3,1))
   metrics=[]
   for seed,cut in enumerate(thresholds[n]):
    metrics.append({'seed':SEEDS[seed],'auroc':float(roc_auc_score(y,s[seed])),'pauc':float(roc_auc_score(y,s[seed],max_fpr=.1)),'tpr':float((s[seed,y==1]>cut).mean()),'fpr':float((s[seed,y==0]>cut).mean()),'fp':int((s[seed,y==0]>cut).sum()),'threshold':cut})
    if role=='OR_test':
     ocut=or_thresholds[n][seed];metrics[-1]['source_calibrated']={'threshold':ocut,'tpr':float((s[seed,y==1]>ocut).mean()),'fpr':float((s[seed,y==0]>ocut).mean()),'fp':int((s[seed,y==0]>ocut).sum())}
   summary[role][n]=metrics
  # Paired primary AUROC bootstrap, same label-stratified draws for all candidates.
  names=['B-F']+['EI-2-'+str(j) for j in range(1,7)];matrix=np.column_stack([scores[role][n][0] for n in names]);rng=np.random.default_rng(20261104)
  from scipy.stats import rankdata
  dist=np.empty((10000,7));classes=[np.flatnonzero(y==j) for j in (0,1)];n0,n1=map(len,classes)
  for b in range(10000):
   ix=np.r_[rng.choice(classes[0],n0,replace=True),rng.choice(classes[1],n1,replace=True)];ranks=rankdata(matrix[ix],method='average',axis=0);dist[b]=(ranks[n0:].sum(0)-n1*(n1+1)/2)/(n0*n1)
  summary[role]['primary_auroc_differences']={n:{'delta':summary[role][n][0]['auroc']-summary[role]['B-F'][0]['auroc'],'ci9875':np.quantile(dist[:,j]-dist[:,0],[.00625,.99375]).tolist()} for j,n in enumerate(names[:5]) if j}
  summary[role]['full_auroc_differences']={names[a]+'_vs_'+names[b]:{'delta':summary[role][names[a]][0]['auroc']-summary[role][names[b]][0]['auroc'],'ci975':np.quantile(dist[:,a]-dist[:,b],[.0125,.9875]).tolist()} for a,b in [(5,0),(6,5)]}
  for n in names[1:]:print('RESULT',role,n,summary[role][n][0],flush=True)
 write(OUT/'evaluation.json',{'history_only':True,'primary':'AUROC','thresholds':thresholds,'OR_thresholds':or_thresholds,'cohorts':summary,'full_vocabulary_endpoints':'executed without compression; same solver and grid','dimensions':{n:{k:m[k] for k in ['raw_dimension','interface_dimension','nonconstant_columns','rank_train']} for n,m in models.items()}})
 np.savez_compressed(OUT/'scores.npz',**{role+'__'+name:values for role,d in scores.items() for name,values in d.items()},**{role+'__y':y for role,y in labels.items()})
 write(OUT/'completion.json',{'status':'six_arms_completed','independent_confirmation':False,'evaluation_sha256':digest(OUT/'evaluation.json'),'score_sha256':digest(OUT/'scores.npz'),'new_training_runs':len(search),'full_endpoints_executed':True})
 print('SIX_ARMS_COMPLETE',flush=True)
if __name__=='__main__':main()
