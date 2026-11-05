from core import *

def main():
 import fcntl,hashlib,sys
 from sklearn.linear_model import LogisticRegression
 from sklearn.metrics import roc_auc_score
 from scipy.stats import rankdata
 OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True);lock=(CACHE/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 rows,roles,f,y=data();train,dev=roles['train'],roles['dev'];manifest=PARENT/'results/ei2-vocabulary/manifest.json'
 files=list((ROOT/'configs/ei2-centroid').glob('*.py'))+[ROOT/'configs/ei2-vocabulary/core.py',ROOT/'configs/ei2-vocabulary/verify.py',ROOT/'docs/ei2-centroid-protocol-2026-11-05.md']
 for p in files:assert p.read_bytes()==subprocess.check_output(['git','-C',str(ROOT),'show','HEAD:'+p.relative_to(ROOT).as_posix()])
 source={'run_id':'EI-C1','source_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),'sources':{p.relative_to(ROOT).as_posix():digest(p) for p in files},'parent_manifest':digest(manifest),'parent_evaluation':digest(PARENT/'results/ei2-vocabulary/evaluation.json'),'F':digest(PARENT/'cache/ei2-vocabulary/F.npy'),'history_only':True}
 if (OUT/'source-lock.json').exists():assert json.loads((OUT/'source-lock.json').read_text())==source
 else:write(OUT/'source-lock.json',source)
 write(OUT/'manifest.json',{'run_id':'EI-C1','samples':rows,'history_only':True})
 texts={};wanted={r['id'] for r in rows}
 for path,sha in json.loads(manifest.read_text())['sources'].items():
  p=ORIGIN/path;assert digest(p)==sha
  for q in read_records(p):
   if q['id'] in wanted and 'prompt' in q:
    if q['id'] in texts:assert texts[q['id']]==q['prompt']
    texts[q['id']]=q['prompt']
 assert set(texts)==wanted
 for row in rows:assert hashlib.sha256(texts[row['id']].encode()).hexdigest()==row['prompt_sha256']
 for artifact in json.loads((ORIGIN/'configs/artifacts-lock.json').read_text())['files']:
  p=ORIGIN/artifact['target'];assert p.stat().st_size==artifact['bytes'] and digest(p)==artifact['sha256']
 author=ORIGIN/'source/HiddenDetect';assert not subprocess.check_output(['git','-C',str(author),'status','--porcelain','--untracked-files=no'],text=True).strip()
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.chdir(ORIGIN)
 import torch
 from llava.model.builder import load_pretrained_model
 torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;sys.path.insert(0,str(author/'code'));mod=module('unchanged_author_test',author/'code/test.py')
 tok,model,_,_=load_pretrained_model(str(ORIGIN/'models/llava-v1.6-vicuna-7b'),None,'llava-v1.6-vicuna-7b',device_map='auto',torch_dtype=torch.float16);assert not model.training
 refs=torch.zeros(model.config.vocab_size,device=model.device,dtype=torch.float32);refs[sorted({tok.encode(s,add_special_tokens=False)[0] for s in mod.refusal_lst})]=1
 template=json.loads((ORIGIN/'cache/oc4c-worktree/results/oc-confirmation/manifest.json').read_text());prefix,suffix=template['prefix'],template['suffix']
 hp=CACHE/'H.npy';h=np.lib.format.open_memmap(hp,mode='r+' if hp.exists() else 'w+',dtype=np.float16,shape=(6582,14,4096));fp=CACHE/'forward-F.npy';fresh=np.lib.format.open_memmap(fp,mode='r+' if fp.exists() else 'w+',dtype=np.float64,shape=(6582,14));cp=CACHE/'checkpoint.json';done=set(json.loads(cp.read_text())['completed']) if cp.exists() else set()
 def collect(indices):
  for i in indices:
   i=int(i)
   if i in done:continue
   ids=tok.encode(prefix+texts[rows[i]['id']]+suffix);assert len(ids)<=512 and hashlib.sha256(np.asarray([ids],dtype=np.int64).tobytes()).hexdigest()==rows[i]['input_ids_sha256']
   with torch.no_grad():
    output=model(torch.tensor([ids],device=model.device),images=None,image_sizes=None,output_hidden_states=True);assert len(output.hidden_states)==33
    hh=[];ff=[]
    for state in output.hidden_states[17:31]:
     hh.append(state[0,-1].half().cpu().numpy());logits=model.lm_head(model.model.norm(state))[:,-1,:];ff.append(float(torch.nn.functional.cosine_similarity(logits,refs).item()))
   assert np.max(np.abs(np.array(ff)-f[i]))<=1e-7,('F identity',i);h[i]=np.array(hh);fresh[i]=ff;done.add(i)
   if len(done)%20==0 or i==int(indices[-1]):h.flush();fresh.flush();write(cp,{'completed':sorted(done)})
   if len(done)%200==0:print('EXTRACTED_C14',len(done),'/6582',flush=True)
 collect(np.r_[train,dev]);centroids=fit_centroids(h[train],y[train]);np.save(CACHE/'centroids.npy',centroids);search=[];chosen={}
 for name in NAMES:
  x,mean,scale=pack(f[train],extra(name,h,centroids,train));d=pack(f[dev],extra(name,h,centroids,dev),mean,scale)[0];choices=[]
  for c in GRID:
   entries=[];aucs=[]
   for seed in SEEDS:
    head=LogisticRegression(C=c,solver='liblinear',max_iter=5000,random_state=seed).fit(x,y[train]);assert max(head.n_iter_)<5000;auc=float(roc_auc_score(y[dev],head.decision_function(d)));entry={'head':head,'mean':mean,'scale':scale,'C':c,'seed':seed,'name':name};path=CACHE/f'{name}-C{c}-S{seed}.joblib';joblib.dump(entry,path);entries.append(entry);aucs.append(auc);search.append({'name':name,'C':c,'seed':seed,'dev_auroc':auc,'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path),'n_iter':head.n_iter_.tolist()})
   choices.append({'C':c,'entries':entries,'dev_auroc_mean':float(np.mean(aucs))})
  chosen[name]=max(enumerate(choices),key=lambda j:(j[1]['dev_auroc_mean'],-j[0]))[1];joblib.dump(chosen[name],CACHE/(name+'.joblib'));print('SELECTED',name,chosen[name]['C'],chosen[name]['dev_auroc_mean'],flush=True)
 assert len(search)==27;write(OUT/'search.json',search)
 write(OUT/'pre-evaluation-seal.json',{'source_lock':digest(OUT/'source-lock.json'),'centroids':digest(CACHE/'centroids.npy'),'search':digest(OUT/'search.json'),'models':{name:digest(CACHE/(name+'.joblib')) for name in NAMES},'fit_indices':train.tolist(),'before_heldout_extraction':True})
 collect(np.array([i for i in range(len(rows)) if i not in set(np.r_[train,dev])]))
 assert len(done)==6582 and np.isfinite(h).all();identity=float(np.max(np.abs(fresh-f)));assert identity<=1e-7
 write(OUT/'extraction.json',{'completed':6582,'hidden_shape':list(h.shape),'dtype':str(h.dtype),'F_max_difference':identity,'H_sha256':digest(hp),'forward_F_sha256':digest(fp),'input_ids_checked':6582,'hidden_state_indices':list(range(17,31)),'decoder_layers_0based':list(range(16,30)),'position':'template last','torch':torch.__version__,'author_commit':subprocess.check_output(['git','-C',str(author),'rev-parse','HEAD'],text=True).strip(),'artifact_lock_sha256':digest(ORIGIN/'configs/artifacts-lock.json')})
 del model;torch.cuda.empty_cache();scores={};labels={};parent=np.load(PARENT/'results/ei2-vocabulary/scores.npz');maxbase=0.
 for role in ['cal','C','RV','OR_cal','OR_test']:
  ix=roles[role];labels[role]=y[ix];scores[role]={}
  for name,m in chosen.items():scores[role][name]=np.array([entry['head'].decision_function(pack(f[ix],extra(name,h,centroids,ix),entry['mean'],entry['scale'])[0]) for entry in m['entries']])
  delta=float(np.max(np.abs(scores[role]['B-F']-parent[role+'__B-F'])));assert delta<=1e-7,(role,delta);maxbase=max(maxbase,delta)
 cuts={role:{name:[float(np.sort(s)[380]) for s in ss] for name,ss in scores[role].items()} for role in ['cal','OR_cal']};cohorts={}
 for role in ['C','RV','OR_test']:
  yy=labels[role];d={}
  for name,ss in scores[role].items():
   ms=[]
   for j,seed in enumerate(SEEDS):
    s=ss[j];cut=cuts['cal'][name][j];m={'seed':seed,'auroc':float(roc_auc_score(yy,s)),'pauc':float(roc_auc_score(yy,s,max_fpr=.1)),'tpr':float((s[yy==1]>cut).mean()),'fpr':float((s[yy==0]>cut).mean()),'fp':int((s[yy==0]>cut).sum()),'threshold':cut}
    if role=='OR_test':
     cut=cuts['OR_cal'][name][j];m['source_calibrated']={'threshold':cut,'tpr':float((s[yy==1]>cut).mean()),'fpr':float((s[yy==0]>cut).mean()),'fp':int((s[yy==0]>cut).sum())}
    ms.append(m)
   d[name]=ms
  groups=[np.flatnonzero(yy==label) for label in [0,1]];n0,n1=map(len,groups);matrix=np.column_stack([scores[role][name][0] for name in ['B-F',ARM]]);rng=np.random.default_rng(20261105);dist=[]
  for _ in range(10000):
   ix=np.r_[rng.choice(groups[0],n0,replace=True),rng.choice(groups[1],n1,replace=True)];rank=rankdata(matrix[ix],axis=0);aucs=(rank[n0:].sum(0)-n1*(n1+1)/2)/(n0*n1);dist.append(aucs[1]-aucs[0])
  d['comparison']={'delta_auroc':d[ARM][0]['auroc']-d['B-F'][0]['auroc'],'ci983333':np.quantile(dist,[.05/6,1-.05/6]).tolist()};cohorts[role]=d
 write(OUT/'evaluation.json',{'run_id':'EI-C1','id':ARM,'history_only':True,'cohorts':cohorts,'thresholds':cuts,'selected':{name:{'C':m['C'],'dev_auroc_mean':m['dev_auroc_mean']} for name,m in chosen.items()},'lr_fits':27,'baseline_reproduction_max_difference':maxbase})
 np.savez_compressed(OUT/'scores.npz',**{role+'__'+name:ss for role,d in scores.items() for name,ss in d.items()},**{role+'__y':yy for role,yy in labels.items()});write(OUT/'completion.json',{'evaluation_sha256':digest(OUT/'evaluation.json'),'scores_sha256':digest(OUT/'scores.npz'),'lr_fits':27,'independent_confirmation':False});print('EI_C1_COMPLETE',flush=True)

if __name__=='__main__':main()
