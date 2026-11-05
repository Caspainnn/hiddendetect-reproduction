from common import *

def main():
 import fcntl
 from sklearn.linear_model import LogisticRegression
 from sklearn.metrics import roc_auc_score
 from sklearn.model_selection import StratifiedKFold
 OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True);lock=(CACHE/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'source-lock.json').exists(),'Inspect existing attempt before restart'
 data=load_data();roles=data['roles'];train,dev=roles['train'],roles['dev'];y=np.array([r['label'] for r in data['rows']]);basis=data['joint_basis'];ranks=[len(b['values']) for b in basis]
 sources=list((ROOT/'configs/ei2-pca-joint').glob('*.py'))+[ROOT/'configs/ei2-pca-select/common.py',ROOT/'configs/ei2-pca-uncapped/common.py',ROOT/'configs/ei2-pca-auto/common.py',ROOT/'configs/ei2-pca-auto/verify.py',ROOT/'configs/ei2-vocabulary/core.py',ROOT/'configs/ei2-vocabulary/verify.py',ROOT/'docs/ei2-pca-joint-protocol-2026-11-05.md']
 for p in sources:assert p.read_bytes()==subprocess.check_output(['git','-C',str(ROOT),'show','HEAD:'+p.relative_to(ROOT).as_posix()]),str(p)
 write(OUT/'source-lock.json',{'run_id':'EI-PJ1','numbering_version':3,'source_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),'sources':{p.relative_to(ROOT).as_posix():digest(p) for p in sources},'parent_verification':digest(BASE/'results/ei2-pca-uncapped/verification.json'),'parent_preseal':digest(BASE/'results/ei2-pca-uncapped/pre-evaluation-seal.json'),'parent_manifest':digest(PARENT/'results/ei2-vocabulary/manifest.json'),'history_only':True})
 write(OUT/'manifest.json',{'run_id':'EI-PJ1','samples':data['rows'],'history_only':True})
 selectors=[]
 def selector(indices,pa,ix,lam,rho,key,fold=None):
  x,mean,scale=pack(data['f'][ix],pa,indices);x=x[:,:14+sum(map(len,indices))];state=s.fit_sparse(x,y[ix],lam,rho);artifact={'state':state,'indices':indices,'mean':mean,'scale':scale,'fit_indices':ix,'fold':fold};path=CACHE/'selectors'/(key+'.joblib');path.parent.mkdir(exist_ok=True);joblib.dump(artifact,path)
  record={'key':key,'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path),'lambda':lam,'rho':rho,'fold':fold,'kkt':state['kkt'],'iterations':state['iterations'],'seconds':state['seconds'],'directions':int(state['mask'].sum()),'fit_indices':ix.tolist()};selectors.append(record);print('SELECTOR',key,record['directions'],record['kkt'],record['seconds'],flush=True);return artifact,record
 cvrecords=[];foldrecords=[];cvarray=np.zeros((3,len(u.TAGS),3));cvcounts=np.zeros((3,len(u.TAGS)),int)
 expected_folds=list(StratifiedKFold(3,shuffle=True,random_state=SEEDS[0]).split(train,y[train]))
 parentpre=json.loads((BASE/'results/ei2-pca-uncapped/pre-evaluation-seal.json').read_text());assert digest(BASE/'results/ei2-pca-uncapped/cv-selection.json')==parentpre['cv_selection']
 for fold in data['parent_cv']['folds']:
  j=fold['fold'];ti,vi=train[expected_folds[j][0]],train[expected_folds[j][1]];np.testing.assert_array_equal(ti,fold['train_indices']);np.testing.assert_array_equal(vi,fold['validation_indices']);path=BASE/fold['basis_path'];assert digest(path)==fold['basis_sha256'];fb=joblib.load(path);franks=[len(b['values']) for b in fb];assert min(franks)>=1024
  for b in fb:np.testing.assert_array_equal(b['train_indices'],ti)
  ta,va=u.project(data,ti,fb),u.project(data,vi,fb);artifact,record=selector(full_indices(franks),ta,ti,.0001,.5,f'fold-{j}-EN',j);order=rank_directions(artifact['state']['weight'],franks)
  foldrecords.append({'fold':j,'train_indices':ti.tolist(),'validation_indices':vi.tolist(),'basis_path':str(path.relative_to(ORIGIN)),'basis_sha256':digest(path),'selector':record,'rank':min(franks),'order':[row.tolist() for row in order]})
  for tagindex,tag in enumerate(u.TAGS):
   k=u.count_for(tag,min(franks));cvcounts[j,tagindex]=k;indices=top_indices(order,[k]*14);x,mean,scale=pack(data['f'][ti],ta,indices);d=pack(data['f'][vi],va,indices,mean,scale)[0]
   for cindex,c in enumerate(GRID):
    start=time.monotonic();head=LogisticRegression(C=c,solver='liblinear',max_iter=5000,random_state=SEEDS[0]).fit(x,y[ti]);assert max(head.n_iter_)<5000;auc=float(roc_auc_score(y[vi],head.decision_function(d)));cvarray[j,tagindex,cindex]=auc
    path=CACHE/'cv'/f'f{j}-{tag}-c{cindex}.joblib';path.parent.mkdir(exist_ok=True);joblib.dump({'head':head,'mean':mean,'scale':scale,'indices':indices},path)
    cvrecords.append({'fold':j,'tag':tag,'k':k,'C':c,'auroc':auc,'n_iter':head.n_iter_.tolist(),'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path),'seconds':time.monotonic()-start})
   print('CV_DONE',j,tag,k,flush=True)
 selection=u.choose_cv(cvarray,u.TAGS,cvcounts);write(OUT/'cv-selection.json',{'selection':selection,'folds':foldrecords,'scores':cvarray.tolist(),'counts':cvcounts.tolist(),'records':cvrecords,'tags':u.TAGS})
 pa={role:u.project(data,roles[role],basis) for role in ['train','dev']};variants={CV:[],POOL:[],GATE:[],'ranked-gate-zero':[],'S-EN':[],'S-L1':[]};full=full_indices(ranks);ranking=None;ranking_record=None
 for family,rho in [('S-EN',.5),('S-L1',1.)]:
  for lam in LAMBDAS:
   artifact,record=selector(full,pa['train'],train,lam,rho,f'{family}-{lam}');indices=mask_indices(artifact['state']['mask'],ranks);variants[family].append({'key':f'lambda-{lam}','lambda':lam,'rho':rho,'indices_by_seed':[indices]*3,'selector':record})
   if family=='S-EN' and lam==.0001:ranking=rank_directions(artifact['state']['weight'],ranks);ranking_record=record
 k=u.count_for(selection['tag'],min(ranks));indices=top_indices(ranking,[k]*14);variants[CV]=[{'key':'cv-'+selection['tag'],'indices_by_seed':[indices]*3,'cv_selection':selection,'rank_selector':ranking_record}]
 for tau in [.8,.9,.95]:
  counts,coverage,capped=u.variance_counts(basis,tau);assert not any(capped);pool=full_indices(counts)
  for lam in LAMBDAS:
   artifact,record=selector(pool,pa['train'],train,lam,1.,f'pool-{tau}-L1-{lam}');indices=map_pool(pool,artifact['state']['mask']);variants[POOL].append({'key':f'tau-{tau}-lambda-{lam}','tau':tau,'lambda':lam,'rho':1.,'pool_counts':counts,'pool_coverage':coverage,'selected_coverage':selected_coverage(basis,indices),'indices_by_seed':[indices]*3,'selector':record})
 gx,_,_=pack(data['f'][train],pa['train'],ranking);gates=[]
 for lam in [0.,.01,.1,1.]:
  indices_list=[];records=[]
  for seed in SEEDS:
   state=u.fit_gate(gx,y[train],ranks,lam,seed);check_prefix_hardening(state['soft_gates'],state['counts']);path=CACHE/f'gate-{lam}-{seed}.joblib';joblib.dump(state,path);indices_list.append(map_prefix(ranking,state['counts']));record={key:state[key] for key in ['lambda','seed','counts','ranks','seconds','steps','loss_start_end']}|{'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path)};gates.append(record);records.append(record);print('GATE_DONE',lam,seed,sum(state['counts']),flush=True)
  variants['ranked-gate-zero' if lam==0 else GATE].append({'key':f'lambda-{lam}','lambda':lam,'rank_selector':ranking_record,'gates':records,'indices_by_seed':indices_list})
 search=[];models={}
 def heads(family,variant,c):
  entries=[];aucs=[]
  for j,seed in enumerate(SEEDS):
   indices=variant['indices_by_seed'][j];x,mean,scale=pack(data['f'][train],pa['train'],indices);d=pack(data['f'][dev],pa['dev'],indices,mean,scale)[0];start=time.monotonic();head=LogisticRegression(C=c,solver='liblinear',max_iter=5000,random_state=seed).fit(x,y[train]);assert max(head.n_iter_)<5000;auc=float(roc_auc_score(y[dev],head.decision_function(d)));aucs.append(auc)
   entry={'head':head,'mean':mean,'scale':scale,'indices':indices,'seed':seed,'C':c,'variant':variant['key']};path=CACHE/'trials'/f'{family}-{variant["key"]}-C{c}-S{seed}.joblib';path.parent.mkdir(exist_ok=True);joblib.dump(entry,path);entries.append(entry)
   search.append({'family':family,'variant':variant['key'],'C':c,'seed':seed,'counts':[len(ix) for ix in indices],'effective_dimension':14+sum(map(len,indices)),'interface_dimension':x.shape[1],'dev_auroc':auc,'n_iter':head.n_iter_.tolist(),'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path),'seconds':time.monotonic()-start})
  return {'variant':variant,'C':c,'dev_auroc_mean':float(np.mean(aucs)),'directions_mean':float(np.mean([sum(map(len,ix)) for ix in variant['indices_by_seed']])),'entries':entries}
 for family,candidates in variants.items():
  choices=[heads(family,variant,c) for variant in candidates for c in GRID];chosen=max(enumerate(choices),key=lambda j:(j[1]['dev_auroc_mean'],-j[1]['directions_mean'],-j[0]))[1];models[family]=chosen;print('SELECTED',family,chosen['variant']['key'],chosen['C'],chosen['directions_mean'],flush=True)
 for family in FAMILIES:
  model=models[family];indices=[full_indices([len(ix) for ix in indices]) for indices in model['variant']['indices_by_seed']];models[family+'-prefix']=heads(family+'-prefix',{'key':'same-count-prefix','indices_by_seed':indices,'matched_to':family},model['C'])
 model=models[POOL];indices=full_indices(model['variant']['pool_counts']);models['same-tau-pool']=heads('same-tau-pool',{'key':'same-tau','indices_by_seed':[indices]*3,'tau':model['variant']['tau'],'matched_to':POOL},model['C'])
 for name,model in models.items():joblib.dump(model,CACHE/(name+'.joblib'))
 write(OUT/'selector-training.json',selectors);write(OUT/'gate-training.json',gates);write(OUT/'final-search.json',search)
 write(OUT/'pre-evaluation-seal.json',{'source_lock':digest(OUT/'source-lock.json'),'cv_selection':digest(OUT/'cv-selection.json'),'selectors':digest(OUT/'selector-training.json'),'gates':digest(OUT/'gate-training.json'),'search':digest(OUT/'final-search.json'),'models':{name:digest(CACHE/(name+'.joblib')) for name in models},'before_evaluation_projection':True})
 scores={};labels={};maximum=0.
 for role in ['cal','C','RV','OR_cal','OR_test']:
  ix=roles[role];a=u.project(data,ix,basis);scores[role],difference=replay_references(data,role,a);maximum=max(maximum,difference);labels[role]=y[ix]
  for name,model in models.items():scores[role][name]=np.array([entry['head'].decision_function(pack(data['f'][ix],a,entry['indices'],entry['mean'],entry['scale'])[0]) for entry in model['entries']])
  print('EVALUATED',role,flush=True)
 thresholds={role:{name:[float(np.sort(ss[j])[380]) for j in range(3)] for name,ss in values.items()} for role,values in scores.items() if role in ['cal','OR_cal']};cohorts={}
 from scipy.stats import rankdata
 names=['Q-CV','Q-VAR','Q-GATE','S-EN','S-L1']+FAMILIES
 for role in ['C','RV','OR_test']:
  yy=labels[role];cohorts[role]={}
  for name,ss in scores[role].items():
   stats=[]
   for j,seed in enumerate(SEEDS):
    cut=thresholds['cal'][name][j];score=ss[j];m={'seed':seed,'auroc':float(roc_auc_score(yy,score)),'pauc':float(roc_auc_score(yy,score,max_fpr=.1)),'threshold':cut,'tpr':float((score[yy==1]>cut).mean()),'fpr':float((score[yy==0]>cut).mean()),'fp':int((score[yy==0]>cut).sum())}
    if role=='OR_test':
     cut=thresholds['OR_cal'][name][j];m['source_calibrated']={'threshold':cut,'tpr':float((score[yy==1]>cut).mean()),'fpr':float((score[yy==0]>cut).mean()),'fp':int((score[yy==0]>cut).sum())}
    stats.append(m)
   cohorts[role][name]=stats
  matrix=np.column_stack([scores[role][n][0] for n in names]);classes=[np.flatnonzero(yy==j) for j in [0,1]];n0,n1=map(len,classes);rng=np.random.default_rng(20261105);dist=np.empty((10000,8))
  for b in range(10000):
   ix=np.r_[rng.choice(classes[0],n0,replace=True),rng.choice(classes[1],n1,replace=True)];rank=rankdata(matrix[ix],axis=0);dist[b]=(rank[n0:].sum(0)-n1*(n1+1)/2)/(n0*n1)
  cohorts[role]['comparisons']={n:{control:{'delta_auroc':cohorts[role][n][0]['auroc']-cohorts[role][control][0]['auroc'],'ci991667':np.quantile(dist[:,names.index(n)]-dist[:,names.index(control)],[.05/12,1-.05/12]).tolist()} for control in CONTROLS[n]} for n in FAMILIES}
  print('COHORT_COMPLETE',role,flush=True)
 np.savez_compressed(OUT/'scores.npz',**{role+'__'+name:ss for role,values in scores.items() for name,ss in values.items()},**{role+'__y':yy for role,yy in labels.items()})
 summary={}
 for name,model in models.items():
  variant={k:v for k,v in model['variant'].items() if k!='indices_by_seed'};summary[name]={'variant':variant,'C':model['C'],'dev_auroc_mean':model['dev_auroc_mean'],'directions_mean':model['directions_mean'],'counts_by_seed':[[len(ix) for ix in entry['indices']] for entry in model['entries']],'indices_1based_by_seed':[[[int(k)+1 for k in ix] for ix in entry['indices']] for entry in model['entries']],'is_pca_prefix_by_seed':[all(np.array_equal(ix,np.arange(len(ix))) for ix in entry['indices']) for entry in model['entries']]}
 write(OUT/'evaluation.json',{'run_id':'EI-PJ1','numbering_version':3,'history_only':True,'primary':'AUROC','controls':CONTROLS,'selected':summary,'cohorts':cohorts,'thresholds':thresholds,'ranks':ranks,'reference_reproduction_max_difference':maximum,'cv_lr_fits':len(cvrecords),'final_lr_fits':len(search),'selector_fits':len(selectors),'gate_fits':len(gates)})
 assert len(cvrecords)==396 and len(search)==192 and len(selectors)==18 and len(gates)==12
 write(OUT/'completion.json',{'lr_fits':588,'selector_fits':18,'gate_fits':12,'evaluation_sha256':digest(OUT/'evaluation.json'),'scores_sha256':digest(OUT/'scores.npz'),'independent_confirmation':False});print('EI_PJ1_COMPLETE',flush=True)

if __name__=='__main__':main()
