"""Recovery entry point: never retrain or reselect after an existing seal.

Appended after review; original frozen run.py and source lock stay unchanged.
If legacy checkpoints lack cost counters, explicitly report incomplete cost.
"""
from core import *

def execution_phase(done,train_dev,sealed):
 done,train_dev=set(done),set(train_dev)
 if sealed:assert train_dev<=done;return 'reuse_sealed'
 assert done<=train_dev,'Unsealed heldout data must not be used to select models'
 return 'collect_train'

def merge_cost(a,b):
 return {'counts':{k:a.get('counts',{}).get(k,0)+b.get('counts',{}).get(k,0) for k in set(a.get('counts',{}))|set(b.get('counts',{}))},'seconds':{k:a.get('seconds',{}).get(k,0.)+b.get('seconds',{}).get(k,0.) for k in set(a.get('seconds',{}))|set(b.get('seconds',{}))},'maximum_allocated_gpu_bytes':max(a.get('maximum_allocated_gpu_bytes',0),b.get('maximum_allocated_gpu_bytes',0))}

def main():
 import fcntl
 from extract import Extractor
 from fit import train,evaluate
 lock=(CACHE/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);source=json.loads((OUT/'source-lock.json').read_text())
 for name,sha in source['sources'].items():assert digest(ROOT/name)==sha,'Original frozen source changed: '+name
 assert digest(OUT/'manifest.json')==source['manifest_sha256'] and digest(OUT/'grouping.json')==source['grouping_sha256']
 rows,private,roles,y=load();cp=CACHE/'checkpoint.json';checkpoint=json.loads(cp.read_text());done=set(checkpoint['completed']);allowed=sorted(np.r_[roles['train'],roles['dev']].tolist());sealed=(OUT/'pre-evaluation-seal.json').exists();phase=execution_phase(done,allowed,sealed);a=arrays(len(rows),'r+')
 if sealed:
  pre=json.loads((OUT/'pre-evaluation-seal.json').read_text());assert pre['source_lock_sha256']==digest(OUT/'source-lock.json') and pre['search_sha256']==digest(OUT/'search.json') and pre['transforms_sha256']==digest(CACHE/'transforms.joblib');assert pre['train_dev_indices']==allowed
  for name,value in a.items():assert hashlib.sha256(np.asarray(value[allowed]).tobytes()).hexdigest()==pre['train_dev_feature_sha256'][name]
  transforms=joblib.load(CACHE/'transforms.joblib');models={}
  for name,sha in pre['models'].items():assert digest(CACHE/(name+'.joblib'))==sha;models[name]=joblib.load(CACHE/(name+'.joblib'))
  print('RECOVERY_REUSED_SEALED_MODELS_WITHOUT_TRAINING',flush=True)
 x=Extractor();x.full_projection=json.loads((OUT/'preflight.json').read_text())['use_full_projection'];cost=checkpoint.get('cost_totals',{'counts':{},'seconds':{},'maximum_allocated_gpu_bytes':0});complete=checkpoint.get('cost_complete',not done) and not checkpoint.get('cost_segment_open',False)
 # An interrupted legacy process may have performed work after its last saved
 # row; never reconstruct exact counts or seconds from sample counts alone.
 checkpoint['cost_complete']=bool(complete);checkpoint['cost_segment_open']=True;checkpoint.setdefault('recovery_segments',[]).append({'starting_completed':len(done),'phase':phase,'code_sha256':digest(Path(__file__)),'legacy_prefix_cost_missing':bool(done and not checkpoint.get('cost_totals')),'preseal_existing_trial_files':len(list((CACHE/'trials').glob('*.joblib'))) if not sealed else 0});write(cp,checkpoint)
 write(OUT/'recovery-provenance.json',{'original_source_lock_sha256':digest(OUT/'source-lock.json'),'recovery_code_sha256':digest(Path(__file__)),'sealed_models_reused':sealed,'cost_complete':bool(complete),'legacy_cost_policy':'Missing or interrupted work is not fabricated; totals are lower bounds when incomplete','segments':checkpoint['recovery_segments']})
 def save():
  for value in a.values():value.flush()
  checkpoint['completed']=sorted(done);checkpoint['cost_totals']=cost;write(cp,checkpoint);write(CACHE/'ocr.json',x.ocr_cache)
 def collect(indices):
  nonlocal cost
  for i in indices:
   i=int(i)
   if i in done:continue
   before_counts=dict(x.counts);before_seconds=dict(x.seconds);s,v,pools,meta,description=x.row(private[i]);a['signals'][i]=s;a['vision'][i]=v
   for j,kind in enumerate(['roi','uniform','random']):a[kind][i]=pools[:,j]
   delta={'counts':{k:value-before_counts.get(k,0) for k,value in x.counts.items()},'seconds':{k:value-before_seconds.get(k,0.) for k,value in x.seconds.items()},'maximum_allocated_gpu_bytes':x.torch.cuda.max_memory_allocated()};cost=merge_cost(cost,delta);checkpoint['metadata'][str(i)]=meta;checkpoint['descriptions'][str(i)]=description;done.add(i);save()
   if len(done)%20==0:print('RECOVERY_IMAGE_FEATURES',len(done),'/',len(rows),flush=True)
 if phase=='collect_train':
  collect(np.r_[roles['train'],roles['dev']]);assert sorted(done)==allowed;transforms,models=train(a,roles,y);write(OUT/'pre-evaluation-seal.json',{'source_lock_sha256':digest(OUT/'source-lock.json'),'search_sha256':digest(OUT/'search.json'),'transforms_sha256':digest(CACHE/'transforms.joblib'),'models':{name:digest(CACHE/(name+'.joblib')) for name in NAMES},'train_dev_indices':allowed,'train_dev_feature_sha256':{name:hashlib.sha256(np.asarray(value[allowed]).tobytes()).hexdigest() for name,value in a.items()},'before_heldout_feature_access':True})
 collect(np.r_[roles['cal'],roles['test'],roles['task_test']]);assert sorted(done)==list(range(len(rows)));checkpoint['cost_segment_open']=False;save();metadata=list(checkpoint['metadata'].values());write(OUT/'extraction.json',{'completed':len(rows),'source_lock_sha256':digest(OUT/'source-lock.json'),'arrays_sha256':{name:digest(CACHE/(name+'.npy')) for name in a},'records':metadata,'runtime':x.runtime,'forward_counts':cost['counts'],'seconds_by_operation':cost['seconds'],'maximum_allocated_gpu_bytes':cost['maximum_allocated_gpu_bytes'],'cost_complete':bool(complete),'cost_is_lower_bound':not complete,'all_vision_tokens_validated':all(m['valid_image_tokens']>=576 for m in metadata),'all_neutral_description_lengths_equal':all(m['neutral_description_length_equal'] for m in metadata),'ocr_failures':sum(m['ocr_returncode']!=0 for m in metadata),'empty_ocr':sum(m['ocr_characters']==0 for m in metadata),'generated_descriptions_sha256':digest(cp)});x.close();evaluate(a,roles,y,rows,transforms,models)

if __name__=='__main__':main()
