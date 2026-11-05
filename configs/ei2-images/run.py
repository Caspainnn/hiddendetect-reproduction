from core import *
from extract import Extractor
from fit import train,evaluate
import subprocess

def main():
 import fcntl
 OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True);lock=(CACHE/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);rows,private,roles,y=load();n=len(rows);assert n==1270
 files=list((ROOT/'configs/ei2-images').glob('*.py'))+[ROOT/'configs/ei2-vocabulary/verify.py',ROOT/'docs/ei2-images-protocol-2026-11-05.md']
 for path in files:assert path.read_bytes()==subprocess.check_output(['git','-C',str(ROOT),'show','HEAD:'+path.relative_to(ROOT).as_posix()]),str(path)
 source={'run_id':'EI-IM1','source_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),'sources':{p.relative_to(ROOT).as_posix():digest(p) for p in files},'manifest_sha256':digest(OUT/'manifest.json'),'grouping_sha256':digest(OUT/'grouping.json'),'acquisition_sha256':digest(ORIGIN/'cache/ei2-image-acquisition/acquisition.json'),'model_lock_sha256':digest(ORIGIN/'configs/artifacts-lock.json'),'ocr_lock_sha256':digest(ORIGIN/'cache/ei2-image-ocr/lock.json'),'preflight_sha256':digest(OUT/'preflight.json')}
 if (OUT/'source-lock.json').exists():assert json.loads((OUT/'source-lock.json').read_text())==source
 else:write(OUT/'source-lock.json',source)
 proof=json.loads((OUT/'preflight.json').read_text());assert proof['passed'] and proof['all_five_interfaces'];x=Extractor();x.full_projection=proof['use_full_projection'];a=arrays(n,'r+');cp=CACHE/'checkpoint.json';checkpoint=json.loads(cp.read_text()) if cp.exists() else {'completed':[],'metadata':{},'descriptions':{}};done=set(checkpoint['completed'])
 def save():
  for values in a.values():values.flush()
  checkpoint['completed']=sorted(done);write(cp,checkpoint);write(CACHE/'ocr.json',x.ocr_cache)
 def collect(indices):
  for i in indices:
   i=int(i)
   if i in done:continue
   signals,vision,pools,meta,description=x.row(private[i]);a['signals'][i]=signals;a['vision'][i]=vision
   for j,kind in enumerate(['roi','uniform','random']):a[kind][i]=pools[:,j]
   checkpoint['metadata'][str(i)]=meta;checkpoint['descriptions'][str(i)]=description;done.add(i)
   if len(done)%5==0 or i==int(indices[-1]):save()
   if len(done)%20==0:print('REAL_IMAGE_FEATURES',len(done),'/',n,'SECONDS',round(sum(x.seconds.values()),1),flush=True)
 collect(np.r_[roles['train'],roles['dev']]);allowed=sorted(np.r_[roles['train'],roles['dev']].tolist());assert sorted(done)==allowed,'heldout accessed before selection';transforms,models=train(a,roles,y)
 write(OUT/'pre-evaluation-seal.json',{'source_lock_sha256':digest(OUT/'source-lock.json'),'search_sha256':digest(OUT/'search.json'),'transforms_sha256':digest(CACHE/'transforms.joblib'),'models':{name:digest(CACHE/(name+'.joblib')) for name in NAMES},'train_dev_indices':allowed,'train_dev_feature_sha256':{name:hashlib.sha256(np.asarray(value[allowed]).tobytes()).hexdigest() for name,value in a.items()},'before_heldout_feature_access':True})
 collect(np.r_[roles['cal'],roles['test'],roles['task_test']]);assert sorted(done)==list(range(n));save();metadata=list(checkpoint['metadata'].values());write(OUT/'extraction.json',{'completed':n,'source_lock_sha256':digest(OUT/'source-lock.json'),'arrays_sha256':{name:digest(CACHE/(name+'.npy')) for name in a},'records':metadata,'runtime':x.runtime,'forward_counts':x.counts,'seconds_by_operation':x.seconds,'maximum_allocated_gpu_bytes':x.torch.cuda.max_memory_allocated(),'all_vision_tokens_validated':all(m['valid_image_tokens']>=576 for m in metadata),'all_neutral_description_lengths_equal':all(m['neutral_description_length_equal'] for m in metadata),'ocr_failures':sum(m['ocr_returncode']!=0 for m in metadata),'empty_ocr':sum(m['ocr_characters']==0 for m in metadata),'generated_descriptions_sha256':digest(cp)});x.close();evaluate(a,roles,y,rows,transforms,models)
if __name__=='__main__':main()
