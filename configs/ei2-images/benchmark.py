"""Post-verification deployment examples on eight fixed TRAIN samples only."""
from core import *
from PIL import Image,ImageEnhance
import io

def main():
 from input_coverage import audit
 audit()
 from extract import Extractor
 proof=json.loads((OUT/'verification.json').read_text());assert proof['passed'] and proof['evaluation_sha256']==digest(OUT/'evaluation.json');rows,private,roles,y=load();pre=json.loads((OUT/'pre-evaluation-seal.json').read_text());t=joblib.load(CACHE/'transforms.joblib');assert digest(CACHE/'transforms.joblib')==pre['transforms_sha256'];a=arrays(len(rows));x=Extractor();x.full_projection=json.loads((OUT/'preflight.json').read_text())['use_full_projection'];records=[]
 for name in [BASE]+ARMS:
  path=CACHE/(name+'.joblib');assert digest(path)==pre['models'][name];entry=joblib.load(path)['entries'][0]
  for i in roles['train'][:8]:
   i=int(i);row=private[i];x.signal_cache.clear();x.ocr_cache.pop(row['image_sha256'],None);x.torch.cuda.reset_peak_memory_stats();x.torch.cuda.synchronize();start=time.monotonic();im=Image.open(RAW/row['image_path']).convert('RGB');f,v,pools,meta=x.mm(row['prompt'],im,name==ARMS[3]);signals=np.zeros((1,9,14));signals[0,0]=f;data={'signals':signals,'vision':v[None]};ocr_seconds=0.
   if name==ARMS[1]:
    ocr=x.ocr(row);ocr_seconds=ocr['seconds'];caption,_=x.caption(row['prompt'],im);signals[0,3]=x.text_signal(x.data_ids(row['prompt'],ocr['text'],caption))
   elif name==ARMS[2]:signals[0,1]=x.text_signal(x.ids(row['prompt']))
   elif name==ARMS[3]:
    for j,kind in enumerate(['roi','uniform','random']):data[kind]=pools[:,j][None]
   elif name==ARMS[4]:
    stream=io.BytesIO();im.save(stream,format='JPEG',quality=90);stream.seek(0)
    for j,view in enumerate([Image.open(stream).convert('RGB'),ImageEnhance.Brightness(im).enhance(.95)],7):signals[0,j]=x.mm(row['prompt'],view)[0]
   ff,extra=feature(name,np.array([0]),data,t);z=pack(ff,extra,entry['mean'],entry['scale'])[0];score=float(entry['head'].decision_function(z)[0]);x.torch.cuda.synchronize();seconds=time.monotonic()-start;peak=x.torch.cuda.max_memory_allocated();expected_f,expected_extra=feature(name,np.array([i]),a,t);expected=float(entry['head'].decision_function(pack(expected_f,expected_extra,entry['mean'],entry['scale'])[0])[0]);assert abs(score-expected)<1e-5,(name,i,score,expected)
   records.append({'name':name,'id':row['id'],'role':'train','seconds':seconds,'maximum_allocated_gpu_bytes':peak,'ocr_seconds':ocr_seconds,'score_replay_difference':abs(score-expected)})
  print('IMAGE_TRAIN_DEPLOYMENT_BENCHMARK',name,flush=True)
 x.close();write(OUT/'benchmark.json',{'passed':True,'evaluation_sha256':digest(OUT/'evaluation.json'),'code_sha256':digest(Path(__file__)),'records':records,'samples_per_arm':8,'scope':'First eight fixed TRAIN samples; warm filesystem, cleared text/OCR feature caches; model already loaded; includes extraction and seed0 head; not a population latency estimate','no_test_used':True})
if __name__=='__main__':main()
