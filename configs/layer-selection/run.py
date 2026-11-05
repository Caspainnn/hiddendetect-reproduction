"""Single resumable server experiment; private prompts stay under data/."""
import ast,csv,hashlib,importlib.util,json,os,subprocess,sys,time,zipfile
from collections import Counter
from pathlib import Path
import numpy as np
from core import INTERVALS,interval_scores,select,paired_bootstrap
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/layer-selection'
PRIVATE=ROOT/'cache/layer-selection'
sys.path.insert(0,str(ROOT/'configs/signal-selection'))
from experiment_core import normalize_prompt,prompt_label

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def read(p):
 with p.open(encoding='utf-8') as stream:return [json.loads(x) for x in stream if x.strip()]
def dedup(rows):
 tree=ast.parse((ROOT/'configs/swarm/core.py').read_text())
 fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='deduplicate')
 ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'frozen-deduplicate','exec'),ns)
 return ns['deduplicate'](rows)

def build(tok):
 mp=OUT/'manifest.json';rp=ROOT/'data/layer-selection/records.jsonl'
 if mp.exists():
  m=json.loads(mp.read_text());assert sha(rp)==m['records_sha256'];return m,read(rp)
 roots=[ROOT]
 extra=ROOT/'cache/pure-oc-worktree'
 if extra.exists():roots.append(extra)
 old={};excluded=set();hashes=set();sources={}
 for base in roots:
  for p in sorted((base/'data').glob('**/*.jsonl')):
   if p.name not in ('records.jsonl','probes.jsonl'):continue
   sources[str(p.relative_to(ROOT))]=sha(p)
   for x in read(p):
    if 'prompt' in x:
     if x['id'] in old:assert old[x['id']]['prompt']==x['prompt']
     old[x['id']]={'id':x['id'],'prompt':x['prompt'],'role':'excluded'}
  for p in sorted((base/'configs').glob('**/*manifest*.json'))+list((base/'configs/benchmarks').glob('*.json')):
   sources[str(p.relative_to(ROOT))]=sha(p)
   for x in json.loads(p.read_text()).get('samples',[]):
    if isinstance(x,dict):
     if 'id' in x:excluded.add(str(x['id']))
     if 'prompt_sha256' in x:hashes.add(x['prompt_sha256'])
 for x in csv.DictReader((ROOT/'data/xstest_prompts.csv').read_text(encoding='utf-8-sig').splitlines()):
  old['xstest:'+x['id']]={'id':'xstest:'+x['id'],'prompt':x['prompt'],'role':'excluded'}
 with zipfile.ZipFile(ROOT/'source/HiddenDetect/data/few_shot.zip') as z:
  for i,x in enumerate(json.loads(z.read('few_shot/few_shot.json'))):old['old-cal:'+str(i)]={'id':'old-cal:'+str(i),'prompt':x['txt'],'role':'excluded'}
 assert not hashes-{hashlib.sha256(x['prompt'].encode()).hexdigest() for x in old.values()},'Missing historical prompt texts'
 excluded.update(old)
 base=json.loads((ROOT/'configs/original-positions/manifest.json').read_text())
 prefix,suffix=base['prefix'],base['suffix']
 historical=read(ROOT/'data/positions/records.jsonl')
 fixed=[{**r,'role':'check' if r['role']=='dev' else 'cal'} for r in historical if r['role']=='dev' or (r['role']=='calibration_aegis' and r['label']==0)]
 train=[{**r,'role':'sel'} for r in historical if r['role']=='train']
 kept,_=dedup(fixed+train)
 assert len([r for r in kept if r['role']=='check'])==600
 assert len([r for r in kept if r['role']=='cal'])==400
 chosen=[r for r in kept if r['role'] in ('check','cal')]
 for y in (0,1):
  pool=sorted([r for r in kept if r['role']=='sel' and r['label']==y],key=lambda r:hashlib.sha256(('LS1:'+r['id']).encode()).hexdigest())
  assert len(pool)>=128;chosen+=pool[:128]
 normalized={normalize_prompt(r['prompt']) for r in old.values()}
 raw=ROOT/'data/signal-selection/raw/nvidia__Aegis-AI-Content-Safety-Dataset-2.0/train.json';sources[str(raw.relative_to(ROOT))]=sha(raw)
 candidates=[]
 for x in sorted(json.loads(raw.read_text()),key=lambda x:hashlib.sha256(('LS1-fresh:'+str(x['id'])).encode()).hexdigest()):
  sid='aegis:train.json:'+str(x['id']);text=x['prompt']
  if sid in excluded or not text or text.strip().upper()=='REDACTED' or '<image>' in text or normalize_prompt(text) in normalized:continue
  ids=tok.encode(prefix+text+suffix)
  if len(ids)>512:continue
  candidates.append({'id':sid,'prompt':text,'label':prompt_label(x['prompt_label']),'role':'test'})
 print('HISTORY',len(old),'CANDIDATES',len(candidates),flush=True)
 kept,removed=dedup(list(old.values())+candidates)
 available={}
 for y in (0,1):
  pool=[r for r in kept if r['role']=='test' and r['label']==y];available[str(y)]=len(pool)
  assert len(pool)>=1000,('Insufficient fresh samples',available)
  chosen+=pool[:1000]
 rows=[]
 for r in chosen:
  ids=tok.encode(prefix+r['prompt']+suffix)
  rows.append({**{k:r[k] for k in ('id','prompt','label','role')},'group_id':hashlib.sha256(normalize_prompt(r['prompt']).encode()).hexdigest(),'input_length':len(ids),'input_ids_sha256':hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest()})
 assert len(rows)==3256 and len({r['group_id'] for r in rows})==len(rows)
 rp.parent.mkdir(parents=True,exist_ok=True);rp.write_text(''.join(json.dumps(r)+'\n' for r in rows))
 m={'samples':[{k:v for k,v in r.items() if k!='prompt'} for r in rows],'prefix':prefix,'suffix':suffix,'records_sha256':sha(rp),'available_fresh_per_class':available,'history_count':len(old),'historical_sources':sources,'roles':dict(Counter(r['role']+':'+str(r['label']) for r in rows))}
 write(mp,m);write(PRIVATE/'dedup.json',removed);return m,rows

def main():
 os.chdir(ROOT);OUT.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True)
 import fcntl
 lock=(PRIVATE/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(ROOT/'models/llava-v1.6-vicuna-7b',use_fast=False,local_files_only=True)
 m,rows=build(tok)
 source={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')}
 freeze={'sources':source,'manifest_sha256':sha(OUT/'manifest.json'),'model_lock_sha256':sha(ROOT/'configs/artifacts-lock.json'),'remote_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'protocol_sha256':sha(ROOT/'docs/layer-selection-execution-2026-10-30.md')}
 fp=OUT/'run-lock.json'
 if fp.exists():assert json.loads(fp.read_text())==freeze
 else:write(fp,freeze)
 print('MANIFEST_FROZEN',m['roles'],m['available_fresh_per_class'],flush=True)
 import torch
 import torch.nn.functional as nnf
 from llava.model.builder import load_pretrained_model
 torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
 for x in json.loads((ROOT/'configs/artifacts-lock.json').read_text())['files']:
  p=ROOT/x['target'];assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256']
 author=ROOT/'source/HiddenDetect'
 assert subprocess.check_output(['git','-C',str(author),'rev-parse','HEAD'],text=True).strip()=='2fd161b1fc47c99b36fe0a299ec64ff0b816f0bb'
 spec=importlib.util.spec_from_file_location('author',author/'code/test.py');module=importlib.util.module_from_spec(spec);sys.path.insert(0,str(author/'code'));spec.loader.exec_module(module)
 tok,model,_,_=load_pretrained_model(str(ROOT/'models/llava-v1.6-vicuna-7b'),None,'llava-v1.6-vicuna-7b',device_map='auto',torch_dtype=torch.float16)
 ref=torch.zeros(32000,device=model.device,dtype=torch.float32);ref[[tok.encode(x,add_special_tokens=False)[0] for x in module.refusal_lst]]=1
 def extract(r):
  ids=tok.encode(m['prefix']+r['prompt']+m['suffix'])
  if 'input_ids_sha256' in r:assert hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest()==r['input_ids_sha256']
  with torch.no_grad():
   out=model(torch.tensor([ids],device=model.device),images=None,image_sizes=None,output_hidden_states=True)
   f=[float(nnf.cosine_similarity(model.lm_head(model.model.norm(h))[:,-1,:],ref).item()) for h in out.hidden_states[1:]]
  return f
 # Identity probes precede reuse and all new confirmation scores.
 old=read(ROOT/'results/original-positions/scores.jsonl');raw={r['id']:r for r in read(ROOT/'data/positions/records.jsonl')}
 probes=[r for r in old if r['id'] in raw][:8]
 diffs=[float(np.max(np.abs(np.asarray(extract(raw[r['id']]))-np.asarray(r['F'])[:,-1]))) for r in probes]
 assert len(probes)==8 and max(diffs)==0.,('Original F identity failed',diffs)
 write(OUT/'identity.json',{'count':len(probes),'maximum_F_difference':max(diffs),'reference_sha256':sha(ROOT/'results/original-positions/scores.jsonl')})
 cache={r['id']:np.asarray(r['F'])[:,-1].tolist() for r in old}
 def collect(subset):
  result=[]
  for i,r in enumerate(subset):
   p=PRIVATE/(hashlib.sha256(r['id'].encode()).hexdigest()+'.json')
   if p.exists():
    x=json.loads(p.read_text());assert x['lock_sha256']==sha(fp);f=x['F']
   else:
    f=cache[r['id']] if r['id'] in cache else extract(r)
    write(p,{'id':r['id'],'F':f,'lock_sha256':sha(fp),'reused':r['id'] in cache})
   result.append(f)
   if (i+1)%50==0:print('SCORED',r['role'],i+1,'/',len(subset),flush=True)
  return np.asarray(result)
 sel=[r for r in rows if r['role']=='sel'];sf=collect(sel)
 selection=select(sf,[r['label'] for r in sel],[r['group_id'] for r in sel])
 sp=OUT/'selection.json'
 if sp.exists():assert json.loads(sp.read_text())==selection
 else:write(sp,selection)
 assert selection['intervals']['B1'] is not None,'FDV undefined; protocol deviation requires report'
 write(OUT/'pre-evaluation-seal.json',{'selection_sha256':sha(sp),'run_lock_sha256':sha(fp),'status':'selectors_fixed_before_confirmation_scoring'})
 print('SELECTED',selection['intervals'],flush=True)
 allf={r['id']:f.tolist() for r,f in zip(sel,sf)}
 for role in ['cal','check','test']:
  subset=[r for r in rows if r['role']==role];f=collect(subset)
  allf.update({r['id']:v.tolist() for r,v in zip(subset,f)})
 score=lambda role,a,b:np.array([np.trapz(np.asarray(allf[r['id']])[a:b+1]) for r in rows if r['role']==role])
 tests={n:score('test',*ab) for n,ab in selection['intervals'].items()}
 y=np.array([r['label'] for r in rows if r['role']=='test'])
 evaluation=paired_bootstrap(y,tests)
 from sklearn.metrics import roc_auc_score
 secondary={};checky=[r['label'] for r in rows if r['role']=='check']
 for n,ab in selection['intervals'].items():
  threshold=float(np.sort(score('cal',*ab))[380]);s=tests[n]
  secondary[n]={'threshold_safe400_ordinal381':threshold,'pauc_standardized_fpr_0_1':float(roc_auc_score(y,s,max_fpr=.1)),'tpr':float((s[y==1]>threshold).mean()),'fpr':float((s[y==0]>threshold).mean()),'historical_check_auroc':float(roc_auc_score(checky,score('check',*ab)))}
 evaluation.update(intervals=selection['intervals'],secondary=secondary,manifest_sha256=sha(OUT/'manifest.json'),selection_sha256=sha(sp),scope='fresh same-source Aegis train; no unseen-domain claim')
 write(OUT/'evaluation.json',evaluation)
 with (OUT/'scores.jsonl').open('w') as out:
  for r in rows:out.write(json.dumps({k:v for k,v in r.items() if k!='prompt'}|{'F':allf[r['id']]})+'\n')
 write(OUT/'completion.json',{'status':'completed','evaluation_sha256':sha(OUT/'evaluation.json'),'scores_sha256':sha(OUT/'scores.jsonl')})
 print('COMPLETE',evaluation,flush=True)

if __name__=='__main__':main()
