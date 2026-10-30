"""Frozen candidate confirmation; no selection or threshold fitting on new inputs."""
import csv,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'configs/original-positions'))
from core import digest,read_jsonl,project_readouts

def score(f,start=16,end=29):
    f=np.asarray(f,dtype=np.float64);assert f.shape==(32,2) and np.isfinite(f).all()
    return [float(np.trapz(f[start:end+1,j])) for j in range(2)]

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def main():
    os.chdir(ROOT);os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    cache=ROOT/'cache/original-template-confirm';cache.mkdir(exist_ok=True)
    lock=(cache/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    seal=json.loads((ROOT/'configs/original-template-confirm/run-lock.json').read_text())
    subprocess.run(['git','merge-base','--is-ancestor',seal['source_commit'],'HEAD'],check=True)
    for name,sha in seal['sha256'].items():assert digest(ROOT/name)==sha,name
    mp=ROOT/'configs/original-template-confirm/manifest.json';m=json.loads(mp.read_text())
    rp=ROOT/'data/original-template-confirm/records.jsonl';assert digest(rp)==m['records_sha256']
    rows=read_jsonl(rp);assert [{k:v for k,v in x.items() if k!='prompt'} for x in rows]==m['samples']
    for item in json.loads((ROOT/'configs/artifacts-lock.json').read_text())['files']:
        path=ROOT/item['target'];assert path.stat().st_size==item['bytes'];h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
        assert h.hexdigest()==item['sha256']
    import torch
    import torch.nn.functional as nnf
    import transformers
    from llava.model.builder import load_pretrained_model
    for directory,commit in [('HiddenDetect','2fd161b1fc47c99b36fe0a299ec64ff0b816f0bb'),('LLaVA','c121f0432da27facab705978f83c4ada465e46fd')]:
        path=ROOT/'source'/directory
        assert subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()==commit
        assert not subprocess.check_output(['git','-C',str(path),'status','--porcelain','--untracked-files=no'],text=True).strip()
    torch.manual_seed(0)
    tokenizer,model,_,_=load_pretrained_model(str(ROOT/'models/llava-v1.6-vicuna-7b'),None,'llava-v1.6-vicuna-7b',device_map='auto',torch_dtype=torch.float16)
    assert not model.training and not any(p.is_meta for p in model.parameters())
    base=json.loads((ROOT/'configs/original-positions/manifest.json').read_text())
    reference=torch.zeros(32000,device=model.device,dtype=torch.float32);reference[[x[0] for x in base['refusal_encodings']]]=1
    def collect(row):
        ids=torch.tensor(tokenizer.encode(m['prefix']+row['prompt']+m['suffix'])).unsqueeze(0).cuda()
        assert hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()==row['input_ids_sha256']
        assert hashlib.sha256(row['prompt'].encode()).hexdigest()==row['prompt_sha256']
        assert ids[0,-3:].tolist()==[9047,13566,29901]
        with torch.no_grad():
            output=model(ids,images=None,image_sizes=None,output_hidden_states=True);f=[]
            for state in output.hidden_states[1:]:
                logits=project_readouts(state,[-2,-1],model.model.norm,model.lm_head)
                f.append([float(nnf.cosine_similarity(logits[:,j,:],reference).item()) for j in range(2)])
                del logits
            del output
        return f
    previous=read_jsonl(ROOT/'results/original-template/scores.jsonl')
    prompts={x['id']:x['prompt'] for x in read_jsonl(ROOT/'data/positions/records.jsonl')}
    prompts.update({'xstest:'+x['id']:x['prompt'] for x in csv.DictReader((ROOT/'data/xstest_prompts.csv').read_text(encoding='utf-8-sig').splitlines())})
    probes=[x for x in previous if x['role']=='identity_xstest'][:8]
    for y in (0,1):probes += [x for x in previous if x['role']=='test_aegis_holdout' and x['label']==y][:4]
    for old in probes:
        row={**old,'prompt':prompts[old['id']],'prompt_sha256':hashlib.sha256(prompts[old['id']].encode()).hexdigest()}
        f=collect(row)
        assert np.array_equal(f,np.asarray(old['F'])[:,[0,2]])
        assert score(f)==[old['score_default'][j] for j in [0,2]]
        assert score(f,17,29)==[old['score_fewshot'][j] for j in [0,2]]
    write(ROOT/'logs/original-template-confirm-identity.json',{'status':'passed','rows':16,'maximum_F_difference':0.,'both_integrals_exact':True,
          'ids':[x['id'] for x in probes],'manifest_sha256':digest(mp),'run_lock_sha256':digest(ROOT/'configs/original-template-confirm/run-lock.json'),
          'torch':torch.__version__,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name()})
    print('IDENTITY_16_PASS',flush=True);began=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for i,row in enumerate(rows):
        path=cache/(hashlib.sha256(row['id'].encode()).hexdigest()+'.json')
        if path.exists():result=json.loads(path.read_text());assert result['manifest_sha256']==digest(mp) and result['id']==row['id']
        else:
            f=collect(row);result={k:v for k,v in row.items() if k!='prompt'}
            result.update(F=f,score_default=score(f),score_fewshot=score(f,17,29),manifest_sha256=digest(mp));write(path,result)
        if (i+1)%100==0:print('SCORED',i+1,'/',len(rows),flush=True)
    directory=ROOT/'results/original-template-confirm';directory.mkdir(exist_ok=True);temp=directory/'scores.tmp'
    with temp.open('w') as f:
        for row in rows:f.write(json.dumps(json.loads((cache/(hashlib.sha256(row['id'].encode()).hexdigest()+'.json')).read_text()))+'\n')
    temp.replace(directory/'scores.jsonl')
    write(directory/'extraction.json',{'status':'completed','rows':len(rows),'manifest_sha256':digest(mp),'scores_sha256':digest(directory/'scores.jsonl'),
          'run_lock_sha256':digest(ROOT/'configs/original-template-confirm/run-lock.json'),'generation_executed':False,
          'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'wall_seconds_scoring':time.monotonic()-began})
    print('ORP_T2_EXTRACTION_COMPLETE',flush=True)

if __name__=='__main__':main()
