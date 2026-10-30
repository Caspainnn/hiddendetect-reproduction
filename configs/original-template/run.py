"""Sequential second group: exact original-rule template neighbors."""
import csv,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'configs/original-positions'))
from core import digest,read_jsonl,project_readouts
IDS=['EI-3-2-2','EI-3-2-3','EI-3-2-1']

def scores(f,start=16,end=29):
    f=np.asarray(f,dtype=np.float64);assert f.shape==(32,3) and np.isfinite(f).all()
    return [float(np.trapz(f[start:end+1,j])) for j in range(3)]

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)

def main():
    os.chdir(ROOT);os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    cache=ROOT/'cache/original-template';cache.mkdir(exist_ok=True)
    lock=(cache/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    frozen=json.loads((ROOT/'configs/original-template/run-lock.json').read_text())
    subprocess.run(['git','merge-base','--is-ancestor',frozen['source_commit'],'HEAD'],check=True)
    for name,sha in frozen['sha256'].items():
        assert digest(ROOT/name)==sha,name
        assert hashlib.sha256(subprocess.check_output(['git','show',frozen['source_commit']+':'+name])).hexdigest()==sha,name
    for item in json.loads((ROOT/'configs/artifacts-lock.json').read_text())['files']:
        path=ROOT/item['target'];assert path.stat().st_size==item['bytes'];h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
        assert h.hexdigest()==item['sha256'],item['target']
    print('LOCKED_MODEL_HASHES_PASS',flush=True)
    import torch
    import torch.nn.functional as nnf
    from llava.model.builder import load_pretrained_model
    for directory,commit in [('HiddenDetect','2fd161b1fc47c99b36fe0a299ec64ff0b816f0bb'),('LLaVA','c121f0432da27facab705978f83c4ada465e46fd')]:
        path=ROOT/'source'/directory
        assert subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()==commit
        assert not subprocess.check_output(['git','-C',str(path),'status','--porcelain','--untracked-files=no'],text=True).strip()
    torch.manual_seed(0);np.random.seed(0)
    tokenizer,model,_,_=load_pretrained_model(str(ROOT/'models/llava-v1.6-vicuna-7b'),None,'llava-v1.6-vicuna-7b',device_map='auto',torch_dtype=torch.float16)
    assert not model.training and not any(p.is_meta for p in model.parameters())
    mp=ROOT/'configs/original-positions/manifest.json';um=json.loads(mp.read_text())
    previous=read_jsonl(ROOT/'results/original-positions/scores.jsonl')
    ex=json.loads((ROOT/'results/original-positions/extraction.json').read_text())
    assert ex['manifest_sha256']==digest(mp) and ex['scores_sha256']==digest(ROOT/'results/original-positions/scores.jsonl')
    assert [x['id'] for x in previous]==[x['id'] for x in um['samples']]
    records=ROOT/'data/positions/records.jsonl';assert digest(records)==um['records_sha256']
    prompts={x['id']:x['prompt'] for x in read_jsonl(records)}
    prompts.update({'xstest:'+x['id']:x['prompt'] for x in csv.DictReader((ROOT/'data/xstest_prompts.csv').read_text(encoding='utf-8-sig').splitlines())})
    reference=torch.zeros(32000,device=model.device,dtype=torch.float32)
    reference[[x[0] for x in um['refusal_encodings']]]=1
    source_names=['configs/original-template/run.py','configs/original-template/evaluate.py','configs/original-template/test_contract.py','configs/original-positions/core.py']
    manifest={'run_id':'ORP-T1','directions':IDS,'names':['template_minus2','template_minus3','template_last'],
              'user_manifest_sha256':digest(mp),'user_scores_sha256':digest(ROOT/'results/original-positions/scores.jsonl'),
              'source_sha256':{n:digest(ROOT/n) for n in source_names},'run_lock_sha256':digest(ROOT/'configs/original-template/run-lock.json'),
              'primary_interval':[16,29],'sensitivity_interval':[17,29],'samples':um['samples']}
    target=ROOT/'configs/original-template/manifest.json'
    if target.exists():assert json.loads(target.read_text())==manifest
    else:write(target,manifest)
    key=digest(target);began=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for i,old in enumerate(previous):
        path=cache/(hashlib.sha256(old['id'].encode()).hexdigest()+'.json')
        if path.exists():result=json.loads(path.read_text());assert result['manifest_sha256']==key
        else:
            text=um['prefix']+prompts[old['id']]+um['suffix'];ids=torch.tensor(tokenizer.encode(text)).unsqueeze(0).cuda()
            assert ids.shape[-1]==old['input_length'] and hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()==old['input_ids_sha256']
            assert ids[0,-3:].tolist()==old['suffix_last3_token_ids']==[9047,13566,29901]
            with torch.no_grad():
                out=model(ids,images=None,image_sizes=None,output_hidden_states=True);f=[]
                for state in out.hidden_states[1:]:
                    logits=project_readouts(state,[-2,-3,-1],model.model.norm,model.lm_head)
                    f.append([float(nnf.cosine_similarity(logits[:,j,:],reference).item()) for j in range(3)])
                    del logits
                del out
            result={k:old[k] for k in ('id','label','role','input_ids_sha256')}
            result.update(manifest_sha256=key,F=f,score_default=scores(f),score_fewshot=scores(f,17,29))
            write(path,result)
        assert np.array_equal(np.asarray(result['F'])[:,-1],np.asarray(old['F'])[:,-1]),('Baseline identity failed',old['id'])
        assert result['score_default'][-1]==old['score_default'][-1] and result['score_fewshot'][-1]==old['score_fewshot'][-1]
        if (i+1)%100==0:print('SCORED',i+1,'/',len(previous),flush=True)
    directory=ROOT/'results/original-template';directory.mkdir(exist_ok=True)
    temp=directory/'scores.tmp'
    with temp.open('w') as f:
        for old in previous:f.write((cache/(hashlib.sha256(old['id'].encode()).hexdigest()+'.json')).read_text().replace('\n','')+'\n')
    temp.replace(directory/'scores.jsonl')
    write(directory/'extraction.json',{'status':'completed','run_id':'ORP-T1','rows':len(previous),'manifest_sha256':key,
          'scores_sha256':digest(directory/'scores.jsonl'),'maximum_same_batch_baseline_F_difference':0.,
          'both_integral_baselines_exact':True,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
          'wall_seconds_scoring_this_invocation':time.monotonic()-began,'generation_executed':False})
    print('ORP_T1_EXTRACTION_COMPLETE',flush=True)

if __name__=='__main__':main()
