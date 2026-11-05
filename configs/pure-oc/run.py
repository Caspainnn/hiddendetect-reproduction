"""One model forward, exact author F at the original last real token; no generation."""
import fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from analysis import ROOT,aggregate,digest,read,write

def main():
    os.chdir(ROOT);os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    directory=ROOT/'cache/pure-oc';directory.mkdir(parents=True,exist_ok=True)
    worker=(directory/'worker.lock').open('w');fcntl.flock(worker,fcntl.LOCK_EX|fcntl.LOCK_NB)
    seal=json.loads((ROOT/'configs/pure-oc/run-lock.json').read_text())
    subprocess.run(['git','merge-base','--is-ancestor',seal['source_commit'],'HEAD'],check=True)
    for name,sha in seal['sha256'].items():
        assert digest(ROOT/name)==sha,name
        assert hashlib.sha256(subprocess.check_output(['git','show',seal['source_commit']+':'+name])).hexdigest()==sha,name
    manifest=json.loads((ROOT/'configs/pure-oc/manifest.json').read_text());rp=ROOT/'data/pure-oc/records.jsonl'
    assert digest(rp)==manifest['records_sha256'];rows=read(rp)
    assert [{k:v for k,v in r.items() if k!='prompt'} for r in rows]==manifest['samples']
    checked={}
    for item in json.loads((ROOT/'configs/artifacts-lock.json').read_text())['files']:
        p=ROOT/item['target'];assert p.stat().st_size==item['bytes'];h=hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
        assert h.hexdigest()==item['sha256'],item['target'];checked[item['target']]=h.hexdigest()
    for name,commit in [('HiddenDetect','2fd161b1fc47c99b36fe0a299ec64ff0b816f0bb'),('LLaVA','c121f0432da27facab705978f83c4ada465e46fd')]:
        p=ROOT/'source'/name
        assert subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()==commit
        assert not subprocess.check_output(['git','-C',str(p),'status','--porcelain','--untracked-files=no'],text=True).strip()
    import torch
    import torch.nn.functional as nnf
    import transformers
    from llava.model.builder import load_pretrained_model
    torch.manual_seed(0)
    tokenizer,model,_,_=load_pretrained_model(str(ROOT/'models/llava-v1.6-vicuna-7b'),None,'llava-v1.6-vicuna-7b',device_map='auto',torch_dtype=torch.float16)
    assert not model.training and not any(p.is_meta for p in model.parameters()) and next(model.parameters()).dtype==torch.float16
    base=json.loads((ROOT/'configs/original-positions/manifest.json').read_text());rv=torch.zeros(32000,device=model.device,dtype=torch.float32)
    rv[[v[0] for v in base['refusal_encodings']]]=1
    def collect(row):
        ids=torch.tensor(tokenizer.encode(manifest['prefix']+row['prompt']+manifest['suffix'])).unsqueeze(0).cuda()
        assert ids.shape[1]<=512 and ids[0,-3:].tolist()==[9047,13566,29901]
        assert hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()==row['input_ids_sha256']
        assert hashlib.sha256(row['prompt'].encode()).hexdigest()==row['prompt_sha256']
        with torch.no_grad():
            output=model(ids,images=None,image_sizes=None,output_hidden_states=True);f=[]
            for state in output.hidden_states[1:]:
                # Full-token projection preserves the author fp16 GEMM shape; only last-token scalar survives.
                logits=model.lm_head(model.model.norm(state))[:,-1,:]
                f.append(float(nnf.cosine_similarity(logits,rv).item()));del logits
            del output
        return f
    # Existing historical private records are used only for identity replay, never new confirmation.
    previous=read(ROOT/'results/original-template-confirm/scores.jsonl')
    prompts={r['id']:r for r in read(ROOT/'data/original-template-confirm/records.jsonl')}
    probes=[]
    for label in (0,1):probes += [r for r in previous if r['label']==label][:8]
    for old in probes:
        row=prompts[old['id']];f=collect(row)
        assert np.array_equal(f,np.asarray(old['F'])[:,-1]),old['id']
        assert aggregate([f])[0,0]==old['score_default'][-1]
    write(ROOT/'logs/pure-oc-identity.json',{'status':'passed','rows':len(probes),'maximum_F_difference':0.,'integral_max_difference':0.,
        'probe_ids':[r['id'] for r in probes],'model_sha256':checked,'torch':torch.__version__,'transformers':transformers.__version__,
        'gpu':torch.cuda.get_device_name(),'manifest_sha256':digest(ROOT/'configs/pure-oc/manifest.json')})
    print('IDENTITY_16_EXACT_PASS',flush=True);began=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for i,row in enumerate(rows):
        p=directory/(hashlib.sha256(row['id'].encode()).hexdigest()+'.json')
        if p.exists():
            result=json.loads(p.read_text());assert result['id']==row['id'] and result['run_lock_sha256']==digest(ROOT/'configs/pure-oc/run-lock.json')
        else:
            f=collect(row);result={k:v for k,v in row.items() if k!='prompt'}
            result.update(F=f,scores=aggregate([f])[0].tolist(),run_lock_sha256=digest(ROOT/'configs/pure-oc/run-lock.json'));write(p,result)
        if (i+1)%100==0:print('OC_SCORED',i+1,'/',len(rows),flush=True)
    out=ROOT/'results/pure-oc';out.mkdir(exist_ok=True)
    target=out/'confirmation-scores.jsonl'
    with target.open('w') as stream:
        for row in rows:stream.write(json.dumps(json.loads((directory/(hashlib.sha256(row['id'].encode()).hexdigest()+'.json')).read_text()))+'\n')
    write(out/'extraction.json',{'status':'completed','rows':len(rows),'wall_seconds_scoring':time.monotonic()-began,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'generation_executed':False,'scores_sha256':digest(target),
        'manifest_sha256':digest(ROOT/'configs/pure-oc/manifest.json'),'run_lock_sha256':digest(ROOT/'configs/pure-oc/run-lock.json')})
    print('OC_A1_N_EXTRACTION_COMPLETE',flush=True)
if __name__=='__main__':main()
