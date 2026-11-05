"""Frozen RV ablations, separate historical calibration and prospective scoring."""
import argparse,fcntl,hashlib,json,os,subprocess,time
import numpy as np
from core import ROOT,NAMES,digest,read,write,references,integrate,threshold,evaluate,validate_confirmation

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['history','confirmation'],required=True);phase=parser.parse_args().phase
    os.chdir(ROOT);os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    out=ROOT/'results/oc-reference';cache=ROOT/'cache/oc-reference';cache.mkdir(parents=True,exist_ok=True)
    worker=(cache/'worker.lock').open('w');fcntl.flock(worker,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock=json.loads((out/'source-lock-v2.json').read_text())
    subprocess.run(['git','merge-base','--is-ancestor',lock['source_commit'],'HEAD'],check=True)
    for name,h in lock['sha256'].items():
        assert digest(ROOT/name)==h,name
        assert hashlib.sha256(subprocess.check_output(['git','show',lock['source_commit']+':'+name])).hexdigest()==h,name
    if phase=='confirmation':
        validate_confirmation(out)
    manifest=json.loads((out/'manifest.json').read_text());rp=ROOT/'data/oc-reference/records.jsonl';assert digest(rp)==manifest['records_sha256']
    rows=read(rp);assert [{k:v for k,v in r.items() if k!='prompt'} for r in rows]==manifest['samples']
    modelhash={}
    for item in json.loads((ROOT/'configs/artifacts-lock.json').read_text())['files']:
        p=ROOT/item['target'];assert p.stat().st_size==item['bytes'];h=hashlib.sha256()
        with p.open('rb') as stream:
            for part in iter(lambda:stream.read(8*1024**2),b''):h.update(part)
        assert h.hexdigest()==item['sha256'];modelhash[item['target']]=h.hexdigest()
    for name,commit in [('HiddenDetect','2fd161b1fc47c99b36fe0a299ec64ff0b816f0bb'),('LLaVA','c121f0432da27facab705978f83c4ada465e46fd')]:
        p=ROOT/'source'/name;assert subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()==commit
        assert not subprocess.check_output(['git','-C',str(p),'status','--porcelain','--untracked-files=no'],text=True).strip()
    import torch
    import torch.nn.functional as nnf
    from llava.model.builder import load_pretrained_model
    torch.manual_seed(0)
    tok,model,_,_=load_pretrained_model(str(ROOT/'models/llava-v1.6-vicuna-7b'),None,'llava-v1.6-vicuna-7b',device_map='auto',torch_dtype=torch.float16)
    assert not model.training and not any(p.is_meta for p in model.parameters()) and next(model.parameters()).dtype==torch.float16
    rules=json.loads((ROOT/'configs/oc-reference/rules.json').read_text());audit=json.loads((out/'token-audit.json').read_text())
    encoded={r['string']:tok.encode(r['string'],add_special_tokens=False) for r in audit['tokens']}
    assert all(encoded[r['string']]==r['full_encoding'] for r in audit['tokens'])
    w=references(rules,encoded);rv=torch.tensor(w,device=model.device);union=np.flatnonzero(w.any(0)).tolist()
    def collect(row):
        ids=torch.tensor(tok.encode(manifest['prefix']+row['prompt']+manifest['suffix'])).unsqueeze(0).cuda()
        assert ids.shape[1]<=512 and ids[0,-3:].tolist()==[9047,13566,29901]
        assert hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()==row['input_ids_sha256']
        assert hashlib.sha256(row['prompt'].encode()).hexdigest()==row['prompt_sha256']
        f=[];selected=[];norm=[]
        with torch.no_grad():
            result=model(ids,images=None,image_sizes=None,output_hidden_states=True)
            for state in result.hidden_states[1:]:
                logits=model.lm_head(model.model.norm(state))[:,-1,:]
                f.append([float(nnf.cosine_similarity(logits,rv[j]).item()) for j in range(4)])
                selected.append(logits[0,union].float().cpu().tolist());norm.append(float(logits.float().norm().item()));del logits
            del result
        return f,selected,norm
    old={r['id']:r for r in read(ROOT/'results/original-positions/scores.jsonl')}
    fixed=[r for r in rows if r['role']!='confirmation'];probes=[]
    for y in (0,1):probes += [r for r in fixed if r['label']==y][:8]
    for r in probes:
        f,_,_=collect(r);assert np.array_equal(np.array(f)[:,0],np.array(old[r['id']]['F'])[:,-1]),r['id']
        assert integrate(f)[0]==old[r['id']]['score_default'][-1]
    write(out/('identity-'+phase+'.json'),{'status':'passed','probe_count':16,'maximum_original_F_difference':0.,'integral_difference':0.,
        'model_sha256':modelhash,'probe_ids':[r['id'] for r in probes],'source_lock_sha256':digest(out/'source-lock-v2.json')})
    selected_rows=fixed if phase=='history' else [r for r in rows if r['role']=='confirmation']
    began=time.monotonic();torch.cuda.reset_peak_memory_stats();saved=[]
    for i,r in enumerate(selected_rows):
        p=cache/(hashlib.sha256(r['id'].encode()).hexdigest()+'.json')
        if p.exists():
            value=json.loads(p.read_text());assert value['source_lock_sha256']==digest(out/'source-lock-v2.json') and value['id']==r['id']
        else:
            f,z,n=collect(r);value={k:v for k,v in r.items() if k!='prompt'}
            value.update(F=f,scores=integrate(f).tolist(),union_token_ids=union,union_logits=z,full_logit_norm=n,source_lock_sha256=digest(out/'source-lock-v2.json'))
            if phase=='confirmation':value['pre_confirmation_sha256']=digest(out/'pre-confirmation-seal.json')
            write(p,value)
        if phase=='history':assert np.array_equal(np.array(value['F'])[:,0],np.array(old[r['id']]['F'])[:,-1])
        saved.append(value)
        if (i+1)%100==0:print('RV_SCORED',phase,i+1,'/',len(selected_rows),flush=True)
    path=out/(phase+'-scores.jsonl');path.write_bytes(''.join(json.dumps(r)+'\n' for r in saved).encode())
    write(out/('extraction-'+phase+'.json'),{'status':'completed','count':len(saved),'seconds':time.monotonic()-began,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'generation':False,'scores_sha256':digest(path),'model_sha256':modelhash})
    if phase=='history':
        cal=[r for r in saved if r['role']=='calibration'];assert len(cal)==400 and all(r['label']==0 for r in cal)
        cut=[threshold(np.array([r['scores'] for r in cal])[:,j]) for j in range(4)]
        write(out/'thresholds.json',{'names':NAMES,'thresholds':cut,'ordinal':381,'calibration_ids':[r['id'] for r in cal]})
        test=[r for r in saved if r['role']=='historical_C'];evaluation=evaluate(test,cut,'historical-C')
        assert abs(evaluation['metrics'][0]['auroc']-.746640)<1e-12
        write(out/'historical-evaluation.json',evaluation)
    else:
        cut=json.loads((out/'thresholds.json').read_text())['thresholds'];write(out/'confirmation-evaluation.json',evaluate(saved,cut,'new-confirmation'))
    print('RV_PHASE_COMPLETE',phase,flush=True)
if __name__=='__main__':main()
