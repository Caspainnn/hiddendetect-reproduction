"""Separate historical extraction/calibration and genuinely fresh scoring, if available."""
import argparse,fcntl,hashlib,json,math,os,subprocess,time
import numpy as np
from core import ROOT,NAMES,digest,read,write,aggregate,threshold,evaluate

def validate_fresh(out):
    pre=json.loads((out/'pre-confirmation-seal.json').read_text())
    required=['source-lock.json','manifest.json','token-audit.json','thresholds.json','history-scores.jsonl',
              'historical-C-evaluation.json','historical-RV-evaluation.json','identity-history.json','extraction-history.json']
    assert set(pre['sha256'])==set(required) and pre['family']==NAMES[1:]
    for name,h in pre['sha256'].items():assert digest(out/name)==h,name
    m=json.loads((out/'manifest.json').read_text());assert m['confirmation_enabled'] and min(m['selected_per_class'].values())>=30
    assert abs(json.loads((out/'historical-C-evaluation.json').read_text())['metrics'][0]['auroc']-.746640)<1e-12
    assert json.loads((out/'identity-history.json').read_text())['maximum_F_difference']==0
    assert json.loads((out/'extraction-history.json').read_text())['count']==2482

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['history','confirmation'],required=True);phase=parser.parse_args().phase
    os.chdir(ROOT);os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    out=ROOT/'results/oc23';cache=ROOT/'cache/oc23';cache.mkdir(parents=True,exist_ok=True)
    worker=(cache/'worker.lock').open('w');fcntl.flock(worker,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock=json.loads((out/'source-lock.json').read_text())
    subprocess.run(['git','merge-base','--is-ancestor',lock['source_commit'],'HEAD'],check=True)
    for name,h in lock['sha256'].items():
        assert digest(ROOT/name)==h,name
        assert hashlib.sha256(subprocess.check_output(['git','show',lock['source_commit']+':'+name])).hexdigest()==h,name
    if phase=='confirmation':validate_fresh(out)
    m=json.loads((out/'manifest.json').read_text());rp=ROOT/'data/oc23/records.jsonl';assert digest(rp)==m['records_sha256'];rows=read(rp)
    assert [{k:v for k,v in r.items() if k!='prompt'} for r in rows]==m['samples']
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
    audit=json.loads((out/'token-audit.json').read_text());refids=audit['token_ids']
    assert all(tok.encode(r['string'],add_special_tokens=False)==r['full_encoding'] for r in audit['tokens'])
    rv=torch.zeros(32000,device=model.device,dtype=torch.float32);rv[refids]=1.
    def collect(row):
        ids=torch.tensor(tok.encode(m['prefix']+row['prompt']+m['suffix'])).unsqueeze(0).cuda()
        assert ids.shape[1]<=512 and ids[0,-3:].tolist()==[9047,13566,29901]
        assert hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()==row['input_ids_sha256']
        assert hashlib.sha256(row['prompt'].encode()).hexdigest()==row['prompt_sha256']
        f=[];layer=[];selected=[];logz=[];norm=[]
        with torch.no_grad():
            result=model(ids,images=None,image_sizes=None,output_hidden_states=True)
            for state in result.hidden_states[1:]:
                logits=model.lm_head(model.model.norm(state))[:,-1,:]
                f.append(float(nnf.cosine_similarity(logits,rv).item()))
                z=logits.float();v=z[0,refids];ls=torch.logsumexp(v,dim=0);den=torch.logsumexp(z,dim=1)[0]
                layer.append([float(v.mean().item()),float((ls-math.log(len(refids))).item()),float(torch.exp(ls-den).item())])
                selected.append(v.cpu().tolist());logz.append(float(den.item()));norm.append(float(z.norm().item()));del logits,z,v
            del result
        return {'F':f,'layer_scores':layer,'refusal_logits':selected,'full_logsumexp':logz,'full_logit_norm':norm,'refusal_token_ids':refids}
    old={r['id']:np.array(r['F'])[:,-1] for r in read(ROOT/'results/original-positions/scores.jsonl')}
    for name in ['history','confirmation']:
        for r in read(ROOT/('results/oc-reference/'+name+'-scores.jsonl')):
            f=np.array(r['F'])[:,0]
            if r['id'] in old:assert np.array_equal(old[r['id']],f)
            old[r['id']]=f
    fixed=[r for r in rows if r['role']!='confirmation'];probes=[]
    for label in (0,1):probes += [r for r in fixed if r['label']==label][:8]
    for r in probes:assert np.array_equal(collect(r)['F'],old[r['id']]),r['id']
    write(out/('identity-'+phase+'.json'),{'status':'passed','probe_count':16,'maximum_F_difference':0.,'model_sha256':modelhash,
        'probe_ids':[r['id'] for r in probes],'source_lock_sha256':digest(out/'source-lock.json')})
    chosen=fixed if phase=='history' else [r for r in rows if r['role']=='confirmation']
    assert chosen;began=time.monotonic();torch.cuda.reset_peak_memory_stats();saved=[]
    for i,r in enumerate(chosen):
        p=cache/(hashlib.sha256(r['id'].encode()).hexdigest()+'.json')
        if p.exists():
            value=json.loads(p.read_text());assert value['source_lock_sha256']==digest(out/'source-lock.json') and value['id']==r['id']
        else:
            data=collect(r);value={k:v for k,v in r.items() if k!='prompt'};value.update(data)
            value.update(scores=aggregate(data['F'],data['layer_scores']).tolist(),source_lock_sha256=digest(out/'source-lock.json'))
            if phase=='confirmation':value['pre_confirmation_sha256']=digest(out/'pre-confirmation-seal.json')
            write(p,value)
        if phase=='history':assert np.array_equal(value['F'],old[r['id']]),r['id']
        saved.append(value)
        if (i+1)%100==0:print('SA_SCORED',phase,i+1,'/',len(chosen),flush=True)
    path=out/(phase+'-scores.jsonl');path.write_bytes(''.join(json.dumps(r)+'\n' for r in saved).encode())
    write(out/('extraction-'+phase+'.json'),{'status':'completed','count':len(saved),'seconds':time.monotonic()-began,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'generation':False,'scores_sha256':digest(path),'model_sha256':modelhash})
    if phase=='history':
        cal=[r for r in saved if r['role']=='calibration'];assert len(cal)==400 and all(r['label']==0 for r in cal)
        cut=[threshold(np.array([r['scores'] for r in cal])[:,j]) for j in range(len(NAMES))]
        write(out/'thresholds.json',{'names':NAMES,'thresholds':cut,'ordinal':381,'calibration_ids':[r['id'] for r in cal]})
        for role in ['historical_C','historical_RV']:
            test=[r for r in saved if r['role']==role];e=evaluate(test,cut,role)
            if role=='historical_C':assert abs(e['metrics'][0]['auroc']-.746640)<1e-12
            write(out/(role.replace('_','-')+'-evaluation.json'),e)
        if not m['confirmation_enabled']:
            write(out/'confirmation-unavailable.json',{'status':'not_performed','reason':'fewer than 30 genuinely unused samples in a class',
                'available_fresh_per_class':m['available_fresh_per_class'],'no_historical_inputs_substituted':True})
    else:write(out/'confirmation-evaluation.json',evaluate(saved,json.loads((out/'thresholds.json').read_text())['thresholds'],'new-confirmation'))
    print('SA_PHASE_COMPLETE',phase,flush=True)
if __name__=='__main__':main()
