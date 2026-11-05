import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import hashlib,json,subprocess,time
import numpy as np
import torch
from core import ROOT,ORIGIN,OUT,digest,read,write,verify_source
def main():
    verify_source();manifest=json.loads((OUT/'manifest.json').read_text());path=ROOT/'data/oc-retraining/records.jsonl'
    assert digest(path)==manifest['records_sha256'];rows=read(path)
    from llava.model.builder import load_pretrained_model
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX
    assert subprocess.check_output(['git','-C',str(ROOT/'source/HiddenDetect'),'rev-parse','HEAD'],text=True).strip()=='2fd161b1fc47c99b36fe0a299ec64ff0b816f0bb'
    tok,model,_,_=load_pretrained_model(str(ROOT/'models/llava-v1.6-vicuna-7b'),None,'llava-v1.6-vicuna-7b',device_map='auto',torch_dtype=torch.float16)
    assert not model.training and next(model.parameters()).dtype==torch.float16
    ids=json.loads((ROOT/'results/oc23/token-audit.json').read_text())['token_ids'];reference=torch.zeros(32000,device='cuda',dtype=torch.float32);reference[ids]=1
    def collect(row,wrapped):
        if wrapped:text=manifest['prefix']+row['prompt']+manifest['suffix'];tokens=torch.tensor(tok.encode(text)).unsqueeze(0).cuda()
        else:
            conv=conv_templates['llava_v1'].copy();conv.append_message(conv.roles[0],row['prompt']);conv.append_message(conv.roles[1],None)
            tokens=tokenizer_image_token(conv.get_prompt(),tok,IMAGE_TOKEN_INDEX,return_tensors='pt').unsqueeze(0).cuda()
        assert tokens.shape[1]<=512 and hashlib.sha256(tokens.cpu().numpy().tobytes()).hexdigest()==row['input_ids_sha256']
        with torch.no_grad():
            result=model(tokens,images=None,image_sizes=None,output_hidden_states=True)
            f=np.array([float(torch.nn.functional.cosine_similarity(model.lm_head(model.model.norm(state))[:,-1,:],reference).item()) for state in result.hidden_states[1:]])
        assert f.shape==(32,) and np.isfinite(f).all();return f
    original=read(ORIGIN/'data/signal-selection/records.jsonl');oldf=np.load(ORIGIN/'cache/signal-selection/F.npy',mmap_mode='r');probes=[]
    for label in (0,1):probes.extend([(i,r) for i,r in enumerate(original) if r['role']=='train' and r['label']==label][:4])
    for i,row in probes:assert np.array_equal(collect(row,False),oldf[i])
    write(OUT/'identity.json',{'status':'passed','old_training_probes':8,'maximum_F_difference':0.,'source_lock_sha256':digest(OUT/'source-lock.json')})
    print('OC_H3_IDENTITY_PASS',flush=True)
    cache=ROOT/'cache/oc-retraining/F';cache.mkdir(parents=True,exist_ok=True);values=[];began=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for i,row in enumerate(rows):
        target=cache/(hashlib.sha256(row['id'].encode()).hexdigest()+'.json')
        if target.exists():
            value=json.loads(target.read_text());assert value['source_lock_sha256']==digest(OUT/'source-lock.json') and value['id']==row['id']
        else:
            assert hashlib.sha256(row['prompt'].encode()).hexdigest()==row['prompt_sha256']
            value={k:v for k,v in row.items() if k!='prompt'};value.update(F=collect(row,True).tolist(),source_lock_sha256=digest(OUT/'source-lock.json'));write(target,value)
        values.append(value)
        if (i+1)%100==0:print('OC_H3_F',i+1,'/',len(rows),flush=True)
    (ROOT/'cache/oc-retraining/F-scores.jsonl').write_bytes(''.join(json.dumps(r)+'\n' for r in values).encode())
    arrays={}
    for role in ['train']:
        group=[r for r in values if r['role']==role];arrays[role+'_X']=np.array([r['F'][16:30] for r in group]);arrays[role+'_y']=np.array([r['label'] for r in group])
    np.savez(ROOT/'cache/oc-retraining/features.npz',**arrays)
    write(OUT/'extraction.json',{'status':'completed','count':len(values),'seconds':time.monotonic()-began,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'features_sha256':digest(ROOT/'cache/oc-retraining/features.npz'),'F_scores_sha256':digest(ROOT/'cache/oc-retraining/F-scores.jsonl'),'generation':False})
    print('OC_H3_EXTRACTION_COMPLETE',flush=True)
if __name__=='__main__':main()
