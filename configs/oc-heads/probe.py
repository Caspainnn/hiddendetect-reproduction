"""A fresh GPU replay of eight training examples against the original F cache."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import hashlib,json,subprocess
import numpy as np
import torch
from core import ROOT,OUT,read,write,digest
def main():
    origin=ROOT.parents[1]
    from llava.model.builder import load_pretrained_model
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX
    assert subprocess.check_output(['git','-C',str(ROOT/'source/HiddenDetect'),'rev-parse','HEAD'],text=True).strip()=='2fd161b1fc47c99b36fe0a299ec64ff0b816f0bb'
    tok,model,_,_=load_pretrained_model(str(ROOT/'models/llava-v1.6-vicuna-7b'),None,'llava-v1.6-vicuna-7b',device_map='auto',torch_dtype=torch.float16)
    ids=json.loads((ROOT/'results/oc23/token-audit.json').read_text())['token_ids']
    reference=torch.zeros(32000,device='cuda',dtype=torch.float32);reference[ids]=1
    rows=read(origin/'data/signal-selection/records.jsonl');f=np.load(origin/'cache/signal-selection/F.npy',mmap_mode='r')
    chosen=[]
    for label in (0,1):chosen.extend([(i,r) for i,r in enumerate(rows) if r['role']=='train' and r['label']==label][:4])
    checks=[]
    for i,row in chosen:
        conv=conv_templates['llava_v1'].copy();conv.append_message(conv.roles[0],row['prompt']);conv.append_message(conv.roles[1],None)
        tokens=tokenizer_image_token(conv.get_prompt(),tok,IMAGE_TOKEN_INDEX,return_tensors='pt').unsqueeze(0).cuda()
        assert hashlib.sha256(tokens.cpu().numpy().tobytes()).hexdigest()==row['input_ids_sha256']
        with torch.no_grad():
            result=model(tokens,images=None,image_sizes=None,output_hidden_states=True)
            fresh=np.array([float(torch.nn.functional.cosine_similarity(model.lm_head(model.model.norm(state))[:,-1,:],reference).item()) for state in result.hidden_states[1:]])
        difference=float(abs(fresh-f[i]).max());assert difference==0,(row['id'],difference)
        checks.append({'id':row['id'],'maximum_32F_difference':difference})
    write(OUT/'identity-training.json',{'status':'passed','checks':checks,'fresh_forward_count':8,'F_difference':0.,'source_lock_sha256':digest(OUT/'source-lock.json'),'generation':False})
    print('HEAD_FRESH_TRAINING_IDENTITY_PASS',len(checks),flush=True)
if __name__=='__main__':main()
