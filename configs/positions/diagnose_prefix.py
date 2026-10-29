"""Label-free shape sensitivity check, train probes only; not a new detector."""
import json,hashlib,os
import numpy as np
from position_core import ROOT,digest,read_jsonl,token_indices

def padded(ids,length,pad):
    assert len(ids)<=length
    return ids+[pad]*(length-len(ids)),[1]*len(ids)+[0]*(length-len(ids))

def main():
    import torch
    from llava.model.builder import load_pretrained_model
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';torch.manual_seed(0)
    m=json.loads((ROOT/'configs/positions/manifest.json').read_text());rows=read_jsonl(ROOT/'data/positions/records.jsonl');sample=[]
    for label in (0,1):sample += [x for x in rows if x['role']=='train' and x['label']==label][:16]
    tokenizer,model,processor,context=load_pretrained_model(str(ROOT/'models/llava-v1.6-vicuna-7b'),None,'llava-v1.6-vicuna-7b',device_map='auto',torch_dtype=torch.float16)
    hidden=[None]*32
    def hook(i):
        def save(module,inputs,output):hidden[i]=output[0].detach()
        return save
    handles=[layer.register_forward_hook(hook(i)) for i,layer in enumerate(model.model.layers)]
    old=np.load(ROOT/'cache/positions/P.npy',mmap_mode='r');fixed=[];variable=[];template_change=[];protected=None
    for row in sample:
        ids=tokenizer.encode(m['prefix']+row['prompt']+m['suffix']);positions=token_indices(row)
        prefix=ids[:positions[0]+1]
        if protected is None:protected=prefix
        assert prefix==protected
        pi,mask=padded(ids,512,tokenizer.pad_token_id or tokenizer.eos_token_id)
        with torch.no_grad():
            output=model(torch.tensor([pi],device=model.device),attention_mask=torch.tensor([mask],device=model.device),images=None,image_sizes=None,output_attentions=True)
            points=[]
            for h in hidden:
                a=h[0].float();a=a/torch.linalg.vector_norm(a,dim=-1).clamp_min(1e-12).unsqueeze(-1);points.append(a[[positions[0],positions[7]]].half().cpu().numpy())
        points=np.stack(points).astype(np.float32);fixed.append(points[:,0]);variable.append(old[row['index'],:,0,:].astype(np.float32))
        template_change.append(float(np.max(np.abs(points[:,1]-old[row['index'],:,7,:].astype(np.float32)))))
        hidden[:]=[None]*32;del output
    fixed=np.stack(fixed);variable=np.stack(variable)
    report={'scope':'train32 probes, label-free protected-prefix equality and shape sensitivity; no metric or threshold selection',
        'manifest_sha256':digest(ROOT/'configs/positions/manifest.json'),'code_sha256':digest(__file__),'rows':32,
        'protected_prefix_ids_identical':True,'variable_shape_prefix_max_difference':float(np.max(np.abs(variable-variable[0]))),
        'fixed512_prefix_max_difference':float(np.max(np.abs(fixed-fixed[0]))),'maximum_template_state_change_due_padding':max(template_change),
        'interpretation':'If drift disappears at fixed shape, variable-shape floating-point execution is a plausible prefix side-channel; not evidence prefix attends to future text.'}
    (ROOT/'logs/positions-prefix-diagnostic.json').write_text(json.dumps(report,indent=2)+'\n');print('PREFIX_DIAGNOSTIC_COMPLETE',report,flush=True)

if __name__=='__main__':main()
