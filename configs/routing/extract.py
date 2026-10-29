"""One fixed512 forward per row, plus variable forward only for fresh rows."""
import json,os,time,hashlib
import numpy as np
from routing_core import ROOT,digest,read_jsonl,write,pad_ids,readouts,stream_hash

def main():
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from llava.model.builder import load_pretrained_model
    torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    mp=ROOT/'configs/routing/manifest.json';m=json.loads(mp.read_text());rp=ROOT/'data/routing/records.jsonl'
    assert digest(rp)==m['records_sha256'];rows=read_jsonl(rp)
    lock=json.loads((ROOT/'configs/artifacts-lock.json').read_text())
    for item in lock['files']:
        path=ROOT/item['target'];assert path.stat().st_size==item['bytes'] and stream_hash(path)==item['sha256'],item['target']
        print('ARTIFACT_VERIFIED',item['target'],flush=True)
    tokenizer,model,_,_=load_pretrained_model(str(ROOT/'models/llava-v1.6-vicuna-7b'),None,'llava-v1.6-vicuna-7b',device_map='auto',torch_dtype=torch.float16)
    assert not model.training and model.config.num_hidden_layers==32 and model.config.hidden_size==4096
    hidden=[None]*32
    def hook(i):
        def save(module,inputs,output):hidden[i]=output[0].detach()
        return save
    handles=[layer.register_forward_hook(hook(i)) for i,layer in enumerate(model.model.layers)]
    directory=ROOT/'cache/routing';directory.mkdir(exist_ok=True)
    shapes={'fixed.npy':(len(rows),32,4,4096),'fresh_variable.npy':(m['fresh_class_count']*2,32,2,4096)}
    assert sum(np.prod(s)*2+128 for s in shapes.values())<7*1024**3
    meta={'manifest_sha256':digest(mp),'extractor_sha256':digest(__file__),'core_sha256':digest(ROOT/'configs/routing/routing_core.py'),'shapes':{k:list(v) for k,v in shapes.items()},'dtype':'float16','torch':torch.__version__}
    exists=(directory/'metadata.json').exists()
    if exists:assert json.loads((directory/'metadata.json').read_text())==meta
    else:write(directory/'metadata.json',meta)
    arrays={k:np.lib.format.open_memmap(directory/k,mode='r+' if exists else 'w+',dtype='float16',shape=s) for k,s in shapes.items()}
    cp=directory/'checkpoint.json';state=json.loads(cp.read_text()) if cp.exists() else {'next_index':0,'prefix_max_difference':0.,'max_fixed_variable_difference':0.,'peak_allocated_bytes':0}
    old=np.load(ROOT/'cache/positions/P.npy',mmap_mode='r');reference_path=directory/'prefix-reference.npy'
    ref=np.load(reference_path) if reference_path.exists() else None
    protected=None;began=time.monotonic()
    def collect(ids,positions,fixed):
        if fixed:pi,mask=pad_ids(ids,512,tokenizer.pad_token_id or tokenizer.eos_token_id)
        else:pi,mask=ids,[1]*len(ids)
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            output=model(torch.tensor([pi],device=model.device),attention_mask=torch.tensor([mask],device=model.device) if fixed else None,images=None,image_sizes=None,output_attentions=True)
            result=[]
            for h in hidden:
                raw=h[0].float();unit=raw/torch.linalg.vector_norm(raw,dim=-1).clamp_min(1e-12).unsqueeze(-1)
                result.append(unit[positions].half().cpu().numpy())
        state['peak_allocated_bytes']=max(state['peak_allocated_bytes'],torch.cuda.max_memory_allocated())
        hidden[:]=[None]*32;del output
        return np.stack(result)
    start=state['next_index']
    for i in range(start,len(rows)):
        row=rows[i];ids=tokenizer.encode(m['prefix']+row['prompt']+m['suffix'])
        assert len(ids)==row['input_length']<=512 and hashlib.sha256(np.array(ids,dtype=np.int64).tobytes()).hexdigest()==row['input_ids_sha256']
        assert hashlib.sha256(row['prompt'].encode()).hexdigest()==row['prompt_sha256']
        anchor=row['prefix_anchor'];prefix=ids[:anchor+1]
        if protected is None:protected=prefix
        assert prefix==protected and readouts(row)==row['routing_indices']
        values=collect(ids,[anchor]+readouts(row),True);arrays['fixed.npy'][i]=values[:,1:]
        if ref is None:ref=values[:,0].copy();np.save(reference_path,ref)
        state['prefix_max_difference']=max(state['prefix_max_difference'],float(np.max(np.abs(values[:,0].astype(float)-ref))))
        if 'previous_index' in row:variable=old[row['previous_index'],:,[6,7],:].transpose(1,0,2)
        else:
            variable=collect(ids,[row['readout_indices'][6],len(ids)-1],False);arrays['fresh_variable.npy'][row['fresh_index']]=variable
        delta=float(np.max(np.abs(values[:,[1,4]].astype(float)-variable.astype(float))))
        state['max_fixed_variable_difference']=max(state['max_fixed_variable_difference'],delta)
        assert state['prefix_max_difference']==0. and state['peak_allocated_bytes']<22*1024**3
        if (i+1)%16==0 or i+1==len(rows):
            for a in arrays.values():a.flush()
            state['next_index']=i+1;write(directory/'checkpoint.tmp',state);(directory/'checkpoint.tmp').replace(cp)
        if (i+1)%100==0:print('CACHED',i+1,'/',len(rows),'SEC',round(time.monotonic()-began,1),flush=True)
    files=['cache/routing/'+k for k in shapes]+['cache/routing/metadata.json','cache/routing/checkpoint.json','cache/routing/prefix-reference.npy','cache/positions/P.npy','cache/positions/seal.json']
    seal={'manifest_sha256':digest(mp),'files':{n:stream_hash(ROOT/n) for n in files},'rows':len(rows),'fresh_rows':2*m['fresh_class_count'],'state':state,'seconds_this_run':time.monotonic()-began,'resumed_at':start,'additional_bytes':sum((directory/k).stat().st_size for k in shapes)}
    write(ROOT/'logs/routing-cache-seal.json',seal);print('ROUTING_CACHE_SEALED',seal,flush=True)

if __name__=='__main__':main()
