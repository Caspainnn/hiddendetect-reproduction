"""Process all valid user tokens online; keep only bounded full-width summaries."""
import hashlib,json,os,subprocess,time
import numpy as np
from processing_core import ROOT,ANCHORS,digest,stream_hash,read_jsonl,write,pad_ids,summarize,open_array

def main():
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from llava.model.builder import load_pretrained_model
    torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    mp=ROOT/'configs/processing/manifest.json';m=json.loads(mp.read_text());rp=ROOT/'data/processing/records.jsonl'
    assert digest(rp)==m['records_sha256'];rows=read_jsonl(rp)
    for item in json.loads((ROOT/'configs/artifacts-lock.json').read_text())['files']:
        p=ROOT/item['target'];assert p.stat().st_size==item['bytes'] and stream_hash(p)==item['sha256'],item['target']
    tokenizer,model,_,_=load_pretrained_model(str(ROOT/'models/llava-v1.6-vicuna-7b'),None,'llava-v1.6-vicuna-7b',device_map='auto',torch_dtype=torch.float16)
    assert not model.training and model.config.num_hidden_layers==32 and model.config.hidden_size==4096
    hidden=[None]*32;att=[None]*32;mlp=[None]*32
    def capture(target,i):
        def save(module,inputs,output):target[i]=(output[0] if isinstance(output,tuple) else output).detach()
        return save
    handles=[]
    for i,l in enumerate(model.model.layers):
        handles.append(l.register_forward_hook(capture(hidden,i)))
        if i in ANCHORS:handles.extend([l.self_attn.register_forward_hook(capture(att,i)),l.mlp.register_forward_hook(capture(mlp,i))])
    directory=ROOT/'cache/processing';directory.mkdir(exist_ok=True)
    shapes={'summaries.npy':[len(rows),4,7,4096],'relations.npy':[len(rows),116],'fresh_B.npy':[m['fresh_class_count']*2,32,4096]}
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    source_names=['configs/processing/extract.py','configs/processing/processing_core.py','configs/routing/routing_core.py']
    source_sha={n:digest(ROOT/n) for n in source_names}
    for n,sha in source_sha.items():assert hashlib.sha256(subprocess.check_output(['git','show',commit+':'+n],cwd=ROOT)).hexdigest()==sha,n
    meta={'manifest_sha256':digest(mp),'shapes':shapes,'extraction_commit':commit,'extraction_sources':source_sha,'fixed_length':512,'anchors':ANCHORS,'torch':torch.__version__}
    exists=(directory/'metadata.json').exists()
    if exists:
        previous=json.loads((directory/'metadata.json').read_text());assert previous['manifest_sha256']==meta['manifest_sha256'] and previous['extraction_sources']==source_sha;meta=previous
    else:write(directory/'metadata.json',meta)
    arrays={n:open_array(directory/n,s,n=='relations.npy',exists) for n,s in shapes.items()}
    assert sum(a.nbytes for a in arrays.values())<2*1024**3
    cp=directory/'checkpoint.json';state=json.loads(cp.read_text()) if cp.exists() else {'next_index':0,'old_B_checked':0,'maximum_old_B_difference':0.,'prefix_difference':0.,'peak_allocated_bytes':0}
    old=np.load(ROOT/'cache/routing/fixed.npy',mmap_mode='r');refpath=directory/'prefix-reference.npy';ref=np.load(refpath) if refpath.exists() else None
    protected=None;start=state['next_index'];began=time.monotonic()
    for i in range(start,len(rows)):
        row=rows[i];ids=tokenizer.encode(m['prefix']+row['prompt']+m['suffix']);assert len(ids)==row['input_length']<=512
        assert hashlib.sha256(np.array(ids,dtype=np.int64).tobytes()).hexdigest()==row['input_ids_sha256']
        assert hashlib.sha256(row['prompt'].encode()).hexdigest()==row['prompt_sha256']
        user=np.concatenate([np.array(a,dtype=int) for a in row['regions']]);assert len(user)>0 and np.array_equal(user,np.unique(user))
        anchor=row['prefix_anchor'];prefix=ids[:anchor+1]
        if protected is None:protected=prefix
        assert prefix==protected
        pi,mask=pad_ids(ids,512,tokenizer.pad_token_id or tokenizer.eos_token_id);last=len(ids)-1
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            output=model(torch.tensor([pi],device=model.device),attention_mask=torch.tensor([mask],device=model.device),images=None,image_sizes=None,output_attentions=True)
            base=[];prefix_values=[]
            for h in hidden:
                value=h[0,[anchor,last]].float();value=value/torch.linalg.vector_norm(value,dim=-1).clamp_min(1e-12)[:,None]
                v=value.half().cpu().numpy();prefix_values.append(v[0]);base.append(v[1])
            base=np.stack(base);prefix_values=np.stack(prefix_values)
            if ref is None:ref=prefix_values.copy();np.save(refpath,ref)
            state['prefix_difference']=max(state['prefix_difference'],float(np.max(np.abs(prefix_values.astype(float)-ref))))
            summary=[];relations=[]
            for l in ANCHORS:
                h=hidden[l][0,user].float().cpu().numpy();pre=hidden[l-1][0,user].float().cpu().numpy()
                a=att[l][0,user].float().cpu().numpy();v=mlp[l][0,user].float().cpu().numpy()
                g=hidden[l][0,last].float().cpu().numpy();aw=output.attentions[l][0,:,last,:][:,user].float().mean(0).cpu().numpy()
                s,r=summarize(h,pre,a,v,g,aw,row['id'],l);summary.append(s);relations.append(r)
        arrays['summaries.npy'][i]=np.asarray(summary,dtype=np.float16);arrays['relations.npy'][i]=np.concatenate(relations).astype(np.float32)
        if 'baseline_index' in row:
            delta=float(np.max(np.abs(base.astype(float)-old[row['baseline_index'],:,3,:].astype(float))))
            state['maximum_old_B_difference']=max(state['maximum_old_B_difference'],delta);state['old_B_checked']+=1
        else:arrays['fresh_B.npy'][row['fresh_index']]=base
        state['peak_allocated_bytes']=max(state['peak_allocated_bytes'],torch.cuda.max_memory_allocated())
        assert state['prefix_difference']==0. and state['maximum_old_B_difference']<=1e-5 and state['peak_allocated_bytes']<22*1024**3
        hidden[:]=[None]*32;att[:]=[None]*32;mlp[:]=[None]*32;del output
        if (i+1)%16==0 or i+1==len(rows):
            for a in arrays.values():a.flush()
            state['next_index']=i+1;write(directory/'checkpoint.tmp',state);(directory/'checkpoint.tmp').replace(cp)
        if (i+1)%100==0:print('PROCESSED',i+1,'/',len(rows),'SEC',round(time.monotonic()-began,1),flush=True)
    files=['cache/processing/'+n for n in shapes]+['cache/processing/metadata.json','cache/processing/checkpoint.json','cache/processing/prefix-reference.npy','cache/routing/fixed.npy']
    seal={'manifest_sha256':digest(mp),'metadata':meta,'state':state,'extraction_commit':meta['extraction_commit'],'extraction_sources':meta['extraction_sources'],'files':{n:stream_hash(ROOT/n) for n in files},'additional_bytes':sum((directory/n).stat().st_size for n in shapes),'seconds':time.monotonic()-began,'resumed_at':start}
    assert state['old_B_checked']==4960
    write(ROOT/'logs/processing-cache-seal.json',seal);print('PROCESSING_CACHE_SEALED',seal,flush=True)

if __name__=='__main__':main()
