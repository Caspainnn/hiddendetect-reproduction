"""Fixed512 full last vectors and matched user summaries; resumable private cache."""
import hashlib,json,os,subprocess,time
import numpy as np
from stability_core import ROOT,ANCHORS,digest,stream_hash,write,read_jsonl,pad_ids,user_indices,summarize,unit,weights

def main():
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from llava.model.builder import load_pretrained_model
    torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;began=time.monotonic()
    mp=ROOT/'configs/stability/manifest.json';m=json.loads(mp.read_text());rp=ROOT/'data/stability/records.jsonl';assert digest(rp)==m['records_sha256'];rows=read_jsonl(rp)
    for r in json.loads((ROOT/'configs/artifacts-lock.json').read_text())['files']:
        p=ROOT/r['target'];assert p.stat().st_size==r['bytes'] and stream_hash(p)==r['sha256']
    tok,model,_,_=load_pretrained_model(str(ROOT/'models/llava-v1.6-vicuna-7b'),None,'llava-v1.6-vicuna-7b',device_map='auto',torch_dtype=torch.float16)
    assert not model.training and model.config.hidden_size==4096 and model.config.num_hidden_layers==32
    hidden=[None]*32;att=[None]*32;mlp=[None]*32;handles=[]
    def capture(target,i):
        def save(module,inputs,output):target[i]=(output[0] if isinstance(output,tuple) else output).detach()
        return save
    for i,l in enumerate(model.model.layers):
        handles.append(l.register_forward_hook(capture(hidden,i)))
        if i in ANCHORS:handles.extend([l.self_attn.register_forward_hook(capture(att,i)),l.mlp.register_forward_hook(capture(mlp,i))])
    directory=ROOT/'cache/stability';directory.mkdir(exist_ok=True)
    shapes={'B.npy':[len(rows),32,4096],'S.npy':[len(rows),4,5,4096]}
    names=['configs/stability/extract.py','configs/stability/stability_core.py','configs/processing/processing_core.py','configs/routing/routing_core.py']
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();shas={n:digest(ROOT/n) for n in names}
    for n,h in shas.items():assert hashlib.sha256(subprocess.check_output(['git','show',commit+':'+n])).hexdigest()==h,n
    meta={'manifest_sha256':digest(mp),'commit':commit,'sources':shas,'artifact_lock_sha256':digest(ROOT/'configs/artifacts-lock.json'),'shapes':shapes,'dtype':'float16','fixed_length':512}
    exists=(directory/'metadata.json').exists()
    if exists:
        old=json.loads((directory/'metadata.json').read_text());assert {k:v for k,v in old.items() if k!='commit'}=={k:v for k,v in meta.items() if k!='commit'};meta=old
    else:write(directory/'metadata.json',meta)
    data={n:np.lib.format.open_memmap(directory/n,mode='r+' if exists else 'w+',dtype='float16',shape=tuple(shape)) for n,shape in shapes.items()}
    cp=directory/'checkpoint.json';state=json.loads(cp.read_text()) if cp.exists() else {'next_index':0,'prefix_difference':0.,'peak_allocated_bytes':0,'pilot_rows':0,'pilot_maximum_B_difference':0.,'pilot_maximum_summary_difference':0.}
    reference=np.load(ROOT/'cache/routing/prefix-reference.npy');oldB=np.load(ROOT/'cache/routing/fixed.npy',mmap_mode='r');oldS=np.load(ROOT/'cache/processing/summaries.npy',mmap_mode='r');protected=None
    def collect(row):
        nonlocal protected
        ids=tok.encode(m['prefix']+row['prompt']+m['suffix']);assert len(ids)==row['input_length']<=512
        assert hashlib.sha256(np.array(ids,dtype=np.int64).tobytes()).hexdigest()==row['input_ids_sha256'] and hashlib.sha256(row['prompt'].encode()).hexdigest()==row['prompt_sha256']
        anchor=row['prefix_anchor'];user=user_indices(row);prefix=ids[:anchor+1]
        if protected is None:protected=prefix
        assert protected==prefix
        pi,mask=pad_ids(ids,512,tok.pad_token_id or tok.eos_token_id);torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            out=model(torch.tensor([pi],device=model.device),attention_mask=torch.tensor([mask],device=model.device),images=None,image_sizes=None,output_attentions=True)
            b=[];controls=[];s=[]
            for h in hidden:
                v=h[0].float();v=v/torch.linalg.vector_norm(v,dim=-1).clamp_min(1e-12)[:,None];v=v[[anchor,len(ids)-1]].half().cpu().numpy();controls.append(v[0]);b.append(v[1])
            for l in ANCHORS:
                h=hidden[l][0,user].float().cpu().numpy();pre=hidden[l-1][0,user].float().cpu().numpy();a=att[l][0,user].float().cpu().numpy();mout=mlp[l][0,user].float().cpu().numpy()
                aw=out.attentions[l][0,:,len(ids)-1,:][:,user].float().mean(0).cpu().numpy();g=hidden[l][0,len(ids)-1].float().cpu().numpy()
                pooled,_=summarize(h,pre,a,mout,g,aw,row['id'],l)
                seed=int(hashlib.sha256((row['id']+':stability-random2:'+str(l)).encode()).hexdigest()[:16],16)
                random2=weights(np.random.default_rng(seed).normal(size=len(h)))@unit(h)
                s.append(np.stack([pooled[0],pooled[1],pooled[2],pooled[6],random2]))
        state['prefix_difference']=max(state['prefix_difference'],float(np.max(np.abs(np.asarray(controls,float)-reference))))
        state['peak_allocated_bytes']=max(state['peak_allocated_bytes'],torch.cuda.max_memory_allocated());assert state['prefix_difference']==0. and state['peak_allocated_bytes']<22*1024**3
        hidden[:]=[None]*32;att[:]=[None]*32;mlp[:]=[None]*32;del out
        return np.asarray(b,np.float16),np.asarray(s,np.float16)
    if state['pilot_rows']==0:
        probes=sum(([r for r in rows if r.get('origin')=='original_train' and r['label']==y][:16] for y in (0,1)),[])
        assert len(probes)==32
        for row in probes:
            b,s=collect(row);ob=oldB[row['baseline_index'],:,3,:];os=oldS[row['processing_index']][:,[0,1,2,6],:]
            state['pilot_maximum_B_difference']=max(state['pilot_maximum_B_difference'],float(np.max(np.abs(b.astype(float)-ob))))
            state['pilot_maximum_summary_difference']=max(state['pilot_maximum_summary_difference'],float(np.max(np.abs(s[:,:4].astype(float)-os))))
        state['pilot_rows']=32;assert state['pilot_maximum_B_difference']<=1e-5 and state['pilot_maximum_summary_difference']<=1e-5;write(cp,state);print('PILOT_PASSED',state,flush=True)
    start=state['next_index']
    for i in range(start,len(rows)):
        b,s=collect(rows[i]);data['B.npy'][i]=b;data['S.npy'][i]=s
        if (i+1)%16==0 or i+1==len(rows):
            for a in data.values():a.flush()
            state['next_index']=i+1;write(directory/'checkpoint.tmp',state);(directory/'checkpoint.tmp').replace(cp)
        if (i+1)%100==0:print('EXTRACTED',i+1,'/',len(rows),'SEC',round(time.monotonic()-began,1),flush=True)
    files=['cache/stability/'+n for n in shapes]+['cache/stability/metadata.json','cache/stability/checkpoint.json']
    write(ROOT/'logs/stability-cache-seal.json',{**meta,**state,'files':{n:stream_hash(ROOT/n) for n in files},'seconds':time.monotonic()-began,'resumed_at':start,'bytes':sum((directory/n).stat().st_size for n in shapes)})
    print('CACHE_SEALED',state,flush=True)

if __name__=='__main__':main()
