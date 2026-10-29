"""Only fresh full template-last vectors; old comparison representation stays sealed."""
import hashlib,json,os,subprocess,time
import numpy as np
from head_core import ROOT,digest,stream_hash,read_jsonl,write,pad_ids

def main():
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch,transformers
    from llava.model.builder import load_pretrained_model
    torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    mp=ROOT/'configs/heads/manifest.json';m=json.loads(mp.read_text());rp=ROOT/'data/heads/records.jsonl';assert digest(rp)==m['records_sha256'];rows=read_jsonl(rp);fresh=[r for r in rows if r['role']=='confirmation_aegis']
    for item in json.loads((ROOT/'configs/artifacts-lock.json').read_text())['files']:
        p=ROOT/item['target'];assert p.stat().st_size==item['bytes'] and stream_hash(p)==item['sha256'],item['target']
    tokenizer,model,_,_=load_pretrained_model(str(ROOT/'models/llava-v1.6-vicuna-7b'),None,'llava-v1.6-vicuna-7b',device_map='auto',torch_dtype=torch.float16)
    assert not model.training and model.config.num_hidden_layers==32 and model.config.hidden_size==4096
    hidden=[None]*32
    def hook(i):
        def save(module,inputs,output):hidden[i]=output[0].detach()
        return save
    handles=[l.register_forward_hook(hook(i)) for i,l in enumerate(model.model.layers)]
    directory=ROOT/'cache/heads';directory.mkdir(exist_ok=True);shape=(len(fresh),32,4096);assert np.prod(shape)*2<.5*1024**3
    names=['configs/heads/extract.py','configs/heads/head_core.py','configs/processing/processing_core.py','configs/routing/routing_core.py']
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();shas={n:digest(ROOT/n) for n in names}
    for n,h in shas.items():assert hashlib.sha256(subprocess.check_output(['git','show',commit+':'+n],cwd=ROOT)).hexdigest()==h,n
    meta={'manifest_sha256':digest(mp),'shape':list(shape),'dtype':'float16','fixed_length':512,'artifact_lock_sha256':digest(ROOT/'configs/artifacts-lock.json'),'extraction_commit':commit,'extraction_sources':shas,'torch':torch.__version__,'numpy':np.__version__,'transformers':transformers.__version__}
    exists=(directory/'metadata.json').exists()
    if exists:
        oldmeta=json.loads((directory/'metadata.json').read_text());assert {k:v for k,v in meta.items() if k!='extraction_commit'}=={k:v for k,v in oldmeta.items() if k!='extraction_commit'};meta=oldmeta
    else:write(directory/'metadata.json',meta)
    old=np.load(ROOT/'cache/routing/fixed.npy',mmap_mode='r');reference=np.load(ROOT/'cache/routing/prefix-reference.npy');protected=None
    def collect(row):
        nonlocal protected
        ids=tokenizer.encode(m['prefix']+row['prompt']+m['suffix']);assert len(ids)==row['input_length']<=512
        assert hashlib.sha256(np.array(ids,dtype=np.int64).tobytes()).hexdigest()==row['input_ids_sha256'] and hashlib.sha256(row['prompt'].encode()).hexdigest()==row['prompt_sha256']
        anchor=row['prefix_anchor'];prefix=ids[:anchor+1]
        if protected is None:protected=prefix
        assert prefix==protected
        pi,mask=pad_ids(ids,512,tokenizer.pad_token_id or tokenizer.eos_token_id);torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            out=model(torch.tensor([pi],device=model.device),attention_mask=torch.tensor([mask],device=model.device),images=None,image_sizes=None,output_attentions=True)
            last=[];control=[]
            for h in hidden:
                v=h[0].float();v=v/torch.linalg.vector_norm(v,dim=-1).clamp_min(1e-12)[:,None];p=v[[anchor,len(ids)-1]].half().cpu().numpy();control.append(p[0]);last.append(p[1])
        hidden[:]=[None]*32;del out
        delta=float(np.max(np.abs(np.asarray(control,dtype=float)-reference)));peak=torch.cuda.max_memory_allocated();assert delta==0. and peak<22*1024**3
        return np.stack(last),delta,peak
    pilot_path=ROOT/'logs/heads-pilot.json'
    if not pilot_path.exists():
        sample=sum(([r for r in rows if r['role']=='train' and r['label']==y][:16] for y in (0,1)),[]);maximum=0.;prefix=0.;peak=0
        for row in sample:
            v,d,p=collect(row);maximum=max(maximum,float(np.max(np.abs(v.astype(float)-old[row['baseline_index'],:,3,:].astype(float)))));prefix=max(prefix,d);peak=max(peak,p)
        assert maximum<=1e-5
        write(pilot_path,{'manifest_sha256':digest(mp),'source_sha256':digest(__file__),'rows':32,'maximum_old_B_difference':maximum,'prefix_difference':prefix,'peak_allocated_bytes':peak});print('PILOT_PASSED',maximum,flush=True)
    pilot=json.loads(pilot_path.read_text());assert pilot['manifest_sha256']==digest(mp) and pilot['source_sha256']==digest(__file__)
    data=np.lib.format.open_memmap(directory/'fresh_B.npy',mode='r+' if exists else 'w+',dtype='float16',shape=shape)
    cp=directory/'checkpoint.json';state=json.loads(cp.read_text()) if cp.exists() else {'next_index':0,'prefix_difference':0.,'peak_allocated_bytes':0};start=state['next_index'];began=time.monotonic()
    for i in range(start,len(fresh)):
        assert fresh[i]['fresh_index']==i;v,d,p=collect(fresh[i]);data[i]=v;state['prefix_difference']=max(state['prefix_difference'],d);state['peak_allocated_bytes']=max(state['peak_allocated_bytes'],p)
        if (i+1)%16==0 or i+1==len(fresh):data.flush();state['next_index']=i+1;write(directory/'checkpoint.tmp',state);(directory/'checkpoint.tmp').replace(cp)
        if (i+1)%100==0:print('FRESH_VECTORS',i+1,'/',len(fresh),'SEC',round(time.monotonic()-began,1),flush=True)
    files=['cache/heads/fresh_B.npy','cache/heads/metadata.json','cache/heads/checkpoint.json','cache/routing/fixed.npy']
    seal={'manifest_sha256':digest(mp),'metadata':meta,'state':state,'pilot':pilot,'files':{n:stream_hash(ROOT/n) for n in files},'bytes':(directory/'fresh_B.npy').stat().st_size,'seconds':time.monotonic()-began,'resumed_at':start}
    write(ROOT/'logs/heads-cache-seal.json',seal);print('HEAD_CACHE_SEALED',state,flush=True)

if __name__=='__main__':main()
