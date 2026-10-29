"""Capture exact token residual states, never pooled or projected."""
import argparse,hashlib,inspect,json,os,time
import numpy as np
from position_core import ROOT,digest,read_jsonl,token_indices,NAMES

def contract():return hashlib.sha256((inspect.getsource(token_indices)+json.dumps(NAMES)).encode()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pilot-only',action='store_true');args=parser.parse_args()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from llava.model.builder import load_pretrained_model
    torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    mp=ROOT/'configs/positions/manifest.json';m=json.loads(mp.read_text());records=ROOT/'data/positions/records.jsonl'
    assert digest(records)==m['records_sha256'];rows=read_jsonl(records);assert len(rows)==len(m['samples'])
    tokenizer,model,processor,context=load_pretrained_model(str(ROOT/'models/llava-v1.6-vicuna-7b'),None,'llava-v1.6-vicuna-7b',device_map='auto',torch_dtype=torch.float16)
    assert not model.training and not any(p.is_meta for p in model.parameters())
    assert model.config.num_hidden_layers==32 and model.config.hidden_size==4096
    hidden=[None]*32
    def hook(i):
        def save(module,inputs,output):hidden[i]=(output[0] if isinstance(output,tuple) else output).detach()
        return save
    hooks=[layer.register_forward_hook(hook(i)) for i,layer in enumerate(model.model.layers)]
    old={name:np.load(ROOT/'cache'/name/'P.npy',mmap_mode='r') for name in ('highdim','fusion','swarm')}
    def collect(row):
        text=m['prefix']+row['prompt']+m['suffix'];ids=torch.tensor(tokenizer.encode(text),device=model.device).unsqueeze(0)
        assert ids.shape[-1]==row['input_length']<=512 and hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()==row['input_ids_sha256']
        assert hashlib.sha256(row['prompt'].encode()).hexdigest()==row['prompt_sha256']
        positions=token_indices(row);assert positions==row['readout_indices']
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            output=model(ids,images=None,image_sizes=None,output_hidden_states=False,output_attentions=True)
            result=[]
            for h in hidden:
                raw=h[0].float();unit=raw/torch.linalg.vector_norm(raw,dim=-1).clamp_min(1e-12).unsqueeze(-1)
                result.append(unit[positions].half().cpu().numpy())
        values=np.stack(result);difference=0.
        if 'origin' in row:
            previous=old[row['origin']][row['origin_index'],:,0,:]
            difference=float(np.max(np.abs(values[:,7,:].astype(np.float32)-previous.astype(np.float32))))
            assert difference<=1e-5,(row['id'],difference)
        allocated=torch.cuda.max_memory_allocated();hidden[:]=[None]*32;del output
        return values,difference,allocated
    pilot_path=ROOT/'logs/positions-pilot.json'
    if not pilot_path.exists():
        sample=[]
        for label in (0,1):sample += [x for x in rows if x['role']=='train' and x['label']==label][:32]
        times=[];peaks=[];difference=0.
        for i,row in enumerate(sample):
            t=time.monotonic();_,delta,allocated=collect(row);times.append(time.monotonic()-t);peaks.append(allocated);difference=max(difference,delta)
            if (i+1)%16==0:print('PILOT',i+1,flush=True)
        pilot={'manifest_sha256':digest(mp),'extractor_sha256':digest(__file__),'readout_contract_sha256':contract(),'rows':64,
            'maximum_old_readout_difference':difference,'peak_allocated_bytes':max(peaks),'median_seconds':float(np.median(times)),
            'passed':bool(max(peaks)<22*1024**3)}
        assert pilot['passed'];pilot_path.write_text(json.dumps(pilot,indent=2)+'\n');print('PILOT_PASSED',pilot,flush=True)
    pilot=json.loads(pilot_path.read_text());assert pilot['manifest_sha256']==digest(mp) and pilot['extractor_sha256']==digest(__file__) and pilot['passed']
    if args.pilot_only:return
    directory=ROOT/'cache/positions';directory.mkdir(exist_ok=True);shape=(len(rows),32,9,4096)
    assert np.prod(shape)*2<15*1024**3
    meta={'manifest_sha256':digest(mp),'count':len(rows),'shape':list(shape),'dtype':'float16',
        'extractor_sha256':digest(__file__),'core_at_extraction_sha256':digest(ROOT/'configs/positions/position_core.py'),
        'readout_contract_sha256':contract(),'pilot_sha256':digest(pilot_path),'torch':torch.__version__,
        'boundary':'unit raw decoder-layer residual output, pre-final-norm, exact token, all4096 coordinates'}
    metadata=directory/'metadata.json';exists=metadata.exists()
    if exists:assert json.loads(metadata.read_text())==meta
    else:metadata.write_text(json.dumps(meta,indent=2)+'\n')
    data=np.lib.format.open_memmap(directory/'P.npy',mode='r+' if exists else 'w+',dtype='float16',shape=shape)
    cp=directory/'checkpoint.json';state=json.loads(cp.read_text()) if cp.exists() else {'next_index':0,'maximum_template_last_difference':0.,'old_rows_checked':0,'prefix_training_max_difference':0.}
    start=state['next_index'];began=time.monotonic();peak=0;prefix_reference=None
    if start:prefix_reference=np.asarray(data[0,:,0,:],np.float32)
    for i in range(start,len(rows)):
        values,delta,allocated=collect(rows[i]);data[i]=values;peak=max(peak,allocated)
        state['maximum_template_last_difference']=max(state['maximum_template_last_difference'],delta)
        if 'origin' in rows[i]:state['old_rows_checked']+=1
        if rows[i]['role']=='train':
            if prefix_reference is None:prefix_reference=values[:,0,:].astype(np.float32)
            state['prefix_training_max_difference']=max(state['prefix_training_max_difference'],float(np.max(np.abs(values[:,0,:].astype(np.float32)-prefix_reference))))
        if (i+1)%16==0 or i+1==len(rows):
            data.flush();state['next_index']=i+1;temp=directory/'checkpoint.tmp';temp.write_text(json.dumps(state));temp.replace(cp)
        if (i+1)%100==0:print('CACHED',i+1,'/',len(rows),'SEC',round(time.monotonic()-began,2),flush=True)
    assert state['old_rows_checked']==5360
    report={'status':'passed','manifest_sha256':digest(mp),'rows':len(rows),'old_rows_checked':state['old_rows_checked'],
        'maximum_template_last_difference':state['maximum_template_last_difference'],'prefix_training_max_difference':state['prefix_training_max_difference'],
        'new_cache_bytes':(directory/'P.npy').stat().st_size,'peak_allocated_bytes':peak,'seconds_this_run':time.monotonic()-began,
        'resumed_at':start,'extractor_sha256':digest(__file__)}
    (ROOT/'logs/positions-readout-identity.json').write_text(json.dumps(report,indent=2)+'\n');print('READOUT_EXTRACTION_COMPLETE',report,flush=True)

if __name__=='__main__':main()
