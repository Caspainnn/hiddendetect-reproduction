"""Stream full-position information into bounded private vector/summary caches."""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from features import ROOT, digest, read_jsonl, projection_matrix, open_cache, checkpoint, TRACE_NAMES, ATTENTION_NAMES, MODULE_NAMES, LAYOUT

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--pilot-only',action='store_true'); args=parser.parse_args()
    os.chdir(ROOT); os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    import torch.nn.functional as nnf
    import transformers
    from scipy.stats import rankdata
    from safetensors import safe_open
    from llava.model.builder import load_pretrained_model
    from llava.conversation import conv_templates
    torch.manual_seed(0)
    mp=Path(__file__).with_name('manifest.json'); manifest=json.loads(mp.read_text()); key=digest(mp)
    records_path=ROOT/'data/highdim/records.jsonl'; assert digest(records_path)==manifest['records_sha256']
    rows=read_jsonl(records_path)
    code=digest(__file__)
    author=ROOT/'source/HiddenDetect'
    assert subprocess.check_output(['git','-C',str(author),'rev-parse','HEAD'],text=True).strip()=='2fd161b1fc47c99b36fe0a299ec64ff0b816f0bb'
    assert not subprocess.check_output(['git','-C',str(author),'status','--porcelain','--untracked-files=no'],text=True).strip()
    spec=importlib.util.spec_from_file_location('author_test',author/'code/test.py'); original=importlib.util.module_from_spec(spec)
    sys.path.insert(0,str(author/'code')); spec.loader.exec_module(original)
    began=time.monotonic()
    tokenizer,model,processor,context=load_pretrained_model(str(ROOT/'models/llava-v1.6-vicuna-7b'),None,
        'llava-v1.6-vicuna-7b',device_map='auto',torch_dtype=torch.float16)
    assert not model.training and not any(p.is_meta for p in model.parameters())
    assert model.config.num_hidden_layers==32 and model.config.hidden_size==4096 and model.config.num_attention_heads==32
    weight_map=json.loads((ROOT/'models/llava-v1.6-vicuna-7b/model.safetensors.index.json').read_text())['weight_map']
    vision=sorted(x for x in weight_map if 'vision_tower' in x)
    for name in [vision[0],vision[len(vision)//2],vision[-1]]:
        with safe_open(ROOT/'models/llava-v1.6-vicuna-7b'/weight_map[name],framework='pt',device='cpu') as stream:
            actual=model.state_dict()[name]; assert torch.equal(actual.cpu(),stream.get_tensor(name).to(actual.dtype))
    reference=torch.zeros(32000,device=model.device,dtype=torch.float32)
    reference[[tokenizer.encode(x,add_special_tokens=False)[0] for x in original.refusal_lst]]=1
    projection=torch.tensor(projection_matrix(),device=model.device)
    hidden=[None]*32; attention_outputs=[None]*32; mlp_outputs=[None]*32; hooks=[]
    def capture(target,index):
        def callback(module,inputs,output): target[index]=(output[0] if isinstance(output,tuple) else output).detach()
        return callback
    for i,layer in enumerate(model.model.layers):
        hooks += [layer.register_forward_hook(capture(hidden,i)),
                  layer.self_attn.register_forward_hook(capture(attention_outputs,i)),
                  layer.mlp.register_forward_hook(capture(mlp_outputs,i))]
    def collect(row,attention):
        text=manifest['prefix']+row['prompt']+manifest['suffix']
        ids=torch.tensor(tokenizer.encode(text),device=model.device).unsqueeze(0)
        assert ids.shape[-1]==row['input_length']<=512
        assert hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()==row['input_ids_sha256']
        assert hashlib.sha256(row['prompt'].encode()).hexdigest()==row['prompt_sha256']
        regions=[torch.tensor(x,device=model.device,dtype=torch.long) for x in row['regions']]
        user=torch.cat(regions); assert len(user)>0 and torch.equal(user,torch.unique_consecutive(user))
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start=time.monotonic()
        with torch.no_grad():
            output=model(ids,images=None,image_sizes=None,output_hidden_states=False,output_attentions=attention)
        torch.cuda.synchronize(); forward_seconds=time.monotonic()-start
        result={'P':[],'X':[],'S':[],'A':[],'M':[],'F':[]}
        previous=None
        with torch.no_grad():
            for layer in range(32):
                raw=hidden[layer][0].float()
                norms=torch.linalg.vector_norm(raw,dim=-1)
                unit=raw/norms.clamp_min(1e-12).unsqueeze(-1)
                pooled=torch.stack([unit[-1]]+[unit[ix].mean(0) if len(ix) else torch.zeros(4096,device=unit.device) for ix in regions])
                result['P'].append(pooled.half().cpu().numpy())
                result['X'].append((pooled@projection).cpu().numpy())
                u=unit[user]; n=norms[user]
                adjacent=(u[1:]*u[:-1]).sum(-1) if len(user)>1 else torch.ones(1,device=u.device)
                cross=(u*previous).sum(-1) if previous is not None else torch.ones(len(user),device=u.device)
                # Author final layer state is already final-normalized; preserve its second norm.
                author_state=hidden[layer] if layer<31 else model.model.norm(hidden[layer])
                logits=model.lm_head(model.model.norm(author_state))[0]
                refusal=nnf.cosine_similarity(logits,reference,dim=-1)
                result['F'].append(float(refusal[-1]))
                logits_user=logits[user].float()
                logz=torch.logsumexp(logits_user,dim=-1)
                probability=torch.softmax(logits_user,dim=-1)
                entropy=logz-(probability*logits_user).sum(-1)
                top=torch.topk(logits_user,2,dim=-1).values
                entropy_np=entropy.cpu().numpy()
                ranks=rankdata(entropy_np)
                trend=float(np.corrcoef(np.arange(len(ranks)),ranks)[0,1]) if len(ranks)>1 and np.std(ranks)>0 else 0.
                r=refusal[user]
                statistics=[n.mean(),n.std(unbiased=False),n.max(),adjacent.mean(),adjacent.min(),(u[0]*u[-1]).sum(),
                    cross.mean(),cross.min(),r.mean(),r.std(unbiased=False),r.max(),r.min(),
                    entropy.mean(),entropy.std(unbiased=False),torch.tensor(trend,device=u.device),(top[:,0]-top[:,1]).mean()]
                result['S'].append(torch.stack(statistics).cpu().numpy())
                if attention:
                    weights=output.attentions[layer][0,:,-1,:].float()
                    prefix_end=int(user[0]); suffix_start=int(user[-1])+1
                    masses=[weights[:,:prefix_end].sum(-1)]
                    masses += [weights[:,ix].sum(-1) if len(ix) else torch.zeros(32,device=u.device) for ix in regions]
                    masses += [weights[:,suffix_start:].sum(-1),
                        -(weights*weights.clamp_min(1e-30).log()).sum(-1)/np.log(max(2,weights.shape[-1])),weights.max(-1).values]
                    values=torch.stack(masses,dim=-1)
                    assert torch.allclose(values[:,:6].sum(-1),torch.ones(32,device=u.device),atol=.003)
                    result['A'].append(values.cpu().numpy())
                else: result['A'].append(np.zeros((32,8),np.float32))
                a=attention_outputs[layer][0].float(); m=mlp_outputs[layer][0].float()
                an=torch.linalg.vector_norm(a,dim=-1); mn=torch.linalg.vector_norm(m,dim=-1)
                cosine=nnf.cosine_similarity(a,m,dim=-1)
                result['M'].append(torch.stack([an[-1],an[user].mean(),an[user].max(),mn[-1],mn[user].mean(),mn[user].max(),cosine[-1],cosine[user].mean()]).cpu().numpy())
                previous=u
                del logits,logits_user,probability,raw,unit,a,m
            final=output.logits[0,-1].detach()
            assert torch.equal(final,final.half().to(final.dtype))
            result['L']=final.half().cpu().numpy()
        result={k:np.asarray(v,dtype=np.float16 if k in ('P','L') else np.float32) for k,v in result.items()}
        result['Q']=np.array([row['input_length'],len(user),*[int(not len(ix)) for ix in regions]],np.float32)
        assert all(np.isfinite(x).all() for x in result.values())
        peak=torch.cuda.max_memory_allocated()
        # Drop per-forward module tensors, not references retained by prior rows.
        for target in (hidden,attention_outputs,mlp_outputs): target[:]=[None]*32
        return result,forward_seconds,peak
    pilot_path=ROOT/'logs/highdim-pilot.json'
    if pilot_path.exists():
        pilot=json.loads(pilot_path.read_text())
        assert pilot['manifest_sha256']==key and pilot['extractor_sha256']==code
    else:
        sample=[]
        for label in (0,1): sample += [x for x in rows if x['role']=='train' and x['label']==label][:32]
        collect(sample[0],False)
        base=[]; with_attention=[]; peaks=[]; features=[]
        try:
            for i,row in enumerate(sample):
                _,seconds,_=collect(row,False); base.append(seconds)
                values,seconds,peak=collect(row,True); with_attention.append(seconds); peaks.append(peak)
                if i<4 or 32<=i<36: features.append(values['F'])
                if (i+1)%16==0: print('RESOURCE_PILOT',i+1,flush=True)
            ratio=float(np.median(np.array(with_attention)/base)); enabled=max(peaks)<22*1024**3 and ratio<=2
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); ratio=None; enabled=False
        pilot={'manifest_sha256':key,'extractor_sha256':code,'rows':len(base),'attention_enabled':enabled,
            'median_forward_time_ratio':ratio,'maximum_allocated_bytes':max(peaks) if peaks else None,
            'rule':'no OOM, max allocated <22GiB and median paired forward ratio<=2',
            'baseline_forward_seconds':base,'attention_forward_seconds':with_attention}
        pilot_path.write_text(json.dumps(pilot,indent=2)+'\n')
        print('PILOT_DECISION',json.dumps({k:v for k,v in pilot.items() if not isinstance(v,list)}),flush=True)
    if args.pilot_only: return
    enabled=pilot['attention_enabled']
    provenance={'extractor_sha256':code,'features_code_sha256':digest(Path(__file__).with_name('features.py')),
        'artifact_lock_sha256':digest(ROOT/'configs/artifacts-lock.json'),'torch':torch.__version__,
        'transformers':transformers.__version__,'numpy':np.__version__,'attention_enabled':enabled,
        'pilot_sha256':digest(pilot_path),'projection_seed':20261029,'boundary':'raw decoder layer output, pre-final-norm; per-token unit directions'}
    cache,start=open_cache(ROOT/'cache/highdim',key,len(rows),provenance)
    smoke=[]
    for label in (0,1): smoke += [x for x in rows if x['role']=='train' and x['label']==label][:4]
    manual=[collect(row,False)[0]['F'] for row in smoke]
    captured=[]; old_loader=original.load_pretrained_model; old_cosine=original.N.cosine_similarity
    def capture_cosine(*args,**kwargs):
        value=old_cosine(*args,**kwargs); captured.append(float(value.item())); return value
    original.load_pretrained_model=lambda *a,**k:(tokenizer,model,processor,context)
    original.N.cosine_similarity=capture_cosine
    try: original.test([{'txt':x['prompt'],'img':None,'toxicity':x['label']} for x in smoke],str(ROOT/'models/llava-v1.6-vicuna-7b'),s=16,e=29)
    finally: original.load_pretrained_model=old_loader; original.N.cosine_similarity=old_cosine
    difference=float(np.max(np.abs(np.array(captured).reshape(8,32)-manual))); assert difference<=1e-7,difference
    (ROOT/'logs/highdim-author-smoke.json').write_text(json.dumps({'rows':8,'train_only':True,'maximum_F_difference':difference,'extractor_sha256':code},indent=2)+'\n')
    peak=0
    for i in range(start,len(rows)):
        values,seconds,allocated=collect(rows[i],enabled); peak=max(peak,allocated)
        for name,value in values.items(): cache[name][i]=value
        if (i+1)%16==0 or i+1==len(rows): checkpoint(cache,i+1)
        if (i+1)%100==0: print('CACHED',i+1,'/',len(rows),'SECONDS',round(time.monotonic()-began,2),flush=True)
    report={'status':'complete','rows':len(rows),'resumed_at':start,'manifest_sha256':key,'extractor_sha256':code,
        'seconds_this_run':time.monotonic()-began,'maximum_allocated_bytes':peak,'attention_enabled':enabled,
        'cache_bytes':sum((cache['directory']/(name+'.npy')).stat().st_size for name in LAYOUT),
        'trace_names':TRACE_NAMES,'attention_names':ATTENTION_NAMES,'module_names':MODULE_NAMES,
        'normalization':'unit raw decoder-layer vectors; module/norm summaries separate; author F boundary preserved',
        'base_model_training':False,'generation':False}
    (ROOT/'logs/highdim-extraction.json').write_text(json.dumps(report,indent=2)+'\n')
    print('FEATURE_CACHE_COMPLETE',json.dumps(report),flush=True)

if __name__=='__main__': main()
