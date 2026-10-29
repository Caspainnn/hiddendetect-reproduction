"""Freeze new pool and fresh confirmation before inference, without score access."""
import json,hashlib,zipfile
from collections import Counter
import numpy as np
from stability_core import ROOT,digest,write,draws
from position_core import read_jsonl,read_csv_rows,deduplicate,normalize_prompt,prompt_label,regions_from_offsets

def main():
    from transformers import AutoTokenizer
    target=ROOT/'configs/stability/manifest.json';assert not target.exists()
    previous=json.loads((ROOT/'configs/processing/manifest.json').read_text());prefix,suffix=previous['prefix'],previous['suffix']
    slow=AutoTokenizer.from_pretrained(ROOT/'models/llava-v1.6-vicuna-7b',use_fast=False,local_files_only=True)
    fast=AutoTokenizer.from_pretrained(ROOT/'models/llava-v1.6-vicuna-7b',use_fast=True,local_files_only=True)
    anchor=len(slow.encode(prefix))-3
    records={f:read_jsonl(ROOT/'data'/f/'records.jsonl') for f in ('signal-selection','highdim','fusion','swarm','positions','routing','processing','heads')}
    old=sum(records.values(),[])
    old += [{'id':'old-xstest:'+x['id'],'prompt':x['prompt'],'label':int('contrast' in x['type'])} for x in read_csv_rows(ROOT/'data/xstest_prompts.csv')]
    with zipfile.ZipFile(ROOT/'source/HiddenDetect/data/few_shot.zip') as z:
        old += [{'id':'old-cal:'+str(i),'prompt':x['txt'],'label':x.get('toxicity',0)} for i,x in enumerate(json.loads(z.read('few_shot/few_shot.json')))]
    old=list({x['id']:x for x in old}.values());oldids={x['id'] for x in old}
    chosen=[]
    for r in records['processing']:
        if r['role']=='train':chosen.append({**r,'role':'pool','origin':'original_train','processing_index':r['index']})
        elif r['role'] in ('test_aegis_holdout','test_toxic_holdout','calibration_toxic') or r['role']=='calibration_aegis' and r['label']==0:
            chosen.append({**r,'processing_index':r['index']})
    candidates=[];reject=Counter()
    raw=json.loads((ROOT/'data/signal-selection/raw/nvidia__Aegis-AI-Content-Safety-Dataset-2.0/train.json').read_text())
    for x in sorted(raw,key=lambda x:hashlib.sha256(('20270200:'+str(x['id'])).encode()).hexdigest()):
        sid='aegis:train.json:'+str(x['id']);label=prompt_label(x['prompt_label'])
        if sid in oldids or label not in (0,1):continue
        text=x['prompt']
        if not text or text.strip().upper()=='REDACTED' or '<image>' in text:reject['empty_redacted_image']+=1;continue
        ids=slow.encode(prefix+text+suffix)
        if len(ids)>512:reject['over512']+=1;continue
        enc=fast(prefix+text+suffix,return_offsets_mapping=True);assert ids==enc['input_ids']
        candidates.append({'id':sid,'prompt':text,'label':label,'role':'candidate','source':'aegis','input_length':len(ids),
            'input_ids_sha256':hashlib.sha256(np.array(ids,dtype=np.int64).tobytes()).hexdigest(),
            'regions':[a.tolist() for a in regions_from_offsets(enc['offset_mapping'],(len(prefix),len(prefix)+len(text)))]})
    print('CANDIDATES',len(candidates),'OLD',len(oldids),flush=True)
    kept,removed=deduplicate([{**x,'role':'excluded'} for x in old]+candidates)
    pools={y:[r for r in kept if r['role']=='candidate' and r['label']==y] for y in (0,1)}
    print('FRESH_COUNTS',{y:len(v) for y,v in pools.items()},flush=True)
    assert all(len(v)>=1300 for v in pools.values()),'Need 800 new train +500 fresh per class'
    for y in (0,1):
        for i,r in enumerate(pools[y][:1300]):
            chosen.append({**r,'role':'pool' if i<800 else 'confirmation_aegis','origin':'new_train' if i<800 else 'fresh_confirmation'})
    for i,r in enumerate(chosen):
        r['index']=i;r['prefix_anchor']=anchor;r['prompt_sha256']=hashlib.sha256(r['prompt'].encode()).hexdigest();r['group_id']=hashlib.sha256(normalize_prompt(r['prompt']).encode()).hexdigest()
    assert len(chosen)==6400 and len({r['group_id'] for r in chosen})==6400
    directory=ROOT/'data/stability';directory.mkdir(exist_ok=True);rp=directory/'records.jsonl'
    rp.write_bytes((''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in chosen)).encode())
    rows=[{k:v for k,v in r.items() if k!='prompt'} for r in chosen];ix=draws(rows)
    write(target,{'samples':rows,'prefix':prefix,'suffix':suffix,'old_unique_exclusions':len(oldids),'records_sha256':digest(rp),
        'counts':dict(Counter(r['role']+':'+str(r['label']) for r in rows)),'draws':[a.tolist() for a in ix],
        'overlap':[[len(set(a)&set(b)) for b in ix] for a in ix],'exclusions':dict(reject),'new_candidate_counts':{str(y):len(v) for y,v in pools.items()},
        'scope':'fresh same-source train-pool perturbations and confirmation, old v1 descriptive; no unseen-domain claim'})
    write(ROOT/'logs/stability-duplicates.json',removed);print('MANIFEST_LOCKED',len(rows),flush=True)

if __name__=='__main__':main()
