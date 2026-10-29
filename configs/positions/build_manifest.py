"""Reuse immutable comparison rows and select fresh confirmation by counts only."""
import json,hashlib,zipfile
from collections import Counter
import numpy as np
from position_core import ROOT,digest,read_jsonl,read_csv_rows,deduplicate,token_indices,NAMES,regions_from_offsets,normalize_prompt,prompt_label

def main():
    from transformers import AutoTokenizer
    target=ROOT/'configs/positions/manifest.json'
    if target.exists():print('MANIFEST_ALREADY_LOCKED');return
    base=json.loads((ROOT/'configs/highdim/manifest.json').read_text());prefix,suffix=base['prefix'],base['suffix']
    slow=AutoTokenizer.from_pretrained(ROOT/'models/llava-v1.6-vicuna-7b',use_fast=False,local_files_only=True)
    fast=AutoTokenizer.from_pretrained(ROOT/'models/llava-v1.6-vicuna-7b',use_fast=True,local_files_only=True)
    prefix_ids=slow.encode(prefix);anchor=len(prefix_ids)-3;assert anchor>=0
    records={family:read_jsonl(ROOT/'data'/family/'records.jsonl') for family in ('signal-selection','highdim','fusion','swarm')}
    old=sum(records.values(),[])
    old += [{'id':'old-xstest:'+x['id'],'prompt':x['prompt'],'label':int('contrast' in x['type'])} for x in read_csv_rows(ROOT/'data/xstest_prompts.csv')]
    with zipfile.ZipFile(ROOT/'source/HiddenDetect/data/few_shot.zip') as z:
        old += [{'id':'old-cal:'+str(i),'prompt':x['txt'],'label':x.get('toxicity',0)} for i,x in enumerate(json.loads(z.read('few_shot/few_shot.json')))]
    old=list({x['id']:x for x in old}.values());oldids={x['id'] for x in old};oldgroups={hashlib.sha256(normalize_prompt(x['prompt']).encode()).hexdigest() for x in old}
    chosen=[]
    for family in ('highdim','fusion','swarm'):
        for x in records[family]:
            take=x['role'] in ('train','dev') if family=='highdim' else x['role']!='probe_train'
            if take:chosen.append({**x,'role':'historical_confirmation' if family=='swarm' else x['role'],'origin':family,'origin_index':x['index']})
    assert len(chosen)==5360 and len({x['id'] for x in chosen})==5360
    candidates=[];reject=Counter()
    raw=json.loads((ROOT/'data/signal-selection/raw/nvidia__Aegis-AI-Content-Safety-Dataset-2.0/train.json').read_text())
    for label in (0,1):
        pool=[x for x in raw if prompt_label(x['prompt_label'])==label and 'aegis:train.json:'+str(x['id']) not in oldids]
        pool=sorted(pool,key=lambda x:hashlib.sha256(('20261118:'+str(x['id'])).encode()).hexdigest())[:1100]
        for x in pool:
            text=x['prompt']
            if not text or text.strip().upper()=='REDACTED' or '<image>' in text:reject['empty_redacted_image']+=1;continue
            full=prefix+text+suffix;ids=slow.encode(full);encoded=fast(full,return_offsets_mapping=True);assert ids==encoded['input_ids']
            if len(ids)>512:reject['over512']+=1;continue
            candidates.append({'id':'aegis:train.json:'+str(x['id']),'prompt':text,'label':label,'role':'confirmation_aegis','source':'aegis',
                'input_length':len(ids),'input_ids_sha256':hashlib.sha256(np.array(ids,dtype=np.int64).tobytes()).hexdigest(),
                'regions':[a.tolist() for a in regions_from_offsets(encoded['offset_mapping'],(len(prefix),len(prefix)+len(text)))]})
    print('NEW_CANDIDATES',len(candidates),'OLD_EXCLUSIONS',len(oldids),flush=True)
    excluded=[{**x,'role':'excluded'} for x in old]
    kept,removed=deduplicate(excluded+candidates)
    pools={y:[x for x in kept if x['role']=='confirmation_aegis' and x['label']==y] for y in (0,1)}
    n=min(500,len(pools[0]),len(pools[1]))//25*25;assert n>=200,('insufficient_fresh_counts',len(pools[0]),len(pools[1]))
    fresh=pools[0][:n]+pools[1][:n]
    for x in fresh:assert x['id'] not in oldids and hashlib.sha256(normalize_prompt(x['prompt']).encode()).hexdigest() not in oldgroups
    chosen+=fresh
    for i,x in enumerate(chosen):
        x['index']=i;x['prefix_anchor']=anchor;x['prompt_sha256']=hashlib.sha256(x['prompt'].encode()).hexdigest();x['group_id']=hashlib.sha256(normalize_prompt(x['prompt']).encode()).hexdigest()
        ids=slow.encode(prefix+x['prompt']+suffix)
        assert ids[:anchor+1]==prefix_ids[:anchor+1] and len(ids)==x['input_length']
        assert hashlib.sha256(np.array(ids,dtype=np.int64).tobytes()).hexdigest()==x['input_ids_sha256']
        x['readout_indices']=token_indices(x)
    assert len({x['group_id'] for x in chosen})==len(chosen)
    directory=ROOT/'data/positions';directory.mkdir(exist_ok=True);path=directory/'records.jsonl'
    path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in chosen),encoding='utf8')
    count=len(chosen);estimate=count*32*9*4096*2+128;assert estimate<15*1024**3
    m={'samples':[{k:v for k,v in x.items() if k!='prompt'} for x in chosen],
        'prefix':prefix,'suffix':suffix,'records_sha256':digest(path),'readouts':NAMES,'counts':dict(Counter(x['role']+':'+str(x['label']) for x in chosen)),
        'benchmark_sha256':digest(ROOT/'configs/benchmarks/v1.json'),'cache_estimated_bytes':estimate,'fresh_class_count':n,
        'old_unique_exclusions':len(oldids),'exclusions':dict(reject),'prefix_control_index':anchor,
        'prefix_control':'prefix-alone token count minus3, verifies protected shared token prefix; avoids user-boundary tokenizer lookahead',
        'confirmation':'fresh same-source Aegis train partition, not official test or unseen-domain',
        'coincident_readout_rows':sum(len(set(x['readout_indices'][1:7]))<6 for x in chosen)}
    target.write_text(json.dumps(m,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    (ROOT/'logs/positions-duplicates.json').write_text(json.dumps(removed,indent=2)+'\n')
    print('MANIFEST_LOCKED',count,n,'CACHE_BYTES',estimate,flush=True)

if __name__=='__main__':main()
