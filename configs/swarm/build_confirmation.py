"""Fresh count-only Aegis confirmation, excludes every prior input before inference."""
import hashlib,json,zipfile
from collections import Counter
import numpy as np
from core import ROOT,digest,deduplicate
from fusion_core import read_jsonl,read_csv_rows,validate_fresh_rows
from features import regions_from_offsets
from experiment_core import normalize_prompt,prompt_label

def main():
    from transformers import AutoTokenizer
    path=ROOT/'configs/swarm/manifest.json'
    if path.exists():print('CONFIRMATION_ALREADY_LOCKED');return
    prior=json.loads((ROOT/'configs/highdim/manifest.json').read_text());prefix,suffix=prior['prefix'],prior['suffix']
    slow=AutoTokenizer.from_pretrained(ROOT/'models/llava-v1.6-vicuna-7b',use_fast=False,local_files_only=True)
    fast=AutoTokenizer.from_pretrained(ROOT/'models/llava-v1.6-vicuna-7b',use_fast=True,local_files_only=True)
    old=[]
    for directory in ('signal-selection','highdim','fusion'):old+=read_jsonl(ROOT/'data'/directory/'records.jsonl')
    old+=[{'id':'old-xstest:'+x['id'],'prompt':x['prompt'],'label':int('contrast' in x['type'])} for x in read_csv_rows(ROOT/'data/xstest_prompts.csv')]
    with zipfile.ZipFile(ROOT/'source/HiddenDetect/data/few_shot.zip') as z:
        old+=[{'id':'old-cal:'+str(i),'prompt':x['txt'],'label':x.get('toxicity',0)} for i,x in enumerate(json.loads(z.read('few_shot/few_shot.json')))]
    old=list({x['id']:x for x in old}.values());ids={x['id'] for x in old}
    oldgroups={hashlib.sha256(normalize_prompt(x['prompt']).encode()).hexdigest() for x in old}
    excluded=[{'id':x['id'],'prompt':x['prompt'],'role':'excluded','label':x['label']} for x in old]
    raw=json.loads((ROOT/'data/signal-selection/raw/nvidia__Aegis-AI-Content-Safety-Dataset-2.0/train.json').read_text())
    new=[];reject=Counter()
    for label in (0,1):
        pool=[x for x in raw if prompt_label(x['prompt_label'])==label and 'aegis:train.json:'+str(x['id']) not in ids]
        pool=sorted(pool,key=lambda x:hashlib.sha256(('20261101:'+str(x['id'])).encode()).hexdigest())[:700]
        for x in pool:
            text=x['prompt']
            if not text or text.strip().upper()=='REDACTED' or '<image>' in text:reject['redacted_image_empty']+=1;continue
            full=prefix+text+suffix;tok=slow.encode(full);encoded=fast(full,return_offsets_mapping=True)
            assert tok==encoded['input_ids']
            if len(tok)>512:reject['over512']+=1;continue
            new.append({'id':'aegis:train.json:'+str(x['id']),'prompt':text,'label':label,'source':'aegis','role':'confirmation_aegis',
                'input_length':len(tok),'input_ids_sha256':hashlib.sha256(np.array(tok,dtype=np.int64).tobytes()).hexdigest(),
                'regions':[v.tolist() for v in regions_from_offsets(encoded['offset_mapping'],(len(prefix),len(prefix)+len(text)))]})
    print('CONFIRM_CANDIDATES',len(new),'OLD_EXCLUSIONS',len(old),flush=True)
    kept,removed=deduplicate(excluded+new);pools={y:[x for x in kept if x['role']=='confirmation_aegis' and x['label']==y] for y in (0,1)}
    target=min(200,len(pools[0]),len(pools[1]))//5*5
    assert target>=100,('insufficient_confirmation',len(pools[0]),len(pools[1]))
    probes=[];training=read_jsonl(ROOT/'data/highdim/records.jsonl')
    for label in (0,1):probes += [{**x,'role':'probe_train','reused_probe':True} for x in training if x['role']=='train' and x['label']==label][:32]
    chosen=probes+pools[0][:target]+pools[1][:target]
    for i,x in enumerate(chosen):
        x['index']=i;x['prompt_sha256']=hashlib.sha256(x['prompt'].encode()).hexdigest();x['group_id']=hashlib.sha256(normalize_prompt(x['prompt']).encode()).hexdigest()
    validate_fresh_rows(chosen,ids,oldgroups,{x['id'] for x in probes})
    directory=ROOT/'data/swarm';directory.mkdir(exist_ok=True);records=directory/'records.jsonl'
    records.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in chosen),encoding='utf8')
    manifest={'samples':[{k:v for k,v in x.items() if k!='prompt'} for x in chosen],
        'prefix':prefix,'suffix':suffix,'records_sha256':digest(records),'counts':dict(Counter(x['role']+':'+str(x['label']) for x in chosen)),
        'seed':20261101,'exclusions':dict(reject),'old_unique_ids':len(ids),'scope':'untouched same-source prospective Aegis official-train holdout; not official test or unseen-domain',
        'dedup':'same TFIDF char3-5 .92 connected grouping; vectorized threshold filter equivalent to legacy unions'}
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (ROOT/'logs/swarm-duplicates.json').write_text(json.dumps(removed,indent=2)+'\n');print('CONFIRMATION_LOCKED',manifest['counts'],flush=True)

if __name__=='__main__':main()
