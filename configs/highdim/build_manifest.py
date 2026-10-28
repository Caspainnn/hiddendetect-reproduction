"""New prompts only, no new model scoring; freeze before extraction."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile
from collections import Counter,defaultdict
import numpy as np
from features import ROOT, regions_from_offsets, digest, read_jsonl
sys.path.insert(0,str(ROOT/'configs/signal-selection'))
from experiment_core import normalize_prompt, prompt_label
from build_manifest import remove_later_duplicates

def main():
    from transformers import AutoTokenizer
    from llava.conversation import conv_templates
    slow=AutoTokenizer.from_pretrained(ROOT/'models/llava-v1.6-vicuna-7b',use_fast=False,local_files_only=True)
    fast=AutoTokenizer.from_pretrained(ROOT/'models/llava-v1.6-vicuna-7b',use_fast=True,local_files_only=True)
    assert fast.is_fast
    marker='HIGH_DIM_USER_CONTENT_46f5a981'
    conv=conv_templates['llava_v1'].copy(); conv.append_message(conv.roles[0],marker); conv.append_message(conv.roles[1],None)
    prefix,suffix=conv.get_prompt().split(marker)
    old=read_jsonl(ROOT/'data/signal-selection/records.jsonl')
    old_ids={x['id'] for x in old}
    excluded=[{'id':x['id'],'prompt':x['prompt'],'role':'excluded','label':x['label']} for x in old]
    for x in csv.DictReader((ROOT/'data/xstest_prompts.csv').read_text(encoding='utf-8-sig').splitlines()):
        excluded.append({'id':'old-xstest:'+x['id'],'prompt':x['prompt'],'role':'excluded','label':int('contrast' in x['type'])})
    with zipfile.ZipFile(ROOT/'source/HiddenDetect/data/few_shot.zip') as z:
        for i,x in enumerate(json.loads(z.read('few_shot/few_shot.json'))):
            excluded.append({'id':'old-cal:'+str(i),'prompt':x['txt'],'role':'excluded','label':x.get('toxicity',0)})
    raw=ROOT/'data/signal-selection/raw'; rejected=Counter(); candidates=[]
    def add(identifier,prompt,label,source,role):
        if identifier in old_ids: return
        if not prompt or prompt.strip().upper()=='REDACTED' or '<image>' in prompt:
            rejected['redacted_empty_or_image_token']+=1; return
        text=prefix+prompt+suffix
        ids=slow.encode(text)
        encoded=fast(text,return_offsets_mapping=True)
        if ids != encoded['input_ids']: raise ValueError('Fast/slow tokenizer mismatch')
        if len(ids)>512: rejected['over512']+=1; return
        regions=regions_from_offsets(encoded['offset_mapping'],(len(prefix),len(prefix)+len(prompt)))
        candidates.append({'id':identifier,'prompt':prompt,'label':int(label),'source':source,'role':role,
            'input_length':len(ids),'input_ids_sha256':hashlib.sha256(np.array(ids,dtype=np.int64).tobytes()).hexdigest(),
            'regions':[x.tolist() for x in regions]})
    seed=20261029
    def key(x): return hashlib.sha256((str(seed)+x['id']).encode()).hexdigest()
    for filename,role,cap in [('train.json','pool',1900),('test.json','test_aegis',600)]:
        pool=[{'id':'aegis:'+filename+':'+str(x['id']),'row':x} for x in json.loads((raw/'nvidia__Aegis-AI-Content-Safety-Dataset-2.0'/filename).read_text())]
        for label in (0,1):
            eligible=sorted([x for x in pool if x['id'] not in old_ids and prompt_label(x['row']['prompt_label'])==label],key=key)
            for x in eligible[:cap]: add(x['id'],x['row']['prompt'],label,'aegis',role)
    toxic=list(csv.DictReader((raw/'lmsys__toxic-chat/data/1123/toxic-chat_annotation_test.csv').read_text(encoding='utf-8-sig').splitlines()))
    for label in (0,1):
        pool=[{'id':'toxicchat:'+x['conv_id'],'row':x} for x in toxic if x['human_annotation'].lower()=='true' and int(x['toxicity'])==label and 'toxicchat:'+x['conv_id'] not in old_ids]
        for x in sorted(pool,key=key)[:250]: add(x['id'],x['row']['user_input'],label,'toxicchat_toxicity_proxy','test_toxicchat')
    print('CANDIDATES',len(candidates),'OLD_EXCLUSIONS',len(excluded),flush=True)
    kept,removed=remove_later_duplicates(excluded+candidates)
    pools=defaultdict(list)
    for x in kept:
        if x['role']!='excluded': pools[(x['role'],x['label'])].append(x)
    selected=[]
    for label in (0,1):
        pool=pools[('pool',label)]
        assert len(pool)>=1500, ('insufficient_train_pool',label,len(pool))
        for index,x in enumerate(pool[:1500]):
            x['role']='train' if index<800 else ('dev' if index<1100 else 'calibration')
            selected.append(x)
    for role,cap in [('test_aegis',250),('test_toxicchat',100)]:
        for label in (0,1):
            assert len(pools[(role,label)])>=cap,(role,label,len(pools[(role,label)]))
            selected+=pools[(role,label)][:cap]
    for i,x in enumerate(selected):
        x['index']=i; x['prompt_sha256']=hashlib.sha256(x['prompt'].encode()).hexdigest()
        x['group_id']=hashlib.sha256(normalize_prompt(x['prompt']).encode()).hexdigest()
    assert len(selected)==3700 and len({x['group_id'] for x in selected})==3700
    directory=ROOT/'data/highdim'; directory.mkdir(exist_ok=True)
    records=directory/'records.jsonl'; records.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in selected),encoding='utf-8')
    report={'seed':seed,'prospective':True,'samples':[{k:v for k,v in x.items() if k!='prompt'} for x in selected],
        'counts':dict(Counter(x['role']+':'+str(x['label']) for x in selected)), 'records_sha256':digest(records),
        'old_manifest_sha256':digest(ROOT/'configs/signal-selection/manifest.json'),
        'raw_lock_sha256':digest(ROOT/'logs/signal-selection-download.json'),'no_truncation':True,'max_tokens':512,
        'tokenizer_offsets_verified_against_original_ids':True,'prefix':prefix,'suffix':suffix,
        'exclusions':dict(rejected),'duplicate_count':len(removed),
        'calibration_source':'unused Aegis train, disjoint from training/development and all old prompts',
        'label_scope':{'aegis':'human prompt harmfulness','toxicchat':'human toxicity proxy, separate task'}}
    (Path(__file__).parent/'manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    (ROOT/'logs/highdim-duplicates.json').write_text(json.dumps(removed,indent=2)+'\n')
    print('MANIFEST_LOCKED',json.dumps(report['counts']),flush=True)

if __name__=='__main__': main()
