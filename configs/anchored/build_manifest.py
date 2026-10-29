"""Count-first fresh confirmation across two sources; no score inspection."""
import json,hashlib,zipfile
from collections import Counter
import numpy as np
from anchored_core import ROOT,digest,write
from position_core import read_jsonl,read_csv_rows,deduplicate,normalize_prompt,prompt_label,regions_from_offsets

def main():
    from transformers import AutoTokenizer
    target=ROOT/'configs/anchored/manifest.json';assert not target.exists()
    previous=json.loads((ROOT/'configs/processing/manifest.json').read_text());prefix,suffix=previous['prefix'],previous['suffix']
    slow=AutoTokenizer.from_pretrained(ROOT/'models/llava-v1.6-vicuna-7b',use_fast=False,local_files_only=True);fast=AutoTokenizer.from_pretrained(ROOT/'models/llava-v1.6-vicuna-7b',use_fast=True,local_files_only=True);anchor=len(slow.encode(prefix))-3
    records={f:read_jsonl(ROOT/'data'/f/'records.jsonl') for f in ('signal-selection','highdim','fusion','swarm','positions','routing','processing','heads','stability')}
    old=sum(records.values(),[])
    old += [{'id':'old-xstest:'+x['id'],'prompt':x['prompt'],'label':int('contrast' in x['type'])} for x in read_csv_rows(ROOT/'data/xstest_prompts.csv')]
    with zipfile.ZipFile(ROOT/'source/HiddenDetect/data/few_shot.zip') as z:
        old += [{'id':'old-cal:'+str(i),'prompt':x['txt'],'label':x.get('toxicity',0)} for i,x in enumerate(json.loads(z.read('few_shot/few_shot.json')))]
    old=list({x['id']:x for x in old}.values());oldids={r['id'] for r in old};chosen=[]
    for r in records['stability']:
        if r['role'] in ('calibration_aegis','calibration_toxic','test_aegis_holdout','test_toxic_holdout'):chosen.append({**r,'stability_index':r['index']})
    assert len(chosen)==2200
    chosen += [{**r,'role':'meta_train'} for r in records['processing'] if r['role']=='dev'];assert len(chosen)==2800
    candidates=[];reject=Counter();rawdir=ROOT/'data/signal-selection/raw'
    def add(sid,text,label,source):
        if sid in oldids:return
        if not text or text.strip().upper()=='REDACTED' or '<image>' in text:reject['empty_redacted_image']+=1;return
        ids=slow.encode(prefix+text+suffix)
        if len(ids)>512:reject['over512']+=1;return
        enc=fast(prefix+text+suffix,return_offsets_mapping=True);assert ids==enc['input_ids']
        candidates.append({'id':sid,'prompt':text,'label':int(label),'role':'candidate_'+source,'source':'aegis' if source=='aegis' else 'toxicchat_toxicity_proxy','input_length':len(ids),'input_ids_sha256':hashlib.sha256(np.array(ids,dtype=np.int64).tobytes()).hexdigest(),'regions':[a.tolist() for a in regions_from_offsets(enc['offset_mapping'],(len(prefix),len(prefix)+len(text)))]})
    raw=json.loads((rawdir/'nvidia__Aegis-AI-Content-Safety-Dataset-2.0/train.json').read_text())
    key=lambda sid:hashlib.sha256(('20270301:'+sid).encode()).hexdigest()
    for x in sorted(raw,key=lambda x:key(str(x['id']))):add('aegis:train.json:'+str(x['id']),x['prompt'],prompt_label(x['prompt_label']),'aegis')
    toxic=[]
    for split in ('train','test'):toxic+=read_csv_rows(rawdir/'lmsys__toxic-chat/data/1123'/('toxic-chat_annotation_'+split+'.csv'))
    for x in sorted(toxic,key=lambda x:key(x['conv_id'])):
        if x['human_annotation'].lower()=='true':add('toxicchat:'+x['conv_id'],x['user_input'],int(x['toxicity']),'toxic')
    print('CANDIDATES',len(candidates),'OLD',len(oldids),flush=True)
    kept,removed=deduplicate([{**r,'role':'excluded'} for r in old]+candidates);counts={}
    for source in ('aegis','toxic'):
        pools={y:[r for r in kept if r['role']=='candidate_'+source and r['label']==y] for y in (0,1)}
        n=500 if source=='aegis' else min(200,*[len(v) for v in pools.values()])//25*25
        print('COUNTS',source,{y:len(v) for y,v in pools.items()},'TAKE',n,flush=True)
        assert all(len(v)>=n for v in pools.values()) and n>=(500 if source=='aegis' else 50)
        counts[source]=n
        for y in (0,1):chosen += [{**r,'role':'confirmation_'+source} for r in pools[y][:n]]
    fresh=0
    for i,r in enumerate(chosen):
        r['index']=i;r['prefix_anchor']=anchor;r['prompt_sha256']=hashlib.sha256(r['prompt'].encode()).hexdigest();r['group_id']=hashlib.sha256(normalize_prompt(r['prompt']).encode()).hexdigest()
        if 'stability_index' not in r:r['fresh_index']=fresh;fresh+=1
    assert len({r['group_id'] for r in chosen})==len(chosen)
    trainix=json.loads((ROOT/'configs/stability/manifest.json').read_text())['draws'][0];trainids={records['stability'][i]['id'] for i in trainix}
    assert not trainids & {r['id'] for r in chosen}
    probes=sum(([r for r in records['stability'] if r.get('origin')=='original_train' and r['label']==y][:16] for y in (0,1)),[])
    directory=ROOT/'data/anchored';directory.mkdir(exist_ok=True);rp=directory/'records.jsonl';rp.write_bytes((''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in chosen)).encode())
    write(directory/'probes.jsonl',None) if False else None
    pp=directory/'probes.jsonl';pp.write_bytes((''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in probes)).encode())
    write(target,{'samples':[{k:v for k,v in r.items() if k!='prompt'} for r in chosen],'prefix':prefix,'suffix':suffix,'records_sha256':digest(rp),'probes_sha256':digest(pp),'old_unique_exclusions':len(oldids),'fresh_class_count':counts,'counts':dict(Counter(r['role']+':'+str(r['label']) for r in chosen)),'new_feature_rows':fresh,'exclusions':dict(reject),'scope':'new same-source Aegis harmfulness and distinct ToxicChat toxicity auxiliary confirmation; separate tasks'})
    write(ROOT/'logs/anchored-duplicates.json',removed);print('MANIFEST_LOCKED',len(chosen),'NEW_FEATURES',fresh,flush=True)

if __name__=='__main__':main()
