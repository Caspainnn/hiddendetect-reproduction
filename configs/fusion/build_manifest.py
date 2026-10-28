"""Count-first fresh calibration/holdouts, with explicit reused training probes."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from fusion_core import ROOT,read_csv_rows,validate_fresh_rows,digest,read_jsonl
sys.path.insert(0,str(ROOT/'configs/signal-selection'))
from build_manifest import remove_later_duplicates
from experiment_core import normalize_prompt,prompt_label
from features import regions_from_offsets

def main():
    from transformers import AutoTokenizer
    slow=AutoTokenizer.from_pretrained(ROOT/'models/llava-v1.6-vicuna-7b',use_fast=False,local_files_only=True)
    fast=AutoTokenizer.from_pretrained(ROOT/'models/llava-v1.6-vicuna-7b',use_fast=True,local_files_only=True)
    prior=json.loads((ROOT/'configs/highdim/manifest.json').read_text()); prefix,suffix=prior['prefix'],prior['suffix']
    old=read_jsonl(ROOT/'data/signal-selection/records.jsonl')+read_jsonl(ROOT/'data/highdim/records.jsonl')
    for x in read_csv_rows(ROOT/'data/xstest_prompts.csv'):
        old.append({'id':'old-xstest:'+x['id'],'prompt':x['prompt'],'label':int('contrast' in x['type'])})
    with zipfile.ZipFile(ROOT/'source/HiddenDetect/data/few_shot.zip') as z:
        old += [{'id':'old-cal:'+str(i),'prompt':x['txt'],'label':x.get('toxicity',0)} for i,x in enumerate(json.loads(z.read('few_shot/few_shot.json')))]
    old_ids={x['id'] for x in old}; old_groups={hashlib.sha256(normalize_prompt(x['prompt']).encode()).hexdigest() for x in old}
    excluded=[{'id':x['id'],'prompt':x['prompt'],'label':x['label'],'role':'excluded'} for x in old]
    raw=ROOT/'data/signal-selection/raw'; candidates=[]; rejected=Counter(); seed=20261030
    def order(x): return hashlib.sha256((str(seed)+x['id']).encode()).hexdigest()
    def add(identifier,text,label,source,role):
        if identifier in old_ids: return
        if not text or text.strip().upper()=='REDACTED' or '<image>' in text: rejected['redacted_empty_image']+=1; return
        full=prefix+text+suffix; ids=slow.encode(full); encoded=fast(full,return_offsets_mapping=True)
        assert ids==encoded['input_ids'],'Tokenizer mismatch'
        if len(ids)>512: rejected['over512']+=1; return
        regions=regions_from_offsets(encoded['offset_mapping'],(len(prefix),len(prefix)+len(text)))
        candidates.append({'id':identifier,'prompt':text,'label':int(label),'source':source,'role':role,
            'input_length':len(ids),'input_ids_sha256':hashlib.sha256(np.array(ids,dtype=np.int64).tobytes()).hexdigest(),
            'regions':[x.tolist() for x in regions]})
    for filename,role,cap in [('train.json','aegis_pool',1400),('test.json','test_aegis_official',1000)]:
        pool=[{'id':'aegis:'+filename+':'+str(x['id']),'row':x} for x in json.loads((raw/'nvidia__Aegis-AI-Content-Safety-Dataset-2.0'/filename).read_text())]
        for label in (0,1):
            eligible=[x for x in pool if x['id'] not in old_ids and prompt_label(x['row']['prompt_label'])==label]
            for x in sorted(eligible,key=order)[:cap]: add(x['id'],x['row']['prompt'],label,'aegis',role)
    for split,role,cap in [('train','toxic_pool',1200),('test','test_toxicchat',300)]:
        rows=read_csv_rows(raw/'lmsys__toxic-chat/data/1123'/('toxic-chat_annotation_'+split+'.csv'))
        for label in (0,1):
            pool=[{'id':'toxicchat:'+x['conv_id'],'row':x} for x in rows if x['human_annotation'].lower()=='true' and int(x['toxicity'])==label and 'toxicchat:'+x['conv_id'] not in old_ids]
            for x in sorted(pool,key=order)[:cap]: add(x['id'],x['row']['user_input'],label,'toxicchat_toxicity_proxy',role)
    print('NEW_CANDIDATES',len(candidates),'OLD_EXCLUSIONS',len(excluded),flush=True)
    kept,removed=remove_later_duplicates(excluded+candidates); pools=defaultdict(list)
    for x in kept:
        if x['role']!='excluded': pools[(x['role'],x['label'])].append(x)
    chosen=[]
    for label in (0,1):
        pool=pools[('aegis_pool',label)]; assert len(pool)>=900,('aegis_pool',label,len(pool))
        for i,x in enumerate(pool[:900]): x['role']='calibration_aegis' if i<400 else 'test_aegis_holdout'; chosen.append(x)
    official=min(100,*[len(pools[('test_aegis_official',label)]) for label in (0,1)])//5*5
    toxic=min(50,*[len(pools[('test_toxicchat',label)]) for label in (0,1)])//5*5
    assert official>=30 and toxic>=10,(official,toxic)
    for role,cap in [('test_aegis_official',official),('test_toxicchat',toxic)]:
        for label in (0,1): chosen+=pools[(role,label)][:cap]
    safe=pools[('toxic_pool',0)]; harmful=pools[('toxic_pool',1)]
    target=min(200,len(harmful),len(safe)-400)//5*5
    assert target>=100,('insufficient_toxic_prospective',len(safe),len(harmful))
    for i,x in enumerate(safe[:400+target]): x['role']='calibration_toxic' if i<400 else 'test_toxic_holdout'; chosen.append(x)
    for x in harmful[:target]: x['role']='test_toxic_holdout'; chosen.append(x)
    old_rows=read_jsonl(ROOT/'data/highdim/records.jsonl'); probes=[]
    for label in (0,1):
        for x in [x for x in old_rows if x['role']=='train' and x['label']==label][:32]: probes.append({**x,'role':'probe_train','reused_probe':True})
    chosen=probes+chosen
    for i,x in enumerate(chosen):
        x['index']=i; x['prompt_sha256']=hashlib.sha256(x['prompt'].encode()).hexdigest()
        x['group_id']=hashlib.sha256(normalize_prompt(x['prompt']).encode()).hexdigest()
    validate_fresh_rows(chosen,old_ids,old_groups,{x['id'] for x in probes})
    directory=ROOT/'data/fusion'; directory.mkdir(exist_ok=True); records=directory/'records.jsonl'
    records.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in chosen),encoding='utf-8')
    manifest={'seed':seed,'samples':[{k:v for k,v in x.items() if k!='prompt'} for x in chosen],
        'counts':dict(Counter(x['role']+':'+str(x['label']) for x in chosen)), 'records_sha256':digest(records),
        'old_highdim_manifest_sha256':digest(ROOT/'configs/highdim/manifest.json'),
        'old_signal_manifest_sha256':digest(ROOT/'configs/signal-selection/manifest.json'),
        'extra_data_sha256':digest(ROOT/'logs/fusion-extra-data.json'),'prefix':prefix,'suffix':suffix,
        'duplicate_count':len(removed),'exclusions':dict(rejected),'no_truncation':True,'max_tokens':512,
        'training':'exact highdim1600 train /600 dev; probe_train64 do not replace full training source',
        'new_holdouts':'fresh unused official-train prospective sample plus small unused official-test partition',
        'toxicity_scope':'independent human toxicity proxy; source-safe calibration is not harmfulness-label equivalence',
        'csv_reader':'physical CSV stream preserves quoted newlines'}
    Path(__file__).with_name('manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    (ROOT/'logs/fusion-duplicates.json').write_text(json.dumps(removed,indent=2)+'\n')
    print('MANIFEST_LOCKED',json.dumps(manifest['counts']),flush=True)

if __name__=='__main__': main()
