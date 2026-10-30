"""Count-only unused confirmation; enumerate actual history, never hard-code exclusions."""
import csv,hashlib,json,sys,zipfile
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'configs/positions'))
from position_core import deduplicate,normalize_prompt,prompt_label,digest,read_jsonl

def main():
    from transformers import AutoTokenizer
    target=ROOT/'configs/original-template-confirm/manifest.json';assert not target.exists()
    sources={};old=[];excluded_ids=set();manifest_shas=set()
    for path in sorted((ROOT/'data').glob('**/*.jsonl')):
        if path.name not in ('records.jsonl','probes.jsonl') or 'original-template-confirm' in path.parts:continue
        sources[str(path.relative_to(ROOT))]=digest(path)
        for x in read_jsonl(path):
            if x.get('prompt') is not None:old.append({'id':x['id'],'prompt':x['prompt'],'role':'excluded'})
    for path in sorted((ROOT/'configs').glob('**/*manifest*.json'))+list((ROOT/'configs/benchmarks').glob('*.json')):
        sources[str(path.relative_to(ROOT))]=digest(path);m=json.loads(path.read_text())
        for x in m.get('samples',[]):
            if isinstance(x,dict):
                if 'id' in x:excluded_ids.add(str(x['id']))
                if 'prompt_sha256' in x:manifest_shas.add(x['prompt_sha256'])
    csvpath=ROOT/'data/xstest_prompts.csv';sources[str(csvpath.relative_to(ROOT))]=digest(csvpath)
    for x in csv.DictReader(csvpath.read_text(encoding='utf-8-sig').splitlines()):
        old.append({'id':'xstest:'+x['id'],'prompt':x['prompt'],'role':'excluded'})
        excluded_ids.update([x['id'],'xstest:'+x['id'],'old-xstest:'+x['id']])
    zpath=ROOT/'source/HiddenDetect/data/few_shot.zip';sources[str(zpath.relative_to(ROOT))]=digest(zpath)
    with zipfile.ZipFile(zpath) as z:
        for i,x in enumerate(json.loads(z.read('few_shot/few_shot.json'))):old.append({'id':'old-cal:'+str(i),'prompt':x['txt'],'role':'excluded'})
    unique={}
    for x in old:
        if x['id'] in unique:assert unique[x['id']]['prompt']==x['prompt']
        unique[x['id']]=x
    old=list(unique.values());excluded_ids.update(unique)
    actual_shas={hashlib.sha256(x['prompt'].encode()).hexdigest() for x in old}
    missing=manifest_shas-actual_shas
    assert not missing,('Historical manifest text unavailable for near-duplicate exclusion',len(missing))
    normalized={normalize_prompt(x['prompt']) for x in old}
    base=json.loads((ROOT/'configs/original-positions/manifest.json').read_text());prefix,suffix=base['prefix'],base['suffix']
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/llava-v1.6-vicuna-7b',use_fast=False,local_files_only=True)
    fast=AutoTokenizer.from_pretrained(ROOT/'models/llava-v1.6-vicuna-7b',use_fast=True,local_files_only=True)
    rawpath=ROOT/'data/signal-selection/raw/nvidia__Aegis-AI-Content-Safety-Dataset-2.0/train.json'
    raw=json.loads(rawpath.read_text());sources[str(rawpath.relative_to(ROOT))]=digest(rawpath)
    candidates=[];reject=Counter()
    for x in sorted(raw,key=lambda x:hashlib.sha256(('ORP-T2-20261030:'+str(x['id'])).encode()).hexdigest()):
        sid='aegis:train.json:'+str(x['id']);text=x['prompt']
        if sid in excluded_ids:reject['historical_id']+=1;continue
        if not text or text.strip().upper()=='REDACTED' or '<image>' in text:reject['empty_redacted_image']+=1;continue
        if normalize_prompt(text) in normalized:reject['historical_normalized_text']+=1;continue
        ids=tokenizer.encode(prefix+text+suffix)
        if len(ids)>512:reject['over512']+=1;continue
        assert ids==fast(prefix+text+suffix)['input_ids'] and ids[-3:]==[9047,13566,29901]
        candidates.append({'id':sid,'prompt':text,'label':prompt_label(x['prompt_label']),'role':'confirmation_aegis',
            'input_length':len(ids),'input_ids_sha256':hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest()})
    print('ACTUAL_HISTORY',len(old),'EXCLUDED_IDS',len(excluded_ids),'CANDIDATES',len(candidates),flush=True)
    kept,removed=deduplicate(old+candidates)
    chosen=[];available={}
    for label in (0,1):
        pool=[x for x in kept if x['role']=='confirmation_aegis' and x['label']==label];available[label]=len(pool)
        assert len(pool)>=500,('Insufficient fresh counts',available)
        chosen+=pool[:500]
    for i,row in enumerate(chosen):
        row.update(index=i,prompt_sha256=hashlib.sha256(row['prompt'].encode()).hexdigest(),
                   group_id=hashlib.sha256(normalize_prompt(row['prompt']).encode()).hexdigest())
        assert row['id'] not in excluded_ids and normalize_prompt(row['prompt']) not in normalized
    assert len({x['id'] for x in chosen})==len({x['group_id'] for x in chosen})==1000
    directory=ROOT/'data/original-template-confirm';directory.mkdir(exist_ok=True);rp=directory/'records.jsonl'
    rp.write_bytes(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in chosen).encode())
    manifest={'run_id':'ORP-T2','samples':[{k:v for k,v in x.items() if k!='prompt'} for x in chosen],
        'records_sha256':digest(rp),'prefix':prefix,'suffix':suffix,'historical_input_rows':len(old),'historical_excluded_ids':len(excluded_ids),
        'historical_source_sha256':sources,'historical_manifest_prompt_hashes_checked':len(manifest_shas),
        'available_after_dedup':available,'rejected':dict(reject),'selected_per_class':500,
        'scope':'previously unused same-source Aegis train confirmation, not new domain; becomes historical after scoring'}
    target.write_text(json.dumps(manifest,indent=2)+'\n')
    (ROOT/'logs/original-template-confirm-duplicates.json').write_text(json.dumps(removed,indent=2)+'\n')
    print('FRESH_MANIFEST_LOCKED',len(chosen),available,flush=True)

if __name__=='__main__':main()
