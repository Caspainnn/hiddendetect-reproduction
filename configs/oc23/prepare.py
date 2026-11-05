"""Audit every historical cohort before deciding whether any new confirmation remains."""
import ast,csv,hashlib,json,sys,zipfile
from collections import Counter
import numpy as np
from core import ROOT,digest,read,write

def main():
    from transformers import AutoTokenizer
    out=ROOT/'results/oc23';out.mkdir(parents=True,exist_ok=True);assert not (out/'manifest.json').exists()
    tok=AutoTokenizer.from_pretrained(ROOT/'models/llava-v1.6-vicuna-7b',use_fast=False,local_files_only=True)
    source=ast.parse((ROOT/'source/HiddenDetect/code/test.py').read_text())
    words=next(ast.literal_eval(n.value) for n in source.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='refusal_lst' for t in n.targets))
    tokens=[{'string':w,'full_encoding':tok.encode(w,add_special_tokens=False)} for w in words]
    ids=sorted({r['full_encoding'][0] for r in tokens});assert len(ids)==20
    write(out/'token-audit.json',{'original_words':words,'tokens':tokens,'token_ids':ids,'support_count':20})
    origin=ROOT.parents[1];roots=[origin,origin/'cache/pure-oc-worktree',origin/'cache/oc-reference-worktree',ROOT]
    old={};excluded=set();hashes=set();sources={}
    for base in roots:
        for p in sorted((base/'data').glob('**/*.jsonl')):
            if p.name not in ('records.jsonl','probes.jsonl'):continue
            sources[str(p.relative_to(origin))]=digest(p)
            for r in read(p):
                if 'prompt' in r:
                    if r['id'] in old:assert old[r['id']]['prompt']==r['prompt']
                    old[r['id']]={'id':r['id'],'prompt':r['prompt'],'role':'excluded'}
        paths=list((base/'configs').glob('**/*manifest*.json'))+list((base/'configs/benchmarks').glob('*.json'))+list((base/'results').glob('*/manifest.json'))
        for p in paths:
            sources[str(p.relative_to(origin))]=digest(p)
            for r in json.loads(p.read_text()).get('samples',[]):
                if isinstance(r,dict):
                    if 'id' in r:excluded.add(str(r['id']))
                    if 'prompt_sha256' in r:hashes.add(r['prompt_sha256'])
    for r in csv.DictReader((ROOT/'data/xstest_prompts.csv').read_text(encoding='utf-8-sig').splitlines()):
        old['xstest:'+r['id']]={'id':'xstest:'+r['id'],'prompt':r['prompt'],'role':'excluded'}
        excluded.update([r['id'],'xstest:'+r['id'],'old-xstest:'+r['id']])
    with zipfile.ZipFile(ROOT/'source/HiddenDetect/data/few_shot.zip') as z:
        for i,r in enumerate(json.loads(z.read('few_shot/few_shot.json'))):old['old-cal:'+str(i)]={'id':'old-cal:'+str(i),'prompt':r['txt'],'role':'excluded'}
    assert not hashes-{hashlib.sha256(r['prompt'].encode()).hexdigest() for r in old.values()},'Historical private text unavailable'
    excluded.update(old)
    sys.path.insert(0,str(ROOT/'configs/signal-selection'))
    from experiment_core import normalize_prompt,prompt_label
    normalized={normalize_prompt(r['prompt']) for r in old.values()}
    tree=ast.parse((ROOT/'configs/swarm/core.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='deduplicate')
    ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'frozen-dedup','exec'),ns)
    base=json.loads((ROOT/'configs/original-positions/manifest.json').read_text());prefix,suffix=base['prefix'],base['suffix']
    reject=Counter();candidates=[]
    for split in ['train.json','validation.json','test.json']:
        p=ROOT/'data/signal-selection/raw/nvidia__Aegis-AI-Content-Safety-Dataset-2.0'/split;sources[str(p.relative_to(origin))]=digest(p)
        for r in sorted(json.loads(p.read_text()),key=lambda r:hashlib.sha256(('OC-SA1:'+split+':'+str(r['id'])).encode()).hexdigest()):
            sid='aegis:'+split+':'+str(r['id']);text=r['prompt']
            if sid in excluded:reject[split+':historical_id']+=1;continue
            if not text or text.strip().upper()=='REDACTED' or '<image>' in text:reject[split+':empty_redacted_image']+=1;continue
            if normalize_prompt(text) in normalized:reject[split+':historical_normalized']+=1;continue
            ids=tok.encode(prefix+text+suffix)
            if len(ids)>512:reject[split+':over512']+=1;continue
            candidates.append({'id':sid,'prompt':text,'label':prompt_label(r['prompt_label']),'role':'confirmation','source_split':split})
    print('SA_HISTORY',len(old),'EXCLUDED_IDS',len(excluded),'CANDIDATES',len(candidates),flush=True)
    kept,removed=ns['deduplicate'](list(old.values())+candidates)
    pools={y:[r for r in kept if r['role']=='confirmation' and r['label']==y] for y in (0,1)}
    available={str(y):len(pools[y]) for y in (0,1)};enabled=min(available.values())>=30
    fresh=[]
    if enabled:
        for y in (0,1):fresh+=sorted(pools[y],key=lambda r:hashlib.sha256(('OC-SA1-selected:'+r['id']).encode()).hexdigest())[:1000]
    prompts={r['id']:r['prompt'] for r in read(ROOT/'data/positions/records.jsonl')};fixed=[]
    for r in base['samples']:
        if r['role'] in ('test_aegis_holdout','calibration_aegis'):
            fixed.append({'id':r['id'],'prompt':prompts[r['id']],'label':r['label'],'role':'historical_C' if r['role']=='test_aegis_holdout' else 'calibration'})
    assert len(fixed)==1400
    for r in read(ROOT/'data/oc-reference/records.jsonl'):
        if r['role']=='confirmation':fixed.append({'id':r['id'],'prompt':r['prompt'],'label':r['label'],'role':'historical_RV'})
    assert len(fixed)==2482
    rows=fixed+fresh
    for i,r in enumerate(rows):
        ids=tok.encode(prefix+r['prompt']+suffix);assert len(ids)<=512 and ids[-3:]==[9047,13566,29901]
        r.update(index=i,input_length=len(ids),input_ids_sha256=hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest(),
                 prompt_sha256=hashlib.sha256(r['prompt'].encode()).hexdigest(),group_id=hashlib.sha256(normalize_prompt(r['prompt']).encode()).hexdigest())
    assert len({r['id'] for r in rows})==len(rows)
    p=ROOT/'data/oc23/records.jsonl';p.parent.mkdir(exist_ok=True);p.write_bytes(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode())
    m={'run_id':'OC-SA1','samples':[{k:v for k,v in r.items() if k!='prompt'} for r in rows],'prefix':prefix,'suffix':suffix,
       'records_sha256':digest(p),'historical_count':len(old),'excluded_ids':len(excluded),'historical_sources':sources,
       'available_fresh_per_class':available,'confirmation_enabled':enabled,'minimum_per_class':30,
       'selected_per_class':{str(y):sum(r['label']==y for r in fresh) for y in (0,1)},'source_pool':'existing frozen Aegis train+validation+test residuals, not cross-domain',
       'remaining_by_split_label':dict(Counter(r['source_split']+':'+str(r['label']) for y in (0,1) for r in pools[y])),
       'rejected':dict(reject),'count_frozen_before_scoring':True}
    write(out/'manifest.json',m)
    write(out/'dedup-summary.json',{'algorithm':'normalized exact + char3-5gram TFIDF65536 cosine>=.92 connected groups, historical priority',
         'removed_count':len(removed),'removed_by_role':dict(Counter(r['role'] for r in removed)),'private_audit_retained_on_server':True})
    write(ROOT/'cache/oc23-dedup-private.json',removed)
    print('SA_MANIFEST_FROZEN',available,'CONFIRMATION_ENABLED',enabled,'SELECTED',m['selected_per_class'],flush=True)
if __name__=='__main__':main()
