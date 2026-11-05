"""Audit fixed references and build count-frozen confirmation before any new scoring."""
import ast,csv,json,sys,zipfile,hashlib
from collections import Counter
import numpy as np
from core import ROOT,digest,read,write,references

def main():
    from transformers import AutoTokenizer
    out=ROOT/'results/oc-reference';out.mkdir(parents=True,exist_ok=True)
    assert not (out/'manifest.json').exists()
    rules=json.loads((ROOT/'configs/oc-reference/rules.json').read_text())
    tok=AutoTokenizer.from_pretrained(ROOT/'models/llava-v1.6-vicuna-7b',use_fast=False,local_files_only=True)
    words=list(dict.fromkeys(rules['original']+rules['expand_add']))
    encoded={w:tok.encode(w,add_special_tokens=False) for w in words};weights=references(rules,encoded)
    source=ast.parse((ROOT/'source/HiddenDetect/code/test.py').read_text())
    author=next(ast.literal_eval(n.value) for n in source.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='refusal_lst' for t in n.targets))
    assert author==rules['original']
    audit={'rules_sha256':digest(ROOT/'configs/oc-reference/rules.json'),'author_list_exact':True,
        'tokens':[{'string':w,'full_encoding':encoded[w],'first_token_id':encoded[w][0],'first_token_decoded':tok.decode([encoded[w][0]])} for w in words],
        'reference_support_counts':[int((v>0).sum()) for v in weights],'coordinates':[{str(i):float(v[i]) for i in np.flatnonzero(v)} for v in weights]}
    write(out/'token-audit.json',audit)
    origin=ROOT.parents[1];roots=[origin,origin/'cache/pure-oc-worktree',ROOT]
    old={};excluded=set();hashes=set();sources={}
    for base in roots:
        for p in sorted((base/'data').glob('**/*.jsonl')):
            if p.name not in ('records.jsonl','probes.jsonl'):continue
            sources[str(p.relative_to(origin))]=digest(p)
            for r in read(p):
                if 'prompt' in r:
                    if r['id'] in old:assert old[r['id']]['prompt']==r['prompt']
                    old[r['id']]={'id':r['id'],'prompt':r['prompt'],'role':'excluded'}
        manifests=sorted((base/'configs').glob('**/*manifest*.json'))+list((base/'configs/benchmarks').glob('*.json'))+list((base/'results').glob('*/manifest.json'))
        for p in manifests:
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
    assert not hashes-{hashlib.sha256(r['prompt'].encode()).hexdigest() for r in old.values()},'Historical text missing'
    excluded.update(old)
    sys.path.insert(0,str(ROOT/'configs/signal-selection'))
    from experiment_core import normalize_prompt,prompt_label
    tree=ast.parse((ROOT/'configs/swarm/core.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='deduplicate')
    ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'frozen-dedup','exec'),ns)
    base=json.loads((ROOT/'configs/original-positions/manifest.json').read_text());prefix,suffix=base['prefix'],base['suffix']
    normalized={normalize_prompt(r['prompt']) for r in old.values()};reject=Counter();candidates=[]
    rawpath=ROOT/'data/signal-selection/raw/nvidia__Aegis-AI-Content-Safety-Dataset-2.0/train.json'
    sources[str(rawpath.relative_to(origin))]=digest(rawpath)
    for r in sorted(json.loads(rawpath.read_text()),key=lambda r:hashlib.sha256(('OC-RV1:'+str(r['id'])).encode()).hexdigest()):
        sid='aegis:train.json:'+str(r['id']);text=r['prompt']
        if sid in excluded:reject['historical_id']+=1;continue
        if not text or text.strip().upper()=='REDACTED' or '<image>' in text:reject['empty_redacted_image']+=1;continue
        if normalize_prompt(text) in normalized:reject['historical_normalized']+=1;continue
        ids=tok.encode(prefix+text+suffix)
        if len(ids)>512:reject['over512']+=1;continue
        candidates.append({'id':sid,'prompt':text,'label':prompt_label(r['prompt_label']),'role':'confirmation'})
    print('HISTORY',len(old),'EXCLUDED_IDS',len(excluded),'CANDIDATES',len(candidates),flush=True)
    kept,removed=ns['deduplicate'](list(old.values())+candidates);fresh=[];available={}
    for label in (0,1):
        pool=[r for r in kept if r['role']=='confirmation' and r['label']==label];available[str(label)]=len(pool)
        assert len(pool)>=2,('No meaningful fresh sample class',available)
        fresh+=pool[:1000]
    prompts={r['id']:r['prompt'] for r in read(ROOT/'data/positions/records.jsonl')}
    fixed=[]
    for r in base['samples']:
        if r['role'] in ('test_aegis_holdout','calibration_aegis'):
            assert r['id'] in prompts
            fixed.append({'id':r['id'],'prompt':prompts[r['id']],'label':r['label'],'role':'historical_C' if r['role']=='test_aegis_holdout' else 'calibration'})
    assert len(fixed)==1400
    rows=fixed+fresh
    for i,r in enumerate(rows):
        ids=tok.encode(prefix+r['prompt']+suffix);assert ids[-3:]==[9047,13566,29901]
        r.update(index=i,input_length=len(ids),input_ids_sha256=hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest(),
            prompt_sha256=hashlib.sha256(r['prompt'].encode()).hexdigest(),group_id=hashlib.sha256(normalize_prompt(r['prompt']).encode()).hexdigest())
    assert len({r['id'] for r in rows})==len(rows)
    path=ROOT/'data/oc-reference/records.jsonl';path.parent.mkdir(exist_ok=True)
    path.write_bytes(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode())
    manifest={'run_id':'OC-RV1','samples':[{k:v for k,v in r.items() if k!='prompt'} for r in rows],
        'prefix':prefix,'suffix':suffix,'records_sha256':digest(path),'historical_count':len(old),'excluded_ids':len(excluded),
        'historical_sources':sources,'available_fresh_per_class':available,'selected_per_class':{str(y):sum(r['label']==y for r in fresh) for y in (0,1)},
        'rejected':dict(reject),'count_frozen_before_scoring':True,'shortfall':'all remaining up to 1000/class; no previously observed inputs added',
        'scope':'same-source Aegis train holdout, not cross-domain; all prior layer-selection and OC inputs excluded'}
    write(out/'manifest.json',manifest);write(out/'dedup.json',removed)
    print('RV_MANIFEST_FROZEN',available,manifest['selected_per_class'],flush=True)
if __name__=='__main__':main()
