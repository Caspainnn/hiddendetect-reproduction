import ast,csv,hashlib,json,os,subprocess,sys,zipfile
from collections import Counter
from pathlib import Path
import numpy as np
from core import ROOT,ORIGIN,H1,OUT,NAMES,SEEDS,digest,read,write
def main():
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    assert not (OUT/'manifest.json').exists()
    acquisition=json.loads((OUT/'acquisition.json').read_text());raw=ROOT/'data/oc-confirmation/raw'
    for item in acquisition:assert digest(raw/item['file'])==item['sha256']
    revision=acquisition[0]['revision']
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'models/llava-v1.6-vicuna-7b',use_fast=False,local_files_only=True)
    template=json.loads((ROOT/'results/oc23/manifest.json').read_text());prefix,suffix=template['prefix'],template['suffix']
    sys.path.insert(0,str(ROOT/'configs/signal-selection'))
    from experiment_core import normalize_prompt
    old={};sources={};data_roots=set()
    worktrees=subprocess.check_output(['git','-C',str(ORIGIN),'worktree','list','--porcelain'],text=True)
    bases=[ORIGIN]+[Path(line.removeprefix('worktree ')) for line in worktrees.splitlines() if line.startswith('worktree ')]
    for base in bases:
        folder=(base/'data').resolve()
        if folder in data_roots or not folder.exists():continue
        data_roots.add(folder)
        for p in sorted(folder.glob('**/*.jsonl')):
            if p.name not in ['records.jsonl','probes.jsonl']:continue
            sources[str(p)]=digest(p)
            for r in read(p):
                if 'prompt' not in r:continue
                if r['id'] in old:assert old[r['id']]['prompt']==r['prompt']
                old[r['id']]={'id':r['id'],'prompt':r['prompt'],'role':'excluded'}
    sources[str(ROOT/'data/xstest_prompts.csv')]=digest(ROOT/'data/xstest_prompts.csv')
    sources[str(ROOT/'source/HiddenDetect/data/few_shot.zip')]=digest(ROOT/'source/HiddenDetect/data/few_shot.zip')
    with (ROOT/'data/xstest_prompts.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):old['xstest:'+row['id']]={'id':'xstest:'+row['id'],'prompt':row['prompt'],'role':'excluded'}
    with zipfile.ZipFile(ROOT/'source/HiddenDetect/data/few_shot.zip') as z:
        for i,r in enumerate(json.loads(z.read('few_shot/few_shot.json'))):old['old-cal:'+str(i)]={'id':'old-cal:'+str(i),'prompt':r['txt'],'role':'excluded'}
    normalized={normalize_prompt(r['prompt']) for r in old.values()}
    rejected=Counter();candidates=[];raw_counts={}
    for filename,label,cap in [('or-bench-80k.csv',0,5000),('or-bench-toxic.csv',1,None)]:
        with (raw/filename).open(encoding='utf-8-sig',newline='') as f:items=list(csv.DictReader(f))
        raw_counts[filename]=len(items)
        ordered=sorted(enumerate(items),key=lambda item:hashlib.sha256(('OC-H2:'+filename+':'+str(item[0])).encode()).hexdigest())
        for index,row in ordered[:cap] if cap else ordered:
            prompt=row['prompt'];sid=f'orbench:{revision}:{filename}:{index}'
            if not prompt or '<image>' in prompt or prompt.strip().upper()=='REDACTED':rejected['empty_redacted_image']+=1;continue
            if sid in old or normalize_prompt(prompt) in normalized:rejected['historical_exact']+=1;continue
            tokens=tok.encode(prefix+prompt+suffix)
            if len(tokens)>512:rejected['over512']+=1;continue
            assert tokens[-3:]==[9047,13566,29901]
            candidates.append({'id':sid,'prompt':prompt,'label':label,'role':'candidate','category':row['category'],'source_file':filename,
                'input_length':len(tokens),'input_ids_sha256':hashlib.sha256(np.asarray(tokens,dtype=np.int64).tobytes()).hexdigest(),
                'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'group_id':hashlib.sha256(normalize_prompt(prompt).encode()).hexdigest()})
    print('OC_H2_PRE_DEDUP',len(old),len(candidates),raw_counts,flush=True)
    tree=ast.parse((ROOT/'configs/swarm/core.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='deduplicate')
    ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'frozen-history-dedup','exec'),ns)
    kept,removed=ns['deduplicate'](sorted(old.values(),key=lambda r:r['id'])+candidates)
    pools={y:sorted([r for r in kept if r['role']=='candidate' and r['label']==y],key=lambda r:hashlib.sha256(('OC-H2-select:'+r['id']).encode()).hexdigest()) for y in (0,1)}
    available={str(y):len(pools[y]) for y in (0,1)}
    write(OUT/'dedup-summary.json',{'raw_counts':raw_counts,'historical_inputs':len(old),'candidate_count':len(candidates),'available':available,
        'removed_count':len(removed),'removed_by_role':dict(Counter(r['role'] for r in removed)),'rejected':dict(rejected),
        'method':'NFKC/casefold word exact, then char3-5 TFIDF65536 cosine>=.92 connected groups, history first','private_audit_retained_on_host':True})
    write(ROOT/'cache/oc-h2-dedup-private.json',removed)
    assert available['0']>=430 and available['1']>=30,('Insufficient unseen data',available)
    chosen=[]
    for r in pools[0][:400]:r['role']='cal';chosen.append(r)
    for y,values in [(0,pools[0][400:1400]),(1,pools[1][:500])]:
        for r in values:r['role']='test';chosen.append(r)
    assert len({r['id'] for r in chosen})==len(chosen) and len({r['group_id'] for r in chosen})==len(chosen)
    records=ROOT/'data/oc-confirmation/records.jsonl';records.write_bytes(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in chosen).encode())
    selection=json.loads((H1/'results/oc-heads/selection.json').read_text());pre=json.loads((H1/'results/oc-heads/pre-evaluation-seal.json').read_text())
    assert digest(H1/'results/oc-heads/selection.json')==pre['selection_sha256']
    models={}
    for name in NAMES[1:]:
        for entry in selection['selected'][name]['models']:
            assert entry['seed'] in SEEDS and digest(H1/entry['path'])==entry['sha256']
            assert pre['model_sha256'][entry['path']]==entry['sha256'];models[entry['path']]=entry['sha256']
    cuts={}
    for seed in SEEDS:
        cut=json.loads((H1/f'results/oc-heads/thresholds-{seed}.json').read_text())
        cuts[str(seed)]=[cut['values'][cut['names'].index(name)] for name in NAMES]
    manifest={'run_id':'OC-H2','samples':[{k:v for k,v in r.items() if k!='prompt'} for r in chosen],'prefix':prefix,'suffix':suffix,
        'records_sha256':digest(records),'acquisition_sha256':digest(OUT/'acquisition.json'),'source_revision':revision,
        'target_counts':{'cal_normal':400,'test_normal':1000,'test_danger':500},'actual_counts':dict(Counter(r['role']+':'+str(r['label']) for r in chosen)),
        'category_counts':dict(Counter(r['role']+':'+str(r['label'])+':'+r['category'] for r in chosen)),
        'available':available,'historical_sources':sources,'history_data_roots':[str(p) for p in sorted(data_roots)],'history_worktree_count':len(bases)-1,'model_selection_sha256':digest(H1/'results/oc-heads/selection.json'),
        'old_thresholds':cuts,'primary_seed':SEEDS[0],'seeds':SEEDS,'new_source_unscored_at_freeze':True}
    write(OUT/'manifest.json',manifest)
    paths=list((ROOT/'configs/oc-confirmation').glob('*.py'))+[ROOT/'configs/oc-heads/core.py',ROOT/'configs/oc-heads/verify_statistics_independent.py',ROOT/'configs/original-positions/core.py',ROOT/'configs/swarm/core.py',ROOT/'configs/signal-selection/experiment_core.py',ROOT/'docs/oc-heads-confirmation-protocol-2026-11-04.md']
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    for p in paths:assert p.read_bytes()==subprocess.check_output(['git','show',commit+':'+str(p.relative_to(ROOT))]),p
    old_results=['results/oc-heads/selection.json','results/oc-heads/pre-evaluation-seal.json','results/oc-heads/evaluation-complete.json']+[f'results/oc-heads/thresholds-{seed}.json' for seed in SEEDS]
    write(OUT/'source-lock.json',{'source_commit':commit,'sha256':{str(p.relative_to(ROOT)):digest(p) for p in paths},'manifest_sha256':digest(OUT/'manifest.json'),'models_sha256':models,
        'h1_result_sha256':{name:digest(H1/name) for name in old_results},'first_new_sample_not_scored':True})
    print('OC_H2_MANIFEST_FROZEN',manifest['actual_counts'],available,flush=True)
if __name__=='__main__':main()
