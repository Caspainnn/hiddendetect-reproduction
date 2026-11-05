import ast,csv,hashlib,json,os,subprocess,sys,zipfile
from collections import Counter
from pathlib import Path
import numpy as np
from core import ROOT,ORIGIN,H1,H2,OUT,HEADS,SEEDS,digest,read,write
def main():
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    assert not (OUT/'manifest.json').exists()
    old={};sources={};folders=set()
    listing=subprocess.check_output(['git','-C',str(ORIGIN),'worktree','list','--porcelain'],text=True)
    for line in listing.splitlines():
        if not line.startswith('worktree '):continue
        folder=(Path(line.removeprefix('worktree '))/'data').resolve()
        if folder in folders or not folder.exists():continue
        folders.add(folder)
        for p in sorted(folder.glob('**/*.jsonl')):
            if p.name not in ['records.jsonl','probes.jsonl']:continue
            sources[str(p)]=digest(p)
            for row in read(p):
                if 'prompt' not in row:continue
                if row['id'] in old:assert old[row['id']]['prompt']==row['prompt']
                old[row['id']]={'id':row['id'],'prompt':row['prompt'],'role':'excluded'}
    with (ROOT/'data/xstest_prompts.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):old['xstest:'+row['id']]={'id':'xstest:'+row['id'],'prompt':row['prompt'],'role':'excluded'}
    with zipfile.ZipFile(ROOT/'source/HiddenDetect/data/few_shot.zip') as z:
        for i,row in enumerate(json.loads(z.read('few_shot/few_shot.json'))):old['fewshot:'+str(i)]={'id':'fewshot:'+str(i),'prompt':row['txt'],'role':'excluded'}
    sources[str(ROOT/'data/xstest_prompts.csv')]=digest(ROOT/'data/xstest_prompts.csv');sources[str(ROOT/'source/HiddenDetect/data/few_shot.zip')]=digest(ROOT/'source/HiddenDetect/data/few_shot.zip')
    sys.path.insert(0,str(ROOT/'configs/signal-selection'))
    from experiment_core import normalize_prompt
    normal={normalize_prompt(row['prompt']) for row in old.values()}
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'models/llava-v1.6-vicuna-7b',use_fast=False,local_files_only=True)
    h2manifest=json.loads((H2/'results/oc-confirmation/manifest.json').read_text());prefix,suffix=h2manifest['prefix'],h2manifest['suffix'];revision=h2manifest['source_revision']
    acquisition=json.loads((H2/'results/oc-confirmation/acquisition.json').read_text());raw=ROOT/'data/oc-confirmation/raw';candidates=[];rejected=Counter()
    for item in acquisition:assert digest(raw/item['file'])==item['sha256']
    for filename,label,cap in [('or-bench-80k.csv',0,5000),('or-bench-toxic.csv',1,None)]:
        with (raw/filename).open(encoding='utf-8-sig',newline='') as f:items=list(csv.DictReader(f))
        ordered=sorted(enumerate(items),key=lambda pair:hashlib.sha256(('OC-H2:'+filename+':'+str(pair[0])).encode()).hexdigest())
        for index,row in (ordered[:cap] if cap else ordered):
            prompt=row['prompt'];sid=f'orbench:{revision}:{filename}:{index}'
            if sid in old or normalize_prompt(prompt) in normal:rejected['historical_exact']+=1;continue
            if not prompt or '<image>' in prompt or prompt.strip().upper()=='REDACTED':rejected['invalid']+=1;continue
            ids=tok.encode(prefix+prompt+suffix)
            if len(ids)>512:rejected['over512']+=1;continue
            candidates.append({'id':sid,'prompt':prompt,'label':label,'role':'candidate','category':row['category'],
                'input_length':len(ids),'input_ids_sha256':hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest(),
                'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'group_id':hashlib.sha256(normalize_prompt(prompt).encode()).hexdigest()})
    tree=ast.parse((ROOT/'configs/swarm/core.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='deduplicate');ns={'np':np}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'h3_dedup','exec'),ns)
    print('OC_H3_PRE_DEDUP',len(old),len(candidates),flush=True)
    kept,removed=ns['deduplicate'](sorted(old.values(),key=lambda r:r['id'])+candidates)
    pools={y:sorted([row for row in kept if row['role']=='candidate' and row['label']==y],key=lambda row:hashlib.sha256(('OC-H3-new:'+row['id']).encode()).hexdigest()) for y in (0,1)}
    n=min(150,len(pools[0]),len(pools[1]));assert n>=30,('Insufficient source training',n)
    rows=[]
    for y in (0,1):
        for row in pools[y][:n]:row['role']='train';rows.append(row)
    trainmanifest=json.loads((H1/'results/oc-heads/manifest.json').read_text());oldrows=[row for row in trainmanifest['samples'] if row['cohort']=='train']
    old_indices=[]
    for y in (0,1):old_indices+=sorted([i for i,row in enumerate(oldrows) if row['label']==y],key=lambda i:hashlib.sha256(('OC-H3-old:'+oldrows[i]['id']).encode()).hexdigest())[:n]
    target=read(ROOT/'data/oc-confirmation/records.jsonl')
    for key in ['id','group_id','prompt_sha256','input_ids_sha256']:
        assert not {row[key] for row in rows}&{row[key] for row in target},key
        assert not {oldrows[i][key] for i in old_indices}&{row[key] for row in target},key
    records=ROOT/'data/oc-retraining/records.jsonl';records.parent.mkdir(exist_ok=True);records.write_bytes(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows).encode())
    write(ROOT/'cache/oc-h3-dedup-private.json',removed)
    summary={'history_count':len(old),'candidate_count':len(candidates),'available':{str(y):len(pools[y]) for y in (0,1)},'n_per_class':n,'new_training_count':2*n,'old_matched_count':2*n,
        'rejected':dict(rejected),'removed_count':len(removed),'historical_worktree_data_roots':len(folders),'target_calibration_count':400,'target_test_counts':[1000,500],'no_target_samples_in_training':True}
    write(OUT/'data-summary.json',summary)
    manifest={'run_id':'OC-H3','samples':[{k:v for k,v in row.items() if k!='prompt'} for row in rows],'prefix':prefix,'suffix':suffix,'source_revision':revision,
        'records_sha256':digest(records),'n_per_class':n,'old_indices':old_indices,'old_train_ids':[oldrows[i]['id'] for i in old_indices],
        'historical_sources':sources,'primary_seed':SEEDS[0],'seeds':SEEDS,'h2_confirmation_sha256':digest(H2/'results/oc-confirmation/confirmation.json')}
    write(OUT/'manifest.json',manifest)
    h1files=['results/oc-heads/selection.json','results/oc-heads/manifest.json','cache/oc-heads/features.npz']
    selection=json.loads((H1/h1files[0]).read_text())
    for name in HEADS:h1files += [entry['path'] for entry in selection['selected'][name]['models']]
    h2files=['results/oc-confirmation/confirmation.json','results/oc-confirmation/manifest.json','cache/oc-confirmation/features.npz','cache/oc-confirmation/frozen-scores.npz']
    paths=list((ROOT/'configs/oc-retraining').glob('*.py'))+[ROOT/'configs/oc-heads/core.py',ROOT/'configs/oc-heads/verify_statistics_independent.py',ROOT/'configs/original-positions/core.py',ROOT/'configs/swarm/core.py',ROOT/'configs/signal-selection/experiment_core.py',ROOT/'docs/oc-heads-retraining-protocol-2026-11-04.md']
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    for p in paths:assert p.read_bytes()==subprocess.check_output(['git','show',commit+':'+str(p.relative_to(ROOT))]),p
    write(OUT/'source-lock.json',{'source_commit':commit,'sha256':{str(p.relative_to(ROOT)):digest(p) for p in paths},'manifest_sha256':digest(OUT/'manifest.json'),
        'h1_files':{name:digest(H1/name) for name in h1files},'h2_files':{name:digest(H2/name) for name in h2files},'before_new_F_or_head_training':True})
    print('OC_H3_TRAINING_FROZEN',summary,flush=True)
if __name__=='__main__':main()
