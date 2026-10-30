"""Remote-only exact author-rule extraction; small F checkpoints, never hidden caches."""
import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from core import ROOT,IDS,NAMES,digest,read_jsonl,positions,integral,project_readouts

def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--identity-only',action='store_true');args=p.parse_args()
    os.chdir(ROOT);os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    directory=ROOT/'cache/original-positions';directory.mkdir(exist_ok=True)
    lock=(directory/'worker.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import torch
    import torch.nn.functional as nnf
    import transformers
    from transformers import AutoTokenizer
    from llava.model.builder import load_pretrained_model
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX
    torch.manual_seed(0);np.random.seed(0)
    frozen=json.loads((ROOT/'configs/original-positions/run-lock.json').read_text())
    subprocess.run(['git','merge-base','--is-ancestor',frozen['source_commit'],'HEAD'],check=True)
    for name,sha in frozen['sha256'].items():
        assert digest(ROOT/name)==sha,name
        assert hashlib.sha256(subprocess.check_output(['git','show',frozen['source_commit']+':'+name])).hexdigest()==sha,name
    artifacts=json.loads((ROOT/'configs/artifacts-lock.json').read_text())
    for item in artifacts['files']:
        path=ROOT/item['target'];assert path.stat().st_size==item['bytes'],item['target']
        checksum=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024**2),b''):checksum.update(chunk)
        assert checksum.hexdigest()==item['sha256'],item['target']
    print('LOCKED_MODEL_HASHES_PASS',flush=True)
    author=ROOT/'source/HiddenDetect'
    assert subprocess.check_output(['git','-C',str(author),'rev-parse','HEAD'],text=True).strip()=='2fd161b1fc47c99b36fe0a299ec64ff0b816f0bb'
    assert not subprocess.check_output(['git','-C',str(author),'status','--porcelain','--untracked-files=no'],text=True).strip()
    spec=importlib.util.spec_from_file_location('original_author',author/'code/test.py')
    original=importlib.util.module_from_spec(spec);sys.path.insert(0,str(author/'code'));spec.loader.exec_module(original)
    tokenizer,model,_,context=load_pretrained_model(str(ROOT/'models/llava-v1.6-vicuna-7b'),None,'llava-v1.6-vicuna-7b',device_map='auto',torch_dtype=torch.float16)
    assert not model.training and not any(x.is_meta for x in model.parameters())
    assert model.config.num_hidden_layers==32 and model.config.hidden_size==4096 and model.config.vocab_size==32000
    assert next(model.parameters()).dtype==torch.float16
    fast=AutoTokenizer.from_pretrained(ROOT/'models/llava-v1.6-vicuna-7b',use_fast=True,local_files_only=True)
    base=json.loads((ROOT/'configs/highdim/manifest.json').read_text());prefix,suffix=base['prefix'],base['suffix']
    benchmark=ROOT/'configs/benchmarks/v1.json';bench=json.loads(benchmark.read_text())
    for name,sha in bench['manifests_sha256'].items():assert digest(ROOT/'configs'/name/'manifest.json')==sha
    pm=json.loads((ROOT/'configs/positions/manifest.json').read_text())
    records=ROOT/'data/positions/records.jsonl';assert digest(records)==pm['records_sha256']
    source_rows={x['id']:x for x in read_jsonl(records)}
    selected=[x for x in bench['samples'] if x['role']=='test_aegis_holdout' or (x['role']=='calibration_aegis' and x['label']==0)]
    assert len(selected)==1400
    rows=[]
    xmanifest=json.loads((ROOT/'configs/xstest450-manifest.json').read_text())
    csvpath=ROOT/'data/xstest_prompts.csv';assert digest(csvpath)==xmanifest['csv_sha256']
    csvrows={x['id']:x for x in csv.DictReader(csvpath.read_text(encoding='utf-8-sig').splitlines())}
    for x in xmanifest['samples']:
        prompt=csvrows[x['id']]['prompt'];assert hashlib.sha256(prompt.encode()).hexdigest()==x['prompt_sha256']
        rows.append({'id':'xstest:'+x['id'],'original_id':x['id'],'prompt':prompt,'label':x['label'],'role':'identity_xstest'})
    for item in selected:
        row=source_rows[item['id']];assert all(row[k]==item[k] for k in ('id','label','role','group_id'))
        rows.append({k:row[k] for k in ('id','prompt','label','role','group_id')})
    assert len({x['id'] for x in rows})==len(rows)
    normgroups={}
    sys.path.insert(0,str(ROOT/'configs/signal-selection'))
    from experiment_core import normalize_prompt
    for row in rows:
        prompt=prefix+row['prompt']+suffix
        conv=conv_templates['llava_v1'].copy();conv.append_message(conv.roles[0],row['prompt']);conv.append_message(conv.roles[1],None)
        assert prompt==conv.get_prompt()
        ids=tokenizer_image_token(prompt,tokenizer,IMAGE_TOKEN_INDEX,return_tensors='pt')
        encoding=fast(prompt,return_offsets_mapping=True)
        assert ids.tolist()==encoding['input_ids']==tokenizer.encode(prompt)
        offsets=encoding['offset_mapping'];a,b=len(prefix),len(prefix)+len(row['prompt'])
        user=[i for i,(s,e) in enumerate(offsets) if e>s and e>a and s<b]
        row.update(input_length=len(ids),input_ids_sha256=hashlib.sha256(np.array(ids,dtype=np.int64).tobytes()).hexdigest(),
                   prompt_sha256=hashlib.sha256(row['prompt'].encode()).hexdigest(),readout_indices=positions(user,len(ids)),
                   user_token_count=len(user),readout_token_ids=[int(ids[i]) for i in positions(user,len(ids))],
                   suffix_last3_token_ids=ids[-3:].tolist(),suffix_last3_tokens=tokenizer.convert_ids_to_tokens(ids[-3:].tolist()),
                   suffix_last3_offsets=offsets[-3:])
        assert all(s>=b for s,e in offsets[-3:]),'Template last3 overlap user boundary'
        key=hashlib.sha256(normalize_prompt(row['prompt']).encode()).hexdigest()
        assert row.get('group_id',key)==key
        normgroups.setdefault(row['role'],set()).add(key)
    assert not normgroups['calibration_aegis']&normgroups['test_aegis_holdout']
    refusal=[tokenizer.encode(x,add_special_tokens=False) for x in original.refusal_lst]
    reference=torch.zeros(32000,device=model.device,dtype=torch.float32)
    reference[[x[0] for x in refusal]]=1
    sources={str(x.relative_to(ROOT)):digest(x) for x in Path(__file__).parent.glob('*.py')}
    manifest={'run_id':'ORP-U1','directions':IDS,'names':NAMES,'primary_interval':[16,29],'sensitivity_interval':[17,29],
              'benchmark_sha256':digest(benchmark),'records_sha256':digest(records),'author_test_sha256':digest(author/'code/test.py'),
              'source_sha256':sources,'run_lock_sha256':digest(ROOT/'configs/original-positions/run-lock.json'),
              'prefix':prefix,'suffix':suffix,'refusal_encodings':refusal,
              'torch':torch.__version__,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(),
              'allow_tf32':torch.backends.cuda.matmul.allow_tf32,'samples':[{k:v for k,v in x.items() if k!='prompt'} for x in rows]}
    mp=ROOT/'configs/original-positions/manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest,'Manifest or code drift; use separate run, never overwrite'
    else:write_json(mp,manifest)
    key=digest(mp);expected={x['id']:x for x in read_jsonl(ROOT/'results/xstest450-default-s16e29.jsonl')}
    identity=[];began=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for i,row in enumerate(rows):
        if args.identity_only and row['role']!='identity_xstest':break
        path=directory/(hashlib.sha256(row['id'].encode()).hexdigest()+'.json')
        if path.exists():
            result=json.loads(path.read_text());assert result['manifest_sha256']==key and result['id']==row['id']
        else:
            ids=tokenizer_image_token(prefix+row['prompt']+suffix,tokenizer,IMAGE_TOKEN_INDEX,return_tensors='pt').unsqueeze(0).cuda()
            assert ids.shape[-1]==row['input_length']
            with torch.no_grad():
                output=model(ids,images=None,image_sizes=None,output_hidden_states=True)
                f=[]
                for state in output.hidden_states[1:]:
                    logits=project_readouts(state,row['readout_indices'],model.model.norm,model.lm_head)
                    f.append([float(nnf.cosine_similarity(logits[:,j,:],reference).item()) for j in range(7)])
                    del logits
                del output
            result={**{k:v for k,v in row.items() if k!='prompt'},'manifest_sha256':key,'F':f,
                    'score_default':integral(f).tolist(),'score_fewshot':integral(f,17,29).tolist()}
            write_json(path,result)
        if row['role']=='identity_xstest':
            old=expected[row['original_id']]
            assert row['input_ids_sha256']==old['input_ids_sha256']
            delta=float(np.max(np.abs(np.asarray(result['F'])[:,-1]-old['F'])))
            identity.append(delta)
            assert delta==0.,('Author identity mismatch; stop before C evaluation',row['id'],delta)
            assert result['score_default'][-1]==old['score'],'Integral differs from original np.trapz'
        if i==449:
            write_json(ROOT/'logs/original-positions-identity.json',{'status':'passed','rows':450,'maximum_F_difference':max(identity),
                       'manifest_sha256':key,'reference_sha256':digest(ROOT/'results/xstest450-default-s16e29.jsonl')})
            print('AUTHOR_IDENTITY_450_PASS',flush=True)
        if (i+1)%25==0:print('SCORED',i+1,'/',len(rows),'SECONDS',round(time.monotonic()-began,1),flush=True)
    if args.identity_only:return
    target=ROOT/'results/original-positions';target.mkdir(exist_ok=True)
    tmp=target/'scores.tmp'
    with tmp.open('w') as out:
        for row in rows:
            result=json.loads((directory/(hashlib.sha256(row['id'].encode()).hexdigest()+'.json')).read_text())
            assert result['manifest_sha256']==key
            out.write(json.dumps(result)+'\n')
    tmp.replace(target/'scores.jsonl')
    write_json(target/'extraction.json',{'status':'completed','rows':len(rows),'manifest_sha256':key,
               'scores_sha256':digest(target/'scores.jsonl'),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
               'wall_seconds_this_invocation':time.monotonic()-began,'generation_executed':False,'hidden_cache_reused':False})
    print('ORP_U1_EXTRACTION_COMPLETE',flush=True)

if __name__=='__main__':main()
