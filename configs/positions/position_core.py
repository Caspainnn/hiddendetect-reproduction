"""Actual-token readout contract; helpers reused without changing older experiments."""
import hashlib,json,sys,subprocess
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'configs/swarm'))
from core import deduplicate,bootstrap_metrics,p_auc
from fusion_core import digest,read_jsonl,read_csv_rows,validate_fresh_rows
from features import regions_from_offsets
from ridge import fit_ridge,ridge_scores
from seal import stream_hash
from experiment_core import normalize_prompt,prompt_label,conservative_threshold
NAMES=['prefix_control','user_first','user_q25','user_q50','user_q75','user_q90','user_last','template_last','random_user']

def token_indices(row):
    user=np.concatenate([np.array(x,dtype=int) for x in row['regions']]);assert len(user)>0 and user[0]>0
    assert np.array_equal(user,np.unique(user)) and user[-1]<row['input_length']
    picked=[int(user[int(q*(len(user)-1))]) for q in [0,.25,.5,.75,.9,1]]
    random=int(hashlib.sha256(('20261117:'+row['id']).encode()).hexdigest(),16)%len(user)
    prefix=int(row.get('prefix_anchor',user[0]-1));assert 0<=prefix<user[0]
    return [prefix,*picked,int(row['input_length']-1),int(user[random])]

def models():
    result=[{'name':n,'positions':[i],'eligible':i in range(1,8),'real_alternative':i in range(1,7)} for i,n in enumerate(NAMES)]
    result += [{'name':'template_plus_'+NAMES[i],'positions':[7,i],'eligible':True,'real_alternative':True} for i in range(1,7)]
    result += [{'name':'template_plus_random','positions':[7,8],'eligible':False,'real_alternative':False},
               {'name':'template_duplicated','positions':[7,7],'eligible':False,'real_alternative':False}]
    return result

def pack(block,positions):
    x=np.asarray(block[:,:,positions,:],np.float32)
    return x.reshape(len(x),-1)/np.sqrt(x.shape[1]*len(positions))

def matrix(data,indices,positions):
    x=np.stack([np.asarray(data[np.asarray(indices),:,p,:],np.float32) for p in positions],axis=2)
    return x.reshape(len(x),-1)/np.sqrt(32*len(positions))

def make_folds(y,groups):
    result=[]
    for repeat,seed in enumerate([20261115,20261116]):
        cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
        for fold,(fit,valid) in enumerate(cv.split(np.zeros(len(y)),y,groups)):
            assert not set(groups[fit])&set(groups[valid])
            result.append({'repeat':repeat,'fold':fold,'fit':fit,'valid':valid})
    return result

def upper_bounds(distributions,family=12):
    bounds={name:float(np.quantile(values,1-.05/family)) for name,values in distributions.items()}
    return {'family':family,'one_sided_quantile':1-.05/family,'margin':.005,'upper_delta_auroc':bounds,
        'supports_margin':bool(bounds and all(v<.005 for v in bounds.values())),
        'scope':'approximate paired bootstrap Bonferroni upper bounds, restricted fixed family only'}

def source_snapshot():
    own=['position_core.py','build_manifest.py','extract.py','seal_cache.py','train.py','evaluate.py','verify.py','diagnose_prefix.py']
    paths=['configs/positions/'+n for n in own]+['configs/swarm/core.py','configs/fusion/fusion_core.py','configs/highdim/features.py','configs/highdim/ridge.py','configs/highdim/seal.py','configs/signal-selection/experiment_core.py','configs/signal-selection/evaluate_frozen.py']
    return {n:digest(ROOT/n) for n in paths}

def sources_match(snapshot):
    for name,sha in snapshot.items():assert digest(ROOT/name)==sha,name

def check_actual_seal(path,expected):assert digest(path)==expected,'Actual cache seal differs from frozen seal'

def verify_cache(actual=True):
    directory=ROOT/'cache/positions';s=json.loads((directory/'seal.json').read_text());m=json.loads((directory/'metadata.json').read_text())
    rows=json.loads((ROOT/'configs/positions/manifest.json').read_text())['samples']
    assert s['manifest_sha256']==m['manifest_sha256']==digest(ROOT/'configs/positions/manifest.json')
    assert s['count']==m['count']==len(rows) and s['metadata_sha256']==digest(directory/'metadata.json')
    assert s['checkpoint_sha256']==digest(directory/'checkpoint.json') and json.loads((directory/'checkpoint.json').read_text())['next_index']==len(rows)
    if actual:assert stream_hash(directory/'P.npy')==s['P_sha256']
    return s

def verify_freeze(out,actual=False):
    f=json.loads((out/'freeze.json').read_text());sources_match(f['source_sha256'])
    for name,sha in f['dependencies_sha256'].items():assert digest(ROOT/name)==sha,name
    for name,key in [('cv.json','cv_sha256'),('oof.npz','oof_sha256'),('selection.json','selection_sha256')]:assert digest(out/name)==f[key]
    for c in f['candidates']:assert digest(out/c['file'])==c['sha256']
    pre=json.loads((out/'pre-evaluation-seal.json').read_text());assert pre['freeze_sha256']==digest(out/'freeze.json')
    assert pre['cache_seal_sha256']==digest(ROOT/'logs/positions-cache-seal.json')==f['cache_seal_sha256']
    report=json.loads((ROOT/'logs/positions-readout-identity.json').read_text())
    assert digest(ROOT/'logs/positions-readout-identity.json')==f['identity_sha256'] and report['status']=='passed'
    assert report['maximum_template_last_difference']<=1e-5
    if actual:
        check_actual_seal(ROOT/'cache/positions/seal.json',f['cache_seal_sha256'])
        verify_cache()
        expected={'results/positions/freeze.json':pre['freeze_sha256'],**f['source_sha256'],**f['dependencies_sha256']}
        for c in f['candidates']:expected['results/positions/'+c['file']]=c['sha256']
        for name,sha in expected.items():assert hashlib.sha256(subprocess.check_output(['git','show',pre['commit']+':'+name],cwd=ROOT)).hexdigest()==sha,name
        subprocess.run(['git','merge-base','--is-ancestor',pre['commit'],'HEAD'],cwd=ROOT,check=True)
    return f
