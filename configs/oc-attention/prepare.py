import json,subprocess,platform
import numpy as np
from core import *

def main():
    assert not (OUT/'source-lock.json').exists()
    h3=module('a2_h3',H3/'configs/oc-retraining/core.py');h3.verify_source()
    s1=module('a2_s1',S1/'configs/oc-svm/core.py');s1.verify_lock()
    a=json.loads((H3/'results/oc-retraining/manifest.json').read_text());b=json.loads((H2/'results/oc-confirmation/manifest.json').read_text())
    for key in ['id','group_id','prompt_sha256','input_ids_sha256']:assert not {r[key] for r in a['samples']}&{r[key] for r in b['samples']},key
    inputs=[H3/'cache/oc-retraining/features.npz',H2/'cache/oc-confirmation/features.npz',
        H3/'results/oc-retraining/source-lock.json',H3/'results/oc-retraining/manifest.json',H3/'results/oc-retraining/training.json',
        H3/'results/oc-retraining/retraining.json',H3/'cache/oc-retraining/frozen-scores.npz',H3/'results/oc-retraining/data-summary.json',
        H2/'results/oc-confirmation/manifest.json',H2/'results/oc-confirmation/extraction.json',
        S1/'results/oc-svm/evaluation.json',S1/'results/oc-svm/verification.json',S1/'cache/oc-svm/frozen-scores.npz',
        ROOT/'configs/oc-heads/core.py',ROOT/'configs/oc-heads/verify_statistics_independent.py',ROOT/'configs/original-positions/core.py']
    for base,part in [(H3,'oc-retraining'),(H2,'oc-confirmation')]:
        e=json.loads((base/f'results/{part}/extraction.json').read_text());assert digest(base/f'cache/{part}/features.npz')==e['features_sha256']
    with np.load(inputs[0]) as data:
        x=data['train_X'];y=data['train_y'];assert x.shape==(300,14) and np.isfinite(x).all() and (y==0).sum()==(y==1).sum()==150
        np.testing.assert_array_equal(y,np.array([r['label'] for r in a['samples']]))
    CACHE.mkdir(parents=True,exist_ok=True);np.savez(CACHE/'training-only.npz',X=x,y=y);inputs.append(CACHE/'training-only.npz')
    old=json.loads((H3/'results/oc-retraining/training.json').read_text())
    inputs.extend(H3/r['path'] for r in old['runs'] if r['id']==ATT and r['group']=='or_small' and r['seed']==SEEDS[0])
    acquisition=json.loads((ROOT/'cache/vendor/acquisition.json').read_text())
    for item in acquisition:assert digest(ROOT/'cache/vendor'/item['filename'])==item['sha256']
    vendor_files=list((ROOT/'cache/vendor/libauc/libauc').rglob('*.py'))+list((ROOT/'cache/vendor/rtdl/rtdl').rglob('*.py'))+[ROOT/'cache/vendor/acquisition.json']
    sources=list((ROOT/'configs/oc-attention').glob('*.py'))+[ROOT/'docs/oc-attention-protocol-2026-11-04.md']
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    for p in sources:assert p.read_bytes()==subprocess.check_output(['git','show',commit+':'+p.relative_to(ROOT).as_posix()]),p
    write(OUT/'source-lock.json',{'run_id':'OC-A2','commit':commit,'sources':{p.relative_to(ROOT).as_posix():digest(p) for p in sources},
        'inputs':{str(p):digest(p) for p in inputs},'vendor_files':{p.relative_to(ROOT).as_posix():digest(p) for p in vendor_files},
        'acquisition':acquisition,'names':NAMES,'learning_rates':LRS,'contrasts':CONTRASTS,'steps':STEPS,'seeds':SEEDS,
        'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__},
        'counts':{'train':[150,150],'calibration_safe':400,'test':[1000,500]},'historical_diagnostic':True,
        'h3_near_duplicate_audit_reused':True,'train_target_four_identity_fields_disjoint':True,'before_training':True})
    print('OC_A2_LOCKED',commit,flush=True)
if __name__=='__main__':main()
