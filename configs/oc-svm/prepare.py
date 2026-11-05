import importlib.util,json,subprocess
from pathlib import Path
import numpy as np
from core import ROOT,OUT,CACHE,H2,H3,digest,write,GRIDS,PRIMARY

def main():
    assert not (OUT/'source-lock.json').exists(),'Existing execution lock must not be overwritten'
    spec=importlib.util.spec_from_file_location('h3_core',H3/'configs/oc-retraining/core.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.verify_source()
    manifests={key:json.loads((base/path).read_text()) for key,base,path in [('h3',H3,'results/oc-retraining/manifest.json'),('h2',H2,'results/oc-confirmation/manifest.json')]}
    h3=manifests['h3'];h2=manifests['h2']
    for key in ['id','group_id','prompt_sha256','input_ids_sha256']:
        assert not {r[key] for r in h3['samples']}&{r[key] for r in h2['samples']},key
    inputs=[H3/'cache/oc-retraining/features.npz',H2/'cache/oc-confirmation/features.npz',
            H3/'results/oc-retraining/source-lock.json',H3/'results/oc-retraining/manifest.json',H3/'results/oc-retraining/extraction.json',
            H3/'results/oc-retraining/data-summary.json',H3/'results/oc-retraining/training.json',H3/'results/oc-retraining/retraining.json',H3/'cache/oc-retraining/frozen-scores.npz',
            H2/'results/oc-confirmation/manifest.json',H2/'results/oc-confirmation/extraction.json',
            H3/'configs/oc-retraining/core.py',ROOT/'configs/original-positions/core.py',ROOT/'configs/oc-heads/verify_statistics_independent.py']
    for base,part in [(H3,'oc-retraining'),(H2,'oc-confirmation')]:
        extraction=json.loads((base/f'results/{part}/extraction.json').read_text())
        assert digest(base/f'cache/{part}/features.npz')==extraction['features_sha256']
    with np.load(inputs[0]) as data:
        x=data['train_X'];y=data['train_y']
        assert x.shape==(300,14) and np.isfinite(x).all() and (y==0).sum()==(y==1).sum()==150
        np.testing.assert_array_equal(y,np.array([r['label'] for r in h3['samples']]))
    # Do not load target feature arrays until all trained models are sealed.
    CACHE.mkdir(parents=True,exist_ok=True)
    np.savez(CACHE/'training-only.npz',X=x,y=y)
    inputs.append(CACHE/'training-only.npz')
    sources=list((ROOT/'configs/oc-svm').glob('*.py'))+[ROOT/'docs/oc-svm-protocol-2026-11-04.md']
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    for p in sources:assert p.read_bytes()==subprocess.check_output(['git','show',commit+':'+p.relative_to(ROOT).as_posix()]),p
    write(OUT/'source-lock.json',{'run_id':'OC-S1','commit':commit,'sources':{p.relative_to(ROOT).as_posix():digest(p) for p in sources},
        'inputs':{str(p):digest(p) for p in inputs},'grids':GRIDS,'primary':PRIMARY,'before_training':True,
        'counts':{'train':[150,150],'calibration_safe':400,'test':[1000,500]},'historical_diagnostic':True,
        'h3_dedup_reused':True,'train_target_ids_groups_prompt_and_input_hashes_disjoint':True})
    print('OC_S1_LOCKED',commit,flush=True)
if __name__=='__main__':main()
