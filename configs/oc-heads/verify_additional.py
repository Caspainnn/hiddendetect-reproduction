"""Post-run audit, without changing the sealed training/scoring implementation."""
import json,subprocess,joblib
import numpy as np
import torch
from core import ROOT,OUT,IDS,digest,write,score
def main():
    torch.set_num_threads(2)
    # Reproduce the dev scorer's math-SDPA context inherited from core.fit().
    # Historical test scores are replayed separately in their evaluation context.
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)
    lock=json.loads((OUT/'source-lock.json').read_text());selection=json.loads((OUT/'selection.json').read_text())
    data=np.load(ROOT/'cache/oc-heads/features.npz');manifest=json.loads((OUT/'manifest.json').read_text())
    assert digest(ROOT/'cache/oc-heads/features.npz')==manifest['features_sha256']
    dependencies={}
    for name in ['configs/original-positions/core.py','configs/oc-heads/core.py']:
        assert (ROOT/name).read_bytes()==subprocess.check_output(['git','show',lock['source_commit']+':'+name])
        dependencies[name]=digest(ROOT/name)
    maxdiff=0.;count=0
    for method in json.loads((OUT/'search.json').read_text()):
        name=method['id'];chosen=selection['selected'][name];entry=method['candidates'][chosen['candidate_index']]
        for saved in chosen['models']:
            assert digest(ROOT/saved['path'])==saved['sha256']
            old=next(r for r in entry['runs'] if r['seed']==saved['seed'])
            values=score(joblib.load(ROOT/saved['path']),data['dev_X'])
            delta=float(abs(values-np.asarray(old['dev_scores'])).max());assert delta<=1e-12,(name,saved['seed'],delta)
            from sklearn.metrics import roc_auc_score
            assert abs(roc_auc_score(data['dev_y'],values,max_fpr=.1)-old['dev_pauc'])<=1e-12
            maxdiff=max(maxdiff,delta);count+=len(values)
    write(OUT/'verification-additional.json',{'status':'passed','dependency_sha256':dependencies,'dependency_bytes_equal_frozen_commit':True,
        'features_sha256_verified':True,'selected_models':24,'dev_score_values_replayed':count,'maximum_dev_score_difference':maxdiff,
        'dev_backend':'training-context math SDPA, deterministic algorithms, threads2; test replay uses original evaluation context'})
    print('OC4_ADDITIONAL_AUDIT_PASS',count,maxdiff,flush=True)
if __name__=='__main__':main()
