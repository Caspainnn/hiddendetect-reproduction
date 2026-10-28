"""Seal executable dependencies and immutable heads before heldout scoring."""
import json
import re
import sys
from datetime import datetime,timezone
from features import ROOT,digest

DEPENDENCIES=['configs/highdim/features.py','configs/highdim/detectors.py','configs/highdim/ridge.py',
    'configs/highdim/seal.py','configs/highdim/train.py','configs/highdim/batch_search.py','configs/highdim/evaluate.py','configs/highdim/seal_code.py',
    'configs/signal-selection/train_search.py','configs/signal-selection/experiment_core.py','configs/signal-selection/evaluate_frozen.py']

def verify_code_seal(out):
    seal=json.loads((out/'pre-evaluation-seal.json').read_text())
    assert digest(out/'freeze.json')==seal['freeze_sha256']
    for name,sha in seal['source_sha256'].items(): assert digest(ROOT/name)==sha,name
    return seal

def main():
    commit=sys.argv[1]; assert re.fullmatch('[0-9a-f]{40}',commit)
    out=ROOT/'results/highdim'; target=out/'pre-evaluation-seal.json'
    assert not (out/'evaluation.json').exists(),'Cannot create a new seal after test evaluation'
    if target.exists(): verify_code_seal(out); print('EXISTING_CODE_SEAL_VERIFIED'); return
    freeze=json.loads((out/'freeze.json').read_text())
    assert digest(ROOT/'configs/highdim/train.py')==freeze['train_code_sha256']
    assert digest(ROOT/'configs/highdim/detectors.py')==freeze['detectors_code_sha256']
    seal={'status':'sealed_before_heldout_scoring','freeze_sha256':digest(out/'freeze.json'),
        'declared_source_commit':commit,'created_utc':datetime.now(timezone.utc).isoformat(),
        'source_sha256':{name:digest(ROOT/name) for name in DEPENDENCIES},
        'scope':'training process source unchanged; dependency sidecar added before any calibration/test scoring'}
    temporary=target.with_suffix('.tmp'); temporary.write_text(json.dumps(seal,indent=2)+'\n'); temporary.replace(target)
    print('EXECUTABLE_DEPENDENCIES_SEALED',len(DEPENDENCIES))

if __name__=='__main__': main()
