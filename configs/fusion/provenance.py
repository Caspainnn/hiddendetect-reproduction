"""Executable dependency seals captured at training start, verified at freeze/eval."""
import json
import hashlib
import re
import subprocess
from pathlib import Path
from fusion_core import ROOT,digest
from seal import verify_seal

SOURCES=['configs/fusion/fusion_core.py','configs/fusion/train.py','configs/fusion/evaluate.py','configs/fusion/provenance.py',
    'configs/fusion/extract.py','configs/fusion/seal_cache.py','configs/fusion/check_probes.py',
    'configs/highdim/features.py','configs/highdim/ridge.py','configs/highdim/detectors.py','configs/highdim/seal.py',
    'configs/signal-selection/train_search.py','configs/signal-selection/experiment_core.py','configs/signal-selection/evaluate_frozen.py']

def source_snapshot(): return {name:digest(ROOT/name) for name in SOURCES}
def verify_sources(sources):
    for name,sha in sources.items(): assert digest(ROOT/name)==sha,name

def check_test_conditions(tests):
    expected={'test_aegis_holdout:global','test_aegis_official:global',
        'test_toxic_holdout:global','test_toxic_holdout:source',
        'test_toxicchat:global','test_toxicchat:source'}
    assert set(tests)==expected,'Missing or unexpected test conditions'

def verify_pre_evaluation_seal(out,require_git=False):
    record=json.loads((out/'pre-evaluation-seal.json').read_text())
    assert record['freeze_sha256']==digest(out/'freeze.json')
    assert re.fullmatch('[0-9a-f]{40}',record['commit'])
    if require_git:
        commit=record['commit']; freeze=json.loads((out/'freeze.json').read_text())
        expected={str((out/'freeze.json').relative_to(ROOT)).replace('\\','/'):record['freeze_sha256'],
                  **freeze['source_sha256']}
        for candidate in freeze['experts']+freeze['candidates']:
            if 'file' in candidate: expected['results/fusion/'+candidate['file']]=candidate['sha256']
        for name,sha in expected.items():
            blob=subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT)
            assert hashlib.sha256(blob).hexdigest()==sha,('Uncommitted frozen dependency',name)
        subprocess.run(['git','merge-base','--is-ancestor',commit,'HEAD'],cwd=ROOT,check=True)

def check_manifest_binding(manifest_path,seal,metadata):
    rows=json.loads(Path(manifest_path).read_text())['samples']
    assert seal['manifest_sha256']==metadata['manifest_sha256']==digest(manifest_path)
    assert seal['count']==metadata['count']==len(rows)
    assert [row['index'] for row in rows]==list(range(len(rows)))

def verify_cache_binding(directory,manifest_path):
    seal=json.loads((directory/'seal.json').read_text()); metadata=json.loads((directory/'metadata.json').read_text())
    check_manifest_binding(manifest_path,seal,metadata)
    verify_seal(directory,seal)

def verify_freeze(out,cache_bytes=False):
    verify_pre_evaluation_seal(out,require_git=cache_bytes)
    freeze=json.loads((out/'freeze.json').read_text()); verify_sources(freeze['source_sha256'])
    assert digest(ROOT/'configs/fusion/manifest.json')==freeze['manifest_sha256']
    assert digest(ROOT/'configs/highdim/manifest.json')==freeze['old_training_manifest_sha256']
    for name,key in [('oof.npz','oof_sha256'),('dev.npz','dev_sha256'),('oof-provenance.json','provenance_sha256'),
                     ('complementarity.json','complementarity_sha256'),('equal.npz','equal_sha256')]: assert digest(out/name)==freeze[key]
    for candidate in freeze['experts']+freeze['candidates']:
        if 'file' in candidate: assert digest(out/candidate['file'])==candidate['sha256']
    for name,key in [('highdim','old_cache_seal_sha256'),('fusion','new_cache_seal_sha256')]:
        path=ROOT/'cache'/name
        if cache_bytes:
            assert digest(path/'seal.json')==freeze[key]
            verify_cache_binding(path,ROOT/'configs'/name/'manifest.json')
    assert digest(ROOT/'results/highdim/scaler.npz')==freeze['legacy_scaler_sha256']
    assert digest(ROOT/'results/highdim/centroids.npy')==freeze['legacy_centroids_sha256']
    assert digest(ROOT/'results/highdim/schema.json')==freeze['legacy_schema_sha256']
    assert digest(ROOT/'logs/fusion-data-usage.json')==freeze['data_usage_sha256']
    return freeze
