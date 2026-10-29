"""Sequential resumable compute-host driver. Start only in the fusion branch."""
import json
import subprocess
import sys
from pathlib import Path
from fusion_core import ROOT,digest

def run(*args):
    subprocess.run(args,cwd=ROOT,check=True)

def main():
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    assert branch=='experiment/conditional-fusion','Preserve old branch: create/switch fusion branch before running'
    py=sys.executable; base=ROOT/'configs/fusion'
    run(py,str(base/'test_core.py'))
    if not (base/'manifest.json').exists(): run(py,str(base/'build_manifest.py'))
    run(py,str(base/'record_usage.py'))
    run('git','add','configs/fusion','logs/fusion-data-usage.json')
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:
        run('git','commit','-m','Freeze fusion input manifest and executable protocol')
    run(py,str(base/'extract.py'),'--pilot-only')
    pilot=json.loads((ROOT/'logs/fusion-pilot.json').read_text())
    assert pilot['attention_enabled'],'Attention resource pilot failed; do not silently change expert set'
    run(py,str(base/'extract.py'))
    run(py,str(base/'seal_cache.py'))
    run(py,str(base/'check_probes.py'))
    run(py,str(base/'train.py'))
    out=ROOT/'results/fusion'
    # Commit the frozen heads before first calibration/test scoring.
    run('git','add','results/fusion','logs/fusion-cache-seal.json','logs/fusion-pilot.json',
        'logs/fusion-probe-compatibility.json','logs/fusion-author-smoke.json','logs/fusion-extraction.json')
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:
        run('git','commit','-m','Seal conditional fusion candidates before prospective evaluation')
    seal={'freeze_sha256':digest(out/'freeze.json'),'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    target=out/'pre-evaluation-seal.json'
    if target.exists():
        previous=json.loads(target.read_text())
        assert previous['freeze_sha256']==seal['freeze_sha256']
        run('git','merge-base','--is-ancestor',previous['commit'],'HEAD')
    else: target.write_text(json.dumps(seal,indent=2)+'\n')
    run(py,str(base/'evaluate.py'))
    run(py,str(base/'verify.py'))
    print('FUSION_ROUND_VERIFIED',flush=True)

if __name__=='__main__': main()
