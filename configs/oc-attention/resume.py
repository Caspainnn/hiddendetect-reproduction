"""Resume sealed models after restoring H3's original scoring backend."""
import json,subprocess,sys,time
from core import *

def main():
    verify_lock()
    assert json.loads((OUT/'status.json').read_text())==json.loads((OUT/'attempt1-evaluate-status.json').read_text())
    assert not (OUT/'evaluation.json').exists()
    original=ROOT/'configs/oc-attention/evaluate.py'
    source=original.read_text()
    old="replay[role]=oldcore.score(old,data[role+'_X']);"
    new="replay[role]=module('a2_h3_scoring_backend',H3/'configs/oc-retraining/core.py').predict(old,data[role+'_X']);setup(SEEDS[0]);"
    assert source.count(old)==1
    target=CACHE/'evaluate-h3-backend-resume.py'
    target.write_bytes(source.replace(old,new).encode())
    revision={'reason':'H3 predict explicitly enabled SDPA backends and disabled deterministic algorithms; original regression called H1 score directly under new backend settings. Restore exact H3 predict for historical replay only, then restore A2 setup.',
      'test_access_before_failure':True,'evaluation_written_before_failure':False,'no_model_or_selection_change':True,
      'source_lock_sha256':digest(OUT/'source-lock.json'),'training_sha256':digest(OUT/'training.json'),
      'original_evaluate_sha256':digest(original),'resume_source_sha256':digest(Path(__file__)),
      'generated_evaluate_sha256':digest(target),'h3_backend_sha256':digest(H3/'configs/oc-retraining/core.py')}
    write(OUT/'evaluation-resume-seal.json',revision)
    for stage,path in [('evaluate',target),('verify',ROOT/'configs/oc-attention/verify.py'),('report',ROOT/'configs/oc-attention/report.py')]:
        write(OUT/'status.json',{'state':'running','stage':stage,'resume':1,'time':time.time()})
        result=subprocess.run([sys.executable,'-u',str(path)],cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'configs/oc-attention')})
        if result.returncode:
            write(OUT/'status.json',{'state':'failed','stage':stage,'code':result.returncode,'resume':1,'time':time.time()});return result.returncode
    write(OUT/'status.json',{'state':'completed','resume':1,'time':time.time()});return 0
if __name__=='__main__':sys.exit(main())
