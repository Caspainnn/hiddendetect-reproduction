import json,joblib,platform,sys,hashlib
from pathlib import Path
import numpy,scipy,sklearn
root=Path.cwd();out=root/'results/oc-svm'
r=json.loads((out/'training.json').read_text());values=[]
for item in r['runs']:
 art=joblib.load(root/item['path']);model=art['model']
 if isinstance(model,dict):details={'linear_coefficients':len(model['coef']),'intercept_parameters':0,'solver':model['solver']}
 elif hasattr(model,'support_vectors_'):details={'support_vectors':len(model.support_vectors_),'dual_coefficients':model.dual_coef_.size,'intercept_parameters':model.intercept_.size,'support_vector_coordinates':model.support_vectors_.size,'solver_fit_status':int(model.fit_status_)}
 else:details={'linear_coefficients':model.coef_.size,'intercept_parameters':model.intercept_.size,'solver_iterations':int(model.n_iter_)}
 values.append({'id':item['id'],'seed':item['seed'],**details})
failures=[]
for name in ['oc-svm-attempt1-pretest-numerical-failure','oc-svm-attempt2-pretest-numerical-failure']:
 d=root/'results'/name
 assert not (d/'evaluation.json').exists() and not (d/'training.json').exists()
 failures.append({'directory':'results/'+name,'status':json.loads((d/'status.json').read_text()),'no_models_sealed_or_test_scoring':True,'source_lock_sha256':hashlib.sha256((d/'source-lock.json').read_bytes()).hexdigest(),'worker_log_sha256':hashlib.sha256((d/'worker.log').read_bytes()).hexdigest()})
value={'models':values,'environment':{'python':platform.python_version(),'numpy':numpy.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},'pretest_failed_attempts':failures,'audit_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'post_evaluation_descriptive_audit':True}
(out/'capacity-environment-failures.json').write_text(json.dumps(value,indent=2)+'\n')
print('OC_S1_CAPACITY_AND_FAILURE_AUDIT',len(values))
