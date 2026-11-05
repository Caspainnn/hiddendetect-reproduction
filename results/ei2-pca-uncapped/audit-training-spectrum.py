import pathlib,json,hashlib,joblib,numpy as np
r=pathlib.Path.cwd();target=r/'cache/ei2-pca-uncapped-worktree';o=target/'results/ei2-pca-uncapped';cache=target/'cache/ei2-pca-uncapped'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
pre=json.loads((o/'pre-evaluation-seal.json').read_text());assert sha(cache/'full-basis.joblib')==pre['basis'];basis=joblib.load(cache/'full-basis.joblib');summary=[]
for layer,b in enumerate(basis,16):
 cumulative=np.cumsum(b['values'])/b['total_energy'];targets={}
 for tau in [.8,.9,.95]:
  k=int(np.searchsorted(cumulative,tau))+1;assert k<=len(cumulative) and cumulative[k-1]>=tau and (k==1 or cumulative[k-2]<tau)
  targets[str(tau)]={'count':k,'coverage':float(cumulative[k-1]),'previous_coverage':float(cumulative[k-2]) if k>1 else 0.}
 summary.append({'layer_0based':layer,'rank':len(b['values']),'n_train':b['n_train'],'retained_energy':b['retained_energy'],'coverage_32':float(cumulative[31]),'targets':targets})
parent=r/'cache/ei2-worktree/results/ei2-vocabulary';base=r/'cache/ei2-pca-extended-worktree/results/ei2-pca-extended';ps=json.loads((parent/'source-lock.json').read_text());pp=json.loads((parent/'pre-evaluation-seal.json').read_text());bs=json.loads((base/'source-lock.json').read_text());bv=json.loads((base/'verification.json').read_text());source=json.loads((o/'source-lock.json').read_text())
assert sha(parent/'manifest.json')==ps['manifest']==bs['parent_manifest']==source['parent_manifest']
assert sha(parent/'source-lock.json')==pp['source_lock'] and sha(base/'source-lock.json')==bv['source_lock_sha256'] and sha(base/'verification.json')==source['parent_verification']
assert json.loads((o/'manifest.json').read_text())['samples']==json.loads((parent/'manifest.json').read_text())['samples']
scores=np.load(o/'scores.npz');identical=True
for role in ['cal','C','RV','OR_cal','OR_test']:identical=identical and np.array_equal(scores[role+'__EI-2-3-2-1'],scores[role+'__EI-2-3-2-1-cap32'])
result={'passed':True,'basis_sha256':pre['basis'],'evaluation_sha256':sha(o/'evaluation.json'),'source_lock_sha256':sha(o/'source-lock.json'),'audit_script_sha256':sha(o/'audit-training-spectrum.py'),'same_frozen_parent_samples_labels_roles':True,'cv_full_and_cap32_scores_bitwise_identical':bool(identical),'layers':summary}
(o/'training-spectrum.json').write_text(json.dumps(result,indent=2)+'\n');print('EI_PA2_SPECTRUM_AND_PARENT_CHAIN_PASS',identical)
