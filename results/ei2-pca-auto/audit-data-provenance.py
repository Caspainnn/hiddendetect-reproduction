import pathlib,json,hashlib
r=pathlib.Path.cwd();p=r/'cache/ei2-worktree';b=r/'cache/ei2-pca-extended-worktree';n=r/'cache/ei2-pca-auto-worktree';o=n/'results/ei2-pca-auto'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
ps=read(p/'results/ei2-vocabulary/source-lock.json');pp=read(p/'results/ei2-vocabulary/pre-evaluation-seal.json');bs=read(b/'results/ei2-pca-extended/source-lock.json');bv=read(b/'results/ei2-pca-extended/verification.json');ns=read(o/'source-lock.json')
checks=[('parent_manifest',p/'results/ei2-vocabulary/manifest.json',ps['manifest']),('parent_source_lock',p/'results/ei2-vocabulary/source-lock.json',pp['source_lock']),('base_source_lock',b/'results/ei2-pca-extended/source-lock.json',bv['source_lock_sha256']),('base_verification',b/'results/ei2-pca-extended/verification.json',ns['parent_verification'])]
assert ps['manifest']==bs['parent_manifest']
for name,path,expected in checks:assert sha(path)==expected,name
assert read(o/'manifest.json')['samples']==read(p/'results/ei2-vocabulary/manifest.json')['samples']
result={'passed':True,'same_samples_labels_and_roles_as_frozen_parent':True,'checks':{name:sha(path) for name,path,_ in checks},'evaluation_sha256':sha(o/'evaluation.json'),'audit_script_sha256':sha(pathlib.Path(__file__))}
(o/'data-provenance-audit.json').write_text(json.dumps(result,indent=2)+'\n');print('PARENT_SAMPLE_LABEL_ROLE_HASH_CHAIN_PASS')
