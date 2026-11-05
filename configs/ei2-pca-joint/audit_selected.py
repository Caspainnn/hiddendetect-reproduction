"""Read-only post-run binding audit; does not fit or select models."""
import numpy as np

def assert_entry_matches(selected,trial):
 for key in ['C','seed','variant']:assert selected[key]==trial[key],key
 for key in ['mean','scale']:
  assert selected[key].dtype==trial[key].dtype,key
  np.testing.assert_array_equal(selected[key],trial[key])
 assert len(selected['indices'])==len(trial['indices'])==14
 for a,b in zip(selected['indices'],trial['indices']):
  assert a.dtype==b.dtype
  np.testing.assert_array_equal(a,b)
 a,b=selected['head'],trial['head']
 assert type(a) is type(b) and a.get_params(deep=True)==b.get_params(deep=True)
 for key in ['coef_','intercept_','classes_','n_iter_']:
  assert getattr(a,key).dtype==getattr(b,key).dtype,key
  np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
 assert a.n_features_in_==b.n_features_in_

def main():
 from common import ROOT,OUT,CACHE,load_data,u,pack,digest,write,joblib,json
 from sklearn.metrics import roc_auc_score
 proof=json.loads((OUT/'verification.json').read_text());assert proof['passed']
 for name,key in [('evaluation.json','evaluation_sha256'),('scores.npz','scores_sha256')]:assert digest(OUT/name)==proof[key]
 pre=json.loads((OUT/'pre-evaluation-seal.json').read_text());assert digest(OUT/'final-search.json')==pre['search'] and digest(OUT/'selector-training.json')==pre['selectors']
 search=json.loads((OUT/'final-search.json').read_text());evaluation=json.loads((OUT/'evaluation.json').read_text())
 data=load_data();ix=data['roles']['dev'];a=u.project(data,ix,data['joint_basis']);y=np.array([r['label'] for r in data['rows']])[ix];bindings=[]
 for name,sha in pre['models'].items():
  path=CACHE/(name+'.joblib');assert digest(path)==sha;model=joblib.load(path);aucs=[]
  for entry in model['entries']:
   records=[r for r in search if (r['family'],r['variant'],r['C'],r['seed'])==(name,model['variant']['key'],model['C'],entry['seed'])];assert len(records)==1
   record=records[0];trialpath=ROOT/record['path'];assert digest(trialpath)==record['sha256'];assert_entry_matches(entry,joblib.load(trialpath))
   x=pack(data['f'][ix],a,entry['indices'],entry['mean'],entry['scale'])[0];auc=float(roc_auc_score(y,entry['head'].decision_function(x)));assert abs(auc-record['dev_auroc'])<1e-12;aucs.append(auc)
   bindings.append({'family':name,'variant':entry['variant'],'C':entry['C'],'seed':entry['seed'],'trial_path':record['path'],'trial_sha256':record['sha256'],'model_sha256':sha,'dev_auroc':auc,'exact_content_match':True})
  assert abs(np.mean(aucs)-model['dev_auroc_mean'])<1e-12 and abs(np.mean(aucs)-evaluation['selected'][name]['dev_auroc_mean'])<1e-12
 assert len(bindings)==30
 write(OUT/'selected-entry-audit.json',{'passed':True,'entries_bound':30,'records':bindings,'evaluation_sha256':digest(OUT/'evaluation.json')})
 zero=[]
 for record in json.loads((OUT/'selector-training.json').read_text()):
  path=ROOT/record['path'];assert digest(path)==record['sha256'];artifact=joblib.load(path);state=artifact['state'];w=state['weight'][14:];offset=0;layers=[]
  for layer,indices in enumerate(artifact['indices'],16):
   row=w[offset:offset+len(indices)];offset+=len(indices);layers.append({'layer_0based':layer,'input_directions':len(row),'exact_zero_count':int((row==0).sum()),'exact_zero_fraction':float((row==0).mean()),'threshold_zero_count':int((np.abs(row)<=1e-8).sum()),'threshold_zero_fraction':float((np.abs(row)<=1e-8).mean())})
  assert offset==len(w)
  zero.append({'key':record['key'],'state_sha256':record['sha256'],'optimizer_success':bool(state['optimizer_success']),'optimizer_message':str(state['optimizer_message']),'layers':layers})
 assert len(zero)==18
 write(OUT/'selector-audit.json',{'passed':True,'threshold':1e-8,'records':zero})
 files=['verification.json','selected-entry-audit.json','selector-audit.json','evaluation.json','scores.npz']
 write(OUT/'final-verification.json',{'passed':True,'run_id':'EI-PJ1','evaluation_sha256':digest(OUT/'evaluation.json'),'scores_sha256':digest(OUT/'scores.npz'),'files':{name:digest(OUT/name) for name in files},'audit_code_sha256':digest(ROOT/'configs/ei2-pca-joint/audit_selected.py'),'selected_entries_bound':30,'selectors_described':18,'training_or_selection_changed':False})
 print('EI_PJ1_FINAL_VERIFICATION_PASS',flush=True)

if __name__=='__main__':main()
