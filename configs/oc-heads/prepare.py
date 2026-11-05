"""Reuse exact author-F caches; partition and provenance checks precede fitting."""
import json,subprocess,sys
from collections import Counter
from pathlib import Path
import numpy as np
from core import ROOT,OUT,IDS,GRIDS,SEEDS,digest,read,write,baseline

def main():
    origin=ROOT.parents[1];assert (origin/'cache/signal-selection/F.npy').exists()
    assert not (OUT/'manifest.json').exists()
    source=origin/'configs/signal-selection/manifest.json';m=json.loads(source.read_text());rows=m['samples']
    metadata=json.loads((origin/'cache/signal-selection/metadata.json').read_text())
    assert metadata['manifest_sha256']==digest(source)
    assert json.loads((origin/'cache/signal-selection/checkpoint.json').read_text())['next_index']==len(rows)
    f=np.load(origin/'cache/signal-selection/F.npy',mmap_mode='r');assert f.shape==(4800,32)
    smoke=json.loads((origin/'logs/signal-selection-feature-smoke.json').read_text())
    assert smoke['scores_match'] and smoke['max_F_difference_from_author']<=1e-7
    cohorts={}
    for role in ['train','dev']:
        cohorts[role]=[dict(r,F=f[i,16:30].astype(float).tolist()) for i,r in enumerate(rows) if r['role']==role]
    history=read(ROOT/'results/oc23/history-scores.jsonl')
    for role,target in [('calibration','cal'),('historical_C','C'),('historical_RV','RV')]:
        chosen=[r for r in history if r['role']==role]
        cohorts[target]=[dict({k:r[k] for k in ['id','label','group_id','prompt_sha256','input_ids_sha256']},F=r['F'][16:30]) for r in chosen]
    assert [len(cohorts[k]) for k in ['train','dev','cal','C','RV']]==[1600,600,400,1000,1082]
    allrows=[dict(r,cohort=k) for k,v in cohorts.items() for r in v]
    for key in ['id','group_id','prompt_sha256','input_ids_sha256']:
        values=[r[key] for r in allrows];assert len(set(values))==len(values),(key,'cross-role overlap')
    assert all(r['label']==0 for r in cohorts['cal'])
    arrays={}
    for key,values in cohorts.items():
        arrays[key+'_X']=np.array([r['F'] for r in values],dtype=float);arrays[key+'_y']=np.array([r['label'] for r in values],dtype=int)
        assert np.isfinite(arrays[key+'_X']).all()
    assert np.array_equal(baseline(arrays['C_X']),np.array([r['scores'][0] for r in history if r['role']=='historical_C']))
    # The last audit covered all three frozen Aegis splits. Verify no new eligible source was added.
    prior=json.loads((ROOT/'results/oc23/manifest.json').read_text());checked={}
    for name,sha in prior['historical_sources'].items():
        if '/data/' in '/'+name or name.startswith('data/'):
            path=origin/name;assert path.exists() and digest(path)==sha,('Changed data source needs new audit',name)
            checked[name]=sha
    actual=list((origin/'data').glob('**/records.jsonl'))+list((origin/'data').glob('**/probes.jsonl'))
    for path in actual:
        name=str(path.relative_to(origin))
        if name=='data/oc23/records.jsonl':
            # This derived cohort was written after the previous residual audit.
            # It contains only that run's already frozen historical inputs.
            assert digest(path)==prior['records_sha256']
            derived=read(path);assert {r['id'] for r in derived}=={r['id'] for r in history}
            assert all(r['role']!='confirmation' for r in derived)
            checked[name]=digest(path)
        else:assert name in checked,('Unregistered records need new audit',name)
    assert min(prior['available_fresh_per_class'].values())<30
    cache=ROOT/'cache/oc-heads';cache.mkdir(parents=True,exist_ok=True)
    np.savez(cache/'features.npz',**arrays)
    manifest={'run_id':'OC-H1','samples':[{k:v for k,v in r.items() if k!='F'} for r in allrows],
        'counts':{k:dict(Counter(str(r['label']) for r in v)) for k,v in cohorts.items()},
        'features_sha256':digest(cache/'features.npz'),'training_F_cache_sha256':digest(origin/'cache/signal-selection/F.npy'),
        'training_manifest_sha256':digest(source),'training_metadata':metadata,'prior_author_smoke':smoke,
        'history_scores_sha256':digest(ROOT/'results/oc23/history-scores.jsonl'),
        'unchanged_audited_data_sources':checked,'available_fresh_per_class':prior['available_fresh_per_class'],
        'independent_confirmation':False,'confirmation_reason':'unchanged frozen source pool; prior full audit leaves fewer than 30 unused safe records',
        'train_label_budget':1600,'selection_label_budget':600,'normal_calibration_budget':400,
        'fixed_features':'original F, last input token, 0-based layers16..29 inclusive, no other model information'}
    write(OUT/'manifest.json',manifest)
    paths=sorted((ROOT/'configs/oc-heads').glob('*.py'))+[ROOT/'docs/oc-heads-protocol-2026-11-04.md']
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    for path in paths:
        assert path.read_bytes()==subprocess.check_output(['git','show',commit+':'+str(path.relative_to(ROOT))]),path
    write(OUT/'source-lock.json',{'source_commit':commit,'sha256':{str(p.relative_to(ROOT)):digest(p) for p in paths},
        'manifest_sha256':digest(OUT/'manifest.json'),'ids':IDS,'grids':GRIDS,'seeds':SEEDS,'before_any_head_fit':True})
    write(OUT/'confirmation-unavailable.json',{'status':'not_performed','available_fresh_per_class':prior['available_fresh_per_class'],'source_pool_unchanged':True,'historical_samples_not_relabelled_as_fresh':True})
    print('HEAD_DATA_FROZEN',manifest['counts'],'NO_NEW_CONFIRMATION',flush=True)
if __name__=='__main__':main()
