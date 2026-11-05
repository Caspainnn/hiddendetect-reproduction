import subprocess
from core import *
def main():
    assert not (OUT/'source-lock.json').exists()
    old=load(H1/'results/oc-heads/source-lock.json')
    for n,s in old['sha256'].items():assert digest(H1/n)==s,n
    manifest=load(H1/'results/oc-heads/manifest.json')
    assert digest(H1/'results/oc-heads/manifest.json')==old['manifest_sha256']
    feature=H1/'cache/oc-heads/features.npz';assert digest(feature)==manifest['features_sha256']
    rows=manifest['samples']
    for key in ['id','group_id','prompt_sha256','input_ids_sha256']:
        values=[r[key] for r in rows];assert len(values)==len(set(values)),key
    data=np.load(feature);assert [data[k+'_X'].shape for k in ['train','dev','cal','C','RV']]==[(1600,14),(600,14),(400,14),(1000,14),(1082,14)]
    assert (data['cal_y']==0).all();assert [(data['train_y']==k).sum() for k in [0,1]]==[800,800]
    CACHE.mkdir(parents=True,exist_ok=True)
    np.savez(CACHE/'training-only.npz',**{key:data[key] for key in ['train_X','train_y','dev_X','dev_y']})
    selection=load(H1/'results/oc-heads/selection.json')
    # Check historical scoring backends before spending the training budget.
    import joblib,torch
    historical=[json.loads(line) for line in (H1/'results/oc-heads/scores.jsonl').read_text().splitlines()]
    oldcore=module('o1_preflight_h1',ROOT/'configs/oc-heads/core.py')
    for n in HISTORICAL:
        item=next(m for m in selection['selected'][n]['models'] if m['seed']==SEEDS[0]);art=joblib.load(H1/item['path'])
        torch.set_num_threads(2);torch.backends.cuda.enable_flash_sdp(True);torch.backends.cuda.enable_mem_efficient_sdp(True);torch.backends.cuda.enable_math_sdp(True);torch.use_deterministic_algorithms(False)
        expected=np.array([r['scores'][(['B']+oldcore.IDS).index(n)] for r in historical if r['seed']==SEEDS[0] and r['role']=='cal'])
        np.testing.assert_array_equal(oldcore.score(art,data['cal_X']),expected)
    inputs=[feature,H1/'results/oc-heads/source-lock.json',H1/'results/oc-heads/manifest.json',H1/'results/oc-heads/selection.json',H1/'results/oc-heads/scores.jsonl',CACHE/'training-only.npz']
    inputs += [H1/m['path'] for n in HISTORICAL for m in selection['selected'][n]['models']]
    lock=load(A2/'results/oc-attention/source-lock.json')
    vendor={}
    for p,s in lock['vendor_files'].items():assert digest(A2/p)==s;vendor[str(A2/p)]=s
    sources=list((ROOT/'configs/oc4-original').glob('*.py'))+[ROOT/'docs/oc4-original-protocol-2026-11-04.md']
    sources += [ROOT/'configs/oc-svm/core.py',ROOT/'configs/oc-attention/core.py',ROOT/'configs/original-positions/core.py',ROOT/'configs/oc-heads/verify_statistics_independent.py',ROOT/'configs/oc-heads/core.py']
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    for p in sources:assert p.read_bytes()==subprocess.check_output(['git','show',commit+':'+p.relative_to(ROOT).as_posix()]),p
    write(OUT/'source-lock.json',{'run_id':'OC-O1','commit':commit,'sources':{p.relative_to(ROOT).as_posix():digest(p) for p in sources},'inputs':{str(p):digest(p) for p in inputs},'vendor':vendor,
       'contrasts':CONTRASTS,'counts':manifest['counts'],'repeats':REPEATS,'primary':'RV','supplementary':'C','historical':True,'train_dev_test_disjoint':True,'before_fit':True})
    print('OC_O1_DATA_SEALED',commit,flush=True)
if __name__=='__main__':main()
