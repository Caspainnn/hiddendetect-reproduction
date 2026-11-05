"""Replay frozen public scores/statistics without private prompts or GPU."""
import json,hashlib,subprocess
import numpy as np
from core import ROOT,NAMES,digest,read,write,references,integrate,threshold,evaluate,validate_confirmation

def sparse_reference_scores(z,norm,selected_weights,weight_norms):
    # Author mixed-dtype cosine normalizes the logits operand in fp16 before promotion.
    normalized=(np.asarray(z,dtype=np.float16)/np.asarray(norm,dtype=np.float16)[:,None]).astype(np.float64)
    return normalized@(selected_weights.astype(np.float64)/weight_norms[:,None]).T

def compare(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys();return sum(compare(a[k],b[k]) for k in a)
    if isinstance(a,list):
        assert len(a)==len(b);return sum(compare(x,y) for x,y in zip(a,b))
    if isinstance(a,(float,int)) and not isinstance(a,bool):assert abs(a-b)<1e-12;return 1
    assert a==b;return 0

def main():
    out=ROOT/'results/oc-reference';lock=json.loads((out/'source-lock-v2.json').read_text())
    for name,h in lock['sha256'].items():assert hashlib.sha256(subprocess.check_output(['git','show',lock['source_commit']+':'+name],cwd=ROOT)).hexdigest()==h
    validate_confirmation(out)
    audit=json.loads((out/'token-audit.json').read_text());rules=json.loads((ROOT/'configs/oc-reference/rules.json').read_text())
    w=references(rules,{r['string']:r['full_encoding'] for r in audit['tokens']});union=np.flatnonzero(w.any(0))
    historical=read(out/'history-scores.jsonl');new=read(out/'confirmation-scores.jsonl')
    manifest=json.loads((out/'manifest.json').read_text());assert {r['id'] for r in historical+new}=={r['id'] for r in manifest['samples']}
    assert not {r['id'] for r in historical}&{r['id'] for r in new}
    assert not {r['group_id'] for r in historical}&{r['group_id'] for r in new}
    maximum=0.
    for r in historical+new:
        assert np.array_equal(integrate(r['F']),r['scores']) and r['union_token_ids']==union.tolist()
        z=np.asarray(r['union_logits'],np.float64);n=np.asarray(r['full_logit_norm'],np.float64)
        reconstructed=sparse_reference_scores(z,n,w[:,union],np.linalg.norm(w.astype(np.float64),axis=1))
        diff=float(np.max(np.abs(reconstructed-r['F'])));assert diff<2e-7,(r['id'],diff)
        maximum=max(maximum,diff)
    cal=[r for r in historical if r['role']=='calibration'];cuts=[threshold(np.array([r['scores'] for r in cal])[:,j]) for j in range(4)]
    assert cuts==json.loads((out/'thresholds.json').read_text())['thresholds']
    numeric={}
    for batch,rows,file in [('historical-C',[r for r in historical if r['role']=='historical_C'],'historical-evaluation.json'),('new-confirmation',new,'confirmation-evaluation.json')]:
        expected=json.loads((out/file).read_text());numeric[batch]=compare(evaluate(rows,cuts,batch),expected)
    write(out/'verification.json',{'status':'passed','numeric_values_checked':numeric,'all_integrals_exact':True,
        'max_cosine_reconstruction_difference':maximum,'cosine_reconstruction_tolerance':2e-7,
        'history_scores_sha256':digest(out/'history-scores.jsonl'),'confirmation_scores_sha256':digest(out/'confirmation-scores.jsonl')})
    print('RV_ALL_REPLAYS_PASS',numeric,'COSINE_MAX_DIFF',maximum)
if __name__=='__main__':main()
