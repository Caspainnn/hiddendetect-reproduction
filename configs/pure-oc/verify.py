"""Numerical replay and absolute binomial intervals; no model or private text needed."""
import json,math,subprocess
import numpy as np
from sklearn.metrics import roc_auc_score
from analysis import ROOT,aggregate,calibrate,digest,evaluate,read,write

def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [center-half,center+half]
def compare(a,b,path=''):
    if isinstance(a,dict):
        assert a.keys()==b.keys(),path
        return sum(compare(a[k],b[k],path+'/'+str(k)) for k in a)
    if isinstance(a,list):
        assert len(a)==len(b),path
        return sum(compare(x,y,path+'/'+str(i)) for i,(x,y) in enumerate(zip(a,b)))
    if isinstance(a,(float,int)) and not isinstance(a,bool):
        assert abs(a-b)<1e-12,(path,a,b)
        return 1
    assert a==b,(path,a,b)
    return 0
def main():
    out=ROOT/'results/pure-oc';q=json.loads((out/'thresholds.json').read_text());cut=q['thresholds']
    assert [calibrate(np.array(q['calibration_scores'])[:,j]) for j in range(5)]==cut
    lock=json.loads((ROOT/'configs/pure-oc/run-lock.json').read_text());verified=[]
    for name,sha in lock['sha256'].items():
        blob=subprocess.check_output(['git','show',lock['source_commit']+':'+name],cwd=ROOT)
        import hashlib
        assert hashlib.sha256(blob).hexdigest()==sha,name
        verified.append(name)
    new=read(out/'confirmation-scores.jsonl');manifest=json.loads((ROOT/'configs/pure-oc/manifest.json').read_text())
    assert len(new)==2000 and [r['id'] for r in new]==[r['id'] for r in manifest['samples']]
    history=read(out/'historical-scores.jsonl')
    assert not set(r['id'] for r in new)&(set(r['id'] for r in history)|set(q['calibration_ids']))
    audits=[];absolute={}
    for batch,file,rows,run in [('C','historical-C.json',[r for r in history if r['batch']=='C'],'OC-A1-H'),
                              ('ORP-T2','historical-ORP-T2.json',[r for r in history if r['batch']=='ORP-T2'],'OC-A1-H-ORP-T2'),
                              ('new','confirmation.json',new,'OC-A1-N')]:
        stored=json.loads((out/file).read_text());replay=evaluate(rows,cut,run);replay.pop('scores')
        numeric=compare(replay,{k:stored[k] for k in replay})
        scores=aggregate([r['F'] for r in rows]);assert np.array_equal(scores,[r['scores'] for r in rows])
        y=np.array([r['label'] for r in rows]);absolute[batch]=[]
        for j,m in enumerate(stored['metrics']):
            item={'name':m['name']}
            for label,name in [(0,'fpr'),(1,'tpr')]:
                n=int((y==label).sum());k=int((scores[y==label,j]>cut[j]).sum())
                assert k/n==m[name];item[name+'_count']=k;item[name+'_n']=n;item[name+'_wilson95']=wilson(k,n)
            absolute[batch].append(item)
        audits.append({'batch':batch,'numeric_values_checked':numeric,'maximum_difference_at_tolerance_1e_12':0.,'all_scores_exact':True})
    write(out/'absolute-rate-intervals.json',{'method':'Wilson two-sided 95%, fixed threshold; not multiplicity-adjusted; no calibration uncertainty','batches':absolute})
    write(out/'verification.json',{'status':'passed','frozen_git_blobs_checked':verified,'replays':audits,
        'new_scores_sha256':digest(out/'confirmation-scores.jsonl'),'manifest_sha256':digest(ROOT/'configs/pure-oc/manifest.json')})
    print('OC_ALL_NUMERICAL_REPLAYS_PASS',audits)
if __name__=='__main__':main()
