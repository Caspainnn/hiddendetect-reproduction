"""One frozen comparison uses strict thresholds and arbitrary two-column scores."""
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'configs/original-positions'))
from core import paired_intervals

def main():
    y=np.array([0,0,0,1,1,1]);s=np.array([[0,1],[1,0],[1,2],[2,2],[3,1],[3,3]],dtype=float)
    cut=[1.,1.]
    result=paired_intervals(y,s,repeats=200,seed=33,family=1,thresholds=cut)[0]
    assert result['ci95']==result['adjusted']
    assert result['delta']==roc_auc_score(y,s[:,0])-roc_auc_score(y,s[:,1])
    pred=s>cut
    for label,name in [(0,'fpr'),(1,'tpr')]:
        assert result['descriptive_'+name]['delta']==(pred[y==label,0].astype(float)-pred[y==label,1]).mean()
    # Equality at a frozen threshold is never an alarm.
    assert not pred[1,0] and not pred[4,1]
    print('ORIGINAL_TEMPLATE_CONFIRM_CONTRACT_PASS')

if __name__=='__main__':main()
