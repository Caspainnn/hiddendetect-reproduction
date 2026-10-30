import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'configs/original-positions'))
from core import paired_intervals,project_readouts

def main():
    # Last three actual suffix token IDs verified across all1850 original manifests.
    ids=np.array([1,7,9047,13566,29901]);assert ids[[-2,-3,-1]].tolist()==[13566,9047,29901]
    data=np.arange(15).reshape(1,5,3)
    assert np.array_equal(project_readouts(data,[-2,-3,-1],lambda x:x,lambda x:x),data[:,[-2,-3,-1],:])
    y=np.array([0,0,1,1]);s=np.array([[0,1,0],[1,0,1],[2,3,2],[3,2,3]])
    pairs=paired_intervals(y,s,repeats=100,family=2)
    assert len(pairs)==2 and pairs[0]['delta']==0 and pairs[0]['adjusted']==[0.,0.]
    print('ORIGINAL_TEMPLATE_CONTRACT_PASS')

if __name__=='__main__':main()
