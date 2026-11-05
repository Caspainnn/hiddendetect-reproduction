import tempfile,joblib
from pathlib import Path
import numpy as np
import torch
from core import *

def main():
    setup(SEEDS[0]);rng=np.random.default_rng(17);x=rng.normal(size=(40,14));y=np.repeat([0,1],20);_,_,knots=preprocessing(x)
    encoder=NumericalEncoding('ple',np.tile([0,1,2,3,4],(14,1)))
    values=torch.tensor(np.repeat(np.array([-1,.5,2.5,5])[:,None],14,axis=1),dtype=torch.float32)
    expected=torch.tensor([[-1,0,0,0],[.5,0,0,0],[1,1,.5,0],[1,1,1,2]],dtype=torch.float32)
    torch.testing.assert_close(encoder(values)[:,0],expected)
    # Same-interface repeated linear encoding remains information-equivalent.
    torch.testing.assert_close(NumericalEncoding('linear',knots)(values).sum(-1),values)
    modified=x.copy();modified[0]+=100;assert not np.array_equal(preprocessing(modified)[2],knots)
    for name in NAMES:
        model=make(name,knots).cpu();s=model(torch.tensor(x,dtype=torch.float32));assert s.shape==(40,1)
        perm=torch.randperm(40);torch.testing.assert_close(model(torch.tensor(x,dtype=torch.float32)[perm]),s[perm],atol=2e-6,rtol=2e-6)
        loss=nn.functional.binary_cross_entropy_with_logits(s,torch.tensor(y,dtype=torch.float32).reshape(-1,1));loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    # Reference library CVaR state update and masked pairwise gradient.
    vendors();from libauc.losses import pAUC_CVaR_Loss
    from libauc.optimizers import SOPA
    pred=torch.tensor([.2,.6,.1,-.3],requires_grad=True).reshape(-1,1);truth=torch.tensor([1,1,0,0]).reshape(-1,1)
    criterion=pAUC_CVaR_Loss(data_len=4,pos_len=2,beta=.1,eta=1.,device=torch.device('cpu'))
    result=criterion(pred,truth,torch.arange(4));manual=(torch.relu(1-(pred[:2]-pred[2:].reshape(1,-1)))**2).mean()/.1
    torch.testing.assert_close(result,manual);assert (criterion.u_pos[:2]==4.5).all()
    for name in [ATT,GATE,PLE_A,PLE_M,PAUC,FT]:
        a,c=fit(name,.001,SEEDS[0],x,y,steps=3)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'m.joblib';joblib.dump(a,p);np.testing.assert_array_equal(score(a,x),score(joblib.load(p),x))
    assert threshold(np.arange(400,dtype=float))==380.
    print('PASS: PLE boundaries/extrapolation, equivalent linear encoding, batch independence, gradients, official CVaR state/loss, six model reloads and threshold.',flush=True)
if __name__=='__main__':main()
