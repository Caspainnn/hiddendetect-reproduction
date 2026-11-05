"""Executable checks: oracle, solver certificate, sklearn identity and reload."""
import itertools,tempfile
from pathlib import Path
import joblib,numpy as np
from sklearn.svm import LinearSVC
from core import oracle,fit,score,BASE,COST,PAUC,threshold

def main():
    rng=np.random.default_rng(137);p=rng.normal(size=(7,3));n=rng.normal(size=(10,3));w=rng.normal(size=3)
    loss,d,a=oracle(w,p,n,beta=.2)
    exhaustive=max(np.maximum(1-(p[:,None,:]-n[list(ix)][None,:,:])@w,0).mean() for ix in itertools.combinations(range(10),2))
    np.testing.assert_allclose(loss,exhaustive,atol=1e-12)
    for j in range(3):
        h=np.eye(3)[j]*1e-6
        np.testing.assert_allclose((oracle(w+h,p,n,.2)[0]-oracle(w-h,p,n,.2)[0])/2e-6,-d[j],atol=1e-7)
    x=np.r_[rng.normal(-.8,.5,(20,3)),rng.normal(.8,.5,(20,3))];y=np.repeat([0,1],20)
    for c in [.001,.1,10.,1000.]:
        art,cost=fit(PAUC,{'C':c},x,y)
        assert score(art,x)[y==1].mean()>score(art,x)[y==0].mean()
        assert cost['solver']['primal_dual_gap']<max(2e-4,2e-5*cost['solver']['primal'])
    a,_=fit(BASE,{'C':10.},x,y);b,_=fit(COST,{'C':10.,'safe_weight':1.},x,y)
    np.testing.assert_array_equal(score(a,x),score(b,x))
    manual=LinearSVC(C=10.,dual=False,max_iter=20000,random_state=20261104).fit((x-x.mean(0))/x.std(0),y)
    np.testing.assert_array_equal(score(a,x),manual.decision_function((x-x.mean(0))/x.std(0)))
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/'m.joblib';joblib.dump(art,path);np.testing.assert_array_equal(score(art,x),score(joblib.load(path),x))
    assert threshold(np.arange(400,dtype=float))==380.
    print('PASS: exhaustive top-negative oracle, finite-difference subgradient, four certified QPs, sklearn/r=1 identity, model reload and threshold.')

if __name__=='__main__':main()
