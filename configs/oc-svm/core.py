"""OC-S1: small-data SVM comparisons using only the frozen fourteen F."""
import hashlib,json,time,warnings
from pathlib import Path
import numpy as np
from scipy.optimize import minimize,nnls
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import LinearSVC,SVC

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/oc-svm'
CACHE=ROOT/'cache/oc-svm'
ORIGIN=ROOT.parents[1]
H2=ORIGIN/'cache/oc4c-worktree'
H3=ORIGIN/'cache/oc4r-worktree'
SEEDS=[20261104,20261105,20261106]
BASE='OC-4-1-3'
COST=BASE+'-1'
RBF=BASE+'-2'
PAUC=BASE+'-3'
CTRL='linear-kernel-SVC'
CS=[.001,.01,.1,1.,3.,10.,30.,100.,1000.]
GRIDS={BASE:[{'C':c} for c in CS],
       COST:[{'C':c,'safe_weight':r} for c in [.1,10.,1000.] for r in [1.,2.,4.]],
       RBF:[{'C':c,'gamma':g} for c in [.1,10.,1000.] for g in [.01,.1,1.]],
       PAUC:[{'C':c} for c in CS],CTRL:[{'C':c} for c in CS]}
PRIMARY=[(COST,BASE),(RBF,BASE),(PAUC,BASE),(RBF,CTRL)]

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())

def oracle(w,pos,neg,beta=.1):
    """Top-k negative pairwise hinge: exact subgradient for beta*n integer.

    At alpha=0 this is the convex top-negative hinge surrogate, inspired by
    Narasimhan & Agarwal 2013. Not an exact reimplementation of their solver.
    Each negative's average hinge is monotone in its score, so selecting
    largest scores also selects the largest negative-wise losses.
    """
    k=int(round(beta*len(neg)))
    if k<1 or abs(k-beta*len(neg))>1e-8:raise ValueError('Integral beta*n required')
    chosen=np.argsort(neg@w,kind='stable')[-k:]
    differences=pos[:,None,:]-neg[chosen][None,:,:]
    margin=1.-differences@w;active=margin>0
    direction=(differences*active[:,:,None]).mean(axis=(0,1))
    intercept=float(active.mean())
    loss=float(np.maximum(margin,0).mean())
    np.testing.assert_allclose(loss,intercept-w@direction,atol=1e-10)
    return loss,direction,intercept

def fit_pauc(x,y,c):
    pos=x[y==1];neg=x[y==0];d=x.shape[1]
    w=np.zeros(d);xi=1.;directions=[];intercepts=[]
    for iteration in range(1000):
        loss,direction,intercept=oracle(w,pos,neg)
        # A 1e-5 loss gap can become 1e-2 objective error at C=1000.
        # Tighten internally while preserving the protocol's error ceiling.
        if iteration and loss-xi<=1e-7:
            # Recover nonnegative QP multipliers from binding constraints.
            D=np.asarray(directions);a=np.asarray(intercepts)
            slack=D@w+xi-a;binding=slack<1e-7
            active_D=D[binding];active_a=a[binding]
            if xi<1e-5:
                active_D=np.vstack([active_D,np.zeros(d)])
                active_a=np.r_[active_a,0.]
            matrix=np.vstack([active_D.T,np.ones(len(active_D))])
            target=np.r_[w,c]
            dual,_=nnls(matrix,target,maxiter=max(1000,10*len(matrix.T)))
            # Scaling makes the multipliers feasible even after NNLS roundoff.
            if dual.sum()>c:dual*=c/dual.sum()
            dual_value=float(active_a@dual-.5*np.linalg.norm(active_D.T@dual)**2)
            primal_value=float(.5*w@w+c*loss)
            gap=primal_value-dual_value
            if gap>max(2e-4,2e-5*max(1.,primal_value)):
                raise RuntimeError(f'pAUC primal-dual gap too large: {gap}')
            return {'coef':w,'solver':{'iterations':iteration,'loss':loss,'xi':float(xi),
                    'surrogate_gap':float(loss-xi),'primal':primal_value,'dual':dual_value,
                    'primal_dual_gap':gap,'beta':.1,'negative_k':len(neg)//10}}
        directions.append(direction);intercepts.append(intercept)
        D=np.asarray(directions);a=np.asarray(intercepts)
        start=np.r_[w,max(xi,loss)]
        normalization=max(1.,c)
        def objective(z):return (.5*z[:-1]@z[:-1]+c*z[-1])/normalization
        def gradient(z):return np.r_[z[:-1],c]/normalization
        A=np.column_stack([D,np.ones(len(D))])
        result=minimize(objective,start,jac=gradient,method='SLSQP',
            bounds=[(None,None)]*d+[(0,None)],
            constraints=[{'type':'ineq','fun':lambda z:A@z-a,'jac':lambda z:A}],
            options={'ftol':1e-11,'maxiter':3000})
        if not result.success or np.min(A@result.x-a)<-1e-6:
            raise RuntimeError(f'pAUC QP failed: {result.message}')
        w=result.x[:-1];xi=result.x[-1]
    raise RuntimeError('pAUC cutting planes failed to converge')

def fit(name,param,x,y,seed=SEEDS[0]):
    started=time.monotonic();mean=x.mean(0);scale=np.maximum(x.std(0),1e-6);z=(x-mean)/scale
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        if name==PAUC:model=fit_pauc(z,y,param['C'])
        elif name in [BASE,COST]:
            weights={0:param['safe_weight'],1:1.} if name==COST else None
            model=LinearSVC(C=param['C'],class_weight=weights,dual=False,max_iter=20000,random_state=seed).fit(z,y)
        elif name in [RBF,CTRL]:
            model=SVC(C=param['C'],kernel='rbf' if name==RBF else 'linear',gamma=param.get('gamma','scale'),
                      probability=False,tol=1e-5,max_iter=100000000,random_state=seed).fit(z,y)
        else:raise ValueError(name)
    artifact={'mean':mean,'scale':scale,'model':model,'name':name,'param':param}
    return artifact,{'seconds':time.monotonic()-started,'solver':model.get('solver') if isinstance(model,dict) else None}

def score(artifact,x):
    z=(np.asarray(x)-artifact['mean'])/artifact['scale'];m=artifact['model']
    s=z@m['coef'] if isinstance(m,dict) else m.decision_function(z)
    if not np.isfinite(s).all():raise ValueError('Nonfinite scores')
    return np.asarray(s,dtype=float)

def threshold(s):
    if s.shape!=(400,) or not np.isfinite(s).all():raise ValueError('400 safe calibration scores required')
    return float(np.sort(s)[380])

def verify_lock():
    seal=json.loads((OUT/'source-lock.json').read_text())
    for path,sha in seal['sources'].items():assert digest(ROOT/path)==sha,path
    for path,sha in seal['inputs'].items():assert digest(path)==sha,path
    return seal
