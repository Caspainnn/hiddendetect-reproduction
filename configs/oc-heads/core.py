"""Eight heads, identical 14-F inputs, predeclared grids and training budgets."""
import hashlib,json,random,time
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/oc-heads'
SEEDS=[20261104,20261105,20261106]
IDS=['OC-4-1-1','OC-4-2-1','OC-4-3-1','OC-4-4-1','OC-4-1-2','OC-4-1-3','OC-4-3-2','OC-4-4-2']
GRIDS={IDS[0]:[.1,1.,10.],IDS[1]:[.003,.001,.0003],IDS[2]:[1,2,3],IDS[3]:[.003,.001,.0003],IDS[4]:[.1,1.,10.],IDS[5]:[.1,1.,10.],IDS[6]:[2,4,6],IDS[7]:[.003,.001,.0003]}
NEURAL={IDS[1],IDS[3],IDS[7]}
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with Path(p).open(encoding='utf-8') as f:return [json.loads(line) for line in f if line.strip()]
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode())
def pauc(y,s):return float(roc_auc_score(y,s,max_fpr=.1))
def baseline(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=14 or not np.isfinite(x).all():raise ValueError('Expected 14 original F per input')
    return np.trapz(x,axis=1)
def threshold(s):
    s=np.asarray(s)
    if s.shape!=(400,) or not np.isfinite(s).all():raise ValueError('Expected 400 safe calibration scores')
    return float(np.sort(s)[380])
def scaler(x):return x.mean(0),np.maximum(x.std(0),1e-6)
def make_neural(name):
    import torch
    from torch import nn
    if name==IDS[1]:return nn.Sequential(nn.Linear(14,16),nn.GELU(),nn.Linear(16,1))
    class AttentionHead(nn.Module):
        def __init__(self):
            super().__init__();self.embed=nn.Linear(1,16);self.layer=nn.Parameter(torch.randn(1,14,16)*.02)
            self.query=nn.Parameter(torch.randn(1,1,16)*.02);self.pool=nn.MultiheadAttention(16,2,dropout=0,batch_first=True)
            self.encoder=nn.TransformerEncoderLayer(16,2,dim_feedforward=32,dropout=0,activation='gelu',batch_first=True,norm_first=True) if name==IDS[7] else nn.Identity()
            self.output=nn.Linear(16,1)
        def forward(self,x):
            z=self.encoder(self.embed(x.unsqueeze(-1))+self.layer)
            pooled,_=self.pool(self.query.expand(len(x),-1,-1),z,z,need_weights=False)
            return self.output(pooled[:,0])
    return AttentionHead()
def fit(name,param,seed,x,y):
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    mean,scale=scaler(x);started=time.monotonic()
    if name in NEURAL:
        torch.cuda.manual_seed_all(seed);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
        torch.use_deterministic_algorithms(True)
        model=make_neural(name).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=param,weight_decay=.01)
        tx=torch.tensor((x-mean)/scale,dtype=torch.float32,device='cuda');ty=torch.tensor(y,dtype=torch.float32,device='cuda').reshape(-1,1)
        generator=torch.Generator(device='cuda').manual_seed(seed)
        for epoch in range(120):
            model.train();order=torch.randperm(len(tx),generator=generator,device='cuda')
            for ix in order.split(256):
                optimizer.zero_grad(set_to_none=True);loss=torch.nn.functional.binary_cross_entropy_with_logits(model(tx[ix]),ty[ix]);loss.backward();optimizer.step()
        model=model.cpu().eval();params=sum(p.numel() for p in model.parameters())
        artifact={'kind':'torch','name':name,'mean':mean,'scale':scale,'state':model.state_dict()}
    else:
        from sklearn.linear_model import LogisticRegression,Ridge
        from sklearn.svm import LinearSVC
        from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
        if name==IDS[0]:model=LogisticRegression(C=param,solver='liblinear',max_iter=5000,random_state=seed)
        elif name==IDS[4]:model=Ridge(alpha=param)
        elif name==IDS[5]:model=LinearSVC(C=param,dual=False,max_iter=20000,random_state=seed)
        elif name==IDS[2]:model=GradientBoostingClassifier(n_estimators=100,learning_rate=.05,max_depth=param,min_samples_leaf=20,random_state=seed)
        elif name==IDS[6]:model=RandomForestClassifier(n_estimators=200,max_depth=param,min_samples_leaf=10,n_jobs=2,random_state=seed)
        else:raise ValueError(name)
        if name in (IDS[2],IDS[6]):mean=np.zeros(14);scale=np.ones(14)
        model.fit((x-mean)/scale,y);params=None
        artifact={'kind':'sklearn','name':name,'mean':mean,'scale':scale,'model':model}
    return artifact,{'seconds':time.monotonic()-started,'parameters':params}
def score(artifact,x):
    x=(np.asarray(x)-artifact['mean'])/artifact['scale']
    if artifact['kind']=='torch':
        import torch
        model=make_neural(artifact['name']);model.load_state_dict(artifact['state']);model.eval()
        with torch.no_grad():return model(torch.tensor(x,dtype=torch.float32)).numpy().reshape(-1).astype(float)
    model=artifact['model']
    if hasattr(model,'decision_function'):return np.asarray(model.decision_function(x)).reshape(-1)
    if hasattr(model,'predict_proba'):return model.predict_proba(x)[:,list(model.classes_).index(1)]
    return np.asarray(model.predict(x)).reshape(-1)
