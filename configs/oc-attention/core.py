"""OC-A2: matched attention experiments, numerical encodings and controls."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import hashlib,importlib.util,json,random,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/oc-attention';CACHE=ROOT/'cache/oc-attention'
ORIGIN=ROOT.parents[1];H2=ORIGIN/'cache/oc4c-worktree';H3=ORIGIN/'cache/oc4r-worktree';S1=ORIGIN/'cache/svm-worktree'
ATT='OC-4-4-1';TRANS='OC-4-4-2';MLP='OC-4-2-1'
GATE=ATT+'-1';PLE_A=ATT+'-2';PAUC=ATT+'-3';FT=TRANS+'-1';PLE_M=MLP+'-1'
UNGATED='ungated-pooling';LIN_A='linear-encoded-attention';LIN_M='linear-encoded-mlp'
NAMES=[ATT,TRANS,MLP,GATE,UNGATED,PLE_A,LIN_A,PLE_M,LIN_M,PAUC,FT]
SEEDS=[20261104,20261105,20261106];LRS=[.003,.001,.0003];STEPS=240
# Five main variants plus mechanism controls, differential encoding gain and
# a common SVM comparison for every main variant: 13 predeclared contrasts.
CONTRASTS=[(GATE,[(GATE,1),(ATT,-1)]),(PLE_A,[(PLE_A,1),(ATT,-1)]),
 (PLE_M,[(PLE_M,1),(LIN_M,-1)]),(PAUC,[(PAUC,1),(ATT,-1)]),(FT,[(FT,1),(TRANS,-1)]),
 ('gate-vs-ungated',[(GATE,1),(UNGATED,-1)]),('encoding-vs-linear-interface',[(PLE_A,1),(LIN_A,-1)]),
 ('encoding-attention-specific',[(PLE_A,1),(LIN_A,-1),(PLE_M,-1),(LIN_M,1)])]
CONTRASTS += [(name+'-vs-SVM',[(name,1),('SVM',-1)]) for name in [GATE,PLE_A,PLE_M,PAUC,FT]]
CONTRASTS += [('ft-vs-attention',[(FT,1),(ATT,-1)]),('attention-vs-SVM',[(ATT,1),('SVM',-1)])]

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def setup(seed):
    torch.set_num_threads(2);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)
def vendors():
    for name in ['libauc','rtdl']:
        path=str(ROOT/'cache/vendor'/name)
        if path not in sys.path:sys.path.insert(0,path)
def preprocessing(x):
    mean=x.mean(0);scale=np.maximum(x.std(0),1e-6);z=(x-mean)/scale
    knots=np.quantile(z,[0,.25,.5,.75,1],axis=0).T
    if not np.all(np.diff(knots,axis=1)>1e-7):raise ValueError('Non-distinct quantile knots; no silent bin fallback')
    return mean,scale,knots

class NumericalEncoding(nn.Module):
    def __init__(self,kind,knots):
        super().__init__();self.kind=kind
        self.register_buffer('knots',torch.tensor(knots,dtype=torch.float32))
    def forward(self,x):
        if self.kind=='scalar':return x.unsqueeze(-1)
        if self.kind=='linear':return x.unsqueeze(-1).expand(-1,-1,4)/4
        ratios=(x.unsqueeze(-1)-self.knots[:,:-1])/(self.knots[:,1:]-self.knots[:,:-1])
        # PLE: first/last intervals linearly extrapolate; inner ones are bounded.
        return torch.cat([ratios[:,:,:1].clamp(max=1),ratios[:,:,1:-1].clamp(0,1),ratios[:,:,-1:].clamp(min=0)],dim=-1)

class AttentionHead(nn.Module):
    def __init__(self,name,knots):
        super().__init__();self.name=name
        kind='ple' if name==PLE_A else 'linear' if name==LIN_A else 'scalar'
        self.encoding=NumericalEncoding(kind,knots);self.embed=nn.Linear(1 if kind=='scalar' else 4,16)
        self.layer=nn.Parameter(torch.randn(1,14,16)*.02)
        if name in [GATE,UNGATED]:
            self.value=nn.Linear(16,16);self.weight=nn.Linear(16,1,bias=False)
            self.gate=nn.Linear(16,16) if name==GATE else None
        else:
            self.query=nn.Parameter(torch.randn(1,1,16)*.02)
            self.pool=nn.MultiheadAttention(16,2,dropout=0,batch_first=True)
            self.encoder=nn.TransformerEncoderLayer(16,2,dim_feedforward=32,dropout=0,activation='gelu',batch_first=True,norm_first=True) if name==TRANS else nn.Identity()
        self.output=nn.Linear(16,1)
    def forward(self,x):
        z=self.embed(self.encoding(x))+self.layer
        if self.name in [GATE,UNGATED]:
            h=torch.tanh(self.value(z))
            if self.gate is not None:h=h*torch.sigmoid(self.gate(z))
            a=torch.softmax(self.weight(h),dim=1);pooled=(a*z).sum(1)
        else:
            z=self.encoder(z);pooled,_=self.pool(self.query.expand(len(x),-1,-1),z,z,need_weights=False);pooled=pooled[:,0]
        return self.output(pooled)

class MLPHead(nn.Module):
    def __init__(self,name,knots):
        super().__init__();kind='ple' if name==PLE_M else 'linear' if name==LIN_M else 'scalar'
        self.encoding=NumericalEncoding(kind,knots)
        self.net=nn.Sequential(nn.Linear(14*(1 if kind=='scalar' else 4),16),nn.GELU(),nn.Linear(16,1))
    def forward(self,x):return self.net(self.encoding(x).flatten(1))

class FTHead(nn.Module):
    def __init__(self):
        super().__init__();vendors();import rtdl
        self.model=rtdl.FTTransformer.make_baseline(n_num_features=14,cat_cardinalities=None,d_token=16,n_blocks=1,
            attention_dropout=0.,ffn_d_hidden=32,ffn_dropout=0.,residual_dropout=0.,d_out=1)
    def forward(self,x):return self.model(x,None)

def make(name,knots):
    if name==FT:return FTHead()
    if name in [MLP,PLE_M,LIN_M]:return MLPHead(name,knots)
    return AttentionHead(ATT if name==PAUC else name,knots)

def fit(name,lr,seed,x,y,steps=STEPS,device='cuda'):
    setup(seed);started=time.monotonic();mean,scale,knots=preprocessing(x)
    model=make(name,knots).to(device);tx=torch.tensor((x-mean)/scale,dtype=torch.float32,device=device)
    ty=torch.tensor(y,dtype=torch.float32,device=device).reshape(-1,1);indices=torch.arange(len(y),device=device)
    if name==PAUC:
        vendors();from libauc.losses import pAUC_CVaR_Loss
        from libauc.optimizers import SOPA
        criterion=pAUC_CVaR_Loss(data_len=len(y),pos_len=int((y==1).sum()),beta=.1,eta=1.,margin=1.,surr_loss='squared_hinge',device=torch.device(device))
        optimizer=SOPA(model.parameters(),mode='adam',lr=lr,weight_decay=.01,epoch_decay=0,clip_value=1.,device=torch.device(device))
    else:criterion=None;optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
    first_loss=None;last_loss=None
    for step in range(steps):
        model.train();optimizer.zero_grad(set_to_none=True);pred=model(tx)
        loss=criterion(pred,ty,indices) if criterion is not None else nn.functional.binary_cross_entropy_with_logits(pred,ty)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('Nonfinite gradient')
        optimizer.step();last_loss=float(loss.detach())
        if first_loss is None:first_loss=last_loss
    if any(not torch.isfinite(p).all() for p in model.parameters()):raise RuntimeError('Nonfinite trained parameter')
    model=model.cpu().eval();params=sum(p.numel() for p in model.parameters())
    artifact={'name':name,'lr':lr,'seed':seed,'mean':mean,'scale':scale,'knots':knots,'state':model.state_dict(),
              'auxiliary_state':None if criterion is None else criterion.u_pos.detach().cpu(),
              'optimizer_state':optimizer.state_dict()}
    dual=None if criterion is None else {'min':float(criterion.u_pos.min()),'max':float(criterion.u_pos.max()),'updated':bool(torch.any(criterion.u_pos!=0))}
    if criterion is not None and not dual['updated']:raise RuntimeError('LibAUC dual states did not update')
    return artifact,{'seconds':time.monotonic()-started,'parameters':params,'steps':steps,'first_loss':first_loss,'last_loss':last_loss,
        'optimizer':'LibAUC-SOPA-adam' if name==PAUC else 'AdamW','libauc_dual':dual,'full_batch_size':len(y)}

def score(artifact,x):
    setup(artifact['seed']);model=make(artifact['name'],artifact['knots']).cpu().eval();model.load_state_dict(artifact['state'])
    with torch.no_grad():s=model(torch.tensor((x-artifact['mean'])/artifact['scale'],dtype=torch.float32)).numpy().reshape(-1).astype(float)
    if not np.isfinite(s).all():raise ValueError('Nonfinite score')
    return s
def threshold(s):
    if s.shape!=(400,) or not np.isfinite(s).all():raise ValueError('400 safe calibration scores required')
    return float(np.sort(s)[380])
def verify_lock():
    lock=json.loads((OUT/'source-lock.json').read_text())
    for group in ['sources','inputs','vendor_files']:
        for path,sha in lock[group].items():assert digest(Path(path) if Path(path).is_absolute() else ROOT/path)==sha,path
    return lock
