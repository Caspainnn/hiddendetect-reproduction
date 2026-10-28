"""Full pooled-vector controls; no projection or test-fitted transformation."""
import numpy as np

def raw_matrix(cache,indices,regions):
    block=np.asarray(cache['P'][np.asarray(indices)][:,:,regions,:],dtype=np.float32)
    return block.reshape(len(indices),-1)/np.sqrt(block.shape[1]*block.shape[2])

def fit_ridge(x,y):
    x=np.asarray(x,dtype=np.float32); y=np.asarray(y,dtype=np.float32)*2-1; offset=float(y.mean())
    try:
        import torch
        gpu=torch.cuda.is_available()
    except ImportError: gpu=False
    if gpu:
        torch.backends.cuda.matmul.allow_tf32=False
        with torch.no_grad():
            tensor=torch.as_tensor(x,device='cuda'); mean=tensor.mean(0); centered=tensor-mean
            kernel=centered@centered.T; kernel.diagonal().add_(1.)
            alpha=torch.linalg.solve(kernel,torch.as_tensor(y-offset,device='cuda'))
            coef=(centered.T@alpha).cpu().numpy(); mean=mean.cpu().numpy()
        torch.cuda.empty_cache()
    else:
        mean=x.mean(0); centered=x-mean; kernel=centered@centered.T+np.eye(len(x),dtype=np.float32)
        coef=centered.T@np.linalg.solve(kernel,y-offset)
    return {'mean':mean,'coef':coef,'intercept':offset,'lambda':1.,'fit_device':'CUDA float32' if gpu else 'CPU float32'}

def ridge_scores(head,x): return (np.asarray(x,dtype=np.float32)-head['mean'])@head['coef']+head['intercept']
