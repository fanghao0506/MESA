"""Shared EIT arithmetic extracted from the manuscript reference implementation.

Public entry points restrict this implementation to a single spatial scale and
LOOCV or LOOCV-1SE. Array ordering is supplied by indexed geometry.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
import torch

def grid_geometry(points, dimension):
    columns=(3,4,5) if dimension=='3d' else ((2,3) if dimension=='biomedical' else (3,4))
    ij=np.rint(np.asarray(points)[:,columns]).astype(np.int64)
    ij-=ij.min(axis=0)
    shape=tuple(ij.max(axis=0)+1)
    if len(np.unique(ij,axis=0))!=len(ij):raise ValueError('Duplicate cells')
    return ij,shape


def smooth_fields(fields, ij, shape, sigma):
    """M G M^T with a symmetric reflected-grid Gaussian; masked holes remain absent."""
    fields=np.asarray(fields,dtype=np.float32)
    single=fields.ndim==1
    if single:fields=fields[None,:]
    grid=np.zeros((len(fields),)+shape,dtype=np.float32)
    index=(slice(None),)+tuple(ij.T)
    grid[index]=fields
    if sigma>0:grid=gaussian_filter(grid,sigma=(0,)+(float(sigma),)*len(shape),mode='reflect')
    result=grid[index]
    return result[0] if single else result


def prepare(context):
    cfg=dict(context.get('config',{}));torch.set_num_threads(int(cfg.get('cpu_threads',4)))
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device(cfg.get('device','cuda' if torch.cuda.is_available() else 'cpu'))
    s=np.asarray(context['sensitivity'],dtype=np.float32)
    ij,shape=grid_geometry(context['points'],context['dimension'])
    scales=tuple(float(v) for v in cfg.get('scales',[.75,1.5,3.0]))
    h=np.concatenate([smooth_fields(s,ij,shape,scale) for scale in scales],axis=1)
    # One common scalar change of units, not a mode-dependent residual metric.
    top=float(np.linalg.eigvalsh(h.astype(np.float64)@h.astype(np.float64).T)[-1])
    h=(h/np.sqrt(top)).astype(np.float32)
    # Automatically selected tiny lambda needs accurate covariance arithmetic.
    # fp32 at lambda/mean(eigenvalue)~1e-5 amplified float32 input rounding by 1%.
    precision=torch.float64 if cfg.get('noise_selection','fixed')!='fixed' else torch.float32
    ht=torch.tensor(h,device=device,dtype=precision)
    energy=torch.sum(ht*ht,dim=0)
    floor=float(cfg.get('sensitivity_floor',.001))*float(energy.max())
    exponent=float(cfg.get('sensitivity_exponent',.5))
    initial=torch.clamp(energy,min=floor).pow(-exponent)
    initial=initial/initial.mean()
    state={'h':ht,'initial':initial,'top':top,'ij':ij,'shape':shape,'scales':scales,'sensitivity':s,
            'n':s.shape[1],'dimension':context['dimension'],'config':cfg,'device':device,
            'identity':torch.eye(s.shape[0],device=device,dtype=precision),'last_diagnostics':{}}
    automatic=cfg.get('noise_selection','fixed')!='fixed'
    if automatic:
        covariance=(ht*initial[None,:])@ht.T
        eigenvalues,eigenvectors=torch.linalg.eigh(covariance.double())
        state['eigenvalues']=torch.clamp(eigenvalues,min=0)
        state['eigenvectors']=eigenvectors
        if int(cfg.get('iterations',16))==0:
            coefficients=(eigenvectors.T.to(ht.dtype)@ht*initial[None,:]).cpu().numpy()
            basis=sum(smooth_fields(block,ij,shape,scale) for block,scale in
                      zip(np.split(coefficients,len(scales),axis=1),scales))
            state['spectral_decoder']=torch.tensor((basis.T/np.sqrt(top)).astype(np.float32),device=device)
    if int(cfg.get('iterations',16))==0 and not automatic:
        covariance=(ht*initial[None,:])@ht.T
        lam=float(cfg.get('noise_ratio',.003))*torch.trace(covariance)/ht.shape[0]+1e-8
        factor=torch.linalg.cholesky(covariance+lam*state['identity'])
        coefficients=(torch.cholesky_solve(ht,factor)*initial[None,:]).cpu().numpy()
        decoder=sum(smooth_fields(block,ij,shape,scale) for block,scale in
                    zip(np.split(coefficients,len(scales),axis=1),scales))
        state['decoder']=torch.tensor((decoder.T/np.sqrt(top)).astype(np.float32),device=device)
    return state


def reset(state):state['last_diagnostics']={}


def contrast_readout(image, config):
    """Monotone amplitude readout, fitted on development only; not a new contour.

    Hill's threshold uses the current field histogram (Otsu), not target area.
    Strictly monotone variants preserve ranking and cannot invent correct edges.
    """
    mode=config.get('readout','power');power=float(config.get('readout_power',1.0))
    norm=float(np.max(np.abs(image)))
    if norm==0:return image.copy()
    values=np.abs(image)/norm
    if mode=='power':out=values**power
    elif mode=='hill':
        histogram,edges=np.histogram(values,bins=128,range=(0,1))
        centers=(edges[:-1]+edges[1:])/2;prob=histogram/max(histogram.sum(),1)
        weight=np.cumsum(prob);moment=np.cumsum(prob*centers)
        between=(moment[-1]*weight-moment)**2/np.maximum(weight*(1-weight),1e-12)
        threshold=max(float(centers[np.argmax(between)]),1/128)
        out=values**power/(values**power+threshold**power)
    else:raise ValueError('Unknown readout')
    return (np.sign(image)*out).astype(np.float32)


def choose_noise(voltage, state):
    """Exact linear LOOCV; optionally use the channel-dispersion 1SE heuristic."""
    d = state['eigenvalues']
    u = state['eigenvectors']
    z = u.T @ voltage.double()
    ratios = torch.logspace(-5, 2, 61, device=d.device, dtype=torch.float64)
    lambdas = ratios * d.mean() + 1e-8
    denominator = d[:, None] + lambdas[None, :]
    residual = u @ (z[:, None] * lambdas[None, :] / denominator)
    diagonal = u.square() @ (lambdas[None, :] / denominator)
    losses = (residual / torch.clamp(diagonal, min=1e-12)) ** 2
    criteria = torch.mean(losses, dim=0)
    index = int(torch.argmin(criteria))
    if state['config']['noise_selection'] == 'loocv_1se':
        margin = losses[:, index].std(unbiased=True) / np.sqrt(len(losses))
        index = int(torch.nonzero(criteria <= criteria[index] + margin)[-1, 0])
    return float(ratios[index]), lambdas[index], z, index in (0, len(ratios) - 1)
