"""Signed/unsigned real-EIT reference solver with one triangular leverage solve."""
import numpy as np
import torch
from . import _components as base
prepare = base.prepare
reset = base.reset

@torch.no_grad()
def reconstruct(voltage,state):
    y=np.asarray(voltage,dtype=np.float64);h=state['h'];cfg=state['config'];device=state['device']
    if y.shape!=(h.shape[0],) or not np.isfinite(y).all():raise ValueError('One finite voltage required')
    amplitude=float(np.linalg.norm(y))
    if amplitude==0:return np.zeros(state['n'],dtype=np.float32)
    ratio,_,_,boundary=base.choose_noise(torch.tensor(y/amplitude,device=device),state)
    b=torch.tensor(y/amplitude,dtype=h.dtype,device=device);gamma=state['initial'].clone();damping=float(cfg.get('damping',.5));count=int(cfg['iterations']);floor=float(cfg.get('variance_floor',1e-5));change=0.
    for step in range(count+1):
        covariance=(h*gamma[None,:])@h.T;lam=ratio*torch.trace(covariance)/h.shape[0]+1e-8;factor=torch.linalg.cholesky(covariance+lam*state['identity']);solved=torch.cholesky_solve(b[:,None],factor)[:,0];response=h.T@solved;coefficients=gamma*response
        if step==count:break
        whitened=torch.linalg.solve_triangular(factor,h,upper=False)
        leverage=torch.clamp(torch.sum(whitened*whitened,dim=0),min=1e-12)
        update=gamma*torch.abs(response)/torch.sqrt(leverage);update=torch.clamp(update/update.mean(),min=floor,max=1/floor);next_gamma=gamma.pow(1-damping)*update.pow(damping);next_gamma=torch.clamp(next_gamma/next_gamma.mean(),min=floor,max=1/floor);change=float(torch.linalg.vector_norm(next_gamma-gamma)/torch.linalg.vector_norm(gamma));gamma=next_gamma
    blocks=coefficients.cpu().numpy().reshape(len(state['scales']),state['n']);image=sum(base.smooth_fields(block,state['ij'],state['shape'],scale) for block,scale in zip(blocks,state['scales']));image=image*(amplitude/np.sqrt(state['top']))
    if bool(cfg.get('nonnegative',state['dimension']!='biomedical')):image=np.maximum(image,0)
    if cfg.get('readout','none')!='none':
        image=base.contrast_readout(image,cfg);fitted=state['sensitivity']@image;gain=float(fitted@y/max(float(fitted@fitted),1e-30));image=image*gain
    state['last_diagnostics']={'iterations':count,'gamma_change':change,'noise_ratio':ratio,'max_gamma':float(gamma.max()),'min_gamma':float(gamma.min()),'basis_scales':state['scales'],'noise_selection':cfg['noise_selection'],'selection_boundary':boundary,'leverage_kernel':'one triangular solve'}
    if not np.isfinite(image).all():raise FloatingPointError('Nonfinite field')
    return image.astype(np.float32)
