"""Manuscript EIT simulation solver, with local imports replacing history paths."""
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
import torch
from . import _components as base

def smooth(fields, ij, shape, sigma, dtype=np.float32):
    arr = np.asarray(fields, dtype=dtype)
    single = arr.ndim == 1
    if single: arr = arr[None, :]
    grid = np.zeros((len(arr),)+tuple(shape), dtype=dtype)
    index = (slice(None),)+tuple(ij.T)
    grid[index] = arr
    grid = gaussian_filter(grid, sigma=(0,)+tuple(sigma), mode='reflect')
    result = grid[index]
    return result[0] if single else result


def prior_diagonal(variance, ij, shape, sigma):
    grid = np.zeros(shape, dtype=float)
    grid[tuple(ij.T)] = variance
    for axis, size in enumerate(shape):
        kernel = gaussian_filter1d(np.eye(size), float(sigma[axis]), axis=0, mode='reflect')
        grid = np.moveaxis(np.tensordot(kernel**2, grid, axes=(1, axis)), 0, axis)
    return grid[tuple(ij.T)]


def geometry(context, cfg):
    ij, shape = base.grid_geometry(context['points'], context['dimension'])
    xyz = np.asarray(context['points'])[:, :3].astype(float)
    design = np.column_stack([ij, np.ones(len(ij))])
    affine = np.linalg.lstsq(design, xyz, rcond=None)[0]
    residual = float(np.max(np.abs(design@affine-xyz)))
    axes = affine[:-1]
    affine_spacing = np.linalg.norm(axes, axis=1)
    unit = axes/affine_spacing[:, None]
    orthogonal_error = float(np.max(np.abs(unit@unit.T-np.eye(len(shape)))))
    # The tactile coordinates describe a warped surface. A single affine map
    # is not exact there. Use an explicitly approximate global grid metric,
    # estimated only from physical lengths of indexed adjacent vertices.
    ids = np.full(shape, -1, dtype=int); ids[tuple(ij.T)] = np.arange(len(ij))
    spacings=[]; distributions=[]
    for axis in range(len(shape)):
        a=[slice(None)]*len(shape); b=a.copy(); a[axis]=slice(None,-1); b[axis]=slice(1,None)
        ia,ib=ids[tuple(a)],ids[tuple(b)]; valid=(ia>=0)&(ib>=0)
        lengths=np.linalg.norm(xyz[ia[valid]]-xyz[ib[valid]],axis=1)
        lengths=lengths[lengths>0]
        if not len(lengths): raise ValueError('No positive physical edge lengths')
        spacings.append(float(np.median(lengths)))
        distributions.append(np.quantile(lengths,[0,.25,.5,.75,1]).tolist())
    spacing=np.asarray(spacings)
    if cfg.get('geometry', 'physical') == 'legacy':
        factors = np.asarray(cfg.get('axis_factors', [1]*len(shape)), dtype=float)
    else:
        factors = np.exp(np.log(spacing).mean())/spacing
    sigmas = [float(s)*factors for s in cfg['scales']]
    return ij, shape, sigmas, dict(spacing=spacing.tolist(), sigmas=[s.tolist() for s in sigmas],
                                  affine_residual=residual, orthogonal_error=orthogonal_error,
                                  neighbor_length_quantiles=distributions,
                                  metric='median adjacent physical lengths; globally separable approximation')


def prepare(context):
    cfg = dict(context['config'])
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    s = np.asarray(context['sensitivity'], dtype=np.float32)
    ij, shape, sigmas, geom = geometry(context, cfg)
    h = np.concatenate([smooth(s, ij, shape, sig) for sig in sigmas], axis=1)
    h64 = h.astype(float)
    top = float(np.linalg.eigvalsh(h64@h64.T)[-1]); del h64
    ht = torch.tensor((h/np.sqrt(top)).astype(np.float32), device=device, dtype=torch.float64); del h
    energy = ht.square().sum(0)
    initial = energy.clamp(min=.001*float(energy.max())).pow(-float(cfg['sensitivity_exponent']))
    initial /= initial.mean()
    cov = (ht*initial[None, :])@ht.T
    ev, u = torch.linalg.eigh(cov)
    prior = sum(prior_diagonal(g, ij, shape, sig) for g, sig in
                zip(initial.cpu().numpy().reshape(len(sigmas), -1), sigmas))/top
    state = dict(config=cfg, h=ht, initial=initial, top=top, ij=ij, shape=shape,
                 sigmas=sigmas, geometry=geom, sensitivity=s, n=s.shape[1], device=device,
                 dimension=context['dimension'], eigenvalues=ev.clamp(min=0), eigenvectors=u,
                 prior_standard_deviation=np.sqrt(np.maximum(prior, max(float(prior.max())*1e-12, 1e-30))),
                 identity=torch.eye(s.shape[0], device=device, dtype=torch.float64), last_diagnostics={})
    # Same exact fixed-prior spectral shortcut is available in both dimensions.
    coefficients = (u.T@ht*initial[None, :]).cpu().numpy()
    basis = sum(smooth(g, ij, shape, sig) for g, sig in zip(np.split(coefficients, len(sigmas), axis=1), sigmas))
    state['spectral_decoder'] = torch.tensor((basis.T/np.sqrt(top)).astype(np.float32), device=device)
    return state


def reset(state):
    state['last_diagnostics'] = {}


@torch.no_grad()
def trajectory(voltage, state, checkpoints=(0,1,2,4)):
    y = np.asarray(voltage, dtype=float)
    if y.shape != (state['h'].shape[0],) or not np.isfinite(y).all():
        raise ValueError('One finite voltage vector required')
    amplitude = float(np.linalg.norm(y))
    if amplitude == 0:
        state['last_diagnostics'] = dict(noise_ratio=0., selection_boundary=False)
        return {k: np.zeros(state['n'], np.float32) for k in checkpoints}
    b = torch.tensor(y/amplitude, device=state['device'], dtype=torch.float64)
    ratio, lam, z, boundary = base.choose_noise(b, state)
    h = state['h']; gamma = state['initial'].clone(); result = {}
    maximum = max(checkpoints)
    if maximum == 0:
        image = (state['spectral_decoder']@(z/(state['eigenvalues']+lam)).float()).cpu().numpy()*amplitude
        result[0] = np.maximum(image, 0).astype(np.float32)
    else:
        for step in range(maximum+1):
            covariance = (h*gamma[None, :])@h.T
            lam = ratio*torch.trace(covariance)/h.shape[0]+1e-8
            factor = torch.linalg.cholesky(covariance+lam*state['identity'])
            response = h.T@torch.cholesky_solve(b[:, None], factor)[:, 0]
            if step in checkpoints:
                coeff = (gamma*response).cpu().numpy().reshape(len(state['sigmas']), state['n'])
                image = sum(smooth(g, state['ij'], state['shape'], sig) for g, sig in zip(coeff, state['sigmas']))
                result[step] = np.maximum(image*(amplitude/np.sqrt(state['top'])), 0).astype(np.float32)
            if step == maximum: break
            whitened = torch.linalg.solve_triangular(factor, h, upper=False)
            leverage = whitened.square().sum(0).clamp(min=1e-12)
            update = gamma*response.abs()/leverage.sqrt()
            update = (update/update.mean()).clamp(min=1e-5, max=1e5)
            gamma = gamma.sqrt()*update.sqrt()
            gamma = (gamma/gamma.mean()).clamp(min=1e-5, max=1e5)
    state['last_diagnostics'] = dict(noise_ratio=float(ratio), selection_boundary=bool(boundary))
    if not all(np.isfinite(a).all() for a in result.values()): raise FloatingPointError('Nonfinite reconstruction')
    return result


def readout(image, voltage, state, standardize=None, power=None):
    cfg = state['config']
    standardize = cfg.get('standardize', False) if standardize is None else standardize
    power = cfg.get('power', 1) if power is None else power
    score = image/state['prior_standard_deviation'] if standardize else image.copy()
    if power:
        score = base.contrast_readout(score, dict(readout='hill', readout_power=power))
    fitted = state['sensitivity']@score
    gain = float(fitted@np.asarray(voltage, dtype=float)/max(float(fitted@fitted), 1e-30))
    return (score*gain).astype(np.float32)


def reconstruct(voltage, state):
    k = int(state['config']['iterations'])
    return readout(trajectory(voltage, state, (k,))[k], voltage, state)
