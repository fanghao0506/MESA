"""Standalone EIT entry points; no dataset or research-folder dependencies."""
import numpy as np
from . import _eit_simulation, _eit_real

def prepare(context):
    """Prepare from operator, indexed points, geometry, and a fixed configuration.

    dimension is '2d', 'practical', or 'biomedical'. See docs/API.md for point
    columns. Neither measurements nor targets are accepted in this context.
    """
    if set(context) != {'dimension', 'sensitivity', 'points', 'grid_shape', 'config'}:
        raise ValueError('prepare accepts only operator, geometry, and configuration')
    cfg = context['config']
    if len(cfg['scales']) != 1 or cfg['noise_selection'] not in ('loocv', 'loocv_1se'):
        raise ValueError('Use one spatial scale and LOOCV or LOOCV-1SE')
    if context['dimension'] not in ('2d', 'practical', 'biomedical'):
        raise ValueError('This release supports the manuscript two-dimensional EIT protocols')
    s = np.asarray(context['sensitivity'])
    if np.iscomplexobj(s):
        raise ValueError('EIT reference operators must be real')
    if s.ndim != 2 or min(s.shape) == 0 or not np.isfinite(s).all() or not np.any(s):
        raise ValueError('sensitivity must be a finite nonzero (m,n) matrix')
    if int(cfg['iterations']) < 0:
        raise ValueError('iterations must be nonnegative')
    if not np.isfinite(cfg['scales'][0]) or cfg['scales'][0] < 0:
        raise ValueError('spatial scale must be finite and nonnegative')
    if not np.isfinite(cfg['sensitivity_exponent']):
        raise ValueError('sensitivity exponent must be finite')
    module = _eit_simulation if context['dimension'] == '2d' else _eit_real
    return module, module.prepare(context)

def reconstruct(measurements, state):
    """Return one amplitude-calibrated vector in the sensitivity column order."""
    if np.iscomplexobj(measurements):
        raise ValueError('EIT reference measurements must be real')
    return state[0].reconstruct(measurements, state[1])

def prepare_operator(sensitivity, image_shape, *, grid_indices=None, sigma=1.0,
                     sensitivity_exponent=1.0, updates=8, hill_power=1.5,
                     selector='loocv', device='cpu'):
    """Convenience adapter for rectangular or masked indexed EIT grids.

    Explicit grid_indices has shape (n,2), is zero-based, and follows sensitivity
    column order. With no indices, the full grid must use row-major ordering.
    """
    shape = tuple(int(v) for v in image_shape)
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError('image_shape must be two positive integers')
    if grid_indices is None:
        grid_indices = np.indices(shape).reshape(2, -1).T
    indices = np.asarray(grid_indices)
    if indices.shape != (np.asarray(sensitivity).shape[1], 2):
        raise ValueError('grid_indices must have shape (sensitivity.shape[1],2)')
    if not np.isfinite(indices).all() or not np.array_equal(indices, np.rint(indices)):
        raise ValueError('grid indices must be finite integers')
    if np.any(indices < 0) or np.any(indices >= np.asarray(shape)):
        raise ValueError('grid indices must lie inside image_shape')
    points = np.column_stack([indices[:, 1], indices[:, 0], np.zeros(len(indices)), indices + 1])
    cfg = dict(scales=[sigma], sensitivity_exponent=sensitivity_exponent,
               iterations=updates, power=hill_power, noise_selection=selector,
               geometry='legacy', standardize=False, device=device)
    return prepare(dict(dimension='2d', sensitivity=sensitivity, points=points,
                        grid_shape=np.asarray(shape), config=cfg))

def stages(measurements, state):
    """Expose initial field, adapted field, and output for an EIT simulation frame."""
    module, core = state
    if module is not _eit_simulation:
        raise ValueError('stage visualization is available for dimension=2d')
    k = int(core['config']['iterations'])
    fields = module.trajectory(measurements, core, (0, k))
    output = module.readout(fields[k], measurements, core)
    return fields[0], fields[k], output, dict(core['last_diagnostics'])
