# Interfaces and numerical conventions

## NumPy ECT/MPI adapter

`mesa.prepare(sensitivity, shape, sigma=1.0, sensitivity_exponent=1.0)` prepares
one real operator. Shape is `(height,width)`; sensitivity is `(m,height*width)`
in row-major order. `mesa.trajectory(y,state,checkpoints=(0,K))` returns field
vectors and regularization diagnostics. `mesa.readout(field,y,state,power)` fits
the output; `power=None` disables Hill but still refits the scalar gain.
`mesa.reconstruct(y,state,updates=K,hill_power=p)` combines these stages.

The NumPy reference adapter clips the recovered field to nonnegative values.
Its signed Hill function does not make the preceding field estimator signed.
Use the EIT biomedical PyTorch adapter for the validated signed configuration.

Candidate ratios are a fixed 61-point log grid from 1e-5 to 1e2. Ridge is
`ratio*trace(covariance)/m + 1e-8`, with the operator energy normalization retained.
The covariance trajectory uses normalized measurements; each frame starts from
the same prepared initial variance. Updates use geometric damping 0.5, a
leverage floor 1e-12, and variance clipping [1e-5,1e5]. The Otsu histogram has
128 bins and a threshold floor 1/128. A signed, unconstrained scalar gain is
fitted using the supplied original sensing operator, with denominator floor 1e-30.

Validate calibration conventions before using external measurements. This API
does not normalize differential capacitances, remove background, or whiten rows
automatically. Rescaling measurements and operators inconsistently changes the
inverse problem. MPI preprocessing (background correction, frequency selection,
real/imaginary stacking and operator normalization) is outside the core solver.

## EIT PyTorch reference adapter

`from mesa import torch_backend` imports the optional PyTorch backend.
`torch_backend.prepare(context)` and `torch_backend.reconstruct(y,state)` match
the manuscript's operator-preparation / one-frame interface. Context must contain
only `dimension`, `sensitivity`, `points`, `grid_shape`, and `config`.

Supported dimensions: `2d` (simulated EIT), `practical` (deformable tactile EIT),
and `biomedical` (signed time-difference EIT). Point rows follow sensitivity column
order. For `2d`/`practical`, columns are `[x,y,z,row_index,column_index]`; for
`biomedical`, columns are `[x,y,row_index,column_index]`. Reference code shifts
index minima to zero and uses their extrema to build the active grid. Do not
reshape column vectors without applying these indices.

Configs have one `scales` entry and `noise_selection` of `loocv` or `loocv_1se`.
Simulation uses `power`; real EIT uses `readout='hill'` and `readout_power`.
Biomedical also uses `nonnegative=false`. LOOCV-1SE is a channel-dispersion
heuristic, not a confidence-coverage assertion when channels are correlated.

For a full rectangular or explicitly indexed masked simulation grid, use
`torch_backend.prepare_operator(S,shape,grid_indices=indices,device='cpu',...)`.
Indices are zero-based `(row,column)` pairs in sensitivity-column order.
For real EIT, use the full context interface and the corresponding fixed config.

The EIT arithmetic intentionally retains float32 smoothing and field/output
rounding with float64 measurement-space solves. The NumPy adapter uses float64
throughout. EIT preparation disables TF32 and sets four Torch CPU threads, as in
the reference implementation. Reference preparation includes spectral/prior
quantities for diagnostic/intervention paths; no latency improvement is claimed.
Nonzero frames must have a nonzero observable component under the prepared
operator. Zero measurements return zeros. No gradients or CUDA Graphs are used.

Outputs retain physical/global amplitude calibration. Peak normalization in the
demo is only for display. Sparse operators and 3D reconstruction are not covered
by this compact release.
