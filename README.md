# MESA

**Measurement-Driven Field–Readout Reconstruction for Inverse Imaging**

MESA is a training-data-free reconstruction framework. A linear or linearized
sensing operator and one current measurement vector determine the reconstruction:
linear LOOCV selects regularization; covariance updates estimate the field; a
histogram-adapted Hill readout forms contrast; a measurement-domain scalar fits
global amplitude. No target image or input SNR enters reconstruction.

This small release contains the core implementation, fixed manuscript settings,
and **three author-generated nonlinear EIT examples**: a circle, an L-shaped target,
and a ring. It does not contain all experiments or raw measurement datasets.

## Installation and first reconstruction

Python 3.10 or newer. From the extracted repository directory:

```bash
python -m pip install -e ".[torch,demo]"
python examples/demo_eit.py --device cpu --output outputs/eit_demo
```

The demo writes a comparison figure, reconstructed vectors, and diagnostics.
For NVIDIA GPUs, install a PyTorch build suitable for your driver first, then:

```bash
python examples/demo_eit.py --device cuda --output outputs/eit_cuda
```

Only NumPy and SciPy are required for the ECT/MPI adapter. An alternate CPU-only
demonstration can run without PyTorch:

```bash
python -m pip install -e ".[demo]"
python examples/demo_eit.py --backend numpy --output outputs/eit_numpy
```

The NumPy adapter uses float64 and a full rectangular grid. Its EIT demonstration
illustrates the shared mechanism; **use the PyTorch backend to check the manuscript
EIT reference outputs**. The original 19.10-ms EIT timing was measured on an RTX
4090 Laptop GPU with four CPU threads and is not a CPU timing or a portability
guarantee. CUDA Graphs are not used.

## Use with your own operator

```python
import mesa

# A: (m,n) real sensing matrix; y: (m,) current measurement vector.
# Full rectangular grid, row-major column order.
state = mesa.prepare(A, (height, width), sigma=0.5, sensitivity_exponent=0)
x = mesa.reconstruct(y, state, updates=2, hill_power=6)
```

Use fixed settings from `configs/` appropriate to the measurement protocol;
settings are not claimed to transfer unchanged between modalities. Complex MPI
system matrices must first be converted consistently to a real representation;
do not silently discard their imaginary parts. See [docs/API.md](docs/API.md).

For an external NPZ containing `operator`, `measurements`, and `image_shape`:

```bash
python examples/reconstruct_npz.py --input input.npz --config configs/mpi_controlled.json --output outputs/mpi.npz
```

## Contents and validation

- `mesa/`: standalone NumPy and PyTorch implementations; no research-folder imports.
- `configs/`: manuscript settings for EIT FEM, tactile and biomedical EIT, ECT, MPI.
- `data/eit_examples.npz`: one operator and three noisy measurements with display targets.
- `examples/`: reconstruction and plotting entry points.
- `tests/`: explicit leave-one-channel-out validation, scalar-fit and framewise
  invariants, and regression against three frozen EIT predictions.
- `docs/`: API, numerical conventions, sample provenance, external dataset sources.

```bash
python -m unittest discover -s tests -v
```

Without PyTorch, EIT-specific tests are reported as skipped. A successful three-case
demo does not reproduce the aggregate manuscript tables or cross-modality statistics.
The reference fixture is used only by tests, never by reconstruction.

## License and citation

The MESA code, documentation, and author-generated example arrays are provided
under the [MIT license](LICENSE). External datasets retain their own terms and
are not redistributed here. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
and [docs/DATA.md](docs/DATA.md). Python dependencies retain their own licenses.

Please cite the accompanying manuscript by Hao Fang, Lin Li, Sihao Teng, Zhe Liu,
and Yunjie Yang, *MESA: Measurement-Driven Field–Readout Reconstruction for Inverse
Imaging*. Publication metadata should be updated when available; no article DOI
is assigned by this release. Machine-readable authorship is in `CITATION.cff`.

[中文说明](README_zh.md)
