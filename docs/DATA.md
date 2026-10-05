# Example data and external datasets

## Included: three author-generated nonlinear EIT examples

`data/eit_examples.npz`, loaded with `numpy.load(...,allow_pickle=False)`:

| Key | Shape | Meaning |
|---|---|---|
| `sensitivity` | 104 x 6400 | Normalized reconstruction operator |
| `points` | 6400 x 5 | Coordinates and indexed pixel ordering |
| `grid_shape` | 2 | 80 x 80 reconstruction grid |
| `measurements` | 3 x 104 | Stored noisy voltage frames |
| `truth` | 3 x 6400 | Target fields for display/evaluation only |
| `clean_measurements` | 3 x 104 | Original noise-free nonlinear FEM frames |
| `case_names` | 3 | circle_01, concave_L_01, ring_01 |
| `observation_indices` | 3 | Original observation rows 432,444,468 |
| `case_indices` | 3 | Original target rows 0,12,36 |
| `snr_db` | scalar | Nominal 60 dB, first noise realization |
| `voltage_scale` | scalar | Original voltage normalization scale |

These are existing simulation arrays, not newly generated replacement data.
Measurements were generated on a 128-cell-per-axis forward FEM mesh and inverted
with an 80-cell-per-axis reconstruction mesh, with point electrodes and additive
Gaussian noise. Original generation used PyEIT 1.2.4 plus SciPy sparse solves.
No PyEIT source or vendor installation is bundled. The example arrays and
reference prediction fixtures are author-generated and released under MIT.

Three fixed cases represent different shape families; they are demonstrations,
not a representative sample for aggregate accuracy claims. All targets and
expected predictions are excluded from the reconstruction API. SHA-256 hashes
and original indices are recorded in `provenance.json` and `MANIFEST.json`.

## External ECT data (not bundled)

- Dataset paper: Shi et al., *An Electrical Capacitance Tomography Dataset for
  Image Reconstruction Benchmarking*, Scientific Data (2026),
  <https://doi.org/10.1038/s41597-026-07433-7>.
- Dataset: <https://doi.org/10.57760/sciencedb.29527>.
- Author repository: <https://github.com/bupt-embodiedai/ECT-bench>.

The manuscript uses a 36 x 10201 sensitivity operator on a 101 x 101 native grid.
Its controlled targets and physical differential-capacitance records are distinct
protocols, with separate configs. Registered real-data evaluation also requires
the fixed mapping to the 100 x 100 label grid. This compact package does not
reproduce that registration, the 1800-case statistics, or dataset-provided LBP.
Follow the original data terms and acquisition/calibration conventions.

## External MPI matrix (not bundled)

- Knopp et al., *OpenMPIData: An Initiative for Freely Accessible Magnetic
  Particle Imaging Data*, Data in Brief, 28, 104971 (2020),
  <https://doi.org/10.1016/j.dib.2019.104971>.
- Data initiative and author documentation:
  <https://magneticparticleimaging.github.io/OpenMPIData.jl/latest>.
- Original calibration file used by the manuscript: `calibration_2.mdf`.

The manuscript's processed real operator is 64 x 361 on a 19 x 19 grid. The
reconstruction targets are generated concentration fields, not independently
scanned physical phantoms. This release does not bundle the calibration matrix
or its raw MDF file. The software package license of OpenMPIData.jl should not
be assumed to be the license of every dataset.

## Real EIT measurements (not bundled)

Tactile measurements come from the authors' manuscript *Flexible Electrical
Impedance Tomography for Tactile Sensing with Deformation Prior* (Teng et al.,
manuscript under review). Biomedical data are attributed to Fang et al.,
*Multifrequency Electrical Impedance Tomography Reconstruction With Multibranch
Attention Image Prior*, DOI <https://doi.org/10.1109/JIOT.2025.3624228>.
Photographs are qualitative references, not registered pixelwise ground truth.
This release provides solver configurations, but no photographs or real EIT data.
