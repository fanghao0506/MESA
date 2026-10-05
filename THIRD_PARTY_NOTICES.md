# Third-party notices

The MIT license in this repository applies to the MESA implementation,
documentation, and included author-generated simulation/reference arrays.
No external benchmark code, raw ECT/MPI data, real EIT measurements,
photographs, IEEE templates, or vendored third-party libraries are included.

NumPy, SciPy, PyTorch and Matplotlib are installed separately and retain their
respective licenses. PyEIT was used in the original simulation generation;
its source code is not redistributed here. See `docs/DATA.md` for provenance.

The mathematical building blocks are established methods: linear LOOCV,
Champagne/SBL-style response-over-leverage covariance fitting, Otsu thresholding,
Hill readout, and least-squares scalar gain. The package does not claim those
individual building blocks as new algorithms. Literature attribution is given
in the accompanying MESA manuscript; this code implements its specified pipeline.
