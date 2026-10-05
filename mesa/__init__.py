"""MESA: measurement-driven field--readout reconstruction for inverse imaging.

NumPy is the default operator interface. Import mesa.torch_backend explicitly
for the manuscript EIT reference solver, which requires PyTorch.
"""
from .numpy_backend import MesaState, prepare, trajectory, readout
__version__ = "0.1.0"

def reconstruct(measurements, state, *, updates=2, hill_power=6.0):
    """Reconstruct on a complete rectangular grid with the NumPy adapter."""
    fields, _ = trajectory(measurements, state, checkpoints=(updates,))
    return readout(fields[updates], measurements, state, hill_power)
