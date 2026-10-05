"""Faithful CPU/NumPy implementation of the current single-scale MESA core.

This is the manuscript ECT/MPI adapter. It uses a complete rectangular grid
and float64 NumPy arithmetic; the EIT reference arithmetic is in torch_backend.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.linalg import solve_triangular


def _smooth_rows(a: np.ndarray, shape: tuple[int, int], sigma: float) -> np.ndarray:
    return gaussian_filter(a.reshape((-1,) + shape), sigma=(0, sigma, sigma), mode="reflect").reshape(a.shape)


def _otsu_hill(x: np.ndarray, power: float) -> np.ndarray:
    peak = float(np.max(np.abs(x)))
    if peak <= 0:
        return np.zeros_like(x)
    u = np.abs(x) / peak
    hist, edges = np.histogram(u, bins=128, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2
    prob = hist / max(int(hist.sum()), 1)
    weight = np.cumsum(prob)
    moment = np.cumsum(prob * centers)
    between = (moment[-1] * weight - moment) ** 2 / np.maximum(weight * (1 - weight), 1e-12)
    tau = max(float(centers[np.argmax(between)]), 1 / 128)
    out = u**power / (u**power + tau**power)
    return np.sign(x) * out


@dataclass
class MesaState:
    sensitivity: np.ndarray
    shape: tuple[int, int]
    sigma: float
    sensitivity_exponent: float
    h: np.ndarray
    top: float
    initial: np.ndarray
    eigvals: np.ndarray
    eigvecs: np.ndarray


def prepare(sensitivity: np.ndarray, shape: tuple[int, int], sigma: float = 1.0,
            sensitivity_exponent: float = 1.0) -> MesaState:
    if np.iscomplexobj(sensitivity):
        raise ValueError("convert complex operators consistently to a real representation first")
    if len(shape) != 2 or any(int(v) != v or v <= 0 for v in shape):
        raise ValueError("shape must contain two positive integers")
    shape = tuple(int(v) for v in shape)
    if not np.isfinite(sigma) or sigma < 0 or not np.isfinite(sensitivity_exponent):
        raise ValueError("sigma must be nonnegative and parameters must be finite")
    s = np.asarray(sensitivity, dtype=np.float64)
    if s.ndim != 2 or s.shape[1] != int(np.prod(shape)):
        raise ValueError("sensitivity must have shape (measurements, prod(shape))")
    if s.shape[0] == 0 or not np.isfinite(s).all():
        raise ValueError("operator must have at least one row and finite entries")
    h = _smooth_rows(s, shape, float(sigma))
    top = float(np.linalg.eigvalsh(h @ h.T)[-1])
    if not np.isfinite(top) or top <= 0:
        raise ValueError("operator has no positive energy")
    h = h / np.sqrt(top)
    energy = np.sum(h * h, axis=0)
    floor = 0.001 * float(energy.max())
    initial = np.maximum(energy, floor) ** (-float(sensitivity_exponent))
    initial /= initial.mean()
    cov = (h * initial[None, :]) @ h.T
    eigvals, eigvecs = np.linalg.eigh(cov)
    return MesaState(s, shape, float(sigma), float(sensitivity_exponent), h, top,
                     initial, np.maximum(eigvals, 0), eigvecs)


def choose_loocv(y_unit: np.ndarray, state: MesaState) -> tuple[float, bool]:
    d, u = state.eigvals, state.eigvecs
    z = u.T @ np.asarray(y_unit, dtype=np.float64)
    ratios = np.logspace(-5, 2, 61)
    lambdas = ratios * d.mean() + 1e-8
    den = d[:, None] + lambdas[None, :]
    residual = u @ (z[:, None] * lambdas[None, :] / den)
    diagonal = (u * u) @ (lambdas[None, :] / den)
    losses = (residual / np.maximum(diagonal, 1e-12)) ** 2
    criterion = losses.mean(axis=0)
    idx = int(np.argmin(criterion))
    return float(ratios[idx]), idx in (0, len(ratios) - 1)


def trajectory(y: np.ndarray, state: MesaState, checkpoints=(0, 4)) -> tuple[dict[int, np.ndarray], dict]:
    if np.iscomplexobj(y):
        raise ValueError("convert complex measurements consistently to a real representation first")
    y = np.asarray(y, dtype=np.float64)
    if y.shape != (state.sensitivity.shape[0],) or not np.isfinite(y).all():
        raise ValueError("one finite measurement vector of shape (m,) is required")
    checkpoints = tuple(checkpoints)
    if not checkpoints or any(int(k) != k or k < 0 for k in checkpoints):
        raise ValueError("checkpoints must contain nonnegative integer update counts")
    amp = float(np.linalg.norm(y))
    if amp == 0:
        return {int(k): np.zeros(state.sensitivity.shape[1]) for k in checkpoints}, {"ratio": 0.0, "boundary": False}
    b = y / amp
    if not np.any(state.h.T @ b):
        raise ValueError("nonzero measurement must have a nonzero observable component")
    ratio, boundary = choose_loocv(b, state)
    h = state.h
    gamma = state.initial.copy()
    out: dict[int, np.ndarray] = {}
    maximum = max(map(int, checkpoints))
    eye = np.eye(h.shape[0])
    for step in range(maximum + 1):
        cov = (h * gamma[None, :]) @ h.T
        lam = ratio * np.trace(cov) / h.shape[0] + 1e-8
        chol = np.linalg.cholesky(cov + lam * eye)
        solved = np.linalg.solve(chol.T, np.linalg.solve(chol, b))
        response = h.T @ solved
        if step in checkpoints:
            coeff = gamma * response
            image = _smooth_rows(coeff[None, :], state.shape, state.sigma)[0]
            out[step] = np.maximum(image * amp / np.sqrt(state.top), 0.0)
        if step == maximum:
            break
        white = solve_triangular(chol, h, lower=True, check_finite=False)
        leverage = np.maximum(np.sum(white * white, axis=0), 1e-12)
        update = gamma * np.abs(response) / np.sqrt(leverage)
        update = np.clip(update / update.mean(), 1e-5, 1e5)
        gamma = np.sqrt(gamma * update)
        gamma = np.clip(gamma / gamma.mean(), 1e-5, 1e5)
    return out, {"ratio": ratio, "boundary": bool(boundary)}


def readout(field: np.ndarray, y: np.ndarray, state: MesaState, power: float | None) -> np.ndarray:
    if np.iscomplexobj(field) or np.iscomplexobj(y):
        raise ValueError("field and measurements must use the prepared real representation")
    score = np.asarray(field, dtype=np.float64).copy()
    y = np.asarray(y, dtype=np.float64)
    if score.shape != (state.sensitivity.shape[1],) or not np.isfinite(score).all():
        raise ValueError("field must be a finite vector of shape (n,)")
    if y.shape != (state.sensitivity.shape[0],) or not np.isfinite(y).all():
        raise ValueError("measurements must be a finite vector of shape (m,)")
    if power is not None and (not np.isfinite(power) or power < 0):
        raise ValueError("power must be finite and nonnegative, or None")
    if power is not None and power > 0:
        score = _otsu_hill(score, float(power))
    fitted = state.sensitivity @ score
    gain = float(fitted @ y / max(float(fitted @ fitted), 1e-30))
    return score * gain
