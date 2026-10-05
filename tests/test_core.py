"""Independent selector checks, invariants, and frozen manuscript regressions."""
from pathlib import Path
import unittest
import numpy as np
import mesa
from mesa.numpy_backend import choose_loocv, _otsu_hill

ROOT = Path(__file__).resolve().parents[1]


class NumpyTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(105)
        self.operator = rng.normal(size=(9, 64))
        self.y = rng.normal(size=9)
        self.state = mesa.prepare(self.operator, (8, 8), sigma=0.5, sensitivity_exponent=0)

    def test_loocv_matches_explicit_leave_one_channel_out(self):
        """Compare eigensystem LOOCV selection with 61*9 actual held-out solves."""
        b = self.y / np.linalg.norm(self.y)
        covariance = (self.state.h * self.state.initial[None, :]) @ self.state.h.T
        ratios = np.logspace(-5, 2, 61)
        losses = []
        for ratio in ratios:
            ridge = ratio * self.state.eigvals.mean() + 1e-8
            errors = []
            for j in range(9):
                keep = np.arange(9) != j
                fitted = covariance[j, keep] @ np.linalg.solve(
                    covariance[np.ix_(keep, keep)] + ridge*np.eye(8), b[keep])
                errors.append((b[j] - fitted)**2)
            losses.append(np.mean(errors))
        selected, _ = choose_loocv(b, self.state)
        self.assertEqual(selected, float(ratios[np.argmin(losses)]))

    def test_zero_measurement(self):
        actual = mesa.reconstruct(np.zeros(9), self.state)
        np.testing.assert_array_equal(actual, np.zeros(64))

    def test_positive_scale_equivariance(self):
        actual = mesa.reconstruct(self.y, self.state)
        scaled = mesa.reconstruct(3.7*self.y, self.state)
        np.testing.assert_allclose(scaled, 3.7*actual, rtol=1e-10, atol=1e-10)

    def test_no_history_between_frames(self):
        first = mesa.reconstruct(self.y, self.state)
        mesa.reconstruct(self.y[::-1], self.state)
        again = mesa.reconstruct(self.y, self.state)
        np.testing.assert_array_equal(again, first)

    def test_gain_satisfies_scalar_normal_equation(self):
        fields, _ = mesa.trajectory(self.y, self.state, checkpoints=(2,))
        q = _otsu_hill(fields[2], 6)
        output = mesa.readout(fields[2], self.y, self.state, power=6)
        self.assertAlmostEqual(float((self.operator@q) @ (self.y-self.operator@output)), 0, places=10)

    def test_invalid_frame_rejected(self):
        for y in [np.ones(8), np.full(9, np.nan), self.y.astype(complex)+1j]:
            with self.subTest(), self.assertRaises(ValueError):
                mesa.reconstruct(y, self.state)

    def test_complex_operator_not_silently_discarded(self):
        with self.assertRaises(ValueError):
            mesa.prepare(self.operator.astype(complex)+1j, (8,8))

    def test_invalid_update_count(self):
        for count in [-1, 1.5]:
            with self.subTest(), self.assertRaises(ValueError):
                mesa.reconstruct(self.y, self.state, updates=count)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is optional; install .[torch] for EIT regressions')
class EitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mesa import torch_backend
        cls.backend = torch_backend
        cls.data = np.load(ROOT / 'data/eit_examples.npz', allow_pickle=False)
        cls.expected = np.load(ROOT / 'tests/fixtures/eit_reference.npz', allow_pickle=False)['predictions']
        cls.state = torch_backend.prepare_operator(cls.data['sensitivity'], cls.data['grid_shape'],
            grid_indices=cls.data['points'][:, 3:5].astype(int)-1, device='cpu')

    def test_three_cases_against_frozen_manuscript_predictions(self):
        for y, expected in zip(self.data['measurements'], self.expected):
            with self.subTest():
                actual = self.backend.reconstruct(y, self.state)
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)

    def test_zero_measurement(self):
        np.testing.assert_array_equal(self.backend.reconstruct(np.zeros(104), self.state), np.zeros(6400))

    def test_positive_scale_equivariance(self):
        y = self.data['measurements'][0]
        x = self.backend.reconstruct(y, self.state)
        np.testing.assert_allclose(self.backend.reconstruct(2.5*y.astype(float), self.state),
                                   2.5*x, rtol=2e-5, atol=2e-6)

    def test_shape_validation(self):
        with self.assertRaises(ValueError):
            self.backend.reconstruct(np.ones(103), self.state)

    def test_selector_1se_is_no_less_regularized(self):
        from mesa._components import choose_noise
        core = self.state[1]
        y = torch.tensor(self.data['measurements'][0].astype(float), dtype=torch.float64)
        y /= torch.linalg.vector_norm(y)
        ordinary = choose_noise(y, core)[0]
        alternative = dict(core, config={**core['config'], 'noise_selection':'loocv_1se'})
        self.assertGreaterEqual(choose_noise(y, alternative)[0], ordinary)


if __name__ == '__main__':
    unittest.main()
