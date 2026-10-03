"""Boundary and negative controls for the new validation utilities."""
import contextlib
import io
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

from validation_common import (
    ROOT, FROZEN, FEATURES, historical_split, load_cleaner, metrics,
    opi_components, strict_prepare, weight_grid,
)


class ValidationTests(unittest.TestCase):
    def raw_fixture(self, size=20):
        return pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=size, freq="10s"),
            "pH": np.full(size, 6.), "TDS": np.full(size, 1000.),
            "water_temp": np.full(size, 20.), "DHT_temp": np.full(size, 24.),
            "DHT_humidity": np.full(size, 70.),
            **{name: ["OFF"] * size for name in
               ["pH_reducer", "add_water", "nutrients_adder", "humidifier", "ex_fan"]},
        })

    def test_future_leak_negative_control_detected_and_split_first_blocks(self):
        raw = self.raw_fixture()
        raw.loc[9, "pH"] = np.nan
        raw.loc[10:, "pH"] = 8.
        altered = raw.copy()
        altered.loc[10:, "pH"] = 4.
        cleaner = load_cleaner()
        with contextlib.redirect_stdout(io.StringIO()):
            all_a = cleaner.clean(raw)
            all_b = cleaner.clean(altered)
            split_a = strict_prepare(raw, raw.timestamp[9], raw.timestamp[10], cleaner)
            split_b = strict_prepare(altered, raw.timestamp[9], raw.timestamp[10], cleaner)
        self.assertNotEqual(all_a.loc[9, "pH"], all_b.loc[9, "pH"])
        pd.testing.assert_frame_equal(split_a["train"], split_b["train"])
        self.assertEqual(float(split_a["train"].iloc[-1].ph), 6.)

    def test_long_outage_not_bridged(self):
        raw = self.raw_fixture()
        raw.loc[10:, "timestamp"] += pd.Timedelta(hours=1)
        raw.loc[9, "pH"] = np.nan
        raw.loc[10:, "pH"] = 8.
        with contextlib.redirect_stdout(io.StringIO()):
            clean = load_cleaner().clean(raw)
        self.assertEqual(float(clean.loc[9, "pH"]), 6.)
        self.assertNotEqual(clean.loc[9, "segment_id"], clean.loc[10, "segment_id"])

    def test_invalid_split_rejected(self):
        raw = self.raw_fixture()
        with self.assertRaises(ValueError):
            strict_prepare(raw, raw.timestamp[10], raw.timestamp[9], load_cleaner())

    def test_purge_has_no_shared_rows(self):
        n = 1200
        frame = pd.DataFrame({"timestamp": pd.date_range("2024", periods=n, freq="10s"),
                              "segment_id": np.zeros(n, dtype=int)})
        train, holdout, all_rows = historical_split(frame)
        self.assertEqual(len(np.intersect1d(train, holdout)), 0)
        self.assertEqual(len(all_rows) - len(train) - len(holdout), 3)

    def test_opi_decomposition(self):
        rng = np.random.default_rng(42)
        bounds = np.array([metrics.OPERATIONAL_BOUNDS[f] for f in FEATURES])
        values = rng.uniform(bounds[:, 0] - .5, bounds[:, 1] + .5, (8, 120, 5))
        parts, _ = opi_components(values)
        self.assertAlmostEqual(float(parts @ [.5, .3, .2]),
                               metrics.operational_plausibility_index(values, FEATURES), places=12)

    def test_nonfinite_data_rejected(self):
        values = np.ones((2, 2, 5))
        values[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            opi_components(values)

    def test_weights_cover_simplex(self):
        weights = weight_grid()
        self.assertEqual(len(weights), 66)
        np.testing.assert_allclose(weights.sum(axis=1), 1.)
        self.assertTrue((weights >= 0).all())
        self.assertTrue(any(np.allclose(w, [.5, .3, .2]) for w in weights))

    def test_scaler_does_not_clip_out_of_range(self):
        train = np.arange(60.).reshape(3, 4, 5)
        test = train + 1000
        scaler = metrics.CanonicalScaler.fit(train, FEATURES)
        np.testing.assert_allclose(scaler.inverse_transform(scaler.transform(test)), test)
        self.assertTrue((scaler.transform(test) > scaler.transform(train).max(axis=(0, 1))).all())

    def test_discrete_clamping_and_bounded_map(self):
        import torch
        sys.path.insert(0, str(FROZEN / "scripts"))
        sys.path.insert(0, str(ROOT / "runtime"))
        from pc_mps import Discretizer
        from pc_cmps_rwgan.model import ResidualTrajectoryDecoder
        discretizer = Discretizer(0., 1., 8)
        self.assertEqual(discretizer.to_index(torch.tensor([-5., 5.])).tolist(), [0, 7])
        decoder = ResidualTrajectoryDecoder(
            2, 1, 1, 8, torch.arange(8.).reshape(1, 8),
            torch.tensor([0.]), torch.tensor([1.]), hidden_dim=4, num_layers=1)
        raw = torch.linspace(-100., 100., 1001).reshape(1, -1, 1)
        result = decoder._smooth_bound(raw)
        self.assertTrue(torch.isfinite(result).all())
        self.assertGreaterEqual(result.min().item(), -1e-5)
        self.assertLessEqual(result.max().item(), 1. + 1e-5)

    def test_historical_minmax_extrapolates_without_clipping(self):
        import ast
        tree = ast.parse((FROZEN / "scripts/pcwgan_train.py").read_text(encoding="utf-8-sig"))
        scaler_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TimeSeriesScaler")
        namespace = {"np": np, "Dict": dict, "Any": object}
        exec(compile(ast.Module(body=[scaler_class], type_ignores=[]), "<frozen scaler class>", "exec"), namespace)
        scaler = namespace["TimeSeriesScaler"]("minmax").fit(np.array([[[0.], [1.]]]))
        unseen = np.array([[[-1.], [2.]]])
        result = scaler.transform(unseen)
        self.assertLess(result.min(), -1.)
        self.assertGreater(result.max(), 1.)
        np.testing.assert_allclose(scaler.inverse_transform(result), unseen)

    def test_mps_cache_matches_original_values_and_gradients(self):
        import importlib.util
        import torch
        sys.path.insert(0, str(ROOT / "runtime"))
        from pc_mps import GenerativeMPS
        spec = importlib.util.spec_from_file_location("reference_mps", FROZEN / "scripts/pc_mps.py")
        reference = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = reference
        spec.loader.exec_module(reference)
        original = reference.GenerativeMPS(6, 3, bond_dim=2, seed=19).double()
        fixed = GenerativeMPS(6, 3, bond_dim=2, seed=19).double()
        fixed.load_state_dict(original.state_dict())
        tokens = torch.tensor([[0, 1, 2, 0, 1, 2], [2, 1, 0, 2, 1, 0]])
        allowed = [torch.tensor([0, 1])] * 6
        old_loss = -original.log_prob(tokens).mean() + original.allowed_log_barrier(allowed)
        norm = fixed.log_norm()
        new_loss = -fixed.log_prob(tokens, log_norm=norm).mean() + fixed.allowed_log_barrier(allowed, log_norm=norm)
        torch.testing.assert_close(old_loss, new_loss, rtol=1e-12, atol=1e-12)
        old_loss.backward()
        new_loss.backward()
        for a, b in zip(original.parameters(), fixed.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-10, atol=1e-10)

    def test_mixture_cached_objective_matches_uncached(self):
        import torch
        sys.path.insert(0, str(FROZEN / "scripts"))
        sys.path.insert(0, str(ROOT / "runtime"))
        from pc_cmps_rwgan.model import ConditionalMPSMixture
        prior = ConditionalMPSMixture(6, 3, 2, experts=2, bond_dim=2, seed=8).double()
        tokens = torch.tensor([[0, 1, 2, 0, 1, 2], [2, 1, 0, 2, 1, 0]])
        conditions = torch.zeros(2, 3, 2, dtype=torch.float64)
        allowed = [torch.tensor([0, 1])] * 6
        separate = prior.nll(tokens, conditions) + prior.allowed_log_barrier(allowed, conditions)
        losses = prior.nll_and_allowed_log_barrier(tokens, allowed, conditions)
        torch.testing.assert_close(separate, sum(losses), rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
