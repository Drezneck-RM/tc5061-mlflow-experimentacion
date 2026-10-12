"""Pruebas de causalidad y protocolo; no requieren descargar datos."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from demand_forecasting.forecasting import (  # noqa: E402
    DAY, TARGET, FEATURES, ForestConfig, build_features, fit_prefix,
    metric_summary, rolling_predictions,
)


class ForecastingTests(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame({
            DAY: ([2, 3, 4, 5, 6] * 12),
            TARGET: 100.0 + np.arange(60) * 2.0 + np.sin(np.arange(60)) * 20.0,
        })
        self.config = ForestConfig(n_estimators=5, max_depth=3, min_samples_leaf=2, seed=42)

    def test_current_and_future_targets_do_not_change_prediction(self):
        origin = 24
        changed = self.raw.copy()
        changed.loc[origin:, TARGET] += 100000.0
        before = build_features(self.raw).iloc[[origin]]
        after = build_features(changed).iloc[[origin]]
        pd.testing.assert_frame_equal(before, after)
        original_model = fit_prefix(self.raw, origin, self.config)
        changed_model = fit_prefix(changed, origin, self.config)
        np.testing.assert_array_equal(original_model.predict(before), changed_model.predict(after))

    def test_same_weekday_uses_last_matching_code_across_gaps(self):
        raw = pd.DataFrame({DAY: [2, 3, 4, 5, 6, 2, 4, 5, 6, 2], TARGET: np.arange(10) * 10.0})
        features = build_features(raw)
        self.assertEqual(features.loc[6, "last_same_weekday"], 20.0)
        self.assertNotEqual(features.loc[6, "last_same_weekday"], raw.loc[1, TARGET])
        self.assertEqual(features.loc[2, "last_same_weekday"], raw.loc[1, TARGET])
        self.assertEqual(features.loc[6, "mean_past_5"], raw.loc[1:5, TARGET].mean())

    def test_pipeline_training_receives_only_prefix_after_warmup(self):
        with patch("demand_forecasting.forecasting.Pipeline.fit", autospec=True) as fitted:
            fit_prefix(self.raw, 24, self.config)
        _, x_train, y_train = fitted.call_args.args
        self.assertEqual(list(x_train.index), list(range(5, 24)))
        self.assertEqual(list(y_train.index), list(range(5, 24)))
        self.assertEqual(list(x_train.columns), FEATURES)
        np.testing.assert_array_equal(y_train, self.raw[TARGET].iloc[5:24])

    def test_daily_update_and_dummy_use_only_available_targets(self):
        origin = 24
        changed = self.raw.copy()
        changed.loc[origin, TARGET] += 100.0
        original = rolling_predictions(self.raw, [origin, origin + 1], self.config)
        modified = rolling_predictions(changed, [origin, origin + 1], self.config)
        self.assertEqual(original.loc[0, "y_pred"], modified.loc[0, "y_pred"])
        self.assertEqual(original.loc[0, "dummy_pred"], modified.loc[0, "dummy_pred"])
        self.assertAlmostEqual(modified.loc[1, "dummy_pred"] - original.loc[1, "dummy_pred"], 100 / 25)
        self.assertEqual(original["training_examples"].tolist(), [19, 20])

    def test_metrics_have_correct_sign_and_dummy_skill(self):
        predictions = pd.DataFrame({"y_true": [10.0, 20.0], "y_pred": [12.0, 22.0], "dummy_pred": [14.0, 24.0]})
        metrics = metric_summary(predictions)
        self.assertEqual(metrics["mae"], 2.0)
        self.assertEqual(metrics["mean_signed_error"], 2.0)
        self.assertEqual(metrics["skill_vs_dummy"], 0.5)

    def test_identical_seed_repeats_predictions(self):
        first = fit_prefix(self.raw, 24, self.config)
        second = fit_prefix(self.raw, 24, self.config)
        x = build_features(self.raw).iloc[[24]]
        np.testing.assert_array_equal(first.predict(x), second.predict(x))


if __name__ == "__main__":
    unittest.main()
