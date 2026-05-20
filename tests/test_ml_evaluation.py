import unittest
from unittest.mock import patch

import numpy as np

from otitenet.ml.evaluation import (
    compare_classifiers,
    evaluate_all_classifiers,
    evaluate_kde_classifiers,
)


class _FakeKDE:
    def __init__(self, valid_preds, train_preds):
        self._valid_preds = np.array(valid_preds)
        self._train_preds = np.array(train_preds)

    def predict(self, features):
        if len(features) == len(self._valid_preds):
            return self._valid_preds
        return self._train_preds


class EvaluateKDEClassifiersTests(unittest.TestCase):
    def setUp(self):
        self.train_encs = np.array([[0.0], [1.0], [2.0], [3.0]])
        self.train_cats = np.array([0, 0, 1, 1])
        self.valid_encs = np.array([[0.1], [0.9], [2.1], [2.9]])
        self.valid_cats = np.array([0, 0, 1, 1])

    def test_evaluate_kde_classifiers_preserves_mcc_key_after_best_update(self):
        calls = []

        def fake_fit_kde_classifier(_train_encs, _train_cats, kernel, bandwidth, learnable=False, soft=True):
            calls.append((kernel, bandwidth))
            if (kernel, bandwidth) == ("gaussian", "scott"):
                return _FakeKDE(valid_preds=[0, 0, 1, 1], train_preds=[0, 0, 1, 1])
            return _FakeKDE(valid_preds=[0, 1, 1, 1], train_preds=[0, 0, 1, 1])

        with patch("otitenet.ml.evaluation.fit_kde_classifier", side_effect=fake_fit_kde_classifier):
            result = evaluate_kde_classifiers(
                self.train_encs,
                self.train_cats,
                self.valid_encs,
                self.valid_cats,
                kernels=["gaussian", "exponential"],
                bandwidths=["scott"],
            )

        self.assertEqual(calls, [("gaussian", "scott"), ("exponential", "scott")])
        self.assertIn("mcc", result)
        self.assertIn("valid_mcc", result)
        self.assertIn("train_mcc", result)
        self.assertEqual(result["kernel"], "gaussian")
        self.assertEqual(result["bandwidth"], "scott")
        self.assertGreaterEqual(result["mcc"], result["valid_mcc"])

    def test_evaluate_all_classifiers_exposes_kde_for_compare(self):
        fake_kde_result = {
            "mcc": 0.75,
            "valid_mcc": 0.75,
            "train_mcc": 1.0,
            "kernel": "gaussian",
            "bandwidth": "scott",
            "classifier": object(),
        }

        with patch("otitenet.ml.evaluation.evaluate_baseline_classifiers", return_value={}):
            with patch("otitenet.ml.evaluation.evaluate_kde_classifiers", return_value=fake_kde_result):
                with patch("otitenet.ml.evaluation.evaluate_knn_with_k_search", return_value=(1, 0.25, [{"k": 1, "valid_mcc": 0.25, "train_mcc": 0.5}])):
                    results = evaluate_all_classifiers(
                        self.train_encs,
                        self.train_cats,
                        self.valid_encs,
                        self.valid_cats,
                        min_k=1,
                        max_k=2,
                        include_kde=True,
                        include_baselines=True,
                    )

        self.assertEqual(results["kde"]["mcc"], 0.75)
        method, best_mcc, best_params = compare_classifiers(results)
        self.assertEqual(method, "kde")
        self.assertEqual(best_mcc, 0.75)
        self.assertEqual(best_params["kernel"], "gaussian")
        self.assertEqual(best_params["bandwidth"], "scott")


if __name__ == "__main__":
    unittest.main()