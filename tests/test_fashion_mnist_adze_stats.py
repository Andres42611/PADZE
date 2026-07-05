import sys
import unittest
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "benchmarks"))

from fashion_mnist_adze_stats import ADZE_VARIANTS, adze_like_features_from_values, make_features


class FashionMnistAdzeStatsTest(unittest.TestCase):
    def test_higher_moments_are_population_central_moments(self):
        vals = np.array(
            [
                [
                    [0.0, 1.0, 2.0, 3.0],
                    [2.0, 2.0, 2.0, 2.0],
                    [1.0, 1.0, 0.0, 0.0],
                ]
            ],
            dtype=np.float32,
        )

        features, names = adze_like_features_from_values(
            vals,
            np.array([0.5], dtype=np.float32),
            include_higher_moments=True,
        )
        row = dict(zip(names, features[0]))

        self.assertAlmostEqual(row["cont_0_moment3"], 0.0)
        self.assertAlmostEqual(row["cont_0_moment4"], 2.5625)
        self.assertAlmostEqual(row["cont_1_moment3"], 0.0)
        self.assertAlmostEqual(row["cont_1_moment4"], 0.0)
        self.assertAlmostEqual(row["cont_2_moment3"], 0.0)
        self.assertAlmostEqual(row["cont_2_moment4"], 0.0625)

    def test_constant_inputs_have_zero_higher_moments(self):
        vals = np.full((2, 3, 5), 0.25, dtype=np.float32)

        features, names = adze_like_features_from_values(
            vals,
            np.array([0.5], dtype=np.float32),
            include_higher_moments=True,
        )
        moment_cols = [i for i, name in enumerate(names) if "moment3" in name or "moment4" in name]

        self.assertTrue(moment_cols)
        np.testing.assert_allclose(features[:, moment_cols], 0.0)

    def test_variants_are_configured_and_scoped(self):
        self.assertEqual(ADZE_VARIANTS["adze_test1"], {"population_count": 3, "higher_moments": True})
        self.assertEqual(ADZE_VARIANTS["adze_test2"], {"population_count": 4, "higher_moments": False})
        self.assertEqual(ADZE_VARIANTS["adze_test3"], {"population_count": 4, "higher_moments": True})

        images = np.zeros((1, 28, 28), dtype=np.uint8)
        _, baseline_names = make_features(images, "adze")
        _, test1_names = make_features(images, "adze_test1")
        _, test2_names = make_features(images, "adze_test2")
        _, test3_names = make_features(images, "adze_test3")

        self.assertNotIn("horizontal_cont_0_moment3", baseline_names)
        self.assertIn("horizontal_cont_0_moment3", test1_names)
        self.assertNotIn("horizontal_cont_mean_3", test1_names)
        self.assertIn("horizontal_cont_mean_3", test2_names)
        self.assertNotIn("horizontal_cont_0_moment3", test2_names)
        self.assertIn("horizontal_cont_mean_3", test3_names)
        self.assertIn("horizontal_cont_0_moment3", test3_names)
        self.assertIn("horizontal_thr0.10_all4", test3_names)


if __name__ == "__main__":
    unittest.main()
