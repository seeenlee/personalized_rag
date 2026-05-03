import unittest

import numpy as np

from new.vector_math import (
    linear_combination,
    moving_average_user_vector,
    normalize_vector,
    spherical_combination,
)


class VectorMathTests(unittest.TestCase):
    def test_normalize_vector_preserves_zero_vector(self):
        self.assertTrue(np.allclose(normalize_vector([0.0, 0.0]), [0.0, 0.0]))

    def test_linear_combination_normalizes_weighted_sum(self):
        result = linear_combination([1.0, 0.0], [0.0, 1.0], alpha=0.5)
        expected = np.array([1.0, 1.0]) / np.sqrt(2.0)
        self.assertTrue(np.allclose(result, expected))

    def test_spherical_combination_midpoint(self):
        result = spherical_combination([1.0, 0.0], [0.0, 1.0], alpha=0.5)
        expected = np.array([1.0, 1.0]) / np.sqrt(2.0)
        self.assertTrue(np.allclose(result, expected))

    def test_moving_average_update_uses_query_weight(self):
        result = moving_average_user_vector([1.0, 0.0], [0.0, 1.0])
        expected = np.array([0.9, 0.1])
        expected = expected / np.linalg.norm(expected)
        self.assertTrue(np.allclose(result, expected))


if __name__ == "__main__":
    unittest.main()
