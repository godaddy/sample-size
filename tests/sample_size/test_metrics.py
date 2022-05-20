import unittest
from itertools import product
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal
from parameterized import parameterized
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.power import TTestIndPower

from sample_size.metrics import BaseMetric
from sample_size.metrics import BooleanMetric
from sample_size.metrics import NumericMetric
from sample_size.metrics import RatioMetric


class BaseMetricTestCase(unittest.TestCase):
    def test_check_positive(self):
        test_negative_number = -10
        test_name = "test"

        with self.assertRaises(Exception) as context:
            BaseMetric.check_positive(test_negative_number, test_name)

        self.assertEqual(
            str(context.exception),
            f"Error: Please provide a positive number for {test_name}.",
        )


class BooleanMetricTestCase(unittest.TestCase):
    def setUp(self):
        self.DEFAULT_MDE = 0.01
        self.DEFAULT_PROBABILITY = 0.05
        self.DEFAULT_MOCK_VARIANCE = 99

    @patch("sample_size.metrics.BooleanMetric._check_probability")
    @patch("sample_size.metrics.BooleanMetric.variance")
    def test_boolean_metric_constructor_sets_params(self, mock_variance, mock_check_probability):
        mock_variance.__get__ = MagicMock(return_value=self.DEFAULT_MOCK_VARIANCE)
        mock_check_probability.return_value = self.DEFAULT_PROBABILITY
        boolean = BooleanMetric(self.DEFAULT_PROBABILITY, self.DEFAULT_MDE)

        mock_check_probability.assert_called_once_with(self.DEFAULT_PROBABILITY)
        self.assertEqual(boolean.probability, self.DEFAULT_PROBABILITY)
        self.assertEqual(boolean.variance, self.DEFAULT_MOCK_VARIANCE)
        self.assertEqual(boolean.mde, self.DEFAULT_MDE)
        self.assertIsInstance(boolean.power_analysis_instance, NormalIndPower)

    def test_boolean_metric_variance(self):
        boolean = BooleanMetric(self.DEFAULT_PROBABILITY, self.DEFAULT_MDE)

        self.assertEqual(boolean.variance, 0.0475)

    def test_boolean_metric_get_probability(self):
        probability = BooleanMetric._check_probability(self.DEFAULT_PROBABILITY)

        self.assertEqual(probability, self.DEFAULT_PROBABILITY)

    def test_boolean_metric_get_probability_too_large(self):
        test_probability = 5

        with self.assertRaises(Exception) as context:
            BooleanMetric._check_probability(test_probability)

        self.assertEqual(
            str(context.exception),
            "Error: Please provide a float between 0 and 1 for probability.",
        )

    def test_boolean_metric_get_probability_too_small(self):
        test_probability = -0.1

        with self.assertRaises(Exception) as context:
            BooleanMetric._check_probability(test_probability)

        self.assertEqual(
            str(context.exception),
            "Error: Please provide a float between 0 and 1 for probability.",
        )

    @parameterized.expand(product([True, False], [2, 5], [200]))
    @patch("sample_size.metrics.BooleanMetric.variance")
    @patch("scipy.stats.uniform.rvs")
    @patch("scipy.stats.norm")
    def test_boolean_generate_p_value(self, true_null, variants, sample_size, mock_norm, mock_uniform, mock_variance):
        p_value_generator = mock_uniform if true_null else mock_norm.sf
        pval = MagicMock()
        mock_norm.rvs.return_value = -ord("🌮")
        p_value_generator.return_value = pval
        mock_variance.__get__ = MagicMock(return_value=self.DEFAULT_MOCK_VARIANCE)

        boolean = BooleanMetric(self.DEFAULT_PROBABILITY, self.DEFAULT_MDE)
        p = boolean.generate_p_value(true_null, sample_size, variants)

        if true_null:
            mock_uniform.assert_called_once_with(0, 1, size=variants - 1)
            mock_norm.assert_not_called()
        else:
            effect_sample_size = self.DEFAULT_MDE / np.sqrt(2 * self.DEFAULT_MOCK_VARIANCE / sample_size)
            mock_uniform.assert_not_called()
            mock_norm.rvs.assert_called_once_with(loc=effect_sample_size, size=variants - 1)
            mock_norm.sf.assert_called_once_with(np.abs(mock_norm.rvs.return_value))
        assert_array_equal(p, pval)


class NumericMetricTestCase(unittest.TestCase):
    def setUp(self):
        self.DEFAULT_MDE = 5
        self.DEFAULT_VARIANCE = 5000

    def test_numeric_metric_constructor_sets_params(self):

        numeric = NumericMetric(self.DEFAULT_VARIANCE, self.DEFAULT_MDE)

        self.assertEqual(numeric.variance, self.DEFAULT_VARIANCE)
        self.assertEqual(numeric.mde, self.DEFAULT_MDE)
        self.assertIsInstance(numeric.power_analysis_instance, TTestIndPower)

    @parameterized.expand(product([True, False], [2, 5], [200]))
    @patch("scipy.stats.uniform.rvs")
    @patch("scipy.stats.nct.rvs")
    @patch("scipy.stats.t.sf")
    def test_numeric_generate_p_value(self, true_null, variants, sample_size, mock_t, mock_nct, mock_uniform):
        p_value_generator = mock_uniform if true_null else mock_t
        pval = MagicMock()
        mock_nct.return_value = -ord("🌮")
        p_value_generator.return_value = pval

        numeric = NumericMetric(self.DEFAULT_VARIANCE, self.DEFAULT_MDE)
        p = numeric.generate_p_value(true_null, sample_size, variants)

        if true_null:
            mock_uniform.assert_called_once_with(0, 1, size=variants - 1)
            mock_nct.assert_not_called()
            mock_t.assert_not_called()
        else:
            nc = np.sqrt(sample_size / 2) * self.DEFAULT_MDE / self.DEFAULT_VARIANCE
            mock_uniform.assert_not_called()
            mock_nct.assert_called_once_with(nc=nc, df=2 * (sample_size - 1), size=variants - 1)
            mock_t.assert_called_once_with(np.abs(mock_nct.return_value), 2 * (sample_size - 1))
        assert_array_equal(p, pval)


class RatioMetricTestCase(unittest.TestCase):
    def setUp(self):
        self.DEFAULT_MDE = 5
        self.DEFAULT_NUMERATOR_MEAN = 2000
        self.DEFAULT_NUMERATOR_VARIANCE = 100000
        self.DEFAULT_DENOMINATOR_MEAN = 200
        self.DEFAULT_DENOMINATOR_VARIANCE = 2000
        self.DEFAULT_COVARIANCE = 5000
        self.DEFAULT_VARIANCE = 99

    @patch("sample_size.metrics.RatioMetric.variance")
    def test_ratio_metric_constructor_sets_params(self, mock_variance):
        mock_variance.__get__ = MagicMock(return_value=self.DEFAULT_VARIANCE)
        ratio = RatioMetric(
            self.DEFAULT_NUMERATOR_MEAN,
            self.DEFAULT_NUMERATOR_VARIANCE,
            self.DEFAULT_DENOMINATOR_MEAN,
            self.DEFAULT_DENOMINATOR_VARIANCE,
            self.DEFAULT_COVARIANCE,
            self.DEFAULT_MDE,
        )

        self.assertEqual(ratio.numerator_mean, self.DEFAULT_NUMERATOR_MEAN)
        self.assertEqual(ratio.numerator_variance, self.DEFAULT_NUMERATOR_VARIANCE)
        self.assertEqual(ratio.denominator_mean, self.DEFAULT_DENOMINATOR_MEAN)
        self.assertEqual(ratio.denominator_variance, self.DEFAULT_DENOMINATOR_VARIANCE)
        self.assertEqual(ratio.covariance, self.DEFAULT_COVARIANCE)
        self.assertEqual(ratio.variance, self.DEFAULT_VARIANCE)
        self.assertEqual(ratio.mde, self.DEFAULT_MDE)
        self.assertIsInstance(ratio.power_analysis_instance, NormalIndPower)

    def test_ratio_metric_variance(self):
        ratio = RatioMetric(
            self.DEFAULT_NUMERATOR_MEAN,
            self.DEFAULT_NUMERATOR_VARIANCE,
            self.DEFAULT_DENOMINATOR_MEAN,
            self.DEFAULT_DENOMINATOR_VARIANCE,
            self.DEFAULT_COVARIANCE,
            self.DEFAULT_MDE,
        )

        self.assertEqual(ratio.variance, 5.0)

    @parameterized.expand(product([True, False], [2, 5], [200]))
    @patch("sample_size.metrics.RatioMetric.variance")
    @patch("scipy.stats.uniform.rvs")
    @patch("scipy.stats.norm")
    def test_ratio_generate_p_value(self, true_null, variants, sample_size, mock_norm, mock_uniform, mock_variance):
        p_value_generator = mock_uniform if true_null else mock_norm.sf
        pval = MagicMock()
        mock_norm.rvs.return_value = -ord("🌮")
        p_value_generator.return_value = pval
        mock_variance.__get__ = MagicMock(return_value=self.DEFAULT_VARIANCE)

        ratio = RatioMetric(
            self.DEFAULT_NUMERATOR_MEAN,
            self.DEFAULT_NUMERATOR_VARIANCE,
            self.DEFAULT_DENOMINATOR_MEAN,
            self.DEFAULT_DENOMINATOR_VARIANCE,
            self.DEFAULT_COVARIANCE,
            self.DEFAULT_MDE,
        )

        p = ratio.generate_p_value(true_null, sample_size, variants)

        if true_null:
            mock_uniform.assert_called_once_with(0, 1, size=variants - 1)
            mock_norm.assert_not_called()
        else:
            effect_sample_size = self.DEFAULT_MDE / float(np.sqrt(2 * self.DEFAULT_VARIANCE / sample_size))
            mock_uniform.assert_not_called()
            mock_norm.rvs.assert_called_once_with(loc=effect_sample_size, size=variants - 1)
            mock_norm.sf.assert_called_once_with(np.abs(mock_norm.rvs.return_value))
        assert_array_equal(p, pval)
