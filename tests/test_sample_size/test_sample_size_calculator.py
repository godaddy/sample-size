import unittest
from unittest.mock import patch

from numpy.testing import assert_equal

from sample_size.metrics import BooleanMetric
from sample_size.metrics import NumericMetric
from sample_size.metrics import RatioMetric
from sample_size.sample_size_calculator import DEFAULT_ALPHA
from sample_size.sample_size_calculator import DEFAULT_POWER
from sample_size.sample_size_calculator import SampleSizeCalculator


class SampleSizeCalculatorTestCase(unittest.TestCase):
    def test_sample_size_calculator_constructor_sets_params(self):
        test_alpha = 0.1
        test_power = 0.9
        calculator = SampleSizeCalculator(
            test_alpha,
            test_power,
        )

        self.assertEqual(calculator.alpha, test_alpha)
        self.assertEqual(calculator.power, test_power)
        self.assertEqual(calculator.boolean_metrics, [])
        self.assertEqual(calculator.numeric_metrics, [])
        self.assertEqual(calculator.ratio_metrics, [])

    def test_sample_size_calculator_constructor_sets_params_with_default_params(self):
        calculator = SampleSizeCalculator()

        self.assertEqual(calculator.alpha, DEFAULT_ALPHA)
        self.assertEqual(calculator.power, DEFAULT_POWER)
        self.assertEqual(calculator.boolean_metrics, [])
        self.assertEqual(calculator.numeric_metrics, [])
        self.assertEqual(calculator.ratio_metrics, [])

    def test_register_bool_metric(self):
        test_probability = 0.05
        test_mde = 0.02

        calculator = SampleSizeCalculator()
        calculator.register_bool_metric(test_probability, test_mde)

        self.assertEqual(len(calculator.boolean_metrics), 1)
        self.assertEqual(calculator.boolean_metrics[0].variance, 0.0475)
        self.assertEqual(calculator.boolean_metrics[0].mde, test_mde)
        self.assertEqual(calculator.numeric_metrics, [])
        self.assertEqual(calculator.ratio_metrics, [])

        calculator.register_bool_metric(test_probability, test_mde)
        self.assertEqual(len(calculator.boolean_metrics), 2)

    def test_register_numeric_metric(self):
        test_variance = 5000
        test_mde = 10

        calculator = SampleSizeCalculator()
        calculator.register_numeric_metric(test_variance, test_mde)

        self.assertEqual(len(calculator.numeric_metrics), 1)
        self.assertEqual(calculator.numeric_metrics[0].variance, test_variance)
        self.assertEqual(calculator.numeric_metrics[0].mde, test_mde)
        self.assertEqual(calculator.boolean_metrics, [])
        self.assertEqual(calculator.ratio_metrics, [])

        calculator.register_numeric_metric(test_variance, test_mde)
        self.assertEqual(len(calculator.numeric_metrics), 2)

    def test_register_ratio_metric(self):
        test_numerator_mean = 2000
        test_numerator_variance = 100000
        test_denominator_mean = 200
        test_denominator_variance = 2000
        test_covariance = 5000
        test_mde = 10
        test_variance = 5

        calculator = SampleSizeCalculator()
        calculator.register_ratio_metric(
            test_numerator_mean,
            test_numerator_variance,
            test_denominator_mean,
            test_denominator_variance,
            test_covariance,
            test_mde,
        )

        self.assertEqual(len(calculator.ratio_metrics), 1)
        self.assertEqual(calculator.ratio_metrics[0].variance, test_variance)
        self.assertEqual(calculator.ratio_metrics[0].mde, test_mde)
        self.assertEqual(calculator.boolean_metrics, [])
        self.assertEqual(calculator.numeric_metrics, [])

        calculator.register_ratio_metric(
            test_numerator_mean,
            test_numerator_variance,
            test_denominator_mean,
            test_denominator_variance,
            test_covariance,
            test_mde,
        )
        self.assertEqual(len(calculator.ratio_metrics), 2)

    @patch("statsmodels.stats.power.NormalIndPower.solve_power")
    def test_get_single_sample_size_normal(self, mock_solve_power):
        test_probability = 0.05
        test_mde = 0.02
        test_sample_size = 2000
        test_metric = BooleanMetric(
            test_probability,
            test_mde,
        )
        mock_solve_power.return_value = test_sample_size

        calculator = SampleSizeCalculator()
        sample_size = calculator._get_single_sample_size(test_metric)

        self.assertEqual(sample_size, test_sample_size)
        mock_solve_power.assert_called_once()
        assert_equal(mock_solve_power.call_args[1]["effect_size"], 0.09176629354822471)
        assert_equal(mock_solve_power.call_args[1]["alpha"], DEFAULT_ALPHA)
        assert_equal(mock_solve_power.call_args[1]["power"], DEFAULT_POWER)
        assert_equal(mock_solve_power.call_args[1]["ratio"], 1)
        assert_equal(mock_solve_power.call_args[1]["alternative"], "two-sided")

    @patch("statsmodels.stats.power.TTestIndPower.solve_power")
    def test_get_single_sample_size_ttest(self, mock_solve_power):
        test_variance = 1000
        test_mde = 5
        test_sample_size = 2000
        test_metric = NumericMetric(
            test_variance,
            test_mde,
        )
        mock_solve_power.return_value = test_sample_size
        calculator = SampleSizeCalculator()

        sample_size = calculator._get_single_sample_size(test_metric)

        self.assertEqual(sample_size, test_sample_size)
        mock_solve_power.assert_called_once()
        assert_equal(mock_solve_power.call_args[1]["effect_size"], 0.15811388300841897)
        assert_equal(mock_solve_power.call_args[1]["alpha"], DEFAULT_ALPHA)
        assert_equal(mock_solve_power.call_args[1]["power"], DEFAULT_POWER)
        assert_equal(mock_solve_power.call_args[1]["ratio"], 1)
        assert_equal(mock_solve_power.call_args[1]["alternative"], "two-sided")

    @patch("sample_size.sample_size_calculator.SampleSizeCalculator._get_single_sample_size")
    def test_get_overall_sample_size_bool(self, mock_get_single_sample_size):
        test_sample_size = 2000
        mock_get_single_sample_size.return_value = test_sample_size

        test_mde = 0.02
        test_probability = 0.05
        calculator = SampleSizeCalculator()
        calculator.register_bool_metric(test_probability, test_mde)

        sample_size = calculator.get_overall_sample_size()

        self.assertEqual(sample_size, test_sample_size)
        mock_get_single_sample_size.assert_called_once()
        self.assertIsInstance(mock_get_single_sample_size.call_args[0][0], BooleanMetric)
        assert_equal(mock_get_single_sample_size.call_args[0][0].probability, test_probability)
        assert_equal(mock_get_single_sample_size.call_args[0][0].mde, test_mde)

    @patch("sample_size.sample_size_calculator.SampleSizeCalculator._get_single_sample_size")
    def test_get_overall_sample_size_numeric(self, mock_get_single_sample_size):
        test_sample_size = 2000
        mock_get_single_sample_size.return_value = test_sample_size

        test_mde = 5
        test_variance = 500
        calculator = SampleSizeCalculator()
        calculator.register_numeric_metric(test_variance, test_mde)

        sample_size = calculator.get_overall_sample_size()

        self.assertEqual(sample_size, test_sample_size)
        mock_get_single_sample_size.assert_called_once()
        self.assertIsInstance(mock_get_single_sample_size.call_args[0][0], NumericMetric)
        assert_equal(mock_get_single_sample_size.call_args[0][0].variance, test_variance)
        assert_equal(mock_get_single_sample_size.call_args[0][0].mde, test_mde)

    @patch("sample_size.sample_size_calculator.SampleSizeCalculator._get_single_sample_size")
    def test_get_overall_sample_size_ratio(self, mock_get_single_sample_size):
        test_sample_size = 2000
        mock_get_single_sample_size.return_value = test_sample_size

        test_numerator_mean = 2000
        test_numerator_variance = 100000
        test_denominator_mean = 200
        test_denominator_variance = 2000
        test_covariance = 5000
        test_mde = 10
        calculator = SampleSizeCalculator()
        calculator.register_ratio_metric(
            test_numerator_mean,
            test_numerator_variance,
            test_denominator_mean,
            test_denominator_variance,
            test_covariance,
            test_mde,
        )

        sample_size = calculator.get_overall_sample_size()

        self.assertEqual(sample_size, test_sample_size)
        mock_get_single_sample_size.assert_called_once()
        self.assertIsInstance(mock_get_single_sample_size.call_args[0][0], RatioMetric)
        assert_equal(mock_get_single_sample_size.call_args[0][0].numerator_mean, test_numerator_mean)
        assert_equal(mock_get_single_sample_size.call_args[0][0].numerator_variance, test_numerator_variance)
        assert_equal(mock_get_single_sample_size.call_args[0][0].denominator_mean, test_denominator_mean)
        assert_equal(mock_get_single_sample_size.call_args[0][0].denominator_variance, test_denominator_variance)
        assert_equal(mock_get_single_sample_size.call_args[0][0].covariance, test_covariance)
        assert_equal(mock_get_single_sample_size.call_args[0][0].mde, test_mde)
