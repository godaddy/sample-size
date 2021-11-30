import unittest
from unittest.mock import patch

from numpy.testing import assert_equal
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.power import TTestIndPower

from sample_size.sample_size_calculator import DEFAULT_ALPHA
from sample_size.sample_size_calculator import DEFAULT_POWER
from sample_size.sample_size_calculator import PowerAnalysisParameters
from sample_size.sample_size_calculator import SampleSizeCalculator


class PowerAnalysisParametersTestCase(unittest.TestCase):
    def setUp(self):
        self.DEFAULT_METRIC_VARIANCE = 500
        self.DEFAULT_MDE = 5

    def test_power_analysis_parameters_sets_params(self):

        power_analysis_parameters = PowerAnalysisParameters(
            self.DEFAULT_METRIC_VARIANCE,
            self.DEFAULT_MDE,
            DEFAULT_ALPHA,
            DEFAULT_POWER,
        )

        self.assertEqual(power_analysis_parameters.metric_variance, self.DEFAULT_METRIC_VARIANCE)
        self.assertEqual(power_analysis_parameters.mde, self.DEFAULT_MDE)
        self.assertEqual(power_analysis_parameters.alpha, DEFAULT_ALPHA)
        self.assertEqual(power_analysis_parameters.power, DEFAULT_POWER)

    def test_power_analysis_parameters_sets_params_with_default_params(self):
        power_analysis_parameters = PowerAnalysisParameters(
            self.DEFAULT_METRIC_VARIANCE,
            self.DEFAULT_MDE,
        )

        self.assertEqual(power_analysis_parameters.metric_variance, self.DEFAULT_METRIC_VARIANCE)
        self.assertEqual(power_analysis_parameters.mde, self.DEFAULT_MDE)
        self.assertEqual(power_analysis_parameters.alpha, DEFAULT_ALPHA)
        self.assertEqual(power_analysis_parameters.power, DEFAULT_POWER)


class SampleSizeCalculatorTestCase(unittest.TestCase):
    def test_sample_size_calculator_constructor_sets_params(self):
        calculator = SampleSizeCalculator(
            DEFAULT_ALPHA,
            DEFAULT_POWER,
        )

        self.assertEqual(calculator.alpha, DEFAULT_ALPHA)
        self.assertEqual(calculator.power, DEFAULT_POWER)
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
        calculator.register_bool_metric(test_mde, test_probability)

        self.assertEqual(len(calculator.boolean_metrics), 1)
        self.assertEqual(calculator.boolean_metrics[0].metric_variance, 0.0475)
        self.assertEqual(calculator.boolean_metrics[0].mde, test_mde)
        self.assertEqual(calculator.boolean_metrics[0].alpha, DEFAULT_ALPHA)
        self.assertEqual(calculator.boolean_metrics[0].power, DEFAULT_POWER)
        self.assertEqual(calculator.numeric_metrics, [])
        self.assertEqual(calculator.ratio_metrics, [])

    def test_register_numeric_metric(self):
        test_variance = 5000
        test_mde = 10

        calculator = SampleSizeCalculator()
        calculator.register_numeric_metric(test_mde, test_variance)

        self.assertEqual(len(calculator.numeric_metrics), 1)
        self.assertEqual(calculator.numeric_metrics[0].metric_variance, test_variance)
        self.assertEqual(calculator.numeric_metrics[0].mde, test_mde)
        self.assertEqual(calculator.numeric_metrics[0].alpha, DEFAULT_ALPHA)
        self.assertEqual(calculator.numeric_metrics[0].power, DEFAULT_POWER)
        self.assertEqual(calculator.boolean_metrics, [])
        self.assertEqual(calculator.ratio_metrics, [])

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
            test_mde,
            test_numerator_mean,
            test_numerator_variance,
            test_denominator_mean,
            test_denominator_variance,
            test_covariance,
        )

        self.assertEqual(len(calculator.ratio_metrics), 1)
        self.assertEqual(calculator.ratio_metrics[0].metric_variance, test_variance)
        self.assertEqual(calculator.ratio_metrics[0].mde, test_mde)
        self.assertEqual(calculator.ratio_metrics[0].alpha, DEFAULT_ALPHA)
        self.assertEqual(calculator.ratio_metrics[0].power, DEFAULT_POWER)
        self.assertEqual(calculator.boolean_metrics, [])
        self.assertEqual(calculator.numeric_metrics, [])

    @patch("statsmodels.stats.power.NormalIndPower.solve_power")
    def test_get_single_sample_size_normal(self, mock_solve_power):
        mock_normal_ind_power = NormalIndPower
        test_variance = 0.05
        test_mde = 0.02
        test_sample_size = 2000
        test_metric = PowerAnalysisParameters(
            test_variance,
            test_mde,
            DEFAULT_ALPHA,
            DEFAULT_POWER,
        )
        mock_solve_power.return_value = test_sample_size

        sample_size = SampleSizeCalculator._get_single_sample_size(test_metric, mock_normal_ind_power)

        self.assertEqual(sample_size, test_sample_size)
        mock_solve_power.assert_called_once()
        assert_equal(mock_solve_power.call_args[1]["effect_size"], 0.08944271909999159)
        assert_equal(mock_solve_power.call_args[1]["alpha"], DEFAULT_ALPHA)
        assert_equal(mock_solve_power.call_args[1]["power"], DEFAULT_POWER)
        assert_equal(mock_solve_power.call_args[1]["ratio"], 1)
        assert_equal(mock_solve_power.call_args[1]["alternative"], "two-sided")

    @patch("statsmodels.stats.power.TTestIndPower.solve_power")
    def test_get_single_sample_size_ttest(self, mock_solve_power):
        mock_ttest_ind_power = TTestIndPower
        test_variance = 1000
        test_mde = 5
        test_sample_size = 2000
        test_metric = PowerAnalysisParameters(
            test_variance,
            test_mde,
            DEFAULT_ALPHA,
            DEFAULT_POWER,
        )
        mock_solve_power.return_value = test_sample_size

        sample_size = SampleSizeCalculator._get_single_sample_size(test_metric, mock_ttest_ind_power)

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
        calculator = SampleSizeCalculator()
        calculator.register_bool_metric(test_mde, 0.05)

        sample_size = calculator.get_overall_sample_size()

        self.assertEqual(sample_size, test_sample_size)
        mock_get_single_sample_size.assert_called_once()
        assert_equal(mock_get_single_sample_size.call_args[0][0].alpha, DEFAULT_ALPHA)
        assert_equal(mock_get_single_sample_size.call_args[0][0].mde, test_mde)
        assert_equal(mock_get_single_sample_size.call_args[0][0].metric_variance, 0.0475)
        assert_equal(mock_get_single_sample_size.call_args[0][0].power, DEFAULT_POWER)
        assert_equal(mock_get_single_sample_size.call_args[0][1], NormalIndPower)

    @patch("sample_size.sample_size_calculator.SampleSizeCalculator._get_single_sample_size")
    def test_get_overall_sample_size_numeric(self, mock_get_single_sample_size):
        test_sample_size = 2000
        mock_get_single_sample_size.return_value = test_sample_size

        test_mde = 5
        test_variance = 500
        calculator = SampleSizeCalculator()
        calculator.register_numeric_metric(test_mde, test_variance)

        sample_size = calculator.get_overall_sample_size()

        self.assertEqual(sample_size, test_sample_size)
        mock_get_single_sample_size.assert_called_once()
        assert_equal(mock_get_single_sample_size.call_args[0][0].alpha, DEFAULT_ALPHA)
        assert_equal(mock_get_single_sample_size.call_args[0][0].mde, test_mde)
        assert_equal(mock_get_single_sample_size.call_args[0][0].metric_variance, test_variance)
        assert_equal(mock_get_single_sample_size.call_args[0][0].power, DEFAULT_POWER)
        assert_equal(mock_get_single_sample_size.call_args[0][1], TTestIndPower)

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
        test_variance = 5
        calculator = SampleSizeCalculator()
        calculator.register_ratio_metric(
            test_mde,
            test_numerator_mean,
            test_numerator_variance,
            test_denominator_mean,
            test_denominator_variance,
            test_covariance,
        )

        sample_size = calculator.get_overall_sample_size()

        self.assertEqual(sample_size, test_sample_size)
        mock_get_single_sample_size.assert_called_once()
        assert_equal(mock_get_single_sample_size.call_args[0][0].alpha, DEFAULT_ALPHA)
        assert_equal(mock_get_single_sample_size.call_args[0][0].mde, test_mde)
        assert_equal(mock_get_single_sample_size.call_args[0][0].metric_variance, test_variance)
        assert_equal(mock_get_single_sample_size.call_args[0][0].power, DEFAULT_POWER)
        assert_equal(mock_get_single_sample_size.call_args[0][1], TTestIndPower)
