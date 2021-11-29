import unittest
from unittest.mock import patch

from numpy.testing import assert_equal

from sample_size.sample_size_calculator.sample_size_calculator import DEFAULT_ALPHA
from sample_size.sample_size_calculator.sample_size_calculator import DEFAULT_POWER
from sample_size.sample_size_calculator.sample_size_calculator import BaseSampleSizeCalculator
from sample_size.sample_size_calculator.sample_size_calculator import PowerAnalysisParameters
from sample_size.sample_size_calculator.sample_size_calculator import PowerAnalysisType
from sample_size.sample_size_calculator.sample_size_calculator import SampleSizeCalculators
from sample_size.sample_size_calculator.variables import Boolean
from sample_size.sample_size_calculator.variables import Numeric
from sample_size.sample_size_calculator.variables import Ratio


class BaseSampleSizeCalculatorTestCase(unittest.TestCase):
    def setUp(self):
        self.DEFAULT_VARIABLE_VARIANCE = 0.25
        self.DEFAULT_VARIABLE_ANALYSIS_TYPE_ZTEST = PowerAnalysisType.ZTestPower
        self.DEFAULT_VARIABLE_ANALYSIS_TYPE_TTEST = PowerAnalysisType.TTestPower
        self.DEFAULT_POWER_ANALYSIS_PARAMETERS = PowerAnalysisParameters()
        self.DEFAULT_POWER_ANALYSIS_PARAMETERS.mde = 0.02
        self.DEFAULT_MOCK_VALUE = 99
        self.DEFAULT_ALPHA = DEFAULT_ALPHA
        self.DEFAULT_POWER = DEFAULT_POWER

    @patch("sample_size.sample_size_calculator.sample_size_calculator.BaseSampleSizeCalculator.get_std_effect_size")
    def test_base_sample_size_calculator_constructor_sets_params(self, mock_get_std_effect_size):
        mock_get_std_effect_size.return_value = self.DEFAULT_MOCK_VALUE

        calculator = BaseSampleSizeCalculator(
            self.DEFAULT_VARIABLE_VARIANCE,
            self.DEFAULT_VARIABLE_ANALYSIS_TYPE_ZTEST,
            self.DEFAULT_POWER_ANALYSIS_PARAMETERS,
        )

        mock_get_std_effect_size.called_once()
        self.assertEqual(calculator.variable_variance, self.DEFAULT_VARIABLE_VARIANCE)
        self.assertEqual(calculator.variable_analysis_type, self.DEFAULT_VARIABLE_ANALYSIS_TYPE_ZTEST)
        self.assertEqual(calculator.power_analysis_parameters, self.DEFAULT_POWER_ANALYSIS_PARAMETERS)
        self.assertEqual(calculator.std_effect_size, self.DEFAULT_MOCK_VALUE)

    def test_get_std_effect_size(self):
        calculator = BaseSampleSizeCalculator(
            self.DEFAULT_VARIABLE_VARIANCE,
            self.DEFAULT_VARIABLE_ANALYSIS_TYPE_ZTEST,
            self.DEFAULT_POWER_ANALYSIS_PARAMETERS,
        )
        std_effect_size = calculator.get_std_effect_size()

        self.assertEqual(std_effect_size, 0.04)

    @patch("statsmodels.stats.power.NormalIndPower.solve_power")
    def test_get_base_sample_size_ztest(self, mock_solve_power):
        power_analysis_parameters = self.DEFAULT_POWER_ANALYSIS_PARAMETERS
        calculator = BaseSampleSizeCalculator(
            self.DEFAULT_VARIABLE_VARIANCE,
            self.DEFAULT_VARIABLE_ANALYSIS_TYPE_ZTEST,
            power_analysis_parameters,
        )

        test_sample_size = 2000
        mock_solve_power.return_value = test_sample_size
        sample_size = calculator.get_base_sample_size()

        self.assertEqual(sample_size, test_sample_size)
        mock_solve_power.assert_called_once()
        assert_equal(mock_solve_power.call_args[1]["effect_size"], calculator.std_effect_size)
        assert_equal(mock_solve_power.call_args[1]["alpha"], self.DEFAULT_ALPHA)
        assert_equal(mock_solve_power.call_args[1]["power"], self.DEFAULT_POWER)
        assert_equal(mock_solve_power.call_args[1]["ratio"], 1)
        assert_equal(mock_solve_power.call_args[1]["alternative"], "two-sided")

    @patch("statsmodels.stats.power.TTestIndPower.solve_power")
    def test_get_base_sample_size_ttest(self, mock_solve_power):
        power_analysis_parameters = self.DEFAULT_POWER_ANALYSIS_PARAMETERS
        calculator = BaseSampleSizeCalculator(
            self.DEFAULT_VARIABLE_VARIANCE,
            self.DEFAULT_VARIABLE_ANALYSIS_TYPE_TTEST,
            power_analysis_parameters,
        )

        test_sample_size = 2000
        mock_solve_power.return_value = test_sample_size
        sample_size = calculator.get_base_sample_size()

        self.assertEqual(sample_size, test_sample_size)
        mock_solve_power.assert_called_once()
        assert_equal(mock_solve_power.call_args[1]["effect_size"], calculator.std_effect_size)
        assert_equal(mock_solve_power.call_args[1]["alpha"], self.DEFAULT_ALPHA)
        assert_equal(mock_solve_power.call_args[1]["power"], self.DEFAULT_POWER)
        assert_equal(mock_solve_power.call_args[1]["ratio"], 1)
        assert_equal(mock_solve_power.call_args[1]["alternative"], "two-sided")


class SampleSizeCalculatorsTestCase(unittest.TestCase):
    def setUp(self):
        self.DEFAULT_MDE = 0.05
        self.DEFAULT_ALPHA = 0.05
        self.DEFAULT_POWER = 0.80
        self.DEFAULT_POWER_ANALYSIS_PARAMETERS = PowerAnalysisParameters()
        self.DEFAULT_POWER_ANALYSIS_PARAMETERS.mde = self.DEFAULT_MDE
        self.DEFAULT_SAMPLE_SIZE = 2000

    def test_sample_size_calculators_constructor_sets_params(self):
        calculators = SampleSizeCalculators(
            self.DEFAULT_MDE,
            self.DEFAULT_ALPHA,
            self.DEFAULT_POWER,
        )
        power_analysis_parameters = self.DEFAULT_POWER_ANALYSIS_PARAMETERS
        power_analysis_parameters.mde = self.DEFAULT_MDE
        power_analysis_parameters.alpha = self.DEFAULT_ALPHA
        power_analysis_parameters.power = self.DEFAULT_POWER

        self.assertEqual(calculators.power_analysis_parameters.alpha, power_analysis_parameters.alpha)
        self.assertEqual(calculators.power_analysis_parameters.power, power_analysis_parameters.power)
        self.assertEqual(calculators.power_analysis_parameters.mde, power_analysis_parameters.mde)

    @patch("sample_size.sample_size_calculator.sample_size_calculator.BaseSampleSizeCalculator.get_base_sample_size")
    def test_get_sample_size(self, mock_get_base_sample_size):
        mock_get_base_sample_size.return_value = self.DEFAULT_SAMPLE_SIZE
        test_variance = 0.25
        test_power_analysis_type = PowerAnalysisType.ZTestPower

        calculators = SampleSizeCalculators(
            self.DEFAULT_MDE,
            self.DEFAULT_ALPHA,
            self.DEFAULT_POWER,
        )
        sample_size = calculators.get_sample_size(test_variance, test_power_analysis_type)

        mock_get_base_sample_size.assert_called_once()
        self.assertEqual(sample_size, self.DEFAULT_SAMPLE_SIZE)

    @patch("sample_size.sample_size_calculator.sample_size_calculator.SampleSizeCalculators.get_sample_size")
    def test_get_boolean_sample_size(self, mock_get_sample_size):
        mock_get_sample_size.return_value = self.DEFAULT_SAMPLE_SIZE
        test_probability = 0.05
        bool_obj = Boolean(test_probability)

        calculators = SampleSizeCalculators(
            self.DEFAULT_MDE,
            self.DEFAULT_ALPHA,
            self.DEFAULT_POWER,
        )

        sample_size = calculators.get_boolean_sample_size(test_probability)

        self.assertEqual(sample_size, self.DEFAULT_SAMPLE_SIZE)
        mock_get_sample_size.assert_called_once()
        assert_equal(mock_get_sample_size.call_args[0][0], bool_obj.variance)
        assert_equal(mock_get_sample_size.call_args[0][1], PowerAnalysisType.ZTestPower)

    @patch("sample_size.sample_size_calculator.sample_size_calculator.SampleSizeCalculators.get_sample_size")
    def test_get_numeric_sample_size(self, mock_get_sample_size):
        mock_get_sample_size.return_value = self.DEFAULT_SAMPLE_SIZE
        test_variance = 5000
        bool_obj = Numeric(test_variance)

        calculators = SampleSizeCalculators(
            self.DEFAULT_MDE,
            self.DEFAULT_ALPHA,
            self.DEFAULT_POWER,
        )

        sample_size = calculators.get_numeric_sample_size(test_variance)

        self.assertEqual(sample_size, self.DEFAULT_SAMPLE_SIZE)
        mock_get_sample_size.assert_called_once()
        assert_equal(mock_get_sample_size.call_args[0][0], bool_obj.variance)
        assert_equal(mock_get_sample_size.call_args[0][1], PowerAnalysisType.TTestPower)

    @patch("sample_size.sample_size_calculator.sample_size_calculator.SampleSizeCalculators.get_sample_size")
    def test_get_ratio_sample_size(self, mock_get_sample_size):
        mock_get_sample_size.return_value = self.DEFAULT_SAMPLE_SIZE
        test_numerator_mean = 2000
        test_numerator_variance = 100000
        test_denominator_mean = 200
        test_denominator_variance = 2000
        test_covariance = 5000
        bool_obj = Ratio(
            test_numerator_mean,
            test_numerator_variance,
            test_denominator_mean,
            test_denominator_variance,
            test_covariance,
        )

        calculators = SampleSizeCalculators(
            self.DEFAULT_MDE,
            self.DEFAULT_ALPHA,
            self.DEFAULT_POWER,
        )

        sample_size = calculators.get_ratio_sample_size(
            test_numerator_mean,
            test_numerator_variance,
            test_denominator_mean,
            test_denominator_variance,
            test_covariance,
        )

        self.assertEqual(sample_size, self.DEFAULT_SAMPLE_SIZE)
        mock_get_sample_size.assert_called_once()
        assert_equal(mock_get_sample_size.call_args[0][0], bool_obj.variance)
        assert_equal(mock_get_sample_size.call_args[0][1], PowerAnalysisType.TTestPower)
