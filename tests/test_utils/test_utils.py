import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from numpy.testing import assert_equal

import sample_size.utils.utils as utils
from sample_size.sample_size_calculator.sample_size_calculator import DEFAULT_ALPHA
from sample_size.sample_size_calculator.sample_size_calculator import DEFAULT_POWER
from sample_size.sample_size_calculator.sample_size_calculator import PowerAnalysisParameters


class UtilsTestCase(unittest.TestCase):
    def setUp(self):
        self.TEST_STR = "TEST"
        self.TEST_SAMPLE_SIZE = 2000

    def compare_power_analysis_parameters(self, parameters_a, parameters_b):
        self.assertEqual(parameters_a.alpha, parameters_b.alpha)
        self.assertEqual(parameters_a.power, parameters_b.power)
        self.assertEqual(parameters_a.mde, parameters_b.mde)

    def test_is_float(self):
        happy_test_str = "0.1"
        result = utils.is_float(happy_test_str)

        self.assertEqual(result, True)

        sad_test_str = "test"
        result = utils.is_float(sad_test_str)

        self.assertEqual(result, False)

        blank_test_str = " "
        result = utils.is_float(blank_test_str)

        self.assertEqual(result, False)

    @patch("sample_size.utils.utils.get_raw_input")
    def test_inputs_float(self, mock_get_raw_input):
        mock_get_raw_input.return_value = "0.05"
        result = utils.get_input(self.TEST_STR)

        self.assertEqual(result, 0.05)

    @patch("sample_size.utils.utils.get_raw_input")
    def test_get_input_blank_str(self, mock_get_raw_input):
        mock_get_raw_input.return_value = ""

        with self.assertRaises(Exception) as context:
            utils.get_input(self.TEST_STR)
            self.assertEqual(
                context.exception,
                Exception(f"Error: Please enter a float for the {self.TEST_STR}."),
            )

    @patch("sample_size.utils.utils.get_raw_input")
    def test_get_input_blank_str_allowed(self, mock_get_raw_input):
        mock_get_raw_input.return_value = " "

        result = utils.get_input(self.TEST_STR, allow_na=True)

        self.assertEqual(result, DEFAULT_ALPHA)

    @patch("sample_size.utils.utils.get_raw_input")
    def test_get_input_blank_str_allowed_non_float(self, mock_get_raw_input):
        mock_get_raw_input.return_value = "test"

        with self.assertRaises(Exception) as context:
            utils.get_input(self.TEST_STR, allow_na=True)
            self.assertEqual(
                context.exception,
                Exception(f"Error: Please enter a float for the {self.TEST_STR}."),
            )

    @patch("sample_size.utils.utils.get_input")
    def test_get_variable_parameters(self, mock_get_input):
        test_input_float = 5
        mock_get_input.return_value = test_input_float
        test_parameter_definitions = {
            "test": "test test",
            "case": "case case",
        }

        result = utils.get_variable_parameters(test_parameter_definitions)

        self.assertEqual(mock_get_input.call_count, 2)
        assert_equal(mock_get_input.call_args_list[0][0][0], "test test")
        assert_equal(mock_get_input.call_args_list[1][0][0], "case case")
        self.assertEqual(
            result,
            {
                "test": test_input_float,
                "case": test_input_float,
            },
        )

    @patch("sample_size.sample_size_calculator.sample_size_calculator.SampleSizeCalculators")
    @patch("sample_size.utils.utils.get_variable_parameters")
    def test_get_sample_size_boolean(self, mock_get_variable_parameters, mock_calculator):
        test_variable_name = "Boolean"
        test_probability = 0.05

        parameters_definitions = {"probability": "baseline probability"}

        mock_get_variable_parameters.return_value = {
            "probability": test_probability,
        }
        mock_calculator.get_boolean_sample_size.return_value = self.TEST_SAMPLE_SIZE

        sample_size = utils.get_sample_size(test_variable_name, mock_calculator)

        self.assertEqual(sample_size, self.TEST_SAMPLE_SIZE)
        mock_get_variable_parameters.assert_called_once_with(parameters_definitions)
        mock_calculator.get_boolean_sample_size.assert_called_once()
        mock_calculator.get_boolean_sample_size.assert_called_once_with(test_probability)

    @patch("sample_size.sample_size_calculator.sample_size_calculator.SampleSizeCalculators")
    @patch("sample_size.utils.utils.get_variable_parameters")
    def test_get_sample_size_numeric(self, mock_get_variable_parameters, mock_calculator):
        test_variable_name = "Numeric"
        test_mean = 50
        test_variance = 5000

        parameters_definitions = {"mean": "mean of the baseline metric", "variance": "variance of the baseline metric"}

        mock_get_variable_parameters.return_value = {
            "mean": test_mean,
            "variance": test_variance,
        }
        mock_calculator.get_numeric_sample_size.return_value = self.TEST_SAMPLE_SIZE

        sample_size = utils.get_sample_size(test_variable_name, mock_calculator)

        self.assertEqual(sample_size, self.TEST_SAMPLE_SIZE)
        mock_get_variable_parameters.assert_called_once_with(parameters_definitions)
        mock_calculator.get_numeric_sample_size.assert_called_once()
        assert_equal(mock_calculator.get_numeric_sample_size.call_args[0][0], test_mean)
        assert_equal(mock_calculator.get_numeric_sample_size.call_args[0][1], test_variance)

    @patch("sample_size.sample_size_calculator.sample_size_calculator.SampleSizeCalculators")
    @patch("sample_size.utils.utils.get_variable_parameters")
    def test_get_sample_size_ratio(self, mock_get_variable_parameters, mock_calculator):
        test_variable_name = "Ratio"
        test_numerator_mean = 2000
        test_numerator_variance = 100000
        test_denominator_mean = 200
        test_denominator_variance = 2000
        test_covariance = 5000

        parameters_definitions = {
            "numerator_mean": "mean of the baseline metric's numerator",
            "numerator_variance": "variance of the baseline metric's numerator",
            "denominator_mean": "mean of the baseline metric's denominator",
            "denominator_variance": "variance of the baseline metric's denominator",
            "covariance": "covariance between the baseline metric's numerator and denominator",
        }

        mock_get_variable_parameters.return_value = {
            "numerator_mean": test_numerator_mean,
            "numerator_variance": test_numerator_variance,
            "denominator_mean": test_denominator_mean,
            "denominator_variance": test_denominator_variance,
            "covariance": test_covariance,
        }
        mock_calculator.get_ratio_sample_size.return_value = self.TEST_SAMPLE_SIZE

        sample_size = utils.get_sample_size(test_variable_name, mock_calculator)

        self.assertEqual(sample_size, self.TEST_SAMPLE_SIZE)
        mock_get_variable_parameters.assert_called_once_with(parameters_definitions)
        mock_calculator.get_ratio_sample_size.assert_called_once()
        assert_equal(mock_calculator.get_ratio_sample_size.call_args[0][0], test_numerator_mean)
        assert_equal(mock_calculator.get_ratio_sample_size.call_args[0][1], test_numerator_variance)
        assert_equal(mock_calculator.get_ratio_sample_size.call_args[0][2], test_denominator_mean)
        assert_equal(mock_calculator.get_ratio_sample_size.call_args[0][3], test_denominator_variance)
        assert_equal(mock_calculator.get_ratio_sample_size.call_args[0][4], test_covariance)

    def test_get_sample_size_other(self):
        test_variable_name = self.TEST_STR
        mock_calculator = MagicMock()

        with self.assertRaises(Exception) as context:
            utils.get_sample_size(test_variable_name, mock_calculator)
            self.assertEqual(
                context.exception,
                Exception("Error: Unexpected variable name. Please use Boolean, Numeric, or Ratio."),
            )

    @patch("sample_size.utils.utils.get_input")
    @patch("sample_size.utils.utils.get_alpha")
    def test_get_power_analysis_input(self, mock_get_alpha, mock_get_input):
        test_alpha = 0.05
        test_mde = 0.1
        mock_get_alpha.return_value = test_alpha
        mock_get_input.side_effect = [test_alpha, test_mde]

        test_parameters = PowerAnalysisParameters
        test_parameters.alpha = test_alpha
        test_parameters.power = DEFAULT_POWER
        test_parameters.mde = test_mde

        parameters = utils.get_power_analysis_input()

        mock_get_alpha.assert_called_once_with(test_alpha)
        self.assertEqual(mock_get_input.call_count, 2)
        assert_equal(mock_get_input.call_args_list[0][0][0], "alpha (default 0.05)")
        assert_equal(mock_get_input.call_args_list[1][0][0], "minimum detectable effect")
        self.compare_power_analysis_parameters(parameters, test_parameters)

    @patch("sample_size.utils.utils.get_input")
    @patch("sample_size.utils.utils.get_alpha")
    def test_get_power_analysis_input_null_alpha(self, mock_get_alpha, mock_get_input):
        test_alpha = None
        test_mde = 0.1
        mock_get_alpha.return_value = test_alpha
        mock_get_input.side_effect = [test_alpha, test_mde]

        test_parameters = PowerAnalysisParameters
        test_parameters.alpha = DEFAULT_ALPHA
        test_parameters.power = DEFAULT_POWER
        test_parameters.mde = test_mde

        parameters = utils.get_power_analysis_input()

        mock_get_alpha.assert_called_once_with(test_alpha)
        self.assertEqual(mock_get_input.call_count, 2)
        assert_equal(mock_get_input.call_args_list[0][0][0], "alpha (default 0.05)")
        assert_equal(mock_get_input.call_args_list[1][0][0], "minimum detectable effect")
        self.compare_power_analysis_parameters(parameters, test_parameters)

    def test_get_alpha(self):
        test_alpha = DEFAULT_ALPHA
        alpha = utils.get_alpha(test_alpha)

        self.assertEqual(alpha, test_alpha)

    def test_get_probability_too_large(self):
        test_alpha = 0.5

        with self.assertRaises(Exception) as context:
            utils.get_alpha(test_alpha)
            self.assertEqual(
                context.exception,
                Exception("Error: Please provide a float between 0 and 0.3 for alpha."),
            )

    def test_get_probability_too_small(self):
        test_alpha = -0.1

        with self.assertRaises(Exception) as context:
            utils.get_alpha(test_alpha)
            self.assertEqual(
                context.exception,
                Exception("Error: Please provide a float between 0 and 0.3 for alpha."),
            )

    @patch("sample_size.utils.utils.get_raw_input")
    def test_get_variable_from_input_boolean(self, mock_get_raw_input):
        mock_get_raw_input.return_value = " Boolean "

        metric_type = utils.get_variable_from_input()

        self.assertEqual(metric_type, "boolean")

    @patch("sample_size.utils.utils.get_raw_input")
    def test_get_variable_from_input_numeric(self, mock_get_raw_input):
        mock_get_raw_input.return_value = " Numeric "

        metric_type = utils.get_variable_from_input()

        self.assertEqual(metric_type, "numeric")

    @patch("sample_size.utils.utils.get_raw_input")
    def test_get_variable_from_input_ratio(self, mock_get_raw_input):
        mock_get_raw_input.return_value = " Ratio "

        metric_type = utils.get_variable_from_input()

        self.assertEqual(metric_type, "ratio")

    @patch("sample_size.utils.utils.get_raw_input")
    def test_get_variable_from_input_error(self, mock_get_raw_input):
        mock_get_raw_input.return_value = "test"

        with self.assertRaises(Exception) as context:
            utils.get_variable_from_input()
            self.assertEqual(
                context.exception,
                Exception("Error: Unexpected metric type. Please enter Boolean, Numeric, or Ratio."),
            )
