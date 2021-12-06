import unittest
from unittest.mock import patch

from numpy.testing import assert_equal

import scripts.utils as utils
from sample_size.sample_size_calculator import SampleSizeCalculator


class UtilsTestCase(unittest.TestCase):
    def setUp(self):
        self.TEST_STR = "TEST"
        self.TEST_SAMPLE_SIZE = 2000

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

    @patch("scripts.utils.input")
    def test_get_float_input_float(self, mock_input):
        mock_input.return_value = " 0.05"
        result = utils.get_float_input(self.TEST_STR)

        self.assertEqual(result, 0.05)
        mock_input.called_once_with(f"Enter the {self.TEST_STR}: ")

    @patch("scripts.utils.input")
    def test_get_float_input_blank_str_allowed_non_float(self, mock_input):
        mock_input.return_value = "test"

        with self.assertRaises(Exception) as context:
            utils.get_float_input(self.TEST_STR)
            self.assertEqual(
                context.exception,
                Exception(f"Error: Please enter a float for the {self.TEST_STR}."),
            )

    @patch("scripts.utils.get_float_input")
    @patch("scripts.utils.input")
    def test_get_alpha(self, mock_input, mock_get_float_input):
        test_input_float = 0.01
        mock_input.return_value = "n"
        mock_get_float_input.return_value = test_input_float

        alpha = utils.get_alpha()

        self.assertEqual(alpha, test_input_float)
        mock_input.called_once_with("Do you want to use default alpha (0.05) for the power analysis? (y/n)")
        mock_get_float_input.called_once_with("alpha (between 0 and 0.3 inclusively)")

    @patch("scripts.utils.input")
    def test_get_alpha_default(self, mock_input):
        mock_input.return_value = "x"

        alpha = utils.get_alpha()

        self.assertEqual(alpha, None)

    @patch("scripts.utils.get_float_input")
    @patch("scripts.utils.input")
    def test_get_alpha_error(self, mock_input, mock_get_float_input):
        test_input_float = 0.5
        mock_input.return_value = "n"
        mock_get_float_input.return_value = test_input_float

        with self.assertRaises(Exception) as context:
            utils.get_alpha()

        self.assertEqual(
            context.exception.args[0],
            "Error: Please provide a float between 0 and 0.3 for alpha.",
        )

    @patch("scripts.utils.get_float_input")
    @patch("scripts.utils.input")
    def test_get_alpha_too_small(self, mock_input, mock_get_float_input):
        test_input_float = -0.1
        mock_input.return_value = "n"
        mock_get_float_input.return_value = test_input_float

        with self.assertRaises(Exception) as context:
            utils.get_alpha()

        self.assertEqual(
            context.exception.args[0],
            "Error: Please provide a float between 0 and 0.3 for alpha.",
        )

    @patch("scripts.utils.get_float_input")
    def test_get_mde(self, mock_get_float_input):
        test_metric_type = "boolean"
        test_mde = 0.01
        mock_get_float_input.return_value = test_mde

        mde = utils.get_mde(test_metric_type)

        self.assertEqual(mde, test_mde)
        mock_get_float_input.called_once_with(
            f"absolute minimum detectable effect for this {test_metric_type} \n"
            f"MDE: targeted treatment metric value minus the baseline value"
        )

    @patch("scripts.utils.input")
    def test_get_variable_from_input_boolean(self, mock_input):
        mock_input.return_value = " Boolean "

        metric_type = utils.get_metric_type_from_input()

        self.assertEqual(metric_type, "boolean")

    @patch("scripts.utils.input")
    def test_get_variable_from_input_numeric(self, mock_input):
        mock_input.return_value = " Numeric "

        metric_type = utils.get_metric_type_from_input()

        self.assertEqual(metric_type, "numeric")

    @patch("scripts.utils.input")
    def test_get_variable_from_input_ratio(self, mock_input):
        mock_input.return_value = " Ratio "

        metric_type = utils.get_metric_type_from_input()

        self.assertEqual(metric_type, "ratio")

    @patch("scripts.utils.input")
    def test_get_variable_from_input_error(self, mock_input):
        mock_input.return_value = "test"

        with self.assertRaises(Exception) as context:
            utils.get_metric_type_from_input()

        self.assertEqual(
            context.exception.args[0],
            "Error: Unexpected metric type. Please enter Boolean, Numeric, or Ratio.",
        )

    @patch("scripts.utils.get_float_input")
    def test_get_variable_parameters(self, mock_get_float_input):
        test_input_float = 5
        mock_get_float_input.return_value = test_input_float
        test_parameter_definitions = {
            "test": "test test",
            "case": "case case",
        }

        result = utils.get_variable_parameters(test_parameter_definitions)

        self.assertEqual(mock_get_float_input.call_count, 2)
        assert_equal(mock_get_float_input.call_args_list[0][0][0], "test test")
        assert_equal(mock_get_float_input.call_args_list[1][0][0], "case case")
        self.assertEqual(
            result,
            {
                "test": test_input_float,
                "case": test_input_float,
            },
        )

    @patch("scripts.utils.get_mde")
    @patch("scripts.utils.get_variable_parameters")
    @patch("scripts.utils.get_metric_type_from_input")
    def test_get_metric_metadata_from_input(
        self, mock_get_metric_type_from_input, mock_get_variable_parameters, mock_get_mde
    ):
        test_metric_type = "Boolean"
        test_metric_type_lower = "boolean"
        test_metric_metadata = {"test": 0.01}
        test_mde = 0.05
        mock_get_metric_type_from_input.return_value = test_metric_type
        mock_get_variable_parameters.return_value = test_metric_metadata
        mock_get_mde.return_value = test_mde
        test_metric_metadata = {"test": 0.01, "mde": test_mde}

        metric_type, metric_metadata = utils.get_metric_metadata_from_input()

        self.assertEqual(metric_type, test_metric_type_lower)
        self.assertEqual(metric_metadata, test_metric_metadata)
        mock_get_metric_type_from_input.assert_called_once()
        mock_get_variable_parameters.assert_called_once_with(utils.METRIC_PARAMETERS[test_metric_type_lower])
        mock_get_mde.assert_called_once_with(test_metric_type_lower)

    @patch("sample_size.sample_size_calculator.SampleSizeCalculator.register_bool_metric")
    def test_register_metric_boolean(self, mock_register_bool_metric):
        test_metric_type = "boolean"
        test_probability = 0.05
        test_mde = 0.02
        test_metric_metadata = {"probability": test_probability, "mde": test_mde}

        calculator = SampleSizeCalculator()
        utils.register_metric(test_metric_type, test_metric_metadata, calculator)

        mock_register_bool_metric.assert_called_once_with(probability=test_probability, mde=test_mde)

    @patch("sample_size.sample_size_calculator.SampleSizeCalculator.register_numeric_metric")
    def test_register_metric_numeric(self, mock_register_numeric_metric):
        test_metric_type = "numeric"
        test_variance = 5000.0
        test_mde = 5.0
        test_metric_metadata = {"variance": test_variance, "mde": test_mde}

        calculator = SampleSizeCalculator()
        utils.register_metric(test_metric_type, test_metric_metadata, calculator)

        mock_register_numeric_metric.assert_called_once_with(variance=test_variance, mde=test_mde)

    @patch("sample_size.sample_size_calculator.SampleSizeCalculator.register_ratio_metric")
    def test_register_metric_ratio(self, mock_register_ratio_metric):
        test_metric_type = "ratio"
        test_numerator_mean = 2000.0
        test_numerator_variance = 100000.0
        test_denominator_mean = 200.0
        test_denominator_variance = 2000.0
        test_covariance = 5000.0
        test_mde = 5.0
        test_metric_metadata = {
            "numerator_mean": test_numerator_mean,
            "numerator_variance": test_numerator_variance,
            "denominator_mean": test_denominator_mean,
            "denominator_variance": test_denominator_variance,
            "covariance": test_covariance,
            "mde": test_mde,
        }

        calculator = SampleSizeCalculator()
        utils.register_metric(test_metric_type, test_metric_metadata, calculator)

        mock_register_ratio_metric.assert_called_once_with(
            numerator_mean=test_numerator_mean,
            numerator_variance=test_numerator_variance,
            denominator_mean=test_denominator_mean,
            denominator_variance=test_denominator_variance,
            covariance=test_covariance,
            mde=test_mde,
        )
