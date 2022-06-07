import unittest
from io import StringIO
from unittest.mock import MagicMock
from unittest.mock import patch

from parameterized import parameterized

from sample_size.sample_size_calculator import DEFAULT_ALPHA
from sample_size.scripts import input_utils


class UtilsTestCase(unittest.TestCase):
    def setUp(self):
        self.DEFAULT_VARIANTS = 2
        self.TEST_STR = "TEST"

    def test_is_float(self):
        happy_test_str = "0.1"
        result = input_utils.is_float(happy_test_str)

        self.assertEqual(result, True)

        sad_test_str = "test"
        result = input_utils.is_float(sad_test_str)

        self.assertEqual(result, False)

        blank_test_str = " "
        result = input_utils.is_float(blank_test_str)

        self.assertEqual(result, False)

    def test_get_float_success(self):
        test_input_str = " 0.05"
        result = input_utils.get_float(test_input_str, self.TEST_STR)

        self.assertEqual(result, 0.05)

    def test_get_float_error(self):
        test_input_str = "test"

        with self.assertRaises(Exception) as context:
            input_utils.get_float(test_input_str, self.TEST_STR)

        self.assertEqual(
            str(context.exception),
            f"Error: Please enter a float for the {self.TEST_STR}.",
        )

    @patch("sample_size.scripts.input_utils.get_float")
    @patch("sample_size.scripts.input_utils.input")
    def test_get_alpha(self, mock_input, mock_get_float):
        test_input_float = 0.01
        test_input_str = "0.01"
        mock_input.return_value = test_input_str
        mock_get_float.return_value = test_input_float

        with patch("sys.stdout", new=StringIO()) as fakeOutput:
            alpha = input_utils.get_alpha()
            self.assertEqual(
                fakeOutput.getvalue().strip(),
                f"Using alpha ({alpha}) and default power (0.8)...",
            )

        self.assertEqual(alpha, test_input_float)
        mock_input.assert_called_once_with(
            "Enter the alpha between (between 0 and 0.3 inclusively) " "or press Enter to use default alpha=0.05: "
        )
        mock_get_float.assert_called_once_with(test_input_str, "alpha")

    @patch("sample_size.scripts.input_utils.get_float")
    @patch("sample_size.scripts.input_utils.input")
    def test_get_alpha_default(self, mock_input, mock_get_float):
        mock_input.return_value = " "

        with patch("sys.stdout", new=StringIO()) as fakeOutput:
            alpha = input_utils.get_alpha()
            self.assertEqual(
                fakeOutput.getvalue().strip(),
                "Using default alpha (0.05) and default power (0.8)...",
            )

        self.assertEqual(alpha, DEFAULT_ALPHA)
        mock_input.assert_called_once()
        mock_get_float.assert_not_called()

    @patch("sample_size.scripts.input_utils.get_float")
    @patch("sample_size.scripts.input_utils.input")
    def test_get_alpha_error(self, mock_input, mock_get_float):
        test_input_float = 0.5
        mock_input.return_value = "0.5"
        mock_get_float.return_value = test_input_float

        with self.assertRaises(Exception) as context:
            input_utils.get_alpha()

        self.assertEqual(
            context.exception.args[0],
            "Error: Please provide a float between 0 and 0.3 for alpha.",
        )
        mock_input.assert_called_once()
        mock_get_float.assert_called_once()

    @patch("sample_size.scripts.input_utils.get_float")
    @patch("sample_size.scripts.input_utils.input")
    def test_get_alpha_too_small(self, mock_input, mock_get_float):
        test_input_float = -0.1
        mock_input.return_value = "-0.1"
        mock_get_float.return_value = test_input_float

        with self.assertRaises(Exception) as context:
            input_utils.get_alpha()

        self.assertEqual(
            context.exception.args[0],
            "Error: Please provide a float between 0 and 0.3 for alpha.",
        )
        mock_input.assert_called_once()
        mock_get_float.assert_called_once()

    @patch("sample_size.scripts.input_utils.get_float")
    @patch("sample_size.scripts.input_utils.input")
    def test_get_mde(self, mock_input, mock_get_float):
        test_metric_type = "boolean"
        test_mde = 0.01
        mock_input.return_value = test_mde
        mock_get_float.return_value = test_mde

        mde = input_utils.get_mde(test_metric_type)

        self.assertEqual(mde, test_mde)
        mock_input.assert_called_once_with(
            f"Enter the absolute minimum detectable effect for this {test_metric_type} \n"
            f"MDE: targeted treatment metric value minus the baseline value: "
        )
        mock_get_float.assert_called_once_with(test_mde, "minimum detectable effect")

    @patch("sample_size.scripts.input_utils.input")
    def test_get_metric_type_boolean(self, mock_input):
        mock_input.return_value = " Boolean "

        metric_type = input_utils.get_metric_type()

        self.assertEqual(metric_type, "boolean")

    @patch("sample_size.scripts.input_utils.input")
    def test_get_metric_type_numeric(self, mock_input):
        mock_input.return_value = " Numeric "

        metric_type = input_utils.get_metric_type()

        self.assertEqual(metric_type, "numeric")

    @patch("sample_size.scripts.input_utils.input")
    def test_get_metric_type_ratio(self, mock_input):
        mock_input.return_value = " Ratio "

        metric_type = input_utils.get_metric_type()

        self.assertEqual(metric_type, "ratio")

    @patch("sample_size.scripts.input_utils.input")
    def test_get_metric_type_error(self, mock_input):
        mock_input.return_value = "test"

        with self.assertRaises(Exception) as context:
            input_utils.get_metric_type()

        self.assertEqual(
            context.exception.args[0],
            "Error: Unexpected metric type. Please enter Boolean, Numeric, or Ratio.",
        )

    @patch("sample_size.scripts.input_utils.get_float")
    @patch("sample_size.scripts.input_utils.input")
    def test_get_metric_parameters(self, mock_input, mock_get_float):
        test_input_float = 5
        mock_input.return_value = test_input_float
        mock_get_float.return_value = test_input_float
        test_parameter_definitions = {
            "test": "test test",
            "case": "case case",
        }

        result = input_utils.get_metric_parameters(test_parameter_definitions)

        self.assertEqual(mock_input.call_count, len(test_parameter_definitions))
        self.assertEqual(mock_input.call_args_list[0][0][0], "Enter the test test: ")
        self.assertEqual(mock_get_float.call_count, len(test_parameter_definitions))
        self.assertEqual(mock_get_float.call_args_list[0][0][0], test_input_float)
        self.assertEqual(mock_get_float.call_args_list[0][0][1], "test test")
        self.assertEqual(
            result,
            {
                "test": test_input_float,
                "case": test_input_float,
            },
        )

    @patch("sample_size.scripts.input_utils.input")
    def test_get_variants(self, mock_input):
        test_input_int = 2
        test_input_str = "2"
        mock_input.return_value = test_input_str

        variants = input_utils.get_variants()

        self.assertEqual(variants, test_input_int)
        mock_input.assert_called_once_with(
            "Enter the number of cohorts for this test or Press Enter to use default variant = 2 if you have only 1 "
            "control and 1 treatment. \n"
            "definition: Control + number of treatments: "
        )

    @patch("sample_size.scripts.input_utils.input")
    def test_get_variants_default(self, mock_input):
        mock_input.return_value = " "

        with patch("sys.stdout", new=StringIO()) as fakeOutput:
            variants = input_utils.get_variants()
            self.assertEqual(
                fakeOutput.getvalue().strip(),
                "Using default variants(2)...",
            )

        self.assertEqual(variants, self.DEFAULT_VARIANTS)
        mock_input.assert_called_once()

    @patch("sample_size.scripts.input_utils.input")
    def test_get_variants_not_int(self, mock_input):
        test_input_str = "2.5"
        mock_input.return_value = test_input_str
        with self.assertRaises(Exception) as context:
            input_utils.get_variants()

        self.assertEqual(
            context.exception.args[0],
            "Error: Please enter a positive integer for the number of variants.",
        )

        mock_input.assert_called_once()

    @patch("sample_size.scripts.input_utils.input")
    def test_get_variants_too_small(self, mock_input):
        test_input_str = "1"
        mock_input.return_value = test_input_str
        with self.assertRaises(Exception) as context:
            input_utils.get_variants()

        self.assertEqual(
            context.exception.args[0],
            "Error: An experiment must contain at least 2 variants.",
        )

        mock_input.assert_called_once()

    @patch("sample_size.scripts.input_utils.input")
    def test_get_variants_negative(self, mock_input):
        test_input_str = "-3"
        mock_input.return_value = test_input_str
        with self.assertRaises(Exception) as context:
            input_utils.get_variants()

        self.assertEqual(
            context.exception.args[0],
            "Error: Please enter a positive integer for the number of variants.",
        )

        mock_input.assert_called_once()

    @patch("sample_size.scripts.input_utils.input")
    def test_register_another_metric_yes(self, mock_input):
        test_input_str = "y"
        mock_input.return_value = test_input_str
        register = input_utils.register_another_metric()

        self.assertTrue(register)
        mock_input.assert_called_once_with("Are you going to register another metric? (y/n)")

    @patch("sample_size.scripts.input_utils.input")
    def test_register_another_metric_no(self, mock_input):
        test_input_str = "n"
        mock_input.return_value = test_input_str
        register = input_utils.register_another_metric()

        self.assertFalse(register)
        mock_input.assert_called_once()

    @patch("sample_size.scripts.input_utils.input")
    def test_register_another_metric_default(self, mock_input):
        test_input_str = ""
        mock_input.return_value = test_input_str
        register = input_utils.register_another_metric()

        self.assertFalse(register)
        mock_input.assert_called_once()

    @patch("sample_size.scripts.input_utils.input")
    def test_register_another_metric_invalid(self, mock_input):
        test_input_str = "yes"
        mock_input.return_value = test_input_str

        with self.assertRaises(Exception) as context:
            input_utils.register_another_metric()

        self.assertEqual(
            context.exception.args[0],
            "Error: Please enter 'y' or 'n'.",
        )

        mock_input.assert_called_once()

    @parameterized.expand(
        [
            ("boolean", {"probability": 0.05, "mde": 0.02}),
            ("numeric", {"variance": 500, "mde": 5}),
            (
                "ratio",
                {
                    "numerator_mean": 2000,
                    "numerator_variance": 100000,
                    "denominator_mean": 200,
                    "denominator_variance": 2000,
                    "covariance": 5000,
                    "mde": 10,
                },
            ),
        ]
    )
    @patch("sample_size.scripts.input_utils.get_mde")
    @patch("sample_size.scripts.input_utils.get_metric_parameters")
    @patch("sample_size.scripts.input_utils.get_metric_type")
    def test_get_metric(
        self, test_metric_type, test_metric_metadata, mock_get_metric_type, mock_get_metric_parameters, mock_get_mde
    ):
        test_mde = 0.05
        mock_get_metric_type.return_value = test_metric_type
        mock_get_metric_parameters.return_value = test_metric_metadata
        mock_get_mde.return_value = test_mde

        metric = input_utils._get_metric()
        self.assertEqual(
            metric,
            {"metric_type": test_metric_type, "metric_metadata": test_metric_metadata},
        )
        mock_get_metric_type.assert_called_once()
        mock_get_metric_parameters.assert_called_once_with(input_utils.METRIC_PARAMETERS[test_metric_type])
        mock_get_mde.assert_called_once_with(test_metric_type)

    @patch("sample_size.scripts.input_utils.register_another_metric")
    @patch("sample_size.scripts.input_utils._get_metric")
    def test_get_metrics_single(self, mock_get_metric, mock_register_another_metric):
        metric = MagicMock()
        mock_register_another_metric.return_value = False
        mock_get_metric.return_value = metric
        metrics = input_utils.get_metrics()
        mock_get_metric.assert_called_once()
        mock_register_another_metric.assert_called_once()
        self.assertEqual(metrics, [metric])

    @patch("sample_size.scripts.input_utils.register_another_metric", side_effect=[True, True, False])
    @patch("sample_size.scripts.input_utils._get_metric")
    def test_get_metrics_multiple(self, mock_get_metric, mock_register_another_metric):
        metric = MagicMock()
        mock_get_metric.return_value = metric
        metrics = input_utils.get_metrics()
        self.assertEqual(mock_get_metric.call_count, 3)
        self.assertEqual(mock_register_another_metric.call_count, 3)
        self.assertEqual(metrics, [metric, metric, metric])
