import unittest
from io import StringIO
from unittest.mock import MagicMock
from unittest.mock import patch

from sample_size.scripts.sample_size_run import main


class TestMain(unittest.TestCase):
    def setUp(self):
        self.DEFAULT_METRIC_TYPE = "boolean"
        self.DEFAULT_METRIC_METADATA = {"test": "case"}
        self.DEFAULT_SAMPLE_SIZE = 2000

    @patch("sample_size.sample_size_calculator.SampleSizeCalculator")
    @patch("sample_size.scripts.utils.register_metric")
    @patch("sample_size.scripts.utils.get_metric_metadata_from_input")
    @patch("sample_size.scripts.utils.get_alpha")
    def test_main_alpha_input(
        self,
        mock_get_alpha,
        get_metric_metadata_from_input,
        mock_register_metric,
        mock_calculator,
    ):
        test_alpha = 0.01
        mock_get_alpha.return_value = test_alpha
        get_metric_metadata_from_input.return_value = (self.DEFAULT_METRIC_TYPE, self.DEFAULT_METRIC_METADATA)
        calculator_obj = MagicMock()
        calculator_obj.get_sample_size.return_value = self.DEFAULT_SAMPLE_SIZE
        mock_calculator.return_value = calculator_obj

        with patch("sys.stdout", new=StringIO()) as fakeOutput:
            main()
            self.assertEqual(
                fakeOutput.getvalue().strip(),
                "Sample size needed in each group: {:.3f}".format(self.DEFAULT_SAMPLE_SIZE),
            )

        mock_get_alpha.assert_called_once()
        get_metric_metadata_from_input.assert_called_once()
        mock_register_metric.assert_called_once_with(
            self.DEFAULT_METRIC_TYPE,
            self.DEFAULT_METRIC_METADATA,
            calculator_obj,
        )
        calculator_obj.get_sample_size.assert_called_once()
        mock_calculator.assert_called_once_with(test_alpha)

    @patch("sample_size.sample_size_calculator.SampleSizeCalculator")
    @patch("sample_size.scripts.utils.register_metric")
    @patch("sample_size.scripts.utils.get_metric_metadata_from_input")
    @patch("sample_size.scripts.utils.get_alpha")
    def test_main_exception_print(
        self, mock_get_alpha, get_metric_metadata_from_input, mock_register_metric, mock_calculator
    ):
        error_message = "wrong alpha"
        mock_get_alpha.side_effect = Exception(error_message)

        with patch("sys.stdout", new=StringIO()) as fakeOutput:
            main()
            self.assertEqual(
                fakeOutput.getvalue().strip(),
                f"Error! The calculator isn't able to calculate sample size due to \n{error_message}",
            )

        mock_get_alpha.assert_called_once()
        mock_calculator.assert_not_called()
        get_metric_metadata_from_input.assert_not_called()
        mock_register_metric.assert_not_called()
