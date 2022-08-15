import unittest
from io import StringIO
from unittest.mock import MagicMock
from unittest.mock import patch

from sample_size.scripts.sample_size_run import main


class TestMain(unittest.TestCase):
    def setUp(self):
        self.DEFAULT_METRIC_TYPE = "boolean"
        self.DEFAULT_VARIANTS = 2
        self.DEFAULT_METRIC_METADATA = {"test": "case"}
        self.DEFAULT_SAMPLE_SIZE = 2000

    @patch("sample_size.sample_size_calculator.SampleSizeCalculator")
    @patch("sample_size.scripts.input_utils.get_metrics")
    @patch("sample_size.scripts.input_utils.get_alpha")
    @patch("sample_size.scripts.input_utils.get_variants")
    def test_main_alpha_input(
        self,
        mock_get_variants,
        mock_get_alpha,
        mock_get_metrics,
        mock_calculator,
    ):
        test_alpha = 0.01
        mock_get_variants.return_value = self.DEFAULT_VARIANTS
        mock_get_alpha.return_value = test_alpha
        mock_get_metrics.return_value = [
            {"metric_type": self.DEFAULT_METRIC_TYPE, "metric_metadata": self.DEFAULT_METRIC_METADATA}
        ]
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
        mock_get_variants.assert_called_once()
        mock_get_metrics.assert_called_once()
        calculator_obj.register_metrics.assert_called_once_with(
            [{"metric_type": self.DEFAULT_METRIC_TYPE, "metric_metadata": self.DEFAULT_METRIC_METADATA}]
        )
        calculator_obj.get_sample_size.assert_called_once()
        mock_calculator.assert_called_once_with(test_alpha, self.DEFAULT_VARIANTS)

    @patch("sample_size.sample_size_calculator.SampleSizeCalculator")
    @patch("sample_size.scripts.input_utils.get_metrics")
    @patch("sample_size.scripts.input_utils.get_alpha")
    def test_main_exception_print(
        self,
        mock_get_alpha,
        mock_get_metrics,
        mock_calculator,
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
        mock_get_metrics.assert_not_called()
