import unittest
from io import StringIO
from unittest.mock import patch

from numpy.testing import assert_equal

from sample_size.sample_size_calculator import DEFAULT_ALPHA
from sample_size.sample_size_calculator import SampleSizeCalculator
from scripts.sample_size_run import main


class TestMain(unittest.TestCase):
    def setUp(self):
        self.DEFAULT_METRIC_TYPE = "boolean"
        self.DEFAULT_METRIC_METADATA = {"test": "case"}
        self.DEFAULT_SAMPLE_SIZE = 2000

    @patch("sample_size.sample_size_calculator.SampleSizeCalculator.get_sample_size")
    @patch("scripts.utils.register_metric")
    @patch("scripts.utils.get_metric_metadata_from_input")
    @patch("scripts.utils.get_alpha")
    def test_main_alpha_input(
        self,
        mock_get_alpha,
        get_metric_metadata_from_input,
        mock_register_metric,
        mock_get_sample_size,
    ):
        test_alpha = 0.01
        mock_get_alpha.return_value = test_alpha
        get_metric_metadata_from_input.return_value = (self.DEFAULT_METRIC_TYPE, self.DEFAULT_METRIC_METADATA)
        mock_get_sample_size.return_value = self.DEFAULT_SAMPLE_SIZE

        with patch("sys.stdout", new=StringIO()) as fakeOutput:
            main()
            self.assertEqual(
                fakeOutput.getvalue().strip(),
                "Sample size needed in each group: {:.3f}".format(self.DEFAULT_SAMPLE_SIZE),
            )

        calculator = SampleSizeCalculator(test_alpha)

        mock_get_alpha.assert_called_once()
        get_metric_metadata_from_input.assert_called_once()
        assert_equal(mock_register_metric.call_args[0][0], self.DEFAULT_METRIC_TYPE)
        assert_equal(mock_register_metric.call_args[0][1], self.DEFAULT_METRIC_METADATA)
        assert_equal(mock_register_metric.call_args[0][2].alpha, test_alpha)
        assert_equal(mock_register_metric.call_args[0][2].power, calculator.power)
        mock_get_sample_size.assert_called_once()

    @patch("sample_size.sample_size_calculator.SampleSizeCalculator.get_sample_size")
    @patch("scripts.utils.register_metric")
    @patch("scripts.utils.get_metric_metadata_from_input")
    @patch("scripts.utils.get_alpha")
    def test_main_default_alpha(
        self, mock_get_alpha, get_metric_metadata_from_input, mock_register_metric, mock_get_sample_size
    ):
        test_alpha = None
        mock_get_alpha.return_value = test_alpha
        get_metric_metadata_from_input.return_value = (self.DEFAULT_METRIC_TYPE, self.DEFAULT_METRIC_METADATA)
        mock_get_sample_size.return_value = self.DEFAULT_SAMPLE_SIZE

        with patch("sys.stdout", new=StringIO()) as fakeOutput:
            main()
            self.assertEqual(
                fakeOutput.getvalue().strip(),
                "Sample size needed in each group: {:.3f}".format(self.DEFAULT_SAMPLE_SIZE),
            )

        calculator = SampleSizeCalculator()

        mock_get_alpha.assert_called_once()
        get_metric_metadata_from_input.assert_called_once()
        assert_equal(mock_register_metric.call_args[0][0], self.DEFAULT_METRIC_TYPE)
        assert_equal(mock_register_metric.call_args[0][1], self.DEFAULT_METRIC_METADATA)
        assert_equal(mock_register_metric.call_args[0][2].alpha, DEFAULT_ALPHA)
        assert_equal(mock_register_metric.call_args[0][2].power, calculator.power)
        mock_get_sample_size.assert_called_once()

    @patch("sample_size.sample_size_calculator.SampleSizeCalculator.get_sample_size")
    @patch("scripts.utils.register_metric")
    @patch("scripts.utils.get_metric_metadata_from_input")
    @patch("scripts.utils.get_alpha")
    def test_main_exception_print(
        self, mock_get_alpha, get_metric_metadata_from_input, mock_register_metric, mock_get_sample_size
    ):
        test_alpha = None
        mock_get_alpha.return_value = test_alpha
        get_metric_metadata_from_input.return_value = (self.DEFAULT_METRIC_TYPE, self.DEFAULT_METRIC_METADATA)
        mock_get_sample_size.return_value = "test"

        with patch("sys.stdout", new=StringIO()) as fakeOutput:
            main()
            self.assertEqual(
                fakeOutput.getvalue().strip(),
                "Error! The calculator isn't able to calculate sample size due to "
                "\nUnknown format code 'f' for object of type 'str'",
            )

        calculator = SampleSizeCalculator()

        mock_get_alpha.assert_called_once()
        get_metric_metadata_from_input.assert_called_once()
        assert_equal(mock_register_metric.call_args[0][0], self.DEFAULT_METRIC_TYPE)
        assert_equal(mock_register_metric.call_args[0][1], self.DEFAULT_METRIC_METADATA)
        assert_equal(mock_register_metric.call_args[0][2].alpha, DEFAULT_ALPHA)
        assert_equal(mock_register_metric.call_args[0][2].power, calculator.power)
        mock_get_sample_size.assert_called_once()
