import unittest
from unittest.mock import patch

from numpy.testing import assert_equal

from sample_size.sample_size_calculator import SampleSizeCalculator
from scripts.sample_size_run import main


class TestMain(unittest.TestCase):
    @patch("sample_size.sample_size_calculator.SampleSizeCalculator.get_overall_sample_size")
    @patch("scripts.utils.register_metric")
    @patch("scripts.utils.get_variable_from_input")
    @patch("scripts.utils.get_alpha")
    def test_main_alpha_input(
        self, mock_get_alpha, mock_get_variable_from_input, mock_register_metric, mock_get_overall_sample_size
    ):
        test_metric_type = "boolean"
        test_alpha = 0.01
        mock_get_alpha.return_value = test_alpha
        mock_get_variable_from_input.return_value = test_metric_type
        mock_get_overall_sample_size.return_value = 2000

        main()
        calculator = SampleSizeCalculator(test_alpha)

        mock_get_alpha.assert_called_once()
        mock_get_variable_from_input.assert_called_once()
        assert_equal(mock_register_metric.call_args[0][0], test_metric_type)
        assert_equal(mock_register_metric.call_args[0][1].alpha, test_alpha)
        assert_equal(mock_register_metric.call_args[0][1].power, calculator.power)
        mock_get_overall_sample_size.assert_called_once()

    @patch("sample_size.sample_size_calculator.SampleSizeCalculator.get_overall_sample_size")
    @patch("scripts.utils.register_metric")
    @patch("scripts.utils.get_variable_from_input")
    @patch("scripts.utils.get_alpha")
    def test_main_default_alpha(
        self, mock_get_alpha, mock_get_variable_from_input, mock_register_metric, mock_get_overall_sample_size
    ):
        test_metric_type = "boolean"
        test_alpha = None
        mock_get_alpha.return_value = test_alpha
        mock_get_variable_from_input.return_value = test_metric_type
        mock_get_overall_sample_size.return_value = 2000

        main()
        calculator = SampleSizeCalculator()

        mock_get_alpha.assert_called_once()
        mock_get_variable_from_input.assert_called_once()
        assert_equal(mock_register_metric.call_args[0][0], test_metric_type)
        assert_equal(mock_register_metric.call_args[0][1].alpha, calculator.alpha)
        assert_equal(mock_register_metric.call_args[0][1].power, calculator.power)
        mock_get_overall_sample_size.assert_called_once()
