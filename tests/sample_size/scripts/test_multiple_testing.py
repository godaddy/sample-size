import unittest
from unittest.mock import patch

from numpy.testing import assert_equal

from sample_size.metrics import BooleanMetric
from sample_size.metrics import NumericMetric
from sample_size.metrics import RatioMetric
from sample_size.sample_size_calculator import DEFAULT_ALPHA
from sample_size.sample_size_calculator import DEFAULT_POWER
from sample_size.sample_size_calculator import SampleSizeCalculator
from sample_size.multiple_testing import MultipleTestingMixin


class MultipleTestingTestCase(unittest.TestCase):

    @patch("sample_size.multiple_testing.MultipleTestingMixin._find_sample_size")
    @patch("sample_size.sample_size_calculator.SampleSizeCalculator._get_single_sample_size")
    def test_get_multiple_sample_size_one_metric(self, mock_get_single_sample_size, mock_find_sample_size):
        test_sample_size = 2000
        mock_get_single_sample_size.return_value = test_sample_size

        test_metric_type = "boolean"
        test_probability = 0.05
        test_mde = 0.02
        test_metric_metadata = {"probability": test_probability, "mde": test_mde}

        calculator = SampleSizeCalculator(MultipleTestingMixin)
        calculator.register_metric(test_metric_type, test_metric_metadata)
        calculator.get_sample_size()
        mock_get_single_sample_size.assert_called_once()
        mock_find_sample_size.assert_not_called()

    @patch("sample_size.multiple_testing.MultipleTestingMixin._find_sample_size")
    @patch("sample_size.sample_size_calculator.SampleSizeCalculator._get_single_sample_size")
    def test_get_multiple_sample_size_multiple_metrics(self, mock_get_single_sample_size, mock_find_sample_size):
        test_sample_size = 2000
        test_m = 2
        mock_get_single_sample_size.return_value = test_sample_size

        test_metric_type = "boolean"
        test_probability = 0.05
        test_mde = 0.02
        test_metric_metadata = {"probability": test_probability, "mde": test_mde}

        calculator = SampleSizeCalculator(MultipleTestingMixin)
        calculator.register_metric(test_metric_type, test_metric_metadata)
        calculator.get_sample_size()

        mock_get_single_sample_size.assert_has_calls([(calculator.alpha, calculator.power),
                                                      (calculator.alpha / test_m, calculator.power)])
        mock_find_sample_size.assert_called_once_with(test_sample_size, test_sample_size)