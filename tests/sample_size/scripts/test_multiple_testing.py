import unittest
from unittest.mock import patch

from parameterized import parameterized

from sample_size.sample_size_calculator import SampleSizeCalculator


class MultipleTestingTestCase(unittest.TestCase):
    def setUp(self):
        self.test_lower = 100
        self.test_upper = 1000
        self.test_metric_type = "boolean"
        self.test_probability = 0.05
        self.test_mde = 0.02
        self.test_metric_metadata = {"probability": self.test_probability, "mde": self.test_mde}
        self.test_metric = {"metric_type": self.test_metric_type, "metric_metadata": self.test_metric_metadata}

    @parameterized.expand([(0,), (1,)])
    @patch(
        "sample_size.sample_size_calculator.SampleSizeCalculator._get_single_sample_size",
        side_effect=[100, 100, 100, 1000, 1000, 1000],
    )
    @patch("sample_size.multiple_testing.MultipleTestingMixin._expected_average_power")
    def test_get_multiple_sample_size_quickly_converge(
        self, power_guess, mock_expected_average_power, mock_get_single_sample_size
    ):
        mock_expected_average_power.side_effect = [power_guess, 0.8]
        calculator = SampleSizeCalculator()
        calculator.register_metrics([self.test_metric, self.test_metric, self.test_metric])
        expected_call_count = len(calculator.metrics) * 2
        sample_size = calculator.get_sample_size()
        self.assertEqual(mock_get_single_sample_size.call_count, expected_call_count)
        self.assertEqual(mock_expected_average_power.call_count, 2)
        self.assertEqual(
            sample_size,
            ((self.test_upper + self.test_lower) / 2 + (self.test_lower if power_guess else self.test_upper)) / 2,
        )

    @patch(
        "sample_size.sample_size_calculator.SampleSizeCalculator._get_single_sample_size",
        side_effect=[100, 100, 100, 1000, 1000, 1000],
    )
    @patch("sample_size.multiple_testing.MultipleTestingMixin._expected_average_power")
    def test_get_multiple_sample_size_no_recursion(self, mock_expected_average_power, mock_get_single_sample_size):
        mock_expected_average_power.return_value = 0.79

        calculator = SampleSizeCalculator()
        calculator.register_metrics([self.test_metric, self.test_metric, self.test_metric])
        expected_call_count = len(calculator.metrics) * 2

        sample_size = calculator.get_sample_size()
        self.assertEqual(mock_get_single_sample_size.call_count, expected_call_count)
        mock_expected_average_power.assert_called_once_with(int((self.test_upper + self.test_lower) / 2))
        self.assertEqual(sample_size, (self.test_upper + self.test_lower) / 2)

    @patch("sample_size.multiple_testing.MultipleTestingMixin._expected_average_power")
    def test_get_multiple_sample_size_no_solution(self, mock_expected_power):
        mock_expected_power.return_value = 0
        test_power = 0.8
        calculator = SampleSizeCalculator()
        calculator.register_metrics([self.test_metric, self.test_metric])

        with self.assertRaises(Exception) as context:
            calculator.get_sample_size()
        self.assertEqual(
            str(context.exception),
            f"Couldn't find a sample size that satisfies the power you requested: {test_power}",
        )

    @parameterized.expand([(10,), (100,), (1000,)])
    def test_expected_average_power(self, test_size):
        calculator = SampleSizeCalculator()
        calculator.register_metrics([self.test_metric, self.test_metric, self.test_metric])
        expected_power = calculator._expected_average_power(test_size)
        inflated_power = calculator._expected_average_power(test_size * 10)
        self.assertTrue(inflated_power > expected_power)
