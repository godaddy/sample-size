import unittest
from unittest.mock import patch

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

    @patch(
        "sample_size.sample_size_calculator.SampleSizeCalculator._get_single_sample_size",
        side_effect=[100, 100, 1000, 1000],
    )
    @patch("sample_size.multiple_testing.MultipleTestingMixin._expected_average_power", side_effect=[0.5, 0.8])
    def test_get_multiple_sample_size(self, mock_expected_average_power, mock_get_single_sample_size):
        calculator = SampleSizeCalculator()
        calculator.register_metrics([self.test_metric, self.test_metric])
        sample_size = calculator.get_sample_size()
        self.assertEqual(mock_get_single_sample_size.call_count, 4)
        self.assertEqual(mock_expected_average_power.call_count, 2)
        self.assertEqual(sample_size, ((self.test_upper + self.test_lower) / 2 + self.test_upper) / 2)

    @patch(
        "sample_size.sample_size_calculator.SampleSizeCalculator._get_single_sample_size",
        side_effect=[100, 100, 1000, 1000],
    )
    @patch("sample_size.multiple_testing.MultipleTestingMixin._expected_average_power")
    def test_get_multiple_sample_size_no_recursion(self, mock_expected_average_power, mock_get_single_sample_size):
        mock_expected_average_power.return_value = 0.79

        calculator = SampleSizeCalculator()
        calculator.register_metrics([self.test_metric, self.test_metric])
        sample_size = calculator.get_sample_size()
        self.assertEqual(mock_get_single_sample_size.call_count, 4)
        mock_expected_average_power.assert_called_once_with(int((self.test_upper + self.test_lower) / 2))
        self.assertEqual(sample_size, (self.test_upper + self.test_lower) / 2)

    @patch(
        "sample_size.sample_size_calculator.SampleSizeCalculator._get_single_sample_size",
        side_effect=[100, 100, 1000, 1000],
    )
    @patch("sample_size.multiple_testing.MultipleTestingMixin._expected_average_power")
    def test_get_multiple_sample_size_too_many_recursion(
        self, mock_expected_average_power, mock_get_single_sample_size
    ):
        test_power = 0.8
        mock_expected_average_power.return_value = 0.1

        calculator = SampleSizeCalculator()
        calculator.register_metrics([self.test_metric, self.test_metric])

        with self.assertRaises(Exception) as context:
            calculator.get_sample_size()

        self.assertEqual(
            str(context.exception),
            f"Couldn't find a sample size that satisfies the power you requested: {test_power}",
        )

        self.assertEqual(mock_get_single_sample_size.call_count, 4)
        mock_expected_average_power.assert_called()

    @patch("numpy.isclose")
    @patch("sample_size.metrics.BooleanMetric.generate_p_value")
    @patch("statsmodels.stats.multitest.multipletests")
    def test_expected_average_power(self, mock_multipletests, mock_generate_p_value, mock_isclose):
        mock_generate_p_value.return_value = 0.01
        mock_multipletests.return_value = [1, 1, 1]
        mock_isclose.return_value = True

        calculator = SampleSizeCalculator()
        calculator.register_metrics([self.test_metric, self.test_metric, self.test_metric])
        replication = len(calculator.metrics) ** 2 * (calculator.variants - 1) * 100

        sample_size = calculator.get_multiple_sample_size(self.test_lower, self.test_upper)
        self.assertEqual(mock_generate_p_value.call_count, replication)
        self.assertEqual(sample_size, ((self.test_upper + self.test_lower) / 2))
