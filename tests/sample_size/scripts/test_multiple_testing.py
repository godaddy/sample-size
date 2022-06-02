import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from sample_size.sample_size_calculator import SampleSizeCalculator


class MultipleTestingTestCase(unittest.TestCase):
    @patch("sample_size.sample_size_calculator.SampleSizeCalculator._get_single_sample_size", side_effect=[100, 1000])
    @patch("sample_size.multiple_testing.MultipleTestingMixin._expected_average_power", side_effect=[0.5, 0.8])
    def test_get_multiple_sample_size(self, mock_expected_average_power, mock_get_single_sample_size):
        test_lower = 100
        test_upper = 1000
        test_metric_type = "boolean"
        test_probability = 0.05
        test_mde = 0.02
        test_metric_metadata = {"probability": test_probability, "mde": test_mde}
        calculator = SampleSizeCalculator()
        calculator.register_metrics(
            [
                {"metric_type": test_metric_type, "metric_metadata": test_metric_metadata},
                {"metric_type": test_metric_type, "metric_metadata": test_metric_metadata},
            ]
        )
        sample_size = calculator.get_sample_size()
        self.assertEqual(mock_get_single_sample_size.call_count, 4)
        self.assertEqual(mock_expected_average_power.call_count, 2)
        self.assertEqual(sample_size, ((test_upper + test_lower) / 2 + test_upper) / 2)

    @patch("sample_size.sample_size_calculator.SampleSizeCalculator._get_single_sample_size", side_effect=[100, 1000])
    @patch("sample_size.multiple_testing.MultipleTestingMixin._expected_average_power")
    def test_get_multiple_sample_size_no_recursion(self, mock_expected_average_power, mock_get_single_sample_size):
        test_lower = 100
        test_upper = 1000
        test_metric_type = "boolean"
        test_probability = 0.05
        test_mde = 0.02
        test_metric_metadata = {"probability": test_probability, "mde": test_mde}
        mock_expected_average_power.return_value = 0.79

        calculator = SampleSizeCalculator()
        calculator.register_metrics(
            [
                {"metric_type": test_metric_type, "metric_metadata": test_metric_metadata},
                {"metric_type": test_metric_type, "metric_metadata": test_metric_metadata},
            ]
        )
        sample_size = calculator.get_sample_size()
        self.assertEqual(mock_get_single_sample_size.call_count, 4)
        mock_expected_average_power.assert_called_once_with(int((test_upper + test_lower) / 2))
        self.assertEqual(sample_size, (test_upper + test_lower) / 2)

    @patch("sample_size.sample_size_calculator.SampleSizeCalculator._get_single_sample_size", side_effect=[100, 1000])
    @patch("sample_size.multiple_testing.MultipleTestingMixin._expected_average_power")
    def test_get_multiple_sample_size_too_many_recursion(
        self, mock_expected_average_power, mock_get_single_sample_size
    ):
        test_lower = 100
        test_upper = 1000
        test_metric_type = "boolean"
        test_probability = 0.05
        test_mde = 0.02
        test_metric_metadata = {"probability": test_probability, "mde": test_mde}
        mock_expected_average_power.return_value = 0.1

        calculator = SampleSizeCalculator()
        calculator.register_metrics(
            [
                {"metric_type": test_metric_type, "metric_metadata": test_metric_metadata},
                {"metric_type": test_metric_type, "metric_metadata": test_metric_metadata},
            ]
        )
        calculator.get_sample_size()
        self.assertEqual(mock_get_single_sample_size.call_count, 4)
        mock_expected_average_power.assert_called()
        with self.assertRaises(Exception):
            calculator.get_multiple_sample_size(test_lower, test_upper)

    @patch("sample_size.multiple_testing.MultipleTestingMixin.generate_p_value")
    @patch("statsmodels.stats.multitest.multipletests")
    def test_expected_average_power(self, mock_multipletests, mock_generate_p_value):
        test_lower = 100
        test_upper = 1000

        test_metric_type = "boolean"
        test_probability = 0.05
        test_mde = 0.02
        test_metric_metadata = {"probability": test_probability, "mde": test_mde}
        mock_generate_p_value.return_value = MagicMock()
        mock_multipletests.return_value = [[1, 1, 1, 1, 0]]

        calculator = SampleSizeCalculator()
        calculator.register_metrics(
            [
                {"metric_type": test_metric_type, "metric_metadata": test_metric_metadata},
                {"metric_type": test_metric_type, "metric_metadata": test_metric_metadata},
            ]
        )
        replication = calculator.metrics * (calculator.variants - 1) * 100

        sample_size = calculator.get_sample_size()
        self.assertEqual(mock_generate_p_value.call_count, replication)
        self.assertEqual(sample_size, (test_upper + test_lower) / 2)
