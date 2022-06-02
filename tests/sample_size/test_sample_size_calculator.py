import unittest
from unittest.mock import call
from unittest.mock import patch

from parameterized import parameterized

from sample_size.metrics import BooleanMetric
from sample_size.metrics import NumericMetric
from sample_size.metrics import RatioMetric
from sample_size.sample_size_calculator import DEFAULT_ALPHA
from sample_size.sample_size_calculator import DEFAULT_POWER
from sample_size.sample_size_calculator import DEFAULT_VARIANTS
from sample_size.sample_size_calculator import SampleSizeCalculator


class SampleSizeCalculatorTestCase(unittest.TestCase):
    def test_sample_size_calculator_constructor_sets_params(self):
        test_alpha = 0.1
        test_variants = 2
        test_power = 0.9
        calculator = SampleSizeCalculator(
            test_alpha,
            test_variants,
            test_power,
        )

        self.assertEqual(calculator.alpha, test_alpha)
        self.assertEqual(calculator.power, test_power)
        self.assertEqual(calculator.metrics, [])

    def test_sample_size_calculator_constructor_sets_params_with_default_params(self):
        calculator = SampleSizeCalculator()

        self.assertEqual(calculator.alpha, DEFAULT_ALPHA)
        self.assertEqual(calculator.variants, DEFAULT_VARIANTS)
        self.assertEqual(calculator.power, DEFAULT_POWER)
        self.assertEqual(calculator.metrics, [])

    @patch("statsmodels.stats.power.NormalIndPower.solve_power")
    def test_get_single_sample_size_normal(self, mock_solve_power):
        test_probability = 0.05
        test_mde = 0.02
        test_sample_size = 2000
        test_metric = BooleanMetric(
            test_probability,
            test_mde,
        )
        mock_solve_power.return_value = test_sample_size

        calculator = SampleSizeCalculator()
        sample_size = calculator._get_single_sample_size(test_metric, calculator.alpha)

        self.assertEqual(sample_size, test_sample_size)
        mock_solve_power.assert_called_once_with(
            effect_size=0.09176629354822471,
            alpha=DEFAULT_ALPHA,
            power=DEFAULT_POWER,
            ratio=1,
            alternative="two-sided",
        )

    @patch("statsmodels.stats.power.TTestIndPower.solve_power")
    def test_get_single_sample_size_ttest(self, mock_solve_power):
        test_variance = 1000
        test_mde = 5
        test_sample_size = 2000
        test_metric = NumericMetric(
            test_variance,
            test_mde,
        )
        mock_solve_power.return_value = test_sample_size
        calculator = SampleSizeCalculator()

        sample_size = calculator._get_single_sample_size(test_metric, calculator.alpha)

        self.assertEqual(sample_size, test_sample_size)
        mock_solve_power.assert_called_once_with(
            effect_size=0.15811388300841897,
            alpha=DEFAULT_ALPHA,
            power=DEFAULT_POWER,
            ratio=1,
            alternative="two-sided",
        )

    @parameterized.expand(
        [
            ("boolean", {"probability": 0.05, "mde": 0.02}, BooleanMetric),
            ("numeric", {"variance": 500, "mde": 5}, NumericMetric),
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
                RatioMetric,
            ),
        ]
    )
    @patch("sample_size.sample_size_calculator.SampleSizeCalculator.get_multiple_sample_size")
    @patch("sample_size.sample_size_calculator.SampleSizeCalculator._get_single_sample_size")
    def test_get_overall_sample_size_single(
        self, metric_type, metadata, metric_class, mock_get_single_sample_size, mock_get_multiple_sample_size
    ):
        test_metric_type = metric_type
        test_sample_size = 2000
        mock_get_single_sample_size.return_value = test_sample_size

        # test_mde = 0.02
        # test_probability = 0.05
        test_metric_metadata = metadata
        calculator = SampleSizeCalculator()
        calculator.register_metrics([{"metric_type": test_metric_type, "metric_metadata": test_metric_metadata}])

        sample_size = calculator.get_sample_size()

        self.assertEqual(sample_size, test_sample_size)
        mock_get_single_sample_size.assert_called_once()
        mock_get_multiple_sample_size.assert_not_called()
        self.assertIsInstance(mock_get_single_sample_size.call_args[0][0], metric_class)
        # assert_equal(mock_get_single_sample_size.call_args[0][0].probability, test_probability)
        # assert_equal(mock_get_single_sample_size.call_args[0][0].mde, test_mde)

    @patch("sample_size.sample_size_calculator.SampleSizeCalculator.get_multiple_sample_size")
    @patch("sample_size.sample_size_calculator.SampleSizeCalculator._get_single_sample_size")
    def test_get_overall_sample_size_multiple(self, mock_get_single_sample_size, mock_get_multiple_sample_size):
        test_metric_type = "boolean"
        test_sample_size = 2000
        mock_get_multiple_sample_size.return_value = test_sample_size
        mock_get_single_sample_size.return_value = test_sample_size

        # test_mde = 0.02
        # test_probability = 0.05
        test_metric_metadata = {"probability": 0.05, "mde": 0.02}
        calculator = SampleSizeCalculator()
        calculator.register_metrics(
            [
                {"metric_type": test_metric_type, "metric_metadata": test_metric_metadata},
                {"metric_type": test_metric_type, "metric_metadata": test_metric_metadata},
            ]
        )

        sample_size = calculator.get_sample_size()

        self.assertEqual(sample_size, test_sample_size)
        mock_get_single_sample_size.assert_has_calls(
            [
                call(calculator.metrics[0], calculator.alpha),
                call(calculator.metrics[1], calculator.alpha),
                call(calculator.metrics[0], calculator.alpha / 2),
                call(calculator.metrics[1], calculator.alpha / 2),
            ]
        )
        mock_get_multiple_sample_size.assert_called_once_with(test_sample_size, test_sample_size)
        # self.assertIsInstance(mock_get_single_sample_size.call_args[0][0], metric_class)
        # assert_equal(mock_get_single_sample_size.call_args[0][0].probability, test_probability)
        # assert_equal(mock_get_single_sample_size.call_args[0][0].mde, test_mde)

    # TODO: parameterize register metric functions
    def test_register_metric_boolean(self):
        test_metric_type = "boolean"
        test_probability = 0.05
        test_mde = 0.02
        test_metric_metadata = {"probability": test_probability, "mde": test_mde}

        calculator = SampleSizeCalculator()
        calculator.register_metrics([{"metric_type": test_metric_type, "metric_metadata": test_metric_metadata}])
        self.assertIsInstance(calculator.metrics[0], BooleanMetric)
        self.assertEqual(len(calculator.metrics), 1)
        self.assertEqual(calculator.metrics[0].variance, 0.0475)
        self.assertEqual(calculator.metrics[0].mde, test_mde)

        calculator.register_metrics([{"metric_type": test_metric_type, "metric_metadata": test_metric_metadata}])
        self.assertEqual(len(calculator.metrics), 2)

    def test_register_metric_numeric(self):
        test_metric_type = "numeric"
        test_variance = 5000.0
        test_mde = 5.0
        test_metric_metadata = {"variance": test_variance, "mde": test_mde}

        calculator = SampleSizeCalculator()
        calculator.register_metrics([{"metric_type": test_metric_type, "metric_metadata": test_metric_metadata}])
        self.assertIsInstance(calculator.metrics[0], NumericMetric)
        self.assertEqual(len(calculator.metrics), 1)
        self.assertEqual(calculator.metrics[0].variance, test_variance)
        self.assertEqual(calculator.metrics[0].mde, test_mde)

        calculator.register_metrics([{"metric_type": test_metric_type, "metric_metadata": test_metric_metadata}])
        self.assertEqual(len(calculator.metrics), 2)

    def test_register_metric_ratio(self):
        test_metric_type = "ratio"
        test_numerator_mean = 2000.0
        test_numerator_variance = 100000.0
        test_denominator_mean = 200.0
        test_denominator_variance = 2000.0
        test_covariance = 5000.0
        test_mde = 5.0
        test_variance = 5
        test_metric_metadata = {
            "numerator_mean": test_numerator_mean,
            "numerator_variance": test_numerator_variance,
            "denominator_mean": test_denominator_mean,
            "denominator_variance": test_denominator_variance,
            "covariance": test_covariance,
            "mde": test_mde,
        }

        calculator = SampleSizeCalculator()
        calculator.register_metrics([{"metric_type": test_metric_type, "metric_metadata": test_metric_metadata}])
        self.assertIsInstance(calculator.metrics[0], RatioMetric)
        self.assertEqual(len(calculator.metrics), 1)
        self.assertEqual(calculator.metrics[0].variance, test_variance)
        self.assertEqual(calculator.metrics[0].mde, test_mde)

        calculator.register_metrics([{"metric_type": test_metric_type, "metric_metadata": test_metric_metadata}])
        self.assertEqual(len(calculator.metrics), 2)

    def test_register_metric_invalid_metadata(self):
        test_metric_type = "numeric"

        calculator = SampleSizeCalculator()
        with self.assertRaises(Exception):
            calculator.register_metrics([{"metric_type": test_metric_type}])
