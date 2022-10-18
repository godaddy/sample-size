import unittest
from itertools import cycle
from itertools import product
from unittest.mock import patch

import numpy as np
from parameterized import parameterized

from sample_size.multiple_testing import DEFAULT_EPSILON
from sample_size.multiple_testing import DEFAULT_REPLICATION
from sample_size.sample_size_calculator import DEFAULT_ALPHA
from sample_size.sample_size_calculator import DEFAULT_POWER
from sample_size.sample_size_calculator import SampleSizeCalculator
from tests.sample_size.test_metrics import ALTERNATIVE

TEST_BOOLEAN = {
    "metric_type": "boolean",
    "metric_metadata": {"probability": 0.05, "mde": 0.02, "alternative": ALTERNATIVE},
}
TEST_NUMERIC = {"metric_type": "numeric", "metric_metadata": {"variance": 5000, "mde": 5, "alternative": ALTERNATIVE}}
TEST_RATIO = {
    "metric_type": "ratio",
    "metric_metadata": {
        "numerator_mean": 500,
        "numerator_variance": 500000,
        "denominator_mean": 20,
        "denominator_variance": 2000,
        "covariance": 25000,
        "mde": 1,
        "alternative": ALTERNATIVE,
    },
}


class MultipleTestingTestCase(unittest.TestCase):
    def setUp(self):
        self.test_lower = 100
        self.test_upper = 1000
        self.test_metric_type = "boolean"
        self.test_probability = 0.05
        self.test_mde = 0.02
        self.test_alternative = ALTERNATIVE
        self.test_metric_metadata = {
            "probability": self.test_probability,
            "mde": self.test_mde,
            "alternative": self.test_alternative,
        }
        self.test_metric = {"metric_type": self.test_metric_type, "metric_metadata": self.test_metric_metadata}

    @parameterized.expand([(0,), (1,), (DEFAULT_POWER + 3 * DEFAULT_EPSILON,)])
    @patch(
        "sample_size.sample_size_calculator.SampleSizeCalculator._get_single_sample_size",
    )
    @patch("sample_size.multiple_testing.MultipleTestingMixin._expected_average_power")
    def test_get_multiple_sample_size_quickly_converge(
        self, power_guess, mock_expected_average_power, mock_get_single_sample_size
    ):
        mock_get_single_sample_size.side_effect = lambda _, alpha: 100 if alpha >= DEFAULT_ALPHA else 1000

        mock_expected_average_power.side_effect = [power_guess, DEFAULT_POWER]
        calculator = SampleSizeCalculator()
        calculator.register_metrics([self.test_metric] * 3)
        expected_call_count = len(calculator.metrics) * 2

        sample_size = calculator.get_sample_size()

        # Initial estimates from first recursive call
        init_candidate = int(np.sqrt(self.test_upper * self.test_lower))
        init_bound = self.test_lower if power_guess > DEFAULT_POWER else self.test_upper

        self.assertEqual(mock_get_single_sample_size.call_count, expected_call_count)
        self.assertEqual(mock_expected_average_power.call_count, 2)
        self.assertEqual(sample_size, int(np.sqrt(init_candidate * init_bound)))

    @patch(
        "sample_size.sample_size_calculator.SampleSizeCalculator._get_single_sample_size",
        side_effect=[100, 100, 100, 1000, 1000, 1000],
    )
    @patch("sample_size.multiple_testing.MultipleTestingMixin._expected_average_power")
    def test_get_multiple_sample_size_no_recursion(self, mock_expected_average_power, mock_get_single_sample_size):
        mock_expected_average_power.return_value = DEFAULT_POWER

        calculator = SampleSizeCalculator()
        calculator.register_metrics([self.test_metric] * 3)
        expected_call_count = len(calculator.metrics) * 2
        geom_mean = int(np.sqrt(self.test_upper * self.test_lower))

        sample_size = calculator.get_sample_size()
        self.assertEqual(mock_get_single_sample_size.call_count, expected_call_count)
        mock_expected_average_power.assert_called_once_with(geom_mean, DEFAULT_REPLICATION)
        self.assertEqual(sample_size, geom_mean)

    @patch("sample_size.multiple_testing.MultipleTestingMixin._expected_average_power")
    def test_get_multiple_sample_size_converges_without_solution(self, mock_expected_power):
        mock_expected_power.return_value = 0

        calculator = SampleSizeCalculator()
        calculator.register_metrics([self.test_metric, self.test_metric])

        with self.assertRaises(Exception) as context:
            calculator.get_sample_size()
        self.assertEqual(
            str(context.exception),
            f"Couldn't find a sample size that satisfies the power you requested: {DEFAULT_POWER}",
        )

    @patch("sample_size.multiple_testing.MultipleTestingMixin._expected_average_power")
    def test_get_multiple_sample_size_does_not_converge(self, mock_expected_power):
        delta = 2 * DEFAULT_EPSILON
        alternating_power = cycle([DEFAULT_POWER - delta, DEFAULT_POWER + delta])
        mock_expected_power.side_effect = lambda *_: next(alternating_power)

        calculator = SampleSizeCalculator()
        calculator.register_metrics([self.test_metric, self.test_metric])

        with self.assertRaises(Exception) as context:
            calculator.get_sample_size()
        self.assertEqual(
            str(context.exception),
            f"Couldn't find a sample size that satisfies the power you requested: {DEFAULT_POWER}",
        )

    @parameterized.expand(
        [
            (TEST_BOOLEAN, 2002, 1),
            (TEST_NUMERIC, 3455, 2),
            (TEST_RATIO, 22115, 3),
            (TEST_BOOLEAN, 2051, 7),
            (TEST_NUMERIC, 3433, 8),
            (TEST_RATIO, 20583, 6),
        ]
    )
    def test_get_multiple_sample_size_fixed_output(self, test_metric, test_sample_size, seed):
        with patch("sample_size.metrics.RANDOM_STATE", np.random.RandomState(seed)):
            calculator = SampleSizeCalculator()
            calculator.register_metrics([test_metric] * 2)
            sample_size = calculator.get_sample_size()
            self.assertEqual(sample_size, test_sample_size)

    @parameterized.expand([(10,), (100,), (1000,)])
    def test_expected_average_power_satisfies_inequality(self, test_size):
        calculator = SampleSizeCalculator()
        calculator.register_metrics([self.test_metric, self.test_metric, self.test_metric])
        expected_power = calculator._expected_average_power(test_size)
        inflated_power = calculator._expected_average_power(test_size * 10)
        self.assertGreater(inflated_power, expected_power)

    @parameterized.expand(product((10, 100, 500, 1000), (0.1, 0.2, 0.5, 0.8, 0.9)))
    @patch("sample_size.multiple_testing.multipletests")
    def test_expected_average_power_is_a_reasonable_approximation(self, replications, true_power, mock_fdr):
        # We are setting the rejection rate to true_power
        # Because we are randomly permuting the indices of the true alternative hypotheses
        # we can conceptualize that aspect as random sampling
        rng = np.random.RandomState(1024)
        mock_fdr.side_effect = lambda a, **kw: [rng.random(len(a)) < true_power]

        sample_size = 10  # arbitrary
        calculator = SampleSizeCalculator()
        calculator.register_metrics([self.test_metric] * 2)
        empirical_power = calculator._expected_average_power(sample_size, replications)
        margin_of_error = 1 / np.sqrt(replications)  # proportional to 1 σ
        self.assertAlmostEqual(true_power, empirical_power, delta=margin_of_error)
