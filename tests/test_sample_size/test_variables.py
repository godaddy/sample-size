import unittest
from unittest.mock import patch

from sample_size.variables import Boolean
from sample_size.variables import Numeric
from sample_size.variables import Ratio


class VariableTestCase(unittest.TestCase):
    def setUp(self):
        self.DEFAULT_PROBABILITY = 0.05
        self.DEFAULT_MEAN = 50
        self.DEFAULT_VARIANCE = 5000
        self.DEFAULT_NUMERATOR_MEAN = 2000
        self.DEFAULT_NUMERATOR_VARIANCE = 100000
        self.DEFAULT_DENOMINATOR_MEAN = 200
        self.DEFAULT_DENOMINATOR_VARIANCE = 2000
        self.DEFAULT_COVARIANCE = 5000
        self.DEFAULT_MOCK_VALUE = 99

    @patch("sample_size.variables.Boolean.get_probability")
    @patch("sample_size.variables.Boolean.get_variance")
    def test_boolean_constructor_sets_params(self, mock_get_variance, mock_get_probability):

        mock_get_variance.return_value = self.DEFAULT_MOCK_VALUE
        mock_get_probability.return_value = self.DEFAULT_PROBABILITY
        boolean = Boolean(self.DEFAULT_PROBABILITY)

        mock_get_variance.assert_called_once()
        mock_get_probability.assert_called_once_with(self.DEFAULT_PROBABILITY)
        self.assertEqual(boolean.probability, self.DEFAULT_PROBABILITY)
        self.assertEqual(boolean.variance, self.DEFAULT_MOCK_VALUE)

    def test_boolean_get_variance(self):
        boolean = Boolean(self.DEFAULT_PROBABILITY)
        variance = boolean.get_variance()

        self.assertEqual(variance, 0.0475)

    def test_get_probability(self):
        test_probability = self.DEFAULT_PROBABILITY
        probability = Boolean.get_probability(test_probability)

        self.assertEqual(probability, test_probability)

    def test_get_probability_too_large(self):
        test_probability = 5

        with self.assertRaises(Exception) as context:
            Boolean.get_probability(test_probability)
            self.assertEqual(
                context.exception,
                Exception("Error: Please provide a float between 0 and 1 for probability."),
            )

    def test_get_probability_too_small(self):
        test_probability = -0.1

        with self.assertRaises(Exception) as context:
            Boolean.get_probability(test_probability)
            self.assertEqual(
                context.exception,
                Exception("Error: Please provide a float between 0 and 1 for probability."),
            )

    def test_numeric_constructor_sets_params(self):

        numeric = Numeric(self.DEFAULT_VARIANCE)

        self.assertEqual(numeric.variance, self.DEFAULT_VARIANCE)

    @patch("sample_size.variables.Ratio.get_variance")
    def test_ratio_constructor_sets_params(self, mock_get_variance):
        mock_get_variance.return_value = self.DEFAULT_MOCK_VALUE
        ratio = Ratio(
            self.DEFAULT_NUMERATOR_MEAN,
            self.DEFAULT_NUMERATOR_VARIANCE,
            self.DEFAULT_DENOMINATOR_MEAN,
            self.DEFAULT_DENOMINATOR_VARIANCE,
            self.DEFAULT_COVARIANCE,
        )

        mock_get_variance.assert_called_once()
        self.assertEqual(ratio.numerator_mean, self.DEFAULT_NUMERATOR_MEAN)
        self.assertEqual(ratio.numerator_variance, self.DEFAULT_NUMERATOR_VARIANCE)
        self.assertEqual(ratio.denominator_mean, self.DEFAULT_DENOMINATOR_MEAN)
        self.assertEqual(ratio.denominator_variance, self.DEFAULT_DENOMINATOR_VARIANCE)
        self.assertEqual(ratio.covariance, self.DEFAULT_COVARIANCE)
        self.assertEqual(ratio.variance, self.DEFAULT_MOCK_VALUE)

    def test_ratio_get_variance(self):
        ratio = Ratio(
            self.DEFAULT_NUMERATOR_MEAN,
            self.DEFAULT_NUMERATOR_VARIANCE,
            self.DEFAULT_DENOMINATOR_MEAN,
            self.DEFAULT_DENOMINATOR_VARIANCE,
            self.DEFAULT_COVARIANCE,
        )
        variance = ratio.get_variance()

        self.assertEqual(variance, 5.0)
