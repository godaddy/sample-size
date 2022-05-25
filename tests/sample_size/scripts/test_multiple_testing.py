import unittest
from unittest.mock import patch

from numpy.testing import assert_equal

from sample_size.metrics import BooleanMetric
from sample_size.metrics import NumericMetric
from sample_size.metrics import RatioMetric
from sample_size.sample_size_calculator import DEFAULT_ALPHA
from sample_size.sample_size_calculator import DEFAULT_POWER
from sample_size.sample_size_calculator import SampleSizeCalculator


class MultipleTestingTestCase(unittest.TestCase):

    def test_get_multiple_sample_size(self):
        self.metrics =

