import unittest

from numpy import nan

from sample_size.sample_size import get_nan
from sample_size.sample_size import get_ttest_power


class TestSampleSize(unittest.TestCase):
    def test_nan(self):
        result = get_nan()
        self.assertEqual("This is to prove numpy works that nan can be called as {}".format(nan), result)

    def test_statsmodel(self):
        result = get_ttest_power(0.5)
        self.assertEqual(
            "This is to prove statsmodels works that the required sample size is {:.1f} "
            "when the effect_size is {:.1f}".format(63.8, 0.5),
            result,
        )
