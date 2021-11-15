import unittest

from sample_size.sample_size import get_mean
from sample_size.sample_size import get_ttest_power


class TestSampleSize(unittest.TestCase):
    def test_numpy(self):
        test_nums = [1, 2, 3, 4]
        result = get_mean(test_nums)
        self.assertEqual(
            "This is to prove numpy works that the mean of {} is {:.1f}".format(str(test_nums), 2.5), result
        )

    def test_statsmodel(self):
        result = get_ttest_power(0.5)
        self.assertEqual(
            "This is to prove statsmodels works that the required sample size is {:.1f} "
            "when the effect_size is {:.1f}".format(63.8, 0.5),
            result,
        )
