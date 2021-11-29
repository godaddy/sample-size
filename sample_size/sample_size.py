from typing import List

from numpy import mean
from statsmodels.stats.power import TTestIndPower


def get_mean(numbers: List[int]) -> str:
    return "This is to prove numpy works that the mean of {} is {:.1f}".format(str(numbers), mean(numbers))


def get_ttest_sample_size(effect_size: float) -> str:
    obj = TTestIndPower()
    n = obj.solve_power(effect_size=effect_size, alpha=0.05, power=0.8, ratio=1, alternative="two-sided")
    return (
        "This is to prove statsmodels works that the required sample size is {:.1f}"
        " when the effect_size is {:.1f} using ttest".format(n, effect_size)
    )
