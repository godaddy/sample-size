from numpy import nan
from statsmodels.stats.power import TTestIndPower


def get_nan() -> str:
    return "This is to prove numpy works that nan can be called as {}".format(nan)


def get_ttest_power(effect_size: float) -> str:
    obj = TTestIndPower()
    n = obj.solve_power(effect_size=effect_size, alpha=0.05, power=0.8, ratio=1, alternative="two-sided")
    return (
        "This is to prove statsmodels works that the required sample size is {:.1f}"
        " when the effect_size is {:.1f}".format(n, effect_size)
    )
