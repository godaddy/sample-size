from itertools import product
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.power import TTestIndPower
from sample_size.metrics import BaseMetric


def get_multiple_sample_size(self, d: float, power_analysis: Union[NormalIndPower, TTestIndPower]) -> int:
    m = (len(self.boolean_metrics) + len(self.numeric_metrics) + len(self.ratio_metrics)) * self.variants
    # calculate required sample size based on minimum standardized effect size since it requires maximum sample size
    lower = power_analysis.solve_power(
        effect_size=d, alpha=self.alpha, power=self.power, ratio=1, alternative="two-sided"
    )
    upper = power_analysis.solve_power(
        effect_size=d, alpha=self.alpha / m, power=self.power, ratio=1, alternative="two-sided"
    )

    # print(f'We look for the minimum required sample size in range [{int(lower)},{int(upper)}]')
    for size in np.linspace(lower, upper, 10):
        expected_power = self.expected_average_power(m, int(size))
        if expected_power >= self.power:
            break
    return int(size)


def expected_average_power(self, number_of_tests: int, size: int) -> float:
    m = number_of_tests
    pp_null, pp_alt = [], []

    if self.boolean_metrics:
        for bool_metric in self.boolean_metrics:
            p_null, p_alt = self.generate_p_value(bool_metric, size)
            pp_null.append(p_null)
            pp_alt.append(p_alt)
    if self.numeric_metrics:
        for numeric_metric in self.numeric_metrics:
            p_null, p_alt = self.generate_p_value(numeric_metric, size)
            pp_null.append(p_null)
            pp_alt.append(p_alt)
    if self.ratio_metrics:
        for ratio_metric in self.ratio_metrics:
            p_null, p_alt = self.generate_p_value(ratio_metric, size)
            pp_null.append(p_null)
            pp_alt.append(p_alt)

    true_H = [*product([0, 1], repeat=m)][1:]
    avg_power = 0
    # 1=rejected, 0=fail to reject
    rejs = np.empty((self.rep, m))

    for t in range(len(true_H)):
        null_index = np.argwhere(np.array(true_H[t]) == 0)
        alt_index = np.argwhere(np.array(true_H[t]) == 1)

        for r in range(self.rep):
            # first len(true_null_p) hypotheses are true null
            true_null_p = [float(np.array(pp_null)[x, r]) for x in null_index]
            true_alt_p = [float(np.array(pp_alt)[x, r]) for x in alt_index]
            pvalues = np.zeros(m)
            pvalues[: len(null_index)] = true_null_p
            pvalues[len(null_index):] = true_alt_p
            rejs[r, :] = multipletests(pvalues, alpha=self.alpha, method="fdr_bh")[0]

        actual_pw = np.sum(rejs[:, len(null_index):], axis=1) / len(alt_index)

        avg_power += np.mean(actual_pw)

    return avg_power / len(true_H)


def generate_p_value(self, metric: BaseMetric, size: int) -> Tuple[List[float], List[float]]:
    metric_type: str = type(metric).__name__
    p_alt = np.zeros(self.rep)
    p_null = stats.norm.rvs(0, 1, self.rep)
    effect_size = metric.mde / float(np.sqrt(metric.variance))
    z_alt = stats.t.rvs(df=size - 1, loc=effect_size, size=self.rep)

    if metric_type == "BooleanMetric":
        p_alt = 2 * stats.norm.sf(np.abs(z_alt))

    elif metric_type == "NumericMetric":
        p_alt = 2 * stats.t.sf(np.abs(z_alt))

    elif metric_type == "RatioMetric":
        p_alt = 2 * stats.norm.sf(np.abs(z_alt))

    return list(p_null), list(p_alt)
