from typing import List
from typing import Tuple

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sample_size.metrics import BaseMetric


class MultipleTesting:
    """
    This class is to calculate sample size required under the case of multiple testing

    Attributes:
    metrics: a list of BaseMetric registered by users
    variants: number of variants, including control
    alpha: statistical significance
    power: statistical power
    """

    def __init__(self, metrics: List[BaseMetric], variants: int, alpha: float, power: float):
        self.REPLICATION: int = 100
        self.metrics = metrics
        self.m = (variants - 1) * len(metrics)
        self.alpha = alpha
        self.power = power

    def get_multiple_sample_size(self) -> int:
        # calculate required sample size based on minimum standardized effect size since it requires maximum sample size
        lower = max([metric.single_sample_size(self.alpha, self.power) for metric in self.metrics])
        upper = max([metric.single_sample_size(self.alpha / self.m, self.power) for metric in self.metrics])

        return self._find_sample_size(lower, upper)

    def _find_sample_size(self, lower: float, upper: float, depth=0):
        MAX_RECURSION_DEPTH = 20
        EPSILON = 0.025

        if depth > MAX_RECURSION_DEPTH:
            raise RecursionError

        candidate = int((upper + lower) / 2)
        expected_power = self._expected_average_power(candidate)

        if np.isclose((self.power, expected_power), EPSILON) or np.isclose((candidate, upper), EPSILON):
            return candidate

        if expected_power > self.power:
            return self._find_sample_size(lower, candidate, depth + 1)
        else:
            return self._find_sample_size(candidate, upper, depth + 1)

    def _expected_average_power(self, sample_size: int):
        power = []
        for m1 in range(1, self.m + 1):
            alts = np.array([True] * m1 + [False] * (self.m - m1))
            for _ in range(self.REPLICATION):
                true_alt = alts[np.random.permutation(self.m)]
                p_values = [self._generate_p_value(m, true_alt[i], sample_size) for i, m in enumerate(self.metrics)]
                rejected = multipletests(p_values, alpha=self.alpha, method="fdr_bh")[0]
                power.append(np.dot(rejected, true_alt) / m1)

        return np.mean(power)

    def _generate_p_value(self, metric, true_alt: bool, size: int) -> Tuple[List[float], List[float]]:
        metric_type: str = type(metric).__name__
        effect_size = metric.mde / np.sqrt(metric.variance / size) if metric_type == 'BooleanMetric' else metric.mde / (
                2 * np.sqrt(metric.variance / size))

        if not true_alt:
            return stats.uniform.rvs(0, 1)

        if metric_type in ["BooleanMetric", "RatioMetric"]:
            z_alt = stats.t.rvs(df=size - 1, loc=effect_size, size=self.REPLICATION)
            return 2 * stats.norm.sf(np.abs(z_alt))

        elif metric_type == "NumericMetric":
            nc = np.sqrt(size / 2) * metric.mde / metric.variance
            t_alt = stats.nct.rvs(nc=nc, df=2 * (size - 1), size=self.REPLICATION)
            return 2 * stats.t.sf(np.abs(t_alt))
