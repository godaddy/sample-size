from typing import List

import numpy as np
from statsmodels.stats.multitest import multipletests

from sample_size.metrics import BaseMetric
from sample_size.sample_size_calculator import SampleSizeCalculator


class MultipleTestingMixin:
    """
    This class calculates sample size required under the case of multiple testing

    Attributes:
    metrics: a list of BaseMetric registered by users
    variants: number of variants, including control
    alpha: statistical significance
    power: statistical power

    """

    def get_multiple_sample_size(self) -> int:
        if len(self.metrics) < 2:
            return self.metrics[0].single_sample_size()
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
