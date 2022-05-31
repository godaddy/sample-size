from typing import List

import numpy as np
from statsmodels.stats.multitest import multipletests

from sample_size.metrics import BaseMetric


class MultipleTestingMixin:
    """
    This class calculates sample size required under the case of multiple testing

    Attributes:
    metrics: a list of BaseMetric registered by users
    variants: number of variants, including control
    alpha: statistical significance
    power: statistical power

    """

    metrics: List[BaseMetric]
    alpha: float
    power: float
    variants: int

    def get_multiple_sample_size(self) -> int:
        if len(self.metrics) < 2:
            return self.get_single_sample_size(self.metrics[0])
        lower = max([self.get_single_sample_size(metric) for metric in self.metrics])
        upper = max([self.get_single_sample_size(metric, self.alpha / len(self.metrics)) for metric in self.metrics])

        return self._find_sample_size(lower, upper)

    def _find_sample_size(self, lower: float, upper: float, depth: int = 0) -> int:
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

    def _expected_average_power(self, sample_size: int, REPLICATION=100) -> float:
        power = []
        m = self.metrics * (self.variants - 1)
        for m1 in range(m):
            nulls = np.array([True] * m1 + [False] * (m - m1))
            for _ in range(REPLICATION):
                true_null = nulls[np.random.permutation(m)]
                p_values = [
                    m.generate_p_value(true_null[i], sample_size, self.variants) for i, m in enumerate(self.metrics)
                ]
                rejected = multipletests(p_values, alpha=self.alpha, method="fdr_bh")[0]
                power.append(sum(rejected[~true_null]) / m1)
                # power.append(int(np.dot(rejected,  1-true_null)) / m1)

        return np.mean(power)
