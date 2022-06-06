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

    REPLICATION: int = 500

    def get_multiple_sample_size(self, lower: float, upper: float, depth: int = 0) -> int:
        max_recursion_depth: int = 20
        epsilon: float = 0.025  # TODO(any): make this configurable by users ML-5429

        if depth > max_recursion_depth:
            raise RecursionError(f"Couldn't find a sample size that satisfies the power you requested: {self.power}")

        candidate = int(lower + (upper - lower) / 2)
        expected_power = self._expected_average_power(candidate)
        if np.isclose(self.power, expected_power, epsilon):
            return candidate

        if expected_power > self.power:
            return self.get_multiple_sample_size(lower, candidate, depth + 1)
        else:
            return self.get_multiple_sample_size(candidate, upper, depth + 1)

    def _expected_average_power(self, sample_size: int, replication: int = REPLICATION) -> float:
        power = []
        num_of_tests = len(self.metrics) * (self.variants - 1)
        for m1 in range(num_of_tests):
            nulls = np.array([True] * m1 + [False] * (num_of_tests - m1))
            for _ in range(replication):
                p_values = []
                true_null = nulls[np.random.permutation(num_of_tests)]
                for v in range(self.variants - 1):
                    p_values.extend(
                        [
                            m.generate_p_value(
                                true_null[v * len(self.metrics) + i],
                                sample_size,
                            )
                            for i, m in enumerate(self.metrics)
                        ]
                    )

                rejected = multipletests(p_values, alpha=self.alpha, method="fdr_bh")[0]
                power.append(sum(rejected[~true_null]) / (num_of_tests - m1))

        return float(np.mean(power))
