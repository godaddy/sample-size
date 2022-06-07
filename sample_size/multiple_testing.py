from typing import List

import numpy as np
from statsmodels.stats.multitest import multipletests

from sample_size.metrics import BaseMetric

REPLICATION: int = 500


class MultipleTestingMixin:
    """
    This class calculates sample size required under the case of multiple testing

    Attributes:
    metrics: a list of BaseMetric registered by users
    variants: number of variants, including control
    alpha: statistical significance
    power: average power, calculated as #correct rejections/#true alternative hypotheses

    """

    metrics: List[BaseMetric]
    alpha: float
    power: float
    variants: int

    def get_multiple_sample_size(self, lower: float, upper: float, depth: int = 0) -> int:
        """
        This method finds minimum required sample size per cohort that generates average power higher than required

        Attributes:
        lower: lower bound of sample size search; maximum of each metric's individually calculated sample size
        upper: upper bound of sample size search; maximum of each metric's individually calculated sample size,
        with Bonferroni adjustment(alpha = alpha/number of tests)
        depth: number of recursions

        Returns minimum required sample size per cohort
        """
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
        """
        This method calculates expected average power of multiple testings. For each possible number of true null
        hypothesis, we simulate each metric/treatment variant's test statistics and calculate their p-values,
        then calculate expected average power = number of True rejection/ true alternative hypotheses

        Attributes:
        sample size: determines the variance/ degrees of freedom of the distribution we sample test statistics from
        replication: number of times we repeat the simulation process

        Returns value expected average power
        """
        power = []
        num_of_tests = len(self.metrics) * (self.variants - 1)
        for num_true_null in range(num_of_tests):
            num_true_alt = num_of_tests - num_true_null
            nulls = np.array([True] * num_true_null + [False] * num_true_alt)
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
                power.append(sum(rejected[~true_null]) / num_true_alt)
        return float(np.mean(power))
