# +
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
    power: average power, calculated as #correct rejections/#true alternative hypotheses

    """

    metrics: List[BaseMetric]
    alpha: float
    power: float
    variants: int

    def get_multiple_sample_size(
        self,
        lower: float,
        upper: float,
        replication: int,
        epsilon: float,
        max_recursion_depth: int,
        depth: int = 0,
    ) -> int:
        """
        This method finds minimum required sample size per cohort that generates average power higher than required

        Attributes:
        lower: lower bound of sample size search; maximum of each metric's individually calculated sample size
        upper: upper bound of sample size search; maximum of each metric's individually calculated sample size,
        with Bonferroni adjustment(alpha = alpha/number of tests)
        depth: number of recursions

        Returns minimum required sample size per cohort
        """

        if depth > max_recursion_depth:
            raise RecursionError(f"Couldn't find a sample size that satisfies the power you requested: {self.power}")

        candidate = int(np.sqrt(lower * upper))
        expected_power = self._expected_average_power(candidate, replication)
        # print(f"lower: {lower}, upper: {upper}, expected_power: {expected_power}, candidate: {candidate}")
        if np.isclose(self.power, expected_power, atol=epsilon):
            return candidate
        elif lower == upper:
            raise RecursionError(f"Couldn't find a sample size that satisfies the power you requested: {self.power}")

        if expected_power > self.power:
            return self.get_multiple_sample_size(lower, candidate, replication, epsilon, max_recursion_depth, depth + 1)
        else:
            return self.get_multiple_sample_size(candidate, upper, replication, epsilon, max_recursion_depth, depth + 1)

    def _expected_average_power(self, sample_size: int, replication) -> float:
        """
        This method calculates expected average power of multiple testings. For each possible number of true null
        hypothesis, we simulate each metric/treatment variant's test statistics and calculate their p-values,
        then calculate expected average power = number of True rejection/ true alternative hypotheses

        Attributes:
        sample size: determines the variance/ degrees of freedom of the distribution we sample test statistics from
        replication: number of times we repeat the simulation process

        Returns value expected average power
        """
        true_alt_count = 0

        # a metric for each test we would conduct
        metrics = self.metrics * (self.variants - 1)

        def fdr_bh(a):
            return multipletests(a, alpha=self.alpha, method="fdr_bh")[0]

        power = []
        for num_true_alt in range(1, len(metrics) + 1):
            true_alt_count += num_true_alt * replication

            true_alt = np.array([np.random.permutation(len(metrics)) < num_true_alt for _ in range(replication)]).T

            p_values = []
            for i, m in enumerate(metrics):
                p_values.append(m.generate_p_values(true_alt[i], sample_size))

            rejected = np.apply_along_axis(fdr_bh, 0, np.array(p_values))

            true_discoveries = rejected & true_alt

            power.append(true_discoveries.sum() / true_alt.sum())

        return np.array(power).mean()


# -
