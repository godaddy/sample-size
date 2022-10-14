from typing import List

import numpy as np
import numpy.typing as npt
from statsmodels.stats.multitest import multipletests

from sample_size.metrics import BaseMetric

DEFAULT_REPLICATION: int = 400
DEFAULT_EPSILON: float = 0.01
DEFAULT_MAX_RECURSION: int = 20


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
        random_state: np.random.RandomState,
        depth: int = 0,
        replication: int = DEFAULT_REPLICATION,
        epsilon: float = DEFAULT_EPSILON,
        max_recursion_depth: int = DEFAULT_MAX_RECURSION,
    ) -> int:
        """
        This method finds minimum required sample size per cohort that generates
        average power higher than required

        Attributes:
            lower: lower bound of sample size search
            upper: upper bound of sample size search
            depth: number of recursions
            replication: number of Monte Carlo simulations to calculate empirical power
            epsilon: absolute difference between our estimate for power and desired power
                needed before we will return
            max_recursion_depth: how many recursive calls can be made before the
                search is abandoned

        Returns
            minimum required sample size per cohort
        """

        if depth > max_recursion_depth:
            raise RecursionError(f"Couldn't find a sample size that satisfies the power you requested: {self.power}")

        candidate = int(np.sqrt(lower * upper))
        expected_power = self._expected_average_power(candidate, random_state, replication)
        if np.isclose(self.power, expected_power, atol=epsilon):
            return candidate
        elif lower == upper:
            raise RecursionError(f"Couldn't find a sample size that satisfies the power you requested: {self.power}")

        if expected_power > self.power:
            return self.get_multiple_sample_size(lower, candidate, random_state, depth + 1)
        else:
            return self.get_multiple_sample_size(candidate, upper, random_state, depth + 1)

    def _expected_average_power(
        self, sample_size: int, random_state: np.random.RandomState, replication: int = DEFAULT_REPLICATION
    ) -> float:
        """
        This method calculates expected average power of multiple testings. For each possible number of true null
        hypothesis, we simulate each metric/treatment variant's test statistics and calculate their p-values,
        then calculate expected average power = number of True rejection/ true alternative hypotheses

        Attributes:
        sample size: determines the variance/ degrees of freedom of the distribution we sample test statistics from
        replication: number of times we repeat the simulation process

        Returns value expected average power
        """
        true_alt_count = 0.0
        true_discovery_count = 0.0

        # a metric for each test we would conduct
        metrics = self.metrics * (self.variants - 1)

        def fdr_bh(a: npt.NDArray[np.float_]) -> npt.NDArray[np.bool_]:
            rejected: npt.NDArray[np.bool_] = multipletests(a, alpha=self.alpha, method="fdr_bh")[0]
            return rejected

        for num_true_alt in range(1, len(metrics) + 1):
            true_alt = np.array([random_state.permutation(len(metrics)) < num_true_alt for _ in range(replication)]).T
            p_values = []
            for i, m in enumerate(metrics):
                p_values.append(m.generate_p_values(true_alt[i], sample_size, random_state))

            rejected = np.apply_along_axis(fdr_bh, 0, np.array(p_values))  # type: ignore[no-untyped-call]

            true_discoveries = rejected & true_alt

            true_discovery_count += true_discoveries.sum()
            true_alt_count += true_alt.sum()

        avg_power = true_discovery_count / true_alt_count

        return avg_power
