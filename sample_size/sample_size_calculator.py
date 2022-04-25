from itertools import product
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.weightstats import ztest

from sample_size.metrics import BaseMetric
from sample_size.metrics import BooleanMetric
from sample_size.metrics import NumericMetric
from sample_size.metrics import RatioMetric

DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.8


class SampleSizeCalculator:
    """
    This class is to calculate sample size based on metric type

    Attributes:
    alpha: statistical significance
    power: statistical power

    """

    def __init__(self, alpha: float = DEFAULT_ALPHA, power: float = DEFAULT_POWER):
        self.alpha = alpha
        self.power = power
        # Consider having a self.metrics to hold all metric types
        self.boolean_metrics: List[BooleanMetric] = []
        self.numeric_metrics: List[NumericMetric] = []
        self.ratio_metrics: List[RatioMetric] = []
        self.rep: int = 100

    def register_bool_metric(self, probability: float, mde: float) -> None:
        metric = BooleanMetric(probability, mde)
        self.boolean_metrics.append(metric)

    def register_numeric_metric(self, variance: float, mde: float) -> None:
        metric = NumericMetric(variance, mde)
        self.numeric_metrics.append(metric)

    def register_ratio_metric(
        self,
        numerator_mean: float,
        numerator_variance: float,
        denominator_mean: float,
        denominator_variance: float,
        covariance: float,
        mde: float,
    ) -> None:
        metric = RatioMetric(
            numerator_mean, numerator_variance, denominator_mean, denominator_variance, covariance, mde
        )
        self.ratio_metrics.append(metric)

    def _get_single_sample_size(self, metric: BaseMetric) -> float:
        effect_size = metric.mde / float(np.sqrt(metric.variance))
        power_analysis = metric.default_power_analysis_instance
        sample_size = int(
            power_analysis.solve_power(
                effect_size=effect_size,
                alpha=self.alpha,
                power=self.power,
                ratio=1,
                alternative="two-sided",
            )
        )
        return sample_size

    def _get_multiple_sample_size(self, d: float, power_analysis: Union[NormalIndPower, TTestIndPower]) -> int:
        m = len(self.boolean_metrics) + len(self.numeric_metrics) + len(self.ratio_metrics)
        # calculate required sample size based on minimum standardized effect size since it requires maximum sample size
        lower = power_analysis.solve_power(
            effect_size=d, alpha=self.alpha, power=self.power, ratio=1, alternative="two-sided"
        )
        upper = power_analysis.solve_power(
            effect_size=d, alpha=self.alpha / m, power=self.power, ratio=1, alternative="two-sided"
        )

        # print(f'We look for the minimum required sample size in range [{int(lower)},{int(upper)}]')
        for size in np.linspace(lower, upper, 10):
            expected_power = self._expected_average_power(m, int(size))
            if expected_power >= self.power:
                break
        return int(size)

    def _expected_average_power(self, number_of_tests: int, size: int) -> float:
        m = number_of_tests
        pp_null, pp_alt = [], []

        if self.boolean_metrics:
            for bool_metric in self.boolean_metrics:
                p_null, p_alt = self._generate_p_value(bool_metric, size)
                pp_null.append(p_null)
                pp_alt.append(p_alt)
        if self.numeric_metrics:
            for numeric_metric in self.numeric_metrics:
                p_null, p_alt = self._generate_p_value(numeric_metric, size)
                pp_null.append(p_null)
                pp_alt.append(p_alt)
        if self.ratio_metrics:
            for ratio_metric in self.ratio_metrics:
                p_null, p_alt = self._generate_p_value(ratio_metric, size)
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
                true_null_p = [float(np.array(pp_null)[x, r]) for x in null_index]
                true_alt_p = [float(np.array(pp_alt)[x, r]) for x in alt_index]
                # first len(true_null_p) hypotheses are true null
                pvalues = np.zeros(m)
                pvalues[: len(null_index)] = true_null_p
                pvalues[len(null_index) :] = true_alt_p
                rejs[r, :] = multipletests(pvalues, alpha=self.alpha, method="fdr_bh")[0]

            actual_pw = np.sum(rejs[:, len(null_index) :], axis=1) / len(alt_index)

            avg_power += np.mean(actual_pw)

        return avg_power / len(true_H)

    def _generate_p_value(self, metric: BaseMetric, size: int) -> Tuple[List[float], List[float]]:
        metric_type: str = type(metric).__name__
        p_null, p_alt = np.zeros(self.rep), np.zeros(self.rep)

        if metric_type == "BooleanMetric":
            delta = metric.mde * size
            var = metric.variance * size

            null_data = np.random.normal(0, np.sqrt(var), self.rep * 2)
            alt_data = np.random.normal(delta, np.sqrt(var), self.rep)
            # resample all negative values
            while np.sum(null_data < 0) > 0:
                null_data[null_data < 0] = np.random.normal(0, np.sqrt(var), len(null_data[null_data < 0]))

            while np.sum(alt_data < 0) > 0:
                alt_data[alt_data < 0] = np.random.normal(delta, np.sqrt(var), len(alt_data[alt_data < 0]))

            for j in range(self.rep):
                p_null[j] = proportions_ztest([null_data[j], null_data[j + self.rep]], [size, size])[1]
                p_alt[j] = proportions_ztest([null_data[j], alt_data[j]], [size, size])[1]

        elif metric_type == "NumericMetric":
            null_data = np.random.normal(0, np.sqrt(metric.variance), (size, self.rep * 2))
            alt_data = np.random.normal(metric.mde, np.sqrt(metric.variance), (size, self.rep))
            # resample all negative values
            while np.sum(null_data < 0) > 0:
                null_data[null_data < 0] = np.random.normal(0, np.sqrt(metric.variance), len(null_data[null_data < 0]))

            while np.sum(alt_data < 0) > 0:
                alt_data[alt_data < 0] = np.random.normal(
                    metric.mde, np.sqrt(metric.variance), len(alt_data[alt_data < 0])
                )

            for j in range(self.rep):
                p_null[j] = stats.ttest_ind(null_data[:, j], null_data[:, j + self.rep])[1]
                p_alt[j] = stats.ttest_ind(null_data[:, j], alt_data[:, j])[1]

        elif metric_type == "RatioMetric":
            null_data = np.random.normal(0, np.sqrt(metric.variance), (size, self.rep * 2))
            alt_data = np.random.normal(metric.mde, np.sqrt(metric.variance), (size, self.rep))
            while np.sum(null_data < 0) > 0:
                null_data[null_data < 0] = np.random.normal(0, np.sqrt(metric.variance), len(null_data[null_data < 0]))

            while np.sum(alt_data < 0) > 0:
                alt_data[alt_data < 0] = np.random.normal(
                    metric.mde, np.sqrt(metric.variance), len(alt_data[alt_data < 0])
                )

            for j in range(self.rep):
                p_null[j] = ztest(null_data[:, j], null_data[:, j + self.rep])[1]
                p_alt[j] = ztest(null_data[:, j], alt_data[:, j])[1]

        return list(p_null), list(p_alt)

    def get_sample_size(self) -> float:
        # Supports the sample size calculation for single metric now.
        # The current structure is set up to support multiple metrics in the future.
        sample_size = float("nan")

        if len(self.boolean_metrics) + len(self.numeric_metrics) + len(self.ratio_metrics) == 1:
            if self.boolean_metrics:
                sample_size = self._get_single_sample_size(self.boolean_metrics[0])
            if self.numeric_metrics:
                sample_size = self._get_single_sample_size(self.numeric_metrics[0])
            if self.ratio_metrics:
                sample_size = self._get_single_sample_size(self.ratio_metrics[0])

        else:
            min_standardized_effect_size: float = 0
            min_metric: BaseMetric
            if self.boolean_metrics:
                for bool_metric in self.boolean_metrics:
                    if min_standardized_effect_size > bool_metric.mde / np.sqrt(bool_metric.variance):
                        min_standardized_effect_size = bool_metric.mde / np.sqrt(bool_metric.variance)
                        min_metric = bool_metric
            if self.numeric_metrics:
                for numeric_metric in self.numeric_metrics:
                    if min_standardized_effect_size > numeric_metric.mde / np.sqrt(numeric_metric.variance):
                        min_standardized_effect_size = numeric_metric.mde / np.sqrt(numeric_metric.variance)
                        min_metric = numeric_metric
            if self.ratio_metrics:
                for ratio_metric in self.ratio_metrics:
                    if min_standardized_effect_size > ratio_metric.mde / np.sqrt(ratio_metric.variance):
                        min_standardized_effect_size = ratio_metric.mde / np.sqrt(ratio_metric.variance)
                        min_metric = ratio_metric
            power_analysis = min_metric.default_power_analysis_instance
            sample_size = self._get_multiple_sample_size(min_standardized_effect_size, power_analysis)

        return sample_size

    def register_metric(self, metric_type: str, metric_metadata: Dict[str, float]) -> None:
        VAR_REGISTER_FUNC_MAP = {
            "boolean": "register_bool_metric",
            "numeric": "register_numeric_metric",
            "ratio": "register_ratio_metric",
        }

        register_func = getattr(self, VAR_REGISTER_FUNC_MAP[metric_type])
        register_func(**metric_metadata)
