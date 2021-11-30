from typing import List

import numpy as np
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.power import TTestIndPower

from sample_size.sample_size_calculator.variables import Boolean
from sample_size.sample_size_calculator.variables import Numeric
from sample_size.sample_size_calculator.variables import Ratio

DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.8


class PowerAnalysisParameters:
    """
    Class to house parameters for power analysis
    metric_variance: variance of the metric
    mde: absolute minimum detectable effect
    alpha: statistical significance
    power: statistical power
    """

    def __init__(
            self,
            metric_variance: float,
            mde: float,
            alpha: float = DEFAULT_ALPHA,
            power: float = DEFAULT_POWER,
    ):
        self.metric_variance = metric_variance
        self.mde = mde
        self.alpha = alpha
        self.power = power


class SampleSizeCalculator:
    """This class is to calculate sample size based on variable type

    Attributes:
    alpha: statistical significance
    power: statistical power

    """

    def __init__(self, alpha: float = DEFAULT_ALPHA, power: float = DEFAULT_POWER):
        self.alpha = alpha
        self.power = power
        self.boolean_metrics: List[PowerAnalysisParameters] = []
        self.numeric_metrics: List[PowerAnalysisParameters] = []
        self.ratio_metrics: List[PowerAnalysisParameters] = []

    def register_bool_metric(self, mde: float, probability: float):
        variable = Boolean(probability)
        metric = PowerAnalysisParameters(variable.variance, mde, self.alpha, self.power)
        self.boolean_metrics.append(metric)

    def register_numeric_metric(self, mde: float, variance: float):
        variable = Numeric(variance)
        metric = PowerAnalysisParameters(variable.variance, mde, self.alpha, self.power)
        self.numeric_metrics.append(metric)

    def register_ratio_metric(
            self,
            mde: float,
            numerator_mean: float,
            numerator_variance: float,
            denominator_mean: float,
            denominator_variance: float,
            covariance: float,
    ):
        variable = Ratio(numerator_mean, numerator_variance, denominator_mean, denominator_variance, covariance)
        metric = PowerAnalysisParameters(variable.variance, mde, self.alpha, self.power)
        self.ratio_metrics.append(metric)

    @staticmethod
    def _get_single_sample_size(metric, power_analysis_type):
        effect_size = metric.mde / float(np.sqrt(metric.metric_variance))
        power_analysis = power_analysis_type()
        sample_size = int(
            power_analysis.solve_power(
                effect_size=effect_size,
                alpha=metric.alpha,
                power=metric.power,
                ratio=1,
                alternative="two-sided",
            )
        )
        return sample_size

    def get_overall_sample_size(self):
        # Supports the sample size calculation for single metric now
        if self.boolean_metrics:
            for metric in self.boolean_metrics:
                return self._get_single_sample_size(metric, NormalIndPower)
        if self.numeric_metrics:
            for metric in self.numeric_metrics:
                return self._get_single_sample_size(metric, TTestIndPower)
        if self.ratio_metrics:
            for metric in self.ratio_metrics:
                return self._get_single_sample_size(metric, TTestIndPower)
