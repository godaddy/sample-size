from typing import Dict
from typing import List

import numpy as np

from sample_size.metrics import BaseMetric
from sample_size.metrics import BooleanMetric
from sample_size.metrics import NumericMetric
from sample_size.metrics import RatioMetric

from sample_size.multiple_testing import MultipleTestingMixin

DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.8


class SampleSizeCalculator(MultipleTestingMixin):
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

    def get_single_sample_size(self, metric: BaseMetric) -> float:
        effect_size = metric.mde / float(np.sqrt(metric.variance))
        power_analysis = metric.power_analysis_instance
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

    def get_sample_size(self) -> float:
        return self.get_multiple_sample_size()

    def register_metric(self, metric_type: str, metric_metadata: Dict[str, float]) -> None:
        VAR_REGISTER_FUNC_MAP = {
            "boolean": "register_bool_metric",
            "numeric": "register_numeric_metric",
            "ratio": "register_ratio_metric",
        }

        register_func = getattr(self, VAR_REGISTER_FUNC_MAP[metric_type])
        register_func(**metric_metadata)
