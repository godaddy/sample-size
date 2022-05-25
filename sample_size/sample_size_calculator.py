from typing import Any
from typing import Dict
from typing import List

import numpy as np

from sample_size.metrics import BaseMetric
from sample_size.metrics import BooleanMetric
from sample_size.metrics import NumericMetric
from sample_size.metrics import RatioMetric

DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.8
DEFAULT_VARIANTS = 2


class SampleSizeCalculator:
    """
    This class is to calculate sample size based on metric type

    Attributes:
    alpha: statistical significance
    power: statistical power

    """

    def __init__(self, alpha: float = DEFAULT_ALPHA, variants: int = DEFAULT_VARIANTS, power: float = DEFAULT_POWER):
        self.alpha = alpha
        self.power = power
        self.metrics: List[BaseMetric] = []
        self.variants: int = variants

    def _get_single_sample_size(self, metric: BaseMetric) -> float:
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
        return self._get_single_sample_size(self.metrics[0])

    def register_metrics(self, metrics: List[Dict[str, Any]]) -> None:
        METRIC_REGISTER_MAP = {
            "boolean": BooleanMetric,
            "numeric": NumericMetric,
            "ratio": RatioMetric,
        }
        for metric in metrics:
            metric_class = METRIC_REGISTER_MAP[metric["metric_type"]]
            registered_metric = metric_class(**metric["metric_metadata"])
            self.metrics.append(registered_metric)
