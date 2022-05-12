from typing import Dict
from typing import List
import numpy as np
from sample_size.metrics import BaseMetric
from sample_size.multiple_testing_funcs import MultipleTesting

DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.8
DEFAULT_VARIANTS = 2


class SampleSizeCalculator:
    """
    This class is to calculate sample size based on metric type

    Attributes:
    alpha: statistical significance
    power: statistical power
    variants: number of variants, including control
    """

    def __init__(self, alpha: float = DEFAULT_ALPHA, variants: int = DEFAULT_VARIANTS, power: float = DEFAULT_POWER):
        self.alpha = alpha
        self.power = power
        self.variants = variants
        self.metrics: List[BaseMetric] = []

    def register_metric(self, metric_type: str, metric_metadata: Dict[str, float]) -> None:
        METRIC_REGISTER_MAP = {
            "boolean": "BooleanMetric",
            "numeric": "NumericMetric",
            "ratio": "RatioMetric",
        }
        metric_register = getattr(self, METRIC_REGISTER_MAP[metric_type])
        metric = metric_register(**metric_metadata)
        self.metrics.append(metric)

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

    def get_sample_size(self) -> float:
        if len(self.metrics) == 1 and self.variants == DEFAULT_VARIANTS:
            sample_size = self._get_single_sample_size(self.metrics[0])
        else:
            sample_size = MultipleTesting(self.metrics, self.variants, self.alpha,
                                          self.power).get_multiple_sample_size()

        return sample_size
