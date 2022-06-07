import json
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

import numpy as np
from jsonschema import validate

from sample_size.metrics import BaseMetric
from sample_size.metrics import BooleanMetric
from sample_size.metrics import NumericMetric
from sample_size.metrics import RatioMetric
from sample_size.multiple_testing import MultipleTestingMixin

DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.8
DEFAULT_VARIANTS = 2

schema_file_path = Path(Path(__file__).parent, "metrics_schema.json")
with open(str(schema_file_path), "r") as schema_file:
    METRICS_SCHEMA = json.load(schema_file)


class SampleSizeCalculator(MultipleTestingMixin):
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

    def _get_single_sample_size(self, metric: BaseMetric, alpha: float) -> float:
        effect_size = metric.mde / float(np.sqrt(metric.variance))
        power_analysis = metric.power_analysis_instance
        sample_size = int(
            power_analysis.solve_power(
                effect_size=effect_size,
                alpha=alpha,
                power=self.power,
                ratio=1,
                alternative="two-sided",
            )
        )
        return sample_size

    def get_sample_size(self) -> float:
        if len(self.metrics) * (self.variants - 1) < 2:
            return self._get_single_sample_size(self.metrics[0], self.alpha)
        lower = min([self._get_single_sample_size(metric, self.alpha) for metric in self.metrics])
        upper = max(
            [
                self._get_single_sample_size(metric, self.alpha / (len(self.metrics) * (self.variants - 1)))
                for metric in self.metrics
            ]
        )
        return self.get_multiple_sample_size(lower, upper)

    def register_metrics(self, metrics: List[Dict[str, Any]]) -> None:
        METRIC_REGISTER_MAP = {
            "boolean": BooleanMetric,
            "numeric": NumericMetric,
            "ratio": RatioMetric,
        }

        validate(instance=metrics, schema=METRICS_SCHEMA)

        for metric in metrics:
            metric_class = METRIC_REGISTER_MAP[metric["metric_type"]]
            registered_metric = metric_class(**metric["metric_metadata"])
            self.metrics.append(registered_metric)
