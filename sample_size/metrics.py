from typing import Union

from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.power import TTestIndPower


class BaseMetric:

    variance: float
    mde: float
    default_power_analysis_instance: Union[NormalIndPower, TTestIndPower]

    @staticmethod
    def check_positive(number: float, name: str) -> None:
        if number < 0:
            raise Exception(f"Please provide a positive number for {name}.")


class BooleanMetric(BaseMetric):

    probability: float

    def __init__(
        self,
        probability: float,
        mde: float,
    ):
        self.probability = self._get_probability(probability)
        self.variance = self._get_variance()
        self.mde = mde
        self.default_power_analysis_instance = NormalIndPower()

    def _get_variance(self) -> float:
        return self.probability * (1 - self.probability)

    @staticmethod
    def _get_probability(probability: float) -> float:
        if 0 <= probability <= 1:
            return probability
        else:
            raise ValueError("Error: Please provide a float between 0 and 1 for probability.")


class NumericMetric(BaseMetric):
    def __init__(
        self,
        variance: float,
        mde: float,
    ):
        self.check_positive(variance, "variance")
        self.variance = variance
        self.mde = mde
        self.default_power_analysis_instance = TTestIndPower()


class RatioMetric(BaseMetric):

    numerator_mean: float
    numerator_variance: float
    denominator_mean: float
    denominator_variance: float
    covariance: float

    def __init__(
        self,
        numerator_mean: float,
        numerator_variance: float,
        denominator_mean: float,
        denominator_variance: float,
        covariance: float,
        mde: float,
    ):
        self.numerator_mean = numerator_mean
        self.check_positive(numerator_variance, "variance")
        self.numerator_variance = numerator_variance
        self.denominator_mean = denominator_mean
        self.check_positive(denominator_variance, "variance")
        self.denominator_variance = denominator_variance
        self.covariance = covariance
        self.variance = self._get_variance()
        self.mde = mde
        self.default_power_analysis_instance = TTestIndPower()

    def _get_variance(self) -> float:

        variance = (
            self.numerator_variance / self.denominator_mean ** 2
            + self.denominator_variance * self.numerator_mean ** 2 / self.denominator_mean ** 4
            - 2 * self.covariance * self.numerator_mean / self.denominator_mean ** 3
        )

        return variance
