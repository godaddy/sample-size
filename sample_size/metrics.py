from abc import ABCMeta
from abc import abstractmethod
from typing import Union

from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.power import TTestIndPower


class BaseMetric:

    __metaclass__ = ABCMeta
    mde: float

    def __init__(self, mde: float):
        self.mde = mde

    @property
    @abstractmethod
    def default_power_analysis_instance(self) -> Union[NormalIndPower, TTestIndPower]:
        raise NotImplementedError

    @property
    @abstractmethod
    def variance(self) -> float:
        raise NotImplementedError

    @staticmethod
    def check_positive(number: float, name: str) -> float:
        if number < 0:
            raise Exception(f"Error: Please provide a positive number for {name}.")
        else:
            return number


class BooleanMetric(BaseMetric):

    probability: float
    mde: float

    def __init__(
        self,
        probability: float,
        mde: float,
    ):
        super(BooleanMetric, self).__init__(mde)
        self.probability = self._get_probability(probability)

    @property
    def variance(self) -> float:
        return self.probability * (1 - self.probability)

    @property
    def default_power_analysis_instance(self) -> NormalIndPower:
        return NormalIndPower()

    @staticmethod
    def _get_probability(probability: float) -> float:
        if 0 <= probability <= 1:
            return probability
        else:
            raise ValueError("Error: Please provide a float between 0 and 1 for probability.")


class NumericMetric(BaseMetric):

    mde: float

    def __init__(
        self,
        variance: float,
        mde: float,
    ):
        super(NumericMetric, self).__init__(mde)
        self._variance = self.check_positive(variance, "variance")

    @property
    def variance(self) -> float:
        return self._variance

    @property
    def default_power_analysis_instance(self) -> TTestIndPower:
        return TTestIndPower()


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
        super(RatioMetric, self).__init__(mde)
        self.numerator_mean = numerator_mean
        self.check_positive(numerator_variance, "variance")
        self.numerator_variance = numerator_variance
        self.denominator_mean = denominator_mean
        self.check_positive(denominator_variance, "variance")
        self.denominator_variance = denominator_variance
        self.covariance = covariance

    @property
    def variance(self) -> float:

        variance = (
            self.numerator_variance / self.denominator_mean ** 2
            + self.denominator_variance * self.numerator_mean ** 2 / self.denominator_mean ** 4
            - 2 * self.covariance * self.numerator_mean / self.denominator_mean ** 3
        )

        return variance

    @property
    def default_power_analysis_instance(self) -> TTestIndPower:
        return TTestIndPower()
