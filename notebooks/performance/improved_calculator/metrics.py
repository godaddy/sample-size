# +
from abc import ABCMeta
from abc import abstractmethod
from typing import Union

import numpy as np
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.power import TTestIndPower


class BaseMetric:
    __metaclass__ = ABCMeta
    mde: float

    def __init__(self, mde: float):
        self.mde = mde

    @property
    @abstractmethod
    def power_analysis_instance(self) -> Union[NormalIndPower, TTestIndPower]:
        raise NotImplementedError

    @property
    @abstractmethod
    def variance(self) -> float:
        raise NotImplementedError

    @staticmethod
    def check_positive(number: float, name: str) -> float:
        if number < 0:
            raise ValueError(f"Error: Please provide a positive number for {name}.")
        else:
            return number

    def generate_p_values(self, true_alt: bool, sample_size: int) -> float:
        """
        This method simulates any registered metric's p-value. The output will later be applied to BH procedure

        Parameters:
            true_null: whether the null hypothesis is true
            sample_size: sample size used for simulations

        Returns:
            p-value: the simulated test statistics' p-value
        """
        total_alt = true_alt.sum()
        total_null = true_alt.size - total_alt

        p_values = np.empty(true_alt.shape)

        p_values[true_alt] = self._generate_alt_p_values(total_alt, sample_size)
        p_values[~true_alt] = stats.uniform.rvs(0, 1, total_null)

        return p_values

    @abstractmethod
    def _generate_alt_p_values(self, size: int, sample_size: int) -> np.array:
        raise NotImplemented


class BooleanMetric(BaseMetric):
    probability: float

    def __init__(
        self, probability: float, mde: float,
    ):
        super(BooleanMetric, self).__init__(mde)
        self.probability = self._check_probability(probability)

    @property
    def variance(self) -> float:
        return self.probability * (1 - self.probability)

    @property
    def power_analysis_instance(self) -> NormalIndPower:
        return NormalIndPower()

    @staticmethod
    def _check_probability(probability: float) -> float:
        if 0 <= probability <= 1:
            return probability
        else:
            raise ValueError("Error: Please provide a float between 0 and 1 for probability.")

    def _generate_alt_p_values(self, size: int, sample_size: int) -> np.array:
        effect_size = self.mde / np.sqrt(2 * self.variance / sample_size)
        z_alt = stats.norm.rvs(loc=effect_size, size=size)
        return 2 * stats.norm.sf(np.abs(z_alt))


class NumericMetric(BaseMetric):
    mde: float

    def __init__(
        self, variance: float, mde: float,
    ):
        super(NumericMetric, self).__init__(mde)
        self._variance = self.check_positive(variance, "variance")

    @property
    def variance(self) -> float:
        return self._variance

    @property
    def power_analysis_instance(self) -> TTestIndPower:
        return TTestIndPower()

    def _generate_alt_p_values(self, size: int, sample_size: int) -> np.array:
        nc = np.sqrt(sample_size / 2 / self.variance) * self.mde
        t_alt = stats.nct.rvs(nc=nc, df=2 * (sample_size - 1), size=size)
        return 2 * stats.t.sf(np.abs(t_alt), 2 * (sample_size - 1))


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
        # TODO: add check for Cauchy-Schwarz inequality
        self.numerator_mean = numerator_mean
        self.numerator_variance = self.check_positive(numerator_variance, "numerator variance")
        self.denominator_mean = denominator_mean
        self.denominator_variance = self.check_positive(denominator_variance, "denominator variance")
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
    def power_analysis_instance(self) -> NormalIndPower:
        return NormalIndPower()

    def _generate_alt_p_values(self, size: int, sample_size: int) -> np.array:
        effect_size = self.mde / np.sqrt(2 * self.variance / sample_size)
        z_alt = stats.norm.rvs(loc=effect_size, size=size)
        return 2 * stats.norm.sf(np.abs(z_alt))
