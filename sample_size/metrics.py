from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.power import TTestIndPower


class BooleanMetric:
    def __init__(
        self,
        probability: float,
        mde: float,
    ):
        self.probability = self._get_probability(probability)
        self.variance = self._get_variance()
        self.mde = mde
        self.default_power_analysis_type = NormalIndPower

    def _get_variance(self) -> float:
        return self.probability * (1 - self.probability)

    @staticmethod
    def _get_probability(probability: float) -> float:
        if 0 <= probability <= 1:
            return probability
        else:
            raise Exception("Error: Please provide a float between 0 and 1 for probability.")


class NumericMetric:
    def __init__(
        self,
        variance: float,
        mde: float,
    ):
        self.variance = variance
        self.mde = mde
        self.default_power_analysis_type = TTestIndPower


class RatioMetric:
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
        self.numerator_variance = numerator_variance
        self.denominator_mean = denominator_mean
        self.denominator_variance = denominator_variance
        self.covariance = covariance
        self.variance = self._get_variance()
        self.mde = mde
        self.default_power_analysis_type = TTestIndPower

    def _get_variance(self) -> float:

        variance = (
            self.numerator_variance / self.denominator_mean ** 2
            + self.denominator_variance * self.numerator_mean ** 2 / self.denominator_mean ** 4
            - 2 * self.covariance * self.numerator_mean / self.denominator_mean ** 3
        )

        return variance
