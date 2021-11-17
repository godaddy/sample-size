from typing import Any

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
    alpha: statistical significance
    power: statistical power
    mde: minimum detectable effect
    """

    mde: float
    alpha: float = DEFAULT_ALPHA
    power: float = DEFAULT_POWER


class PowerAnalysisType:
    """
    Class to house power analysis methods
    ZTestPower: power analysis method based on Z-test (used by Boolean variable)
    TTestPower: power analysis method based on T-test (used by Numeric and Ratio variables)
    """

    ZTestPower = NormalIndPower
    TTestPower = TTestIndPower


class BaseSampleSizeCalculator:
    def __init__(
        self, variable_variance: float, variable_analysis_type: Any, power_analysis_parameters: PowerAnalysisParameters
    ):
        self.variable_variance = variable_variance
        self.variable_analysis_type = variable_analysis_type
        self.power_analysis_parameters = power_analysis_parameters
        self.std_effect_size = self.get_std_effect_size()

    def get_std_effect_size(self) -> float:
        return self.power_analysis_parameters.mde / float(np.sqrt(self.variable_variance))

    def get_base_sample_size(self) -> float:

        # perform power analysis to find sample size for given effect
        power_analysis = self.variable_analysis_type
        power_analysis_obj = power_analysis()

        sample_size = float(
            power_analysis_obj.solve_power(
                effect_size=self.std_effect_size,
                alpha=self.power_analysis_parameters.alpha,
                power=self.power_analysis_parameters.power,
                ratio=1,
                alternative="two-sided",
            )
        )

        return sample_size


class SampleSizeCalculators:
    """This class is to calculate sample size based on variable type

    Attributes:
        power_analysis_parameters (PowerAnalysisParameters): Parameters for power analysis.

    """

    def __init__(self, mde: float, alpha: float = DEFAULT_ALPHA, power: float = DEFAULT_POWER):
        self.power_analysis_parameters = PowerAnalysisParameters()
        self.power_analysis_parameters.alpha = alpha
        self.power_analysis_parameters.power = power
        self.power_analysis_parameters.mde = mde

    def get_sample_size(self, variance: float, power_analysis_type: Any) -> float:
        sample_size_calculator = BaseSampleSizeCalculator(variance, power_analysis_type, self.power_analysis_parameters)
        return sample_size_calculator.get_base_sample_size()

    def get_boolean_sample_size(self, probability: float) -> float:
        bool_obj = Boolean(probability)
        return self.get_sample_size(bool_obj.variance, NormalIndPower)

    def get_numeric_sample_size(self, mean: float, variance: float) -> float:
        numeric_obj = Numeric(mean, variance)
        return self.get_sample_size(numeric_obj.variance, TTestIndPower)

    def get_ratio_sample_size(
        self,
        numerator_mean: float,
        numerator_variance: float,
        denominator_mean: float,
        denominator_variance: float,
        covariance: float,
    ) -> float:
        ratio_obj = Ratio(numerator_mean, numerator_variance, denominator_mean, denominator_variance, covariance)
        return self.get_sample_size(ratio_obj.variance, TTestIndPower)
