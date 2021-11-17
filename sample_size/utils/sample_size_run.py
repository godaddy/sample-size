from sample_size.sample_size_calculator.sample_size_calculator import SampleSizeCalculators
from sample_size.utils.utils import get_power_analysis_input
from sample_size.utils.utils import get_sample_size
from sample_size.utils.utils import get_variable_from_input

if __name__ == "__main__":
    """
    Calculate sample size based on user inputs for
        1. metric type: Boolean, Numeric, or Ratio (case insensitive)
        2. power analysis parameters: alpha, power, and mde
        3. variable input:
            * Boolean: probability
            * Numeric: mean and variance
            * Ratio: mean and variance of numerator and denominator and their covariance

    For future use in Hivemind, this is a sample code to calculate sample sizes with given parameters
        calculator = SampleSizeCalculators(mde=0.02)
        bool_sample_size = calculator.get_boolean_sample_size(probability=0.05)
        numeric_sample_size = calculator.get_numeric_sample_size(mean=109, variance=941)
        ratio__sample_size = calculator.get_ratio_sample_size(
            numerator_mean=109,
            numerator_variance=941,
            denominator_mean=2046,
            denominator_variance=9668,
            covariance=7236
        )
    """

    # Get variable object based on user input for metric type
    variable_name = get_variable_from_input()

    # Get power analysis parameters object based on user input
    power_analysis_parameters = get_power_analysis_input()

    # Get calculator
    calculator = SampleSizeCalculators(
        power_analysis_parameters.mde, power_analysis_parameters.alpha, power_analysis_parameters.power
    )

    # Get and print sample size based on variable and power analysis parameters
    sample_size = get_sample_size(variable_name, calculator)
    print(sample_size)
    print("\n Sample size needed in each group: {:.3f}".format(sample_size))
