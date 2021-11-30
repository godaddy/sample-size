def main() -> None:
    """
    Calculate sample size based on user inputs for
        1. metric type: Boolean, Numeric, or Ratio (case insensitive)
        2. power analysis parameters: alpha, power, and mde
        3. inputs specific to metric type:
            * Boolean: probability
            * Numeric: variance
            * Ratio: mean and variance of numerator and denominator and their covariance

    """
    from sample_size.sample_size_calculator import SampleSizeCalculator
    from scripts.utils import get_alpha
    from scripts.utils import get_variable_from_input
    from scripts.utils import register_metric

    # Get alpha for power analysis
    alpha = get_alpha()
    if alpha:
        calculator = SampleSizeCalculator(alpha)
    else:
        calculator = SampleSizeCalculator()

    # register metric
    metric_type = get_variable_from_input()
    register_metric(metric_type, calculator)

    # Get and print sample size based on variable and power analysis parameters
    sample_size = calculator.get_overall_sample_size()
    print("\n Sample size needed in each group: {:.3f}".format(sample_size))


if __name__ == "__main__":
    main()
