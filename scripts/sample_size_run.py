def main() -> None:
    """
    Calculate sample size based on user inputs for
        1. metric type: Boolean, Numeric, or Ratio (case insensitive)
        2. power analysis parameters: alpha, power, and mde
        3. inputs specific to metric type:
            * Boolean: probability
            * Numeric: variance
            * Ratio: mean and variance of numerator and denominator and their covariance

    NOTE: the calculator supports single metric per calculator for now.
    """
    from sample_size.sample_size_calculator import SampleSizeCalculator
    from scripts.utils import get_alpha
    from scripts.utils import get_metric_metadata_from_input
    from scripts.utils import register_metric

    try:
        # Get alpha for power analysis
        alpha = get_alpha()
        if alpha:
            calculator = SampleSizeCalculator(alpha)
        else:
            calculator = SampleSizeCalculator()

        # register metric
        metric_type, metric_metadata = get_metric_metadata_from_input()
        register_metric(metric_type, metric_metadata, calculator)

        # Get and print sample size based on variable and power analysis parameters
        sample_size = calculator.get_sample_size()
        print("\nSample size needed in each group: {:.3f}".format(sample_size))
    except Exception as e:
        print(f"Error! The calculator isn't able to calculate sample size due to \n{e}")
        return


if __name__ == "__main__":
    main()
