def main() -> None:
    """
    Calculate sample size based on user inputs for
        1. metric type: Boolean, Numeric, or Ratio (case insensitive)
        2. power analysis parameters: alpha, power, and mde
        3. inputs specific to metric type:
            * Boolean: probability
            * Numeric: variance
            * Ratio: mean and variance of numerator and denominator and their covariance

    NOTES:
        1. default statistical power is used in this script all the time
        2. the calculator supports single metric per calculator for now
    """
    from sample_size.sample_size_calculator import SampleSizeCalculator
    from sample_size.scripts.input_utils import get_alpha
    from sample_size.scripts.input_utils import get_metric_metadata
    from sample_size.scripts.input_utils import get_mde
    from sample_size.scripts.input_utils import get_variants
    from sample_size.scripts.input_utils import register_another_metric

    try:
        # Get alpha for power analysis
        alpha = get_alpha()
        variants = get_variants()
        calculator = SampleSizeCalculator(alpha, variants)

        register = True
        while register:
            metric_type, metric_metadata = get_metric_metadata()
            for v in range(int(variants) - 1):
                print(f'Treatment {v + 1}')
                if v > 0:
                    metric_metadata["mde"] = get_mde(metric_type)

                calculator.register_metric(metric_type, metric_metadata)
            register = register_another_metric()

        # Get and print sample size based on variable and power analysis parameters
        sample_size = calculator.get_sample_size()

        print("\nSample size needed in each group: {:.3f}".format(sample_size))

    except Exception as e:
        print(f"Error! The calculator isn't able to calculate sample size due to \n{e}")


if __name__ == "__main__":
    main()
