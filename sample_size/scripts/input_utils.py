from typing import Collection
from typing import Dict
from typing import List

from sample_size.sample_size_calculator import DEFAULT_ALPHA

METRIC_PARAMETERS = {
    "boolean": {"probability": "baseline probability (between 0 and 1)"},
    "numeric": {"variance": "variance of the baseline metric"},
    "ratio": {
        "numerator_mean": "mean of the baseline metric's numerator",
        "numerator_variance": "variance of the baseline metric's numerator",
        "denominator_mean": "mean of the baseline metric's denominator",
        "denominator_variance": "variance of the baseline metric's denominator",
        "covariance": "covariance between the baseline metric's numerator and denominator",
    },
}


def is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def get_float(input_str: str, input_name: str) -> float:
    input_str = input_str.strip()
    if is_float(input_str):
        return float(input_str)
    else:
        raise ValueError(f"Error: Please enter a float for the {input_name}.")


def get_alpha() -> float:
    alpha_input = input(
        "Enter the alpha between (between 0 and 0.3 inclusively) or press Enter to use default alpha=0.05: "
    ).strip()

    if alpha_input == "":
        print("Using default alpha (0.05) and default power (0.8)...")
        return DEFAULT_ALPHA
    else:
        alpha = get_float(alpha_input, "alpha")
        if 0 < alpha <= 0.3:
            print(f"Using alpha ({alpha}) and default power (0.8)...")
            return alpha
        else:
            raise ValueError("Error: Please provide a float between 0 and 0.3 for alpha.")


def get_mde(metric_type: str) -> float:
    mde = get_float(
        input(
            f"Enter the absolute minimum detectable effect for this {metric_type} \n"
            f"MDE: targeted treatment metric value minus the baseline value: "
        ),
        "minimum detectable effect",
    )
    return mde


def get_metric_type() -> str:
    metric_type = input("Enter metric type (Boolean, Numeric, Ratio): ").strip().lower()
    if metric_type in ["boolean", "numeric", "ratio"]:
        return metric_type
    else:
        raise ValueError("Error: Unexpected metric type. Please enter Boolean, Numeric, or Ratio.")


def get_metric_parameters(parameter_definitions: Dict[str, str]) -> Dict[str, float]:
    parameters = {}

    for parameter_name, parameter_definition in parameter_definitions.items():
        parameters[parameter_name] = get_float(input(f"Enter the {parameter_definition}: "), parameter_definition)

    return parameters


def get_variants() -> int:
    number_of_variants = (
        input(
            "Enter the number of cohorts for this test or Press Enter to use default variant = 2 if you have only 1 "
            "control and 1 treatment. \n"
            "definition: Control + number of treatments: "
        )
        .strip()
        .lower()
    )
    if number_of_variants.isdigit():
        if int(number_of_variants) < 2:
            raise ValueError("Error: An experiment must contain at least 2 variants.")
        return int(number_of_variants)
    elif number_of_variants == "":
        print("Using default variants(2)...")
        return 2
    else:
        raise ValueError("Error: Please enter a positive integer for the number of variants.")


def register_another_metric() -> bool:
    register = input("Are you going to register another metric? (y/n)").strip().lower()
    if register == "y":
        return True
    elif register in ["n", ""]:
        return False
    else:
        raise ValueError("Error: Please enter 'y' or 'n'.")


def _get_metric() -> Dict[str, Collection[str]]:
    metric_type = get_metric_type()
    metric_metadata = get_metric_parameters(METRIC_PARAMETERS[metric_type])
    metric_metadata["mde"] = get_mde(metric_type)
    return {"metric_type": metric_type, "metric_metadata": metric_metadata}


def get_metrics() -> List[Dict[str, Collection[str]]]:
    metrics = [_get_metric()]
    while register_another_metric():
        metrics.append(_get_metric())

    return metrics
