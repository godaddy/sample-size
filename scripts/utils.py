from typing import Dict
from typing import Tuple
from typing import Union

from sample_size.sample_size_calculator import SampleSizeCalculator

DEFAULT_ALPHA = 0.05
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


def is_float(value: str) -> float:
    try:
        float(value)
        return True
    except ValueError:
        return False


def get_float_input(input_name: str) -> float:
    input_str = input(f"Enter the {input_name}: ")
    if is_float(input_str):
        return float(input_str)
    else:
        raise Exception(f"Error: Please enter a float for the {input_name}.")


def get_alpha() -> Union[float, None]:
    alpha_default_check = input("Do you want to use default alpha (0.05) for the power analysis? (y/n)")
    if alpha_default_check.lower() == "n":
        alpha = get_float_input("alpha (between 0 and 0.3 inclusively)")
        if 0 < alpha <= 0.3:
            print(f"Using alpha ({alpha}) and default power (0.8)...")
            return alpha
        else:
            raise Exception("Error: Please provide a float between 0 and 0.3 for alpha.")
    else:
        print("Using default alpha (0.05) and power (0.8)...")
        return None


def get_mde(metric_type: str) -> float:
    mde = get_float_input(
        f"absolute minimum detectable effect for this {metric_type} \n"
        f"MDE: targeted treatment metric value minus the baseline value"
    )
    return mde


def get_metric_type_from_input() -> str:
    metric_type = input("Enter metric type (Boolean, Numeric, Ratio): ").strip().lower()
    if metric_type in ["boolean", "numeric", "ratio"]:
        return metric_type
    else:
        raise Exception("Error: Unexpected metric type. Please enter Boolean, Numeric, or Ratio.")


def get_variable_parameters(parameter_definitions: Dict[str, str]) -> Dict[str, float]:

    parameters = {}

    for parameter_name, parameter_definition in parameter_definitions.items():
        parameters[parameter_name] = get_float_input(parameter_definition)

    return parameters


def get_metric_metadata_from_input() -> Tuple[str, Dict[str, float]]:
    metric_type = get_metric_type_from_input().lower()
    metric_metadata = get_variable_parameters(METRIC_PARAMETERS[metric_type])
    metric_metadata["mde"] = get_mde(metric_type)

    return metric_type, metric_metadata


def register_metric(
    metric_type: str,
    metric_metadata: Dict[str, float],
    calculator: SampleSizeCalculator,
) -> None:
    VAR_REGISTER_FUNC_MAP = {
        "boolean": "register_bool_metric",
        "numeric": "register_numeric_metric",
        "ratio": "register_ratio_metric",
    }

    register_func = getattr(calculator, VAR_REGISTER_FUNC_MAP[metric_type])
    register_func(**metric_metadata)
