from typing import Dict
from typing import Union

from sample_size.sample_size_calculator import SampleSizeCalculator

DEFAULT_ALPHA = 0.05


def is_float(value: str) -> float:
    try:
        float(value)
        return True
    except ValueError:
        return False


def get_raw_input(text: str) -> str:
    return input(text)


def get_input(input_name: str) -> float:
    input_str = get_raw_input(f"Enter the {input_name}: ")
    if is_float(input_str):
        return float(input_str)
    else:
        raise Exception(f"Error: Please enter a float for the {input_name}.")


def get_alpha() -> Union[float, None]:
    alpha_default_check = get_raw_input("Do you want to use default alpha (0.05) for the power analysis? (y/n)")
    if alpha_default_check.lower() == "n":
        alpha = get_input("alpha")
        if 0 <= alpha <= 0.3:
            return alpha
        else:
            raise Exception("Error: Please provide a float between 0 and 0.3 for alpha.")
    else:
        print("Using default alpha...")
        return None


def get_mde(metric_type: str) -> float:
    mde = get_input(f"absolute minimum detectable effect for this {metric_type}")
    return mde


def get_variable_from_input() -> str:
    metric_type = get_raw_input("Enter metric type (Boolean, Numeric, Ratio): ").strip().lower()
    if metric_type in ["boolean", "numeric", "ratio"]:
        return metric_type
    else:
        raise Exception("Error: Unexpected metric type. Please enter Boolean, Numeric, or Ratio.")


def get_variable_parameters(parameter_definitions: Dict[str, str]) -> Dict[str, float]:

    parameters = {}

    for parameter_name, parameter_definition in parameter_definitions.items():
        parameters[parameter_name] = get_input(parameter_definition)

    return parameters


def register_metric(metric_type: str, calculator: SampleSizeCalculator) -> None:
    metric_type = metric_type.lower()
    if metric_type == "boolean":
        parameters_definitions = {"probability": "baseline probability"}
        parameters = get_variable_parameters(parameters_definitions)
        mde = get_mde(metric_type)
        calculator.register_bool_metric(mde, parameters["probability"])
    elif metric_type == "numeric":
        parameters_definitions = {"variance": "variance of the baseline metric"}
        parameters = get_variable_parameters(parameters_definitions)
        mde = get_mde(metric_type)
        calculator.register_numeric_metric(mde, parameters["variance"])
    elif metric_type == "ratio":
        parameters_definitions = {
            "numerator_mean": "mean of the baseline metric's numerator",
            "numerator_variance": "variance of the baseline metric's numerator",
            "denominator_mean": "mean of the baseline metric's denominator",
            "denominator_variance": "variance of the baseline metric's denominator",
            "covariance": "covariance between the baseline metric's numerator and denominator",
        }
        parameters = get_variable_parameters(parameters_definitions)
        mde = get_mde(metric_type)
        calculator.register_ratio_metric(
            mde,
            parameters["numerator_mean"],
            parameters["numerator_variance"],
            parameters["denominator_mean"],
            parameters["denominator_variance"],
            parameters["covariance"],
        )
    else:
        raise Exception("Error: Unexpected variable name. Please use Boolean, Numeric, or Ratio.")
