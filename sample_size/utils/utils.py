from typing import Dict

from sample_size.sample_size_calculator.sample_size_calculator import PowerAnalysisParameters
from sample_size.sample_size_calculator.sample_size_calculator import SampleSizeCalculators

DEFAULT_ALPHA = 0.05


def is_float(value: str) -> float:
    try:
        float(value)
        return True
    except ValueError:
        return False


def get_raw_input(text: str) -> str:
    return input(text)


def get_input(input_name: str, allow_na: bool = False) -> float:
    input_str = get_raw_input(f"Enter the {input_name}: ")
    if allow_na and input_str.strip() == "":
        return DEFAULT_ALPHA
    if is_float(input_str):
        return float(input_str)
    else:
        raise Exception(f"Error: Please enter a float for the {input_name}.")


def get_variable_parameters(parameter_definitions: Dict[str, str]) -> Dict[str, float]:

    parameters = {}

    for parameter_name, parameter_definition in parameter_definitions.items():
        parameters[parameter_name] = get_input(parameter_definition)

    return parameters


def get_sample_size(variable_name: str, calculator: SampleSizeCalculators) -> float:
    variable_name = variable_name.lower()
    if variable_name == "boolean":
        parameters_definitions = {"probability": "baseline probability"}
        parameters = get_variable_parameters(parameters_definitions)
        return calculator.get_boolean_sample_size(parameters["probability"])
    elif variable_name == "numeric":
        parameters_definitions = {"mean": "mean of the baseline metric", "variance": "variance of the baseline metric"}
        parameters = get_variable_parameters(parameters_definitions)
        return calculator.get_numeric_sample_size(parameters["mean"], parameters["variance"])
    elif variable_name == "ratio":
        parameters_definitions = {
            "numerator_mean": "mean of the baseline metric's numerator",
            "numerator_variance": "variance of the baseline metric's numerator",
            "denominator_mean": "mean of the baseline metric's denominator",
            "denominator_variance": "variance of the baseline metric's denominator",
            "covariance": "covariance between the baseline metric's numerator and denominator",
        }
        parameters = get_variable_parameters(parameters_definitions)
        return calculator.get_ratio_sample_size(
            parameters["numerator_mean"],
            parameters["numerator_variance"],
            parameters["denominator_mean"],
            parameters["denominator_variance"],
            parameters["covariance"],
        )
    else:
        raise Exception("Error: Unexpected variable name. Please use Boolean, Numeric, or Ratio.")


def get_power_analysis_input() -> PowerAnalysisParameters:

    parameters = PowerAnalysisParameters()
    alpha = get_alpha(get_input("alpha (default 0.05)", allow_na=True))
    if alpha:
        parameters.alpha = alpha
    parameters.mde = get_input("minimum detectable effect")

    return parameters


def get_alpha(alpha: float) -> float:
    if 0 <= alpha <= 0.3:
        return alpha
    else:
        raise Exception("Error: Please provide a float between 0 and 0.3 for alpha.")


def get_variable_from_input() -> str:
    metric_type = get_raw_input("Enter metric type (Boolean, Numeric, Ratio): ").strip().lower()
    if metric_type in ["boolean", "numeric", "ratio"]:
        return metric_type
    else:
        raise Exception("Error: Unexpected metric type. Please enter Boolean, Numeric, or Ratio.")
