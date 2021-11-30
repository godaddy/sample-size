#!/usr/bin/env python3
import subprocess
import sys
from typing import List


def execute(name: str, command: List[str], failure_message: str) -> None:
    """
    Execute a command with output sent to the console. If exit code is non-zero will exit with non-zero and
    display the provided failure message.
    """
    print("--------------------------------------------------------------------------------")
    print(f"Running task: {name} ({command})")
    print("--------------------------------------------------------------------------------")
    completed_process = subprocess.run(command)
    if completed_process.returncode != 0:
        print("FAILURE")
        print(failure_message)
        sys.exit(1)


# The following section defines individual methods to run specific sub-tasks, used by poetry
# hooks given in the pyproject.toml file. Add new tasks to qa() below as well.
def test() -> None:
    execute("tests", ["pytest", "tests", "--cov-fail-under", "95"], "Please fix the failing test.")


def format_check() -> None:
    execute("format", ["isort", "--check-only", "."], 'Import ordering incorrect. Run "poetry run isort ." to fix.')
    execute("format", ["black", "--check", "."], 'Running "poetry run black ." will fix most issues.')


def format_fix() -> None:
    execute("format", ["isort", "."], "isort formatting failed. Address issues manually.")
    execute("format", ["black", "."], "black formatting failed. Address issues manually.")


def lint() -> None:
    execute("lint", ["flake8"], "Address any remaining flake8 issues manually.")


def type_check() -> None:
    execute(
        "type_check", ["mypy", "."], 'Mypy check failed. See above for errors, run "poetry run mypy ." to confirm fix.'
    )


# This routine runs all the defined tasks in order
def qa() -> None:
    format_fix()
    type_check()
    lint()
    test()


if __name__ == "__main__":
    qa()
