from typeguard import typechecked


@typechecked
def print_info(text: str):
    """Prints info message.

    Args:
        text (str): Info message.
    """
    print(f"\033[0;32m[INFO]: {text}\033[0m")


@typechecked
def print_warn(text: str):
    """Prints warning message.

    Args:
        text (str): Warning message.
    """
    print(f"\033[0;33m[WARN]: {text}\033[0m")


@typechecked
def exit_error(text: str):
    """Prints error message and exits application with error code 1.

    Args:
        text (str): Error message.
    """
    print(f"\033[0;31m[ERROR]: {text}\033[0m")
    exit(1)
