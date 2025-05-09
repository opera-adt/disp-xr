import json
import logging
import logging.config
import os
import sys
import time
from functools import wraps
from pathlib import Path
from typing import Callable, Optional, TypeVar

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

# Define the type variables for the decorator function
P = ParamSpec("P")
T = TypeVar("T")


def setup_logging(
    *, logger_name: str = "disp_xr", debug: bool = False, filename: Optional[str] = None
) -> None:
    """Set up logging configuration for the application.

    Parameters
    ----------
    logger_name : str, optional
        Name of the logger to configure (default is "disp_xr").
    debug : bool, optional
        If True, sets the logging level to DEBUG (default is False).
    filename : Optional[str], optional
        If provided, logs will be written to the specified file (default is None).

    Returns
    -------
    None
        Configures logging based on a JSON configuration file.

    """
    config_file = Path(__file__).parent / "log-config.json"

    with open(config_file) as f_in:
        config = json.load(f_in)

    if logger_name not in config["loggers"]:
        config["loggers"][logger_name] = {"level": "INFO", "handlers": ["stderr"]}

    if debug:
        config["loggers"][logger_name]["level"] = "DEBUG"

    # Ensure "stderr" is always present to print logs to the screen
    if "stderr" not in config["loggers"][logger_name]["handlers"]:
        config["loggers"][logger_name]["handlers"].append("stderr")

    if filename:
        if "file" not in config["loggers"][logger_name]["handlers"]:
            config["loggers"][logger_name]["handlers"].append("file")
        config["handlers"]["file"]["filename"] = os.fspath(filename)
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

    if "filename" not in config["handlers"]["file"]:
        config["handlers"].pop("file", None)

    logging.config.dictConfig(config)


def log_runtime(f: Callable[P, T]) -> Callable[P, T]:
    """Decorate a function to time how long it takes to run.

    Usage
    -----
    @log_runtime
    def test_func():
        return 2 + 4
    """
    logger = logging.getLogger(__name__)

    @wraps(f)
    def wrapper(*args: P.args, **kwargs: P.kwargs):
        t1 = time.time()

        result = f(*args, **kwargs)

        t2 = time.time()
        elapsed_seconds = t2 - t1
        elapsed_minutes = elapsed_seconds / 60.0

        time_string = (
            f"Total elapsed time for {f.__module__}.{f.__name__}: "
            f"{elapsed_minutes:.2f} minutes ({elapsed_seconds:.2f} seconds)"
        )

        logger.info(time_string)

        return result

    return wrapper
