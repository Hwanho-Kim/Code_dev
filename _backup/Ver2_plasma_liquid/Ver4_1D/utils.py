"""
Utility functions for Plasma-Liquid Interaction simulation.

This module provides logging, error handling, and common utility functions.
"""

import functools
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import numpy as np

# Type variable for generic decorator
F = TypeVar('F', bound=Callable[..., Any])


# =============================================================================
# Custom Exceptions
# =============================================================================

class PlasmaLiquidError(Exception):
    """Base exception for plasma-liquid simulation errors"""
    pass


class PreprocessingError(PlasmaLiquidError):
    """Error during data preprocessing"""
    pass


class ChemistryError(PlasmaLiquidError):
    """Error in chemistry calculations"""
    pass


class ODESolverError(PlasmaLiquidError):
    """Error in ODE solver"""
    pass


class FileLoadError(PlasmaLiquidError):
    """Error loading data file"""
    pass


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.DEBUG,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and/or console handlers.

    Parameters
    ----------
    name : str
        Logger name
    log_file : Path, optional
        Path to log file
    level : int
        Logging level (default: DEBUG)
    console_output : bool
        Whether to output to console

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler (INFO 이상만 출력, DEBUG 숨김)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Default application logger
_app_logger: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    """Get the application logger, creating it if necessary."""
    global _app_logger
    if _app_logger is None:
        _app_logger = setup_logger('plasma_liquid')
    return _app_logger


# =============================================================================
# Error Handling Decorators
# =============================================================================

def safe_calculation(
    default_return: Any = None,
    log_errors: bool = True
) -> Callable[[F], F]:
    """
    Decorator for safe numerical calculations.

    Catches numerical errors and returns a default value instead of crashing.

    Parameters
    ----------
    default_return : Any
        Value to return on error
    log_errors : bool
        Whether to log errors

    Returns
    -------
    Callable
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                # Check for invalid results
                if isinstance(result, (int, float)):
                    if np.isnan(result) or np.isinf(result):
                        if log_errors:
                            get_logger().warning(
                                f"{func.__name__} returned invalid value, using default"
                            )
                        return default_return
                return result
            except (OverflowError, FloatingPointError, ValueError, ZeroDivisionError) as e:
                if log_errors:
                    get_logger().warning(f"{func.__name__} calculation error: {e}")
                return default_return
            except Exception as e:
                if log_errors:
                    get_logger().error(f"{func.__name__} unexpected error: {e}")
                return default_return
        return wrapper  # type: ignore
    return decorator


def log_execution(func: F) -> F:
    """
    Decorator to log function entry and exit.

    Parameters
    ----------
    func : Callable
        Function to decorate

    Returns
    -------
    Callable
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger()
        logger.debug(f"Entering {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Exiting {func.__name__} successfully")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise
    return wrapper  # type: ignore


# =============================================================================
# Global Exception Handler
# =============================================================================

def setup_global_exception_handler(error_log_path: Optional[Path] = None):
    """
    Set up a global exception handler to prevent GUI crashes.

    Parameters
    ----------
    error_log_path : Path, optional
        Path to error log file
    """
    def handle_exception(exc_type, exc_value, exc_traceback):
        """Global exception handler"""
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))

        # Log to console
        print("=" * 80)
        print("UNHANDLED EXCEPTION:")
        print("=" * 80)
        print(error_msg)
        print("=" * 80)

        # Log to file
        if error_log_path:
            try:
                with open(error_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 80 + "\n")
                    f.write(error_msg)
                    f.write("=" * 80 + "\n\n")
            except Exception:
                pass

        # Try to show messagebox if GUI is available
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk._default_root
            if root and root.winfo_exists():
                messagebox.showerror(
                    "Unexpected Error",
                    f"An unexpected error occurred:\n\n{str(exc_value)}\n\n"
                    f"Check error log for details."
                )
        except Exception:
            pass

    sys.excepthook = handle_exception


# =============================================================================
# Numerical Utilities
# =============================================================================

def safe_array(
    arr: np.ndarray,
    min_val: float = 0.0,
    max_val: float = 1.0,
    replace_nan: float = 0.0,
    replace_inf: float = 0.0
) -> np.ndarray:
    """
    Ensure array contains only valid, bounded values.

    Parameters
    ----------
    arr : np.ndarray
        Input array
    min_val : float
        Minimum allowed value
    max_val : float
        Maximum allowed value
    replace_nan : float
        Value to replace NaN with
    replace_inf : float
        Value to replace Inf with

    Returns
    -------
    np.ndarray
        Sanitized array
    """
    arr = np.array(arr, dtype=np.float64, copy=True)
    arr = np.nan_to_num(arr, nan=replace_nan, posinf=replace_inf, neginf=replace_inf)
    arr = np.clip(arr, min_val, max_val)
    return arr


def is_valid_array(arr: np.ndarray) -> bool:
    """
    Check if array contains only finite values.

    Parameters
    ----------
    arr : np.ndarray
        Array to check

    Returns
    -------
    bool
        True if array is valid
    """
    return np.all(np.isfinite(arr))


# =============================================================================
# File Utilities
# =============================================================================

def ensure_directory(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Parameters
    ----------
    path : Path
        Directory path

    Returns
    -------
    Path
        The directory path
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_timestamp() -> str:
    """
    Generate a timestamp string for file naming.

    Returns
    -------
    str
        Timestamp in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')
