"""
Data I/O utilities for loading CSV files using NumPy.
"""

from __future__ import annotations
from typing import Generator, Optional, Union
import numpy as np


def load_csv(
    filepath: str,
    delimiter: str = ",",
    missing_values: Optional[Union[str, float]] = None,
    fill_value: Optional[float] = None,
    skip_header: int = 0,
    dtype: type = np.float64,
) -> np.ndarray:
    """
    Load numeric CSV data into a NumPy array with missing value handling.
    """
    # TODO: Implement CSV reader using np.loadtxt or np.genfromtxt
    # TODO: Handle missing values (skip rows or fill with fill_value)
    raise NotImplementedError


def load_csv_chunks(
    filepath: str,
    chunk_size: int = 1000,
    delimiter: str = ",",
    skip_header: int = 0,
    dtype: type = np.float64,
) -> Generator[np.ndarray, None, None]:
    """
    Stream/chunk-based CSV reader for large datasets.
    """
    # TODO: Yield chunks of CSV data as NumPy arrays
    raise NotImplementedError