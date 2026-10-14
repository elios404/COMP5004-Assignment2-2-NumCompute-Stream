"""
Data I/O utilities for loading CSV files using NumPy.
"""

from __future__ import annotations
import io
from itertools import islice
from typing import Generator, Optional, Union
import numpy as np


def load_csv(
    filepath: Union[str, io.IOBase],
    delimiter: str = ",",
    missing_values: Optional[Union[str, float]] = None,
    fill_value: Optional[float] = np.nan,
    skip_header: int = 0,
    dtype: type = np.float64,
) -> np.ndarray:
    """
    Load numeric CSV data into a 2D NumPy array with missing value handling.

    Parameters
    ----------
    filepath : Union[str, io.IOBase]
        Path to the CSV file or a file-like buffer (e.g., io.StringIO).
    delimiter : str, default=','
        Field delimiter character.
    missing_values : Optional[Union[str, Sequence[str]]], default=None
        Strings recognized as missing values. If None, defaults to common
        representations: '', '?', 'NA', 'nan', 'NaN', 'null'.
    fill_value : Optional[float], default=np.nan
        Value used to replace missing entries.
    skip_header : int, default=0
        Number of leading lines to skip (e.g., column names).
    dtype : type, default=np.float64
        Target data type for the returned array.

    Returns
    -------
    np.ndarray
        2D array of shape (N, D) containing the loaded numeric data.

    Raises
    ------
    FileNotFoundError
        If filepath is a string and the file does not exist.
    ValueError
        If the data cannot be parsed into the requested dtype.

    Complexity
    ----------
    Time: O(N * D) where N is number of rows and D is number of columns.
    Space: O(N * D) for storing the loaded array in memory.
    """
    if missing_values is None:
        missing_values = ("", "?", "NA", "nan", "NaN", "null")

    data = np.genfromtxt(
        filepath,
        delimiter=delimiter,
        skip_header=skip_header,
        dtype=dtype,
        missing_values=missing_values,
        filling_values=fill_value,
    )

    if data.size == 0: # empty file or no valid data
        return np.empty((0, 0), dtype=dtype)

    # in case of a single row, ensure the output is 2D
    return np.atleast_2d(data) 


def load_csv_chunks(
    filepath: str,
    chunk_size: int = 100,
    delimiter: str = ",",
    skip_header: int = 0,
    dtype: type = np.float64,
) -> Generator[np.ndarray, None, None]:
    """
    Stream/chunk-based CSV reader for large datasets.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    chunk_size : int, default=1000
        Number of rows to yield in each chunk.
    delimiter : str, default=','
        Field delimiter character.
    skip_header : int, default=0
        Number of leading lines to skip before chunking.
    dtype : type, default=np.float64
        Target data type for the yielded arrays.

    Yields
    ------
    np.ndarray
        2D array of shape (chunk_size, D) or (remaining_rows, D).

    Raises
    ------
    ValueError
        If chunk_size is less than or equal to 0.
    FileNotFoundError
        If the file does not exist.

    Complexity
    ----------
    Time: O(N * D) total over the entire stream.
    Space: O(chunk_size * D) peak memory at any given time.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    with open(filepath, "r", encoding="utf-8") as f:
        for _ in range(skip_header):
            if f.readline() == "":
                return  # reached EOF before skipping all header lines

        while True:
            lines = list(islice(f, chunk_size))
            if not lines:
                break

            chunk_buffer = io.StringIO("".join(lines))
            chunk_array = load_csv(
                chunk_buffer,
                delimiter=delimiter,
                dtype=dtype,
            )

            if chunk_array.size > 0:
                yield chunk_array
