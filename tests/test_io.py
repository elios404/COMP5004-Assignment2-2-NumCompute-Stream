"""
Unit tests for data I/O utilities (numcompute/io.py).
"""

import io
import numpy as np
import pytest

from numcompute.io import load_csv, load_csv_chunks


def test_load_csv_standard(tmp_path):
    """Test loading a clean standard CSV file."""
    file = tmp_path / "data.csv"
    file.write_text("1.0,2.0,3.0\n4.0,5.0,6.0\n7.0,8.0,9.0\n", encoding="utf-8")

    arr = load_csv(str(file))

    assert arr.shape == (3, 3)
    assert np.allclose(arr, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])


def test_load_csv_with_header_and_delimiter(tmp_path):
    """Test skipping header lines and using a custom delimiter."""
    file = tmp_path / "custom.tsv"
    file.write_text("colA\tcolB\n10.5\t20.5\n30.5\t40.5\n", encoding="utf-8")

    arr = load_csv(str(file), delimiter="\t", skip_header=1)

    assert arr.shape == (2, 2)
    assert np.allclose(arr, [[10.5, 20.5], [30.5, 40.5]])


def test_load_csv_missing_values(tmp_path):
    """Edge Case: Test handling missing values (NaNs, '?', 'NA') with default and custom fill."""
    file = tmp_path / "missing.csv"
    # CSV Content with missing values represented as '', '?', and 'NA'
    file.write_text("1.0,2.0,3.0\n4.0,,6.0\n?,8.0,NA\n", encoding="utf-8")

    arr_nan = load_csv(str(file))
    assert arr_nan.shape == (3, 3)
    assert np.isnan(arr_nan[1, 1])
    assert np.isnan(arr_nan[2, 0])
    assert np.isnan(arr_nan[2, 2])
    assert arr_nan[0, 0] == 1.0

    # Test custom fill_value for missing entries
    arr_filled = load_csv(str(file), fill_value=-999.0)
    assert arr_filled[1, 1] == -999.0
    assert arr_filled[2, 0] == -999.0


def test_load_csv_single_row(tmp_path):
    """Edge Case: Ensure 1D row array is converted to 2D (1, D) matrix."""
    file = tmp_path / "single_row.csv"
    file.write_text("10.0,20.0,30.0\n", encoding="utf-8")

    arr = load_csv(str(file))
    assert arr.ndim == 2
    assert arr.shape == (1, 3)


def test_load_csv_empty_file(tmp_path):
    """Edge Case: Ensure empty file returns an empty (0, 0) array without crashing."""
    file = tmp_path / "empty.csv"
    file.write_text("", encoding="utf-8")

    arr = load_csv(str(file))
    assert arr.shape == (0, 0)
    assert arr.size == 0


def test_load_csv_from_stringio():
    """Test loading directly from an in-memory io.StringIO buffer."""
    buffer = io.StringIO("1.0,2.0\n3.0,4.0\n")
    arr = load_csv(buffer)

    assert arr.shape == (2, 2)
    assert np.allclose(arr, [[1.0, 2.0], [3.0, 4.0]])


def test_load_csv_file_not_found():
    """Test error handling when the file does not exist."""
    with pytest.raises(FileNotFoundError):
        load_csv("non_existent_file_path_12345.csv")


def test_load_csv_chunks(tmp_path):
    """Test generator streaming of large CSV files chunk by chunk."""
    file = tmp_path / "large.csv"
    # Create a CSV with 5 rows of data
    lines = ["id,val\n"] + [f"{i},{i*10}\n" for i in range(1, 6)]
    file.write_text("".join(lines), encoding="utf-8")

    chunks = list(load_csv_chunks(str(file), chunk_size=2, skip_header=1))

    assert len(chunks) == 3
    assert chunks[0].shape == (2, 2)
    assert chunks[1].shape == (2, 2)
    assert chunks[2].shape == (1, 2)

    merged = np.vstack(chunks)
    assert merged.shape == (5, 2)
    assert np.allclose(merged[:, 0], [1, 2, 3, 4, 5])


def test_load_csv_chunks_invalid_size():
    """Test exception when chunk_size <= 0."""
    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        next(load_csv_chunks("dummy.csv", chunk_size=0))