"""
Mathematical and data utility helpers.
"""

from __future__ import annotations
from typing import Generator, Optional, Tuple
import numpy as np


def euclidean_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distance between rows of X and Y."""
    # TODO: Fully vectorized pairwise distance
    raise NotImplementedError


def cosine_similarity(X: np.ndarray, Y: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Compute pairwise Cosine similarity between rows of X and Y."""
    # TODO: Fully vectorized cosine similarity
    raise NotImplementedError


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute numerically stable softmax using max-subtraction trick."""
    # TODO: Subtract max along axis before exponentiation
    raise NotImplementedError


def logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute numerically stable log(sum(exp(x))) along an axis."""
    # TODO: Numerically stable logsumexp
    raise NotImplementedError


def batch_generator(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    batch_size: int = 32,
    shuffle: bool = True,
) -> Generator[Tuple[np.ndarray, Optional[np.ndarray]], None, None]:
    """Yield mini-batches of data."""
    # TODO: Batch slicing with optional shuffle
    raise NotImplementedError