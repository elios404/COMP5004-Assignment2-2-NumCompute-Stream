"""
Ranking with tie-breaking and percentile calculations.
"""

from __future__ import annotations
from typing import Optional, Sequence, Union
import numpy as np


def rank(
    data: np.ndarray,
    method: str = "average",
    axis: int = -1,
) -> np.ndarray:
    """
    Compute ranks of data along an axis.
    Supported methods: 'average', 'dense', 'ordinal'.
    """
    # TODO: Vectorized rank computation with tie-handling
    raise NotImplementedError


def percentile(
    data: np.ndarray,
    q: Union[float, Sequence[float]],
    interpolation: str = "linear",
    axis: Optional[int] = None,
) -> np.ndarray:
    """
    Compute the q-th percentile of data along an axis.
    Supported interpolation: 'linear', 'lower', 'higher', 'midpoint'.
    """
    # TODO: Compute percentiles without external libraries
    raise NotImplementedError