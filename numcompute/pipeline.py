"""
Pipeline abstractions and protocol interfaces for NumCompute.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np


class BaseTransformer:
    """
    Base class / protocol for all data transformers in NumCompute.
    Subclasses must implement `fit` and `transform`.
    """

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> BaseTransformer:
        """
        Fit transformer parameters from training data X.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (N, D).
        y : Optional[np.ndarray], default=None
            Target array of shape (N,) or (N, K).

        Returns
        -------
        BaseTransformer
            The fitted transformer instance.
        """
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform input data X using fitted parameters.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (N, D).

        Returns
        -------
        np.ndarray
            Transformed array.
        """
        raise NotImplementedError

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (N, D).
        y : Optional[np.ndarray], default=None
            Target array of shape (N,) or (N, K).

        Returns
        -------
        np.ndarray
            Transformed array.
        """
        return self.fit(X, y).transform(X)


class BaseEstimator:
    """
    Base class / protocol for all predictive models in NumCompute.
    Subclasses must implement `fit` and `predict`.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseEstimator:
        """
        Fit model parameters from training features X and targets y.

        Parameters
        ----------
        X : np.ndarray
            Training features of shape (N, D).
        y : np.ndarray
            Target labels or values of shape (N,) or (N, K).

        Returns
        -------
        BaseEstimator
            The fitted model instance.
        """
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict outcomes for samples in X.

        Parameters
        ----------
        X : np.ndarray
            Input features of shape (N, D).

        Returns
        -------
        np.ndarray
            Predicted values or class labels of shape (N,).
        """
        raise NotImplementedError


class Pipeline:
    """
    Sequentially chain multiple transformers with an optional final estimator.

    Intermediate steps must implement `fit` and `transform`.
    The final step can be a transformer (with `transform`) or an estimator (with `predict`).

    Parameters
    ----------
    steps : List[Tuple[str, Any]]
        List of (name, transform_or_estimator_step) tuples chaining the pipeline.
    """

    def __init__(self, steps: List[Tuple[str, Any]]) -> None:
        if not steps:
            raise ValueError("Pipeline steps cannot be empty.")

        for name, step in steps[:-1]:
            if not hasattr(step, "transform") or not hasattr(step, "fit"):
                raise TypeError(f"Intermediate step '{name}' must implement both 'fit' and 'transform'.")

        self.steps = steps
        self._is_fitted = False

    @property
    def named_steps(self) -> Dict[str, Any]:
        """Access pipeline steps by name as a dictionary."""
        return dict(self.steps)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Pipeline:
        """
        Sequentially fit all transformers and the final step.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (N, D).
        y : Optional[np.ndarray], default=None
            Target array of shape (N,) or (N, K).

        Returns
        -------
        Pipeline
            Fitted pipeline instance.

        Complexity
        ----------
        Time: O(Sum of time complexities across all steps).
        Space: O(Max memory required by intermediate representations).
        """
        X_curr = X

        for _, step in self.steps[:-1]:
            X_curr = step.fit_transform(X_curr, y)

        _, last_step = self.steps[-1]
        if hasattr(last_step, "fit"):
            last_step.fit(X_curr, y)

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Pass data through all transformers sequentially.

        Parameters
        ----------
        X : np.ndarray
            Data to transform of shape (N, D).

        Returns
        -------
        np.ndarray
            Transformed data from the final transformer step.

        Raises
        ------
        RuntimeError
            If called before fit.
        AttributeError
            If the final step does not implement `transform`.
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline is not fitted yet.")

        X_curr = X
        for name, step in self.steps:
            if not hasattr(step, "transform"):
                raise AttributeError(f"Final step '{name}' does not implement 'transform'. Use predict() instead.")
            X_curr = step.transform(X_curr)

        return X_curr

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit all steps and transform data through all pipeline transformers.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (N, D).
        y : Optional[np.ndarray], default=None
            Target array of shape (N,) or (N, K).

        Returns
        -------
        np.ndarray
            Transformed array.
        """
        X_curr = X
        for _, step in self.steps[:-1]:
            X_curr = step.fit_transform(X_curr, y)

        name, last_step = self.steps[-1]
        if not hasattr(last_step, "transform"):
            raise AttributeError(f"Final step '{name}' does not implement 'transform'. Cannot call fit_transform.")

        res = last_step.fit_transform(X_curr, y)
        self._is_fitted = True

        return res

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Transform X through all preprocessors and generate predictions using final estimator.

        Parameters
        ----------
        X : np.ndarray
            Data of shape (N, D) to predict.

        Returns
        -------
        np.ndarray
            Predictions from the final model step.

        Raises
        ------
        RuntimeError
            If called before fit.
        AttributeError
            If the final step does not implement `predict`.
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline is not fitted yet.")

        X_curr = X
        for _, step in self.steps[:-1]:
            X_curr = step.transform(X_curr)

        name, last_step = self.steps[-1]
        if not hasattr(last_step, "predict"):
            raise AttributeError(f"Final step '{name}' does not implement 'predict'.")

        return last_step.predict(X_curr)


class FeatureUnion(BaseTransformer):
    """
    Concatenate results of multiple transformer objects in parallel.

    Parameters
    ----------
    transformer_list : List[Tuple[str, BaseTransformer]]
        List of (name, transformer) tuples applied in parallel to the input data.
    """

    def __init__(self, transformer_list: List[Tuple[str, Any]]) -> None:
        if not transformer_list:
            raise ValueError("transformer_list cannot be empty.")
        self.transformer_list = transformer_list

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> FeatureUnion:
        """Fit all transformers in parallel on X."""
        for _, trans in self.transformer_list:
            trans.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform X with each transformer and concatenate outputs horizontally.

        Returns
        -------
        np.ndarray
            Horizontally stacked feature matrix of shape (N, sum(D_i)).
        """
        transformed = [trans.transform(X) for _, trans in self.transformer_list]
        return np.hstack(transformed)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)