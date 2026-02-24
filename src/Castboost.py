from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

# -------------------------------
# Default hyperparameters
# -------------------------------
DEFAULT_PARAMS = dict(
    depth=6,
    learning_rate=0.05,
    iterations=2000,
    loss_function="Logloss",
    eval_metric="AUC",
    l2_leaf_reg=3.0,
    random_seed=42,
    verbose=200,
    # For GPU, uncomment the following line if you installed CatBoost with GPU support
    task_type="GPU",
)


def build_catboost(params: Optional[dict] = None) -> CatBoostClassifier:
    """Return a CatBoostClassifier with sensible defaults (override via params)."""
    cfg = dict(DEFAULT_PARAMS)
    if params:
        cfg.update(params)
    return CatBoostClassifier(**cfg)


class CatBoostMatchModel:
    """Thin wrapper around CatBoostClassifier.

    Accepts numpy arrays or pandas DataFrames. You can pass categorical feature
    indices (ints) or names (if X is a DataFrame).
    """

    def __init__(
        self,
        params: Optional[dict] = None,
        cat_feature_indices: Optional[Sequence[Union[int, str]]] = None,
    ) -> None:
        self.params = dict(DEFAULT_PARAMS)
        if params:
            self.params.update(params)
        self.cat_features = list(cat_feature_indices) if cat_feature_indices is not None else None
        self.model: CatBoostClassifier = CatBoostClassifier(**self.params)

    @staticmethod
    def _to_pool(
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, Iterable]] = None,
        cat_features: Optional[Sequence[Union[int, str]]] = None,
    ) -> Pool:
        """Create a CatBoost Pool from X/y and cat_features.

        If X is a DataFrame and cat_features are strings, CatBoost will match by name.
        If X is a numpy array, cat_features must be integer indices.
        """
        return Pool(data=X, label=y, cat_features=cat_features)

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, Iterable],
        eval_set: Optional[Tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, Iterable]]] = None,
        early_stopping_rounds: Optional[int] = 300,
    ) -> None:
        train_pool = self._to_pool(X, y, self.cat_features)

        if eval_set is not None:
            X_val, y_val = eval_set
            valid_pool = self._to_pool(X_val, y_val, self.cat_features)
            self.model.fit(train_pool, eval_set=valid_pool, use_best_model=True, early_stopping_rounds=early_stopping_rounds)
        else:
            self.model.fit(train_pool)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        pool = self._to_pool(X, None, self.cat_features)
        # returns shape [N, 2]; we take positive class column
        proba = self.model.predict_proba(pool)
        if isinstance(proba, list):
            proba = np.array(proba)
        return np.asarray(proba)[:, 1]

    def predict(self, X: Union[np.ndarray, pd.DataFrame], threshold: float = 0.5) -> np.ndarray:
        p = self.predict_proba(X)
        return (p >= threshold).astype(int)

    def save(self, path: str) -> None:
        self.model.save_model(path)  # default format is 'cbm'

    @classmethod
    def load(cls, path: str) -> "CatBoostMatchModel":
        # Load into a fresh CatBoostClassifier and wrap
        model = build_catboost()
        model.load_model(path)
        wrapper = cls()
        wrapper.model = model
        return wrapper