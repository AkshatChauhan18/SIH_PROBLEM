"""
src/models/baseline.py
Baseline model using Logistic Regression on flat temporal-window aggregated statistics.
Uses strictly information available at prediction time to forecast future attack events.
"""

from pathlib import Path
from typing import List, Dict, Any, Union
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class BaselineLogisticRegression:
    """
    Flat statistical baseline:
    Features at time t:
    [total_flows, total_bytes, total_pkts, syn_ratio, rst_ratio, mean_duration, n_src, n_dst]
    Target:
    Future attack event y[t+1] (or y[t+k])
    """

    def __init__(self, random_state: int = 42):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fits baseline scaler and logistic regression on training sequences."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def save(self, filepath: Union[str, Path]):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "BaselineLogisticRegression":
        path = Path(filepath)
        with open(path, "rb") as f:
            return pickle.load(f)
