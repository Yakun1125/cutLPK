"""High-level interface mirroring scikit-learn's estimator pattern."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from . import _cutlpk


@dataclass
class OrdinaryKMeans:
    """Minimal estimator-style wrapper for the cutLPK ordinary k-means solver.

    Parameters mirror the command-line options but keep sensible defaults. Additional
    keyword arguments accepted by :func:`_cutlpk.run_ordinary_kmeans` can be supplied
    at construction or during :meth:`fit`.
    """

    n_clusters: int
    warm_start: bool = True
    random_state: int = 42
    lloyd_random_starts: int = 100
    solver: str = "cupdlpx"
    bnb_node_limit: int = 0
    extra_params: Dict[str, Any] = field(default_factory=dict)

    # Learned attributes (populated after ``fit``)
    cost_: Optional[float] = field(init=False, default=None)
    relative_gap_: Optional[float] = field(init=False, default=None)
    lower_bound_: Optional[float] = field(init=False, default=None)
    upper_bound_: Optional[float] = field(init=False, default=None)
    warm_start_cost_: Optional[float] = field(init=False, default=None)
    status_: Optional[str] = field(init=False, default=None)
    retcode_: Optional[int] = field(init=False, default=None)
    bnb_executed_: Optional[bool] = field(init=False, default=None)
    bnb_status_: Optional[str] = field(init=False, default=None)
    labels_: Optional[np.ndarray] = field(init=False, default=None)

    def _collect_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "warm_start": self.warm_start,
            "random_seed": self.random_state,
            "lloyd_random_starts": self.lloyd_random_starts,
            "solver": self.solver,
            "bnb_node_limit": self.bnb_node_limit,
        }
        params.update(self.extra_params)
        return params

    def fit(self, X: Any, **override_params: Any) -> "OrdinaryKMeans":
        """Run the solver on ``X`` and store the resulting metrics."""

        data = np.asarray(X, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("Input data must be a 2D array-like structure")

        params = self._collect_params()
        params.update(override_params)

        result = _cutlpk.run_ordinary_kmeans(data, int(self.n_clusters), **params)

        self.cost_ = float(result["cost"])
        self.relative_gap_ = float(result["relative_gap"])
        self.lower_bound_ = float(result["lower_bound"])
        self.upper_bound_ = float(result["upper_bound"])
        self.warm_start_cost_ = float(result["warm_start_cost"])
        self.status_ = str(result["status"])
        self.retcode_ = int(result["retcode"])
        self.bnb_executed_ = bool(result["bnb_executed"])
        self.bnb_status_ = result["bnb_status"] if self.bnb_executed_ else None
        if result["labels"] is not None:
            self.labels_ = np.asarray(result["labels"], dtype=np.int32)
        else:
            self.labels_ = None

        return self

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        params = {
            "n_clusters": self.n_clusters,
            "warm_start": self.warm_start,
            "random_state": self.random_state,
            "lloyd_random_starts": self.lloyd_random_starts,
            "solver": self.solver,
            "bnb_node_limit": self.bnb_node_limit,
        }
        params.update(self.extra_params)
        return params

    def set_params(self, **params: Any) -> "OrdinaryKMeans":
        for key, value in params.items():
            if key == "n_clusters":
                self.n_clusters = int(value)
            elif key == "warm_start":
                self.warm_start = bool(value)
            elif key == "random_state":
                self.random_state = int(value)
            elif key == "lloyd_random_starts":
                self.lloyd_random_starts = int(value)
            elif key == "solver":
                self.solver = str(value)
            elif key == "bnb_node_limit":
                self.bnb_node_limit = int(value)
            else:
                self.extra_params[key] = value
        return self


def solve_kmeans(X: Any, n_clusters: int, **params: Any) -> Dict[str, Any]:
    """Functional-style helper returning the solver dictionary result."""

    data = np.asarray(X, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array-like structure")

    return _cutlpk.run_ordinary_kmeans(data, int(n_clusters), **params)
