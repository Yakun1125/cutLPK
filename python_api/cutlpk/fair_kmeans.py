"""Fair K-means clustering via cutLPK.

Supports both *alpha*-fair (proportional representation) and *tau*-fair
(minimum group fraction) constraints.  Provides a scikit-learn-style estimator
``FairKMeans`` and a functional helper ``solve_fair_kmeans``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from . import _cutlpk


def _resolve_output_dir(dir_: Optional[str]) -> Path:
    if dir_ is None:
        return Path.cwd()
    p = Path(dir_)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _default_output_prefix(n_clusters: int, kind: str = "fair") -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"cutlpk_{kind}_K{n_clusters}_{ts}"


@dataclass
class FairKMeans:
    """Global LP-based solver for fair K-means clustering.

    Parameters
    ----------
    n_clusters : int
        Number of clusters *K*.
    groups : array-like
        Group membership.  Accepts two forms:

        - **1D array of labels**: each entry is the group index for that point.
        - **2D boolean array** of shape (n, G).

    fairness_type : str
        ``"tau"`` (default) or ``"alpha"``.
    fairness_param : float
        Fairness parameter in (0, 1].
    """
    n_clusters: int
    groups: Any
    fairness_type: str = "tau"
    fairness_param: float = 0.99
    solver: str = "cupdlpx"
    solver_warm_start: bool = True
    max_cuts_init: int = 15_000_000
    max_cuts_per_iter: int = 30_000_000
    max_cuts_added_iter: int = 10_000_000
    max_separation_size: int = 15_000_000
    max_active_cuts_size: int = 30_000_000
    max_iter: int = 3000
    num_iter_no_improve: int = 2
    exact_separation: bool = True
    remove_inactive_cuts: bool = True
    warm_start: int = 1
    first_lp_time_limit: float = 360.0
    lp_time_limit: float = 180.0
    time_limit: float = 7200.0
    max_separation_time: float = 300.0
    first_lp_solver_tol: float = 1e-6
    solver_tol: float = 1e-6
    lb_solver_tol: float = 1e-6
    cuts_vio_tol: float = 1e-4
    cuts_act_tol: float = 1e-4
    opt_gap: float = 1e-4
    t_upper_bound: Optional[int] = None
    random_seed: int = 42
    lloyd_random_starts: int = 100
    heuristic_only: bool = False
    bnb_node_limit: int = 0
    bnb_time_limit: float = 3600.0
    bnb_gap_tol: float = 1e-4
    bnb_global_ub: float = float("inf")
    bnb_cut_iter_limit: int = 30
    verbose: int = 1
    output_level: int = 3
    save_output: bool = True
    output_dir: Optional[str] = None
    output_prefix: Optional[str] = None
    cost_: Optional[float] = field(init=False, default=None)
    relative_gap_: Optional[float] = field(init=False, default=None)
    lower_bound_: Optional[float] = field(init=False, default=None)
    upper_bound_: Optional[float] = field(init=False, default=None)
    warm_start_cost_: Optional[float] = field(init=False, default=None)
    solver_time_: Optional[float] = field(init=False, default=None)
    separation_time_: Optional[float] = field(init=False, default=None)
    heuristic_time_: Optional[float] = field(init=False, default=None)
    status_: Optional[str] = field(init=False, default=None)
    retcode_: Optional[int] = field(init=False, default=None)
    bnb_executed_: Optional[bool] = field(init=False, default=None)
    bnb_status_: Optional[str] = field(init=False, default=None)
    labels_: Optional[np.ndarray] = field(init=False, default=None)
    _STATUS_MAP = {0: "optimal", 1: "no_violated_cuts", 2: "no_improvement", 3: "time_or_limit", 4: "error", 5: "max_iter", 6: "infeasible", 7: "heuristic_only"}
    _BNB_STATUS_MAP = {0: "optimal", 1: "node_limit", 2: "time_limit", 3: "all_nodes_explored", 4: "error"}

    def _normalize_groups(self) -> np.ndarray:
        g = np.asarray(self.groups)
        if g.ndim == 1:
            n = len(g)
            unique = np.unique(g)
            G = len(unique)
            bool_mat = np.zeros((n, G), dtype=np.float64)
            for idx, label in enumerate(unique):
                bool_mat[g == label, idx] = 1.0
            return bool_mat
        elif g.ndim == 2:
            return g.astype(np.float64)
        else:
            raise ValueError("groups must be 1D (labels) or 2D (boolean matrix)")

    def _collect_params(self) -> Dict[str, Any]:
        t_ub = self.t_upper_bound if self.t_upper_bound is not None else self.n_clusters
        if self.fairness_type not in ("tau", "alpha"):
            raise ValueError("fairness_type must be 'tau' or 'alpha'")
        if not (0.0 < self.fairness_param <= 1.0):
            raise ValueError("fairness_param must be in (0, 1]")
        return {
            "fairness_type": self.fairness_type, "fairness_param": self.fairness_param,
            "solver": self.solver, "solver_warm_start": self.solver_warm_start,
            "max_cuts_init": self.max_cuts_init, "max_cuts_per_iter": self.max_cuts_per_iter,
            "max_cuts_added_iter": self.max_cuts_added_iter, "max_separation_size": self.max_separation_size,
            "max_active_cuts_size": self.max_active_cuts_size,
            "max_iter": self.max_iter, "num_iter_no_improve": self.num_iter_no_improve,
            "exact_separation": self.exact_separation, "remove_inactive_cuts": self.remove_inactive_cuts,
            "warm_start": self.warm_start,
            "first_lp_time_limit": self.first_lp_time_limit, "lp_time_limit": self.lp_time_limit,
            "time_limit": self.time_limit, "max_separation_time": self.max_separation_time,
            "first_lp_solver_tol": self.first_lp_solver_tol, "solver_tol": self.solver_tol,
            "lb_solver_tol": self.lb_solver_tol, "cuts_vio_tol": self.cuts_vio_tol,
            "cuts_act_tol": self.cuts_act_tol, "opt_gap": self.opt_gap,
            "t_upper_bound": t_ub, "random_seed": self.random_seed,
            "lloyd_random_starts": self.lloyd_random_starts, "heuristic_only": self.heuristic_only,
            "bnb_node_limit": self.bnb_node_limit, "bnb_time_limit": self.bnb_time_limit,
            "bnb_gap_tol": self.bnb_gap_tol, "bnb_global_ub": self.bnb_global_ub,
            "bnb_cut_iter_limit": self.bnb_cut_iter_limit,
            "verbose": self.verbose, "output_level": self.output_level,
        }

    def _build_output_paths(self) -> Dict[str, str]:
        out_dir = _resolve_output_dir(self.output_dir)
        kind = f"fair_{self.fairness_type}"
        prefix = self.output_prefix or _default_output_prefix(self.n_clusters, kind)
        return {
            "output_file": str(out_dir / f"{prefix}_output.txt") if self.save_output else "",
            "bnb_output_file": str(out_dir / f"{prefix}_bnb_log.txt") if self.save_output else "",
        }

    def fit(self, X: Any, **override_params: Any) -> "FairKMeans":
        data = np.asarray(X, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("Input data must be a 2D array-like structure")
        groups_mat = self._normalize_groups()
        if groups_mat.shape[0] != data.shape[0]:
            raise ValueError(f"groups length ({groups_mat.shape[0]}) must match number of data points ({data.shape[0]})")
        params = self._collect_params()
        params.update(self._build_output_paths())
        params.update(override_params)
        result = _cutlpk.run_fair_kmeans(data, int(self.n_clusters), groups_mat, params)
        self.cost_ = float(result["cost"])
        self.relative_gap_ = float(result["relative_gap"])
        self.lower_bound_ = float(result["lower_bound"])
        self.upper_bound_ = float(result["upper_bound"])
        self.warm_start_cost_ = float(result["warm_start_cost"])
        self.solver_time_ = float(result.get("solver_time", 0.0))
        self.separation_time_ = float(result.get("separation_time", 0.0))
        self.heuristic_time_ = float(result.get("heuristic_time", 0.0))
        self.status_ = self._STATUS_MAP.get(int(result["status"]), "unknown")
        self.retcode_ = int(result["retcode"])
        self.bnb_executed_ = bool(result["bnb_executed"])
        self.bnb_status_ = self._BNB_STATUS_MAP.get(int(result["bnb_status"])) if self.bnb_executed_ else None
        labels = result["labels"]
        self.labels_ = np.asarray(labels, dtype=np.int32) if labels is not None else None
        return self

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in [
            "n_clusters", "groups", "fairness_type", "fairness_param",
            "solver", "solver_warm_start", "max_cuts_init", "max_cuts_per_iter",
            "max_cuts_added_iter", "max_separation_size", "max_active_cuts_size",
            "max_iter", "num_iter_no_improve", "exact_separation", "remove_inactive_cuts",
            "warm_start", "first_lp_time_limit", "lp_time_limit", "time_limit",
            "max_separation_time", "first_lp_solver_tol", "solver_tol", "lb_solver_tol",
            "cuts_vio_tol", "cuts_act_tol", "opt_gap", "t_upper_bound",
            "random_seed", "lloyd_random_starts", "heuristic_only",
            "bnb_node_limit", "bnb_time_limit", "bnb_gap_tol", "bnb_global_ub", "bnb_cut_iter_limit",
            "verbose", "output_level", "save_output", "output_dir", "output_prefix",
        ]}

    def set_params(self, **params: Any) -> "FairKMeans":
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key!r}")
        return self

    def summary(self) -> str:
        if self.cost_ is None:
            return "FairKMeans (not fitted)"
        return f"FairKMeans(K={self.n_clusters}, type={self.fairness_type}, rho={self.fairness_param}, cost={self.cost_:.6f}, gap={self.relative_gap_:.2e}, status={self.status_})"

    def __repr__(self) -> str:
        return self.summary()


def solve_fair_kmeans(X: Any, n_clusters: int, groups: Any, **params: Any) -> Dict[str, Any]:
    data = np.asarray(X, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array-like structure")
    groups_mat = np.asarray(groups)
    if groups_mat.ndim == 1:
        n = len(groups_mat)
        unique = np.unique(groups_mat)
        G = len(unique)
        gm = np.zeros((n, G), dtype=np.float64)
        for idx, label in enumerate(unique):
            gm[groups_mat == label, idx] = 1.0
        groups_mat = gm
    return _cutlpk.run_fair_kmeans(data, int(n_clusters), groups_mat, params)
