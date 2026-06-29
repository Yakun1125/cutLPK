"""Ordinary (unfair) K-means clustering via cutLPK.

Provides a scikit-learn-style estimator ``OrdinaryKMeans`` and a functional
helper ``solve_kmeans``.  All solver parameters are exposed as explicit fields
with sensible defaults matching the C++ reference implementation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from . import _cutlpk


# ---------------------------------------------------------------------------
# Default output directory helper
# ---------------------------------------------------------------------------
def _resolve_output_dir(dir_: Optional[str]) -> Path:
    if dir_ is None:
        return Path.cwd()
    p = Path(dir_)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _default_output_prefix(n_clusters: int, kind: str = "ordinary") -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"cutlpk_{kind}_K{n_clusters}_{ts}"


# ===================================================================
# Estimator
# ===================================================================

@dataclass
class OrdinaryKMeans:
    """Global LP-based solver for ordinary K-means clustering.

    Parameters
    ----------
    n_clusters : int
        Number of clusters *K*.

    Solver
    ------
    solver : str
        LP solver backend. Currently ``"cupdlpx"`` (GPU, default) and
        ``"gurobi"`` are supported.
    solver_warm_start : bool
        Use primal/dual warm-start across iterations (recommended).

    Cutting-plane – cut management
    ------------------------------
    max_cuts_init : int
        Maximum number of inequalities in the initial LP (default 15 000 000).
    max_cuts_per_iter : int
        Maximum total cuts allowed in the LP at any iteration (default 30 000 000).
    max_cuts_added_iter : int
        Maximum cuts added in a *single* iteration (default 10 000 000).
    max_separation_size : int
        Maximum number of cuts examined during separation (default 15 000 000).
    max_active_cuts_size : int
        Maximum number of active (tight) cuts retained (default 30 000 000).

    Cutting-plane – algorithm control
    ---------------------------------
    max_iter : int
        Maximum cutting-plane iterations (default 3000).
    num_iter_no_improve : int
        Stop if the gap does not improve for this many consecutive iterations
        after *t* has reached its upper bound (default 2).
    exact_separation : bool
        If True, use exact (greedy) separation; otherwise heuristic (default True).
    remove_inactive_cuts : bool
        Drop cuts that are not tight at the current LP solution (default True).
    warm_start : int
        Warm-start level for the cutting-plane (0 = off, 1 = on; default 1).

    Cutting-plane – time limits (seconds)
    -------------------------------------
    first_lp_time_limit : float
        Time limit for the very first LP solve (default 360.0).
    lp_time_limit : float
        Time limit for each subsequent LP solve (default 180.0).
    time_limit : float
        Overall wall-clock time limit for the cutting-plane algorithm
        (default 7200.0).
    max_separation_time : float
        Maximum time spent in one separation call (default 300.0).

    Cutting-plane – tolerances
    --------------------------
    first_lp_solver_tol : float
        Solver tolerance for the first LP (default 1e-6).
    solver_tol : float
        Solver tolerance for subsequent LPs (default 1e-6).
    lb_solver_tol : float
        Tolerance used when computing safe lower bounds (default 1e-6).
    cuts_vio_tol : float
        Violation tolerance for cut separation (default 1e-4).
    cuts_act_tol : float
        Activity tolerance for deciding if a cut is tight (default 1e-4).
    opt_gap : float
        Target relative optimality gap (default 1e-4).

    Cutting-plane – *t* parameter
    -----------------------------
    t_upper_bound : int or None
        Maximum subset size *t* for inequalities (3k).  If ``None``, defaults
        to *K*.

    Heuristic
    ---------
    random_seed : int
        Random seed for Lloyd's algorithm initialisation (default 42).
    lloyd_random_starts : int
        Number of random restarts for Lloyd's algorithm (default 100).
    heuristic_only : bool
        If True, only run Lloyd's heuristic and skip the LP (default False).

    Branch & bound
    --------------
    bnb_node_limit : int
        Maximum BnB nodes; 0 disables BnB (default 0).
    bnb_time_limit : float
        BnB time limit in seconds (default 3600.0).
    bnb_gap_tol : float
        BnB optimality gap tolerance (default 1e-4).
    bnb_global_ub : float
        Known global upper bound for BnB (default inf).
    bnb_cut_iter_limit : int
        Maximum cut iterations inside each BnB node (default 30).

    Output control
    --------------
    verbose : int
        Console verbosity: 0 = silent, 1 = progress, 2 = debug (default 1).
    output_level : int
        File output detail: 0 = none, 1 = summary, 2 = iterations,
        3 = full (default 3).
    save_output : bool
        Whether to write log files to disk at all (default True).
    output_dir : str or Path or None
        Directory for output files. ``None`` means current working directory.
    output_prefix : str or None
        Prefix for auto-generated file names. ``None`` generates a timestamped
        name like ``cutlpk_ordinary_K3_20260628_143022``.

    Attributes (populated by ``fit``)
    ---------------------------------
    cost_ : float
        Best (upper bound) clustering cost.
    relative_gap_ : float
        Relative optimality gap at termination.
    lower_bound_ : float
        Best lower bound.
    upper_bound_ : float
        Best upper bound (same as ``cost_``).
    warm_start_cost_ : float
        Cost of the Lloyd-heuristic solution used to warm-start.
    solver_time_ : float
        Total time (seconds) spent inside the LP solver.
    separation_time_ : float
        Total time (seconds) spent on cut separation.
    heuristic_time_ : float
        Total time (seconds) spent on the rounding heuristic.
    status_ : str
        Human-readable termination status.
    retcode_ : int
        Numeric return code (0 = optimal).
    bnb_executed_ : bool
        Whether branch & bound was triggered.
    bnb_status_ : str or None
        BnB status string (if BnB ran).
    labels_ : np.ndarray or None
        Cluster label for each point (0 .. K-1).
    """

    # ---- mandatory ----
    n_clusters: int

    # ---- solver ----
    solver: str = "cupdlpx"
    solver_warm_start: bool = True

    # ---- cut management ----
    max_cuts_init: int = 15_000_000
    max_cuts_per_iter: int = 30_000_000
    max_cuts_added_iter: int = 10_000_000
    max_separation_size: int = 15_000_000
    max_active_cuts_size: int = 30_000_000

    # ---- algorithm control ----
    max_iter: int = 3000
    num_iter_no_improve: int = 2
    exact_separation: bool = True
    remove_inactive_cuts: bool = True
    warm_start: int = 1

    # ---- time limits ----
    first_lp_time_limit: float = 360.0
    lp_time_limit: float = 180.0
    time_limit: float = 7200.0
    max_separation_time: float = 300.0

    # ---- tolerances ----
    first_lp_solver_tol: float = 1e-6
    solver_tol: float = 1e-6
    lb_solver_tol: float = 1e-6
    cuts_vio_tol: float = 1e-4
    cuts_act_tol: float = 1e-4
    opt_gap: float = 1e-4

    # ---- t parameter ----
    t_upper_bound: Optional[int] = None

    # ---- heuristic ----
    random_seed: int = 42
    lloyd_random_starts: int = 100
    heuristic_only: bool = False

    # ---- branch & bound ----
    bnb_node_limit: int = 0
    bnb_time_limit: float = 3600.0
    bnb_gap_tol: float = 1e-4
    bnb_global_ub: float = float("inf")
    bnb_cut_iter_limit: int = 30

    # ---- output control ----
    verbose: int = 1
    output_level: int = 3
    save_output: bool = True
    output_dir: Optional[str] = None
    output_prefix: Optional[str] = None

    # ---- learned attributes ----
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

    # ---- internal ----
    _STATUS_MAP = {
        0: "optimal",
        1: "no_violated_cuts",
        2: "no_improvement",
        3: "time_or_limit",
        4: "error",
        5: "max_iter",
        6: "infeasible",
        7: "heuristic_only",
    }
    _BNB_STATUS_MAP = {
        0: "optimal",
        1: "node_limit",
        2: "time_limit",
        3: "all_nodes_explored",
        4: "error",
    }

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------
    def _collect_params(self) -> Dict[str, Any]:
        t_ub = self.t_upper_bound if self.t_upper_bound is not None else self.n_clusters
        return {
            "solver": self.solver,
            "solver_warm_start": self.solver_warm_start,
            "max_cuts_init": self.max_cuts_init,
            "max_cuts_per_iter": self.max_cuts_per_iter,
            "max_cuts_added_iter": self.max_cuts_added_iter,
            "max_separation_size": self.max_separation_size,
            "max_active_cuts_size": self.max_active_cuts_size,
            "max_iter": self.max_iter,
            "num_iter_no_improve": self.num_iter_no_improve,
            "exact_separation": self.exact_separation,
            "remove_inactive_cuts": self.remove_inactive_cuts,
            "warm_start": self.warm_start,
            "first_lp_time_limit": self.first_lp_time_limit,
            "lp_time_limit": self.lp_time_limit,
            "time_limit": self.time_limit,
            "max_separation_time": self.max_separation_time,
            "first_lp_solver_tol": self.first_lp_solver_tol,
            "solver_tol": self.solver_tol,
            "lb_solver_tol": self.lb_solver_tol,
            "cuts_vio_tol": self.cuts_vio_tol,
            "cuts_act_tol": self.cuts_act_tol,
            "opt_gap": self.opt_gap,
            "t_upper_bound": t_ub,
            "random_seed": self.random_seed,
            "lloyd_random_starts": self.lloyd_random_starts,
            "heuristic_only": self.heuristic_only,
            "bnb_node_limit": self.bnb_node_limit,
            "bnb_time_limit": self.bnb_time_limit,
            "bnb_gap_tol": self.bnb_gap_tol,
            "bnb_global_ub": self.bnb_global_ub,
            "bnb_cut_iter_limit": self.bnb_cut_iter_limit,
            "verbose": self.verbose,
            "output_level": self.output_level,
        }

    def _build_output_paths(self) -> Dict[str, str]:
        out_dir = _resolve_output_dir(self.output_dir)
        prefix = self.output_prefix or _default_output_prefix(self.n_clusters, "ordinary")
        return {
            "output_file": str(out_dir / f"{prefix}_output.txt") if self.save_output else "",
            "bnb_output_file": str(out_dir / f"{prefix}_bnb_log.txt") if self.save_output else "",
        }

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def fit(self, X: Any, **override_params: Any) -> "OrdinaryKMeans":
        """Run the solver on ``X`` (shape ``(n, d)``) and store results."""
        data = np.asarray(X, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("Input data must be a 2D array-like structure")

        params = self._collect_params()
        params.update(self._build_output_paths())
        params.update(override_params)

        result = _cutlpk.run_ordinary_kmeans(data, int(self.n_clusters), params)

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
        self.bnb_status_ = (
            self._BNB_STATUS_MAP.get(int(result["bnb_status"]))
            if self.bnb_executed_
            else None
        )
        labels = result["labels"]
        self.labels_ = np.asarray(labels, dtype=np.int32) if labels is not None else None

        return self

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        """Return all constructor parameters as a dictionary."""
        return {
            "n_clusters": self.n_clusters,
            "solver": self.solver,
            "solver_warm_start": self.solver_warm_start,
            "max_cuts_init": self.max_cuts_init,
            "max_cuts_per_iter": self.max_cuts_per_iter,
            "max_cuts_added_iter": self.max_cuts_added_iter,
            "max_separation_size": self.max_separation_size,
            "max_active_cuts_size": self.max_active_cuts_size,
            "max_iter": self.max_iter,
            "num_iter_no_improve": self.num_iter_no_improve,
            "exact_separation": self.exact_separation,
            "remove_inactive_cuts": self.remove_inactive_cuts,
            "warm_start": self.warm_start,
            "first_lp_time_limit": self.first_lp_time_limit,
            "lp_time_limit": self.lp_time_limit,
            "time_limit": self.time_limit,
            "max_separation_time": self.max_separation_time,
            "first_lp_solver_tol": self.first_lp_solver_tol,
            "solver_tol": self.solver_tol,
            "lb_solver_tol": self.lb_solver_tol,
            "cuts_vio_tol": self.cuts_vio_tol,
            "cuts_act_tol": self.cuts_act_tol,
            "opt_gap": self.opt_gap,
            "t_upper_bound": self.t_upper_bound,
            "random_seed": self.random_seed,
            "lloyd_random_starts": self.lloyd_random_starts,
            "heuristic_only": self.heuristic_only,
            "bnb_node_limit": self.bnb_node_limit,
            "bnb_time_limit": self.bnb_time_limit,
            "bnb_gap_tol": self.bnb_gap_tol,
            "bnb_global_ub": self.bnb_global_ub,
            "bnb_cut_iter_limit": self.bnb_cut_iter_limit,
            "verbose": self.verbose,
            "output_level": self.output_level,
            "save_output": self.save_output,
            "output_dir": self.output_dir,
            "output_prefix": self.output_prefix,
        }

    def set_params(self, **params: Any) -> "OrdinaryKMeans":
        """Set parameters from keyword arguments."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key!r}")
        return self

    def summary(self) -> str:
        """Return a one-line summary of the last ``fit`` result."""
        if self.cost_ is None:
            return "OrdinaryKMeans (not fitted)"
        return (
            f"OrdinaryKMeans(K={self.n_clusters}, cost={self.cost_:.6f}, "
            f"gap={self.relative_gap_:.2e}, status={self.status_}, "
            f"time={self.solver_time_:.1f}s)"
        )

    def __repr__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# Functional helper
# ---------------------------------------------------------------------------

def solve_kmeans(X: Any, n_clusters: int, **params: Any) -> Dict[str, Any]:
    """Functional interface returning the raw solver result dictionary."""
    data = np.asarray(X, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array-like structure")
    return _cutlpk.run_ordinary_kmeans(data, int(n_clusters), params)