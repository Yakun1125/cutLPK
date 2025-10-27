"""Smoke tests for the cutlpk Python bindings.

Run with ``PYTHONPATH=python_api python python_api/test_python_api.py`` from the
repository root.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from cutlpk import OrdinaryKMeans, solve_kmeans


def load_iris() -> np.ndarray:
    repo_root = Path(__file__).resolve().parents[1]
    iris_path = repo_root / "src" / "build" / "iris.csv"
    data = np.loadtxt(iris_path, delimiter=",")
    return data


def run_solver_functional(data: np.ndarray) -> None:
    result = solve_kmeans(data, n_clusters=3, lloyd_random_starts=5, bnb_node_limit=0)
    print("Functional API result:", {k: result[k] for k in ("cost", "relative_gap", "status")})


def run_solver_estimator(data: np.ndarray) -> None:
    model = OrdinaryKMeans(n_clusters=3, lloyd_random_starts=5, bnb_node_limit=0)
    model.fit(data)
    print("Estimator API result:", {"cost": model.cost_, "relative_gap": model.relative_gap_, "status": model.status_})


def main() -> None:
    data = load_iris()
    run_solver_functional(data)
    run_solver_estimator(data)


if __name__ == "__main__":
    main()
