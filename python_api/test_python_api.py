"""Smoke tests for the cutlpk Python bindings.

Run with ``PYTHONPATH=python_api python python_api/test_python_api.py`` from the
repository root.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from cutlpk import OrdinaryKMeans, solve_kmeans, FairKMeans, solve_fair_kmeans, SpectralKMeans, solve_spectral_kmeans


def generate_synthetic_data(n_samples: int = 50) -> np.ndarray:
    """Generate synthetic data with two vertical lines."""
    np.random.seed(42)  # For reproducibility
    
    # Two vertical lines: one at x=0, one at x=2
    # Points distributed along y-axis from -1 to 1
    n_per_line = n_samples // 2
    
    # First line: x=0, y uniform in [-1, 1]
    line1 = np.column_stack([
        np.zeros(n_per_line),
        np.random.uniform(-1, 1, n_per_line)
    ])
    
    # Second line: x=2, y uniform in [-1, 1]  
    line2 = np.column_stack([
        np.full(n_per_line, 2.0),
        np.random.uniform(-1, 1, n_per_line)
    ])
    
    # Combine the lines
    data = np.vstack([line1, line2])
    
    # Add some noise
    data += np.random.normal(0, 0.05, data.shape)
    
    return data


def load_iris() -> np.ndarray:
    repo_root = Path(__file__).resolve().parents[1]
    iris_path = repo_root / "HC_data.csv"
    data = np.loadtxt(iris_path, delimiter=",")
    return data


def generate_synthetic_groups(n_samples: int, n_groups: int = 3) -> tuple[list[list[bool]], list[int]]:
    """Generate synthetic group assignments and ratios for fair clustering."""
    # Randomly assign each point to groups
    groups = []
    for _ in range(n_groups):
        group = [False] * n_samples
        groups.append(group)
    
    # Assign each point to exactly one group randomly
    for i in range(n_samples):
        group_idx = np.random.randint(0, n_groups)
        groups[group_idx][i] = True
    
    # Generate ratios - roughly equal distribution
    ratios = [1] * n_groups  # Equal ratios for simplicity
    
    return groups, ratios


def compute_laplacian(data: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Compute Laplacian matrix from data using RBF kernel."""
    n_samples = data.shape[0]
    
    # Compute pairwise distances
    distances = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            distances[i, j] = np.sum((data[i] - data[j]) ** 2)
    
    # Compute adjacency matrix using RBF kernel
    A = np.exp(-distances / (2 * sigma ** 2))
    
    # Compute degree matrix
    D = np.diag(np.sum(A, axis=1))
    
    # Compute Laplacian
    L = D - A
    
    return L

def main() -> None:
    print("Generating synthetic data...")
    data = generate_synthetic_data(n_samples=10)
    
    # Validate data
    if not isinstance(data, np.ndarray):
        raise ValueError("Data is not a numpy array")
    if data.ndim != 2:
        raise ValueError("Data is not 2-dimensional")
    if np.isnan(data).any():
        raise ValueError("Data contains NaN values")
    
    print(f"Data shape: {data.shape}")
    print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
    
    # Generate synthetic fair clustering info
    # For this simple case, let's assign points to groups based on which line they're on
    n_groups = 2
    n_rows = data.shape[0]
    group_size = n_rows // n_groups
    groups = [[False] * n_groups for _ in range(n_rows)]

    for i in range(n_rows):
        group_idx = i // group_size
        groups[i][group_idx] = True
    
    print(f"Generated groups matrix of shape ({len(groups)}, {len(groups[0])})")
    for i, point_groups in enumerate(groups):
        print(f"Point {i}: {point_groups}")
    
    # Compute Laplacian for spectral clustering
    laplacian = compute_laplacian(data, sigma=0.5)  # Smaller sigma for this scale
    print(f"Computed Laplacian matrix of shape {laplacian.shape}")

    ordinary_kmeans = OrdinaryKMeans(n_clusters=2, bnb_node_limit=0,
        extra_params={
            "cutting_plane_output_file": "ordinary_kmeans_output.txt",
            "bnb_output_file": "ordinary_kmeans_bnb_output.txt"
        })
    ordinary_kmeans.fit(data)

    print(f"Cost: {ordinary_kmeans.cost_:.4f}, Status: {ordinary_kmeans.status_}")

    fair_kmeans = FairKMeans(n_clusters=2, groups=groups, fairness_type="alpha", fairness_param=0.8,
        bnb_node_limit=0,
        extra_params={
            "cutting_plane_output_file": "fair_kmeans_output.txt",
            "bnb_output_file": "fair_kmeans_bnb_output.txt"
        })
    fair_kmeans.fit(data)
    print(f"Cost: {fair_kmeans.cost_:.4f}, Status: {fair_kmeans.status_}")

    fair_tau_kmeans = FairKMeans(n_clusters=2, groups=groups, fairness_type="tau", fairness_param=0.2,
        bnb_node_limit=0,
        extra_params={
            "cutting_plane_output_file": "fair_tau_kmeans_output.txt",
            "bnb_output_file": "fair_tau_kmeans_bnb_output.txt"
        })
    fair_tau_kmeans.fit(data)
    print(f"Cost: {fair_tau_kmeans.cost_:.4f}, Status: {fair_tau_kmeans.status_}")

    spectral_kmeans = SpectralKMeans(n_clusters=2, bnb_node_limit=0,
        extra_params={
            "cutting_plane_output_file": "spectral_kmeans_output.txt",
            "bnb_output_file": "spectral_kmeans_bnb_output.txt"
        })
    spectral_kmeans.fit(laplacian)
    print(f"Cost: {spectral_kmeans.cost_:.4f}, Status: {spectral_kmeans.status_}")

    
    print("All tests completed successfully!")

    # read HC_data.csv AND HC_labels.csv as group using numpy
    data = np.loadtxt("HC_data.csv", delimiter=",")
    # groups: 1D array of group labels
    group_labels = np.loadtxt("HC_labels.csv", delimiter=",").astype(int)
    n_points = len(group_labels)
    n_groups = np.max(group_labels) + 1

    # Convert to one-hot matrix
    groups_onehot = np.zeros((n_points, n_groups), dtype=bool)
    for i, g in enumerate(group_labels):
        groups_onehot[i, g] = True

    #specify output file names and write some info to it first, including n_clusters, fairness_type, fairness_param
    cutting_plane_output_file = "fair_kmeans_HC_output.txt"
    bnb_output_file = "fair_kmeans_HC_bnb_output.txt"
    with open(cutting_plane_output_file, "w") as f:
        f.write(f"n_clusters: 3\n")
        f.write(f"fairness_type: alpha\n")
        f.write(f"fairness_param: 0.8\n")
    with open(bnb_output_file, "w") as f:
        f.write(f"n_clusters: 3\n")
        f.write(f"fairness_type: alpha\n")
        f.write(f"fairness_param: 0.8\n")
    # run fair kmeans on data
    fair_kmeans = FairKMeans(n_clusters=3, groups=groups_onehot, fairness_type="alpha", fairness_param=0.8,
        bnb_node_limit=0,
        extra_params={
            "cutting_plane_output_file": "fair_kmeans_HC_output.txt",
            "bnb_output_file": "fair_kmeans_HC_bnb_output.txt",
            "max_cuts_firstLP": 10000
        })
    fair_kmeans.fit(data)
    print(f"Cost: {fair_kmeans.cost_:.4f}, Status: {fair_kmeans.status_}")

if __name__ == "__main__":
    main()
