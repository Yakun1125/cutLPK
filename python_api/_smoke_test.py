"""Smoke test for cutlpk Python bindings — all synthetic data."""
import os
import sys
import numpy as np

# Must add DLL directories BEFORE any cutlpk import
for dll_dir in [
    r"C:\gurobi1203\win64\bin",
    r"C:\Users\WYKge\Desktop\clustering_paper_2026\experiment\cuPDLPx\build-msvc",
    r"C:\Users\WYKge\Desktop\clustering_paper_2026\experiment\cuPDLPx\build-msvc\Release",
    r"C:\Users\WYKge\Desktop\clustering_paper_2026\experiment\cuPDLPx\build-msvc\_deps\pslp-build\Release",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
]:
    if os.path.isdir(dll_dir):
        os.add_dll_directory(dll_dir)

sys.path.insert(0, r"c:\Users\WYKge\Desktop\clustering_paper_2026\experiment\cutLPK\python_api")

from cutlpk import OrdinaryKMeans, FairKMeans, SpectralKMeans, solve_kmeans

errors = []

# ---- 1. OrdinaryKMeans (heuristic_only) ----
print("=== Test 1: OrdinaryKMeans (heuristic_only) ===")
try:
    X = np.random.rand(30, 2).astype(np.float64)
    model = OrdinaryKMeans(n_clusters=2, verbose=0, save_output=False, heuristic_only=True)
    model.fit(X)
    assert model.cost_ is not None, "cost_ is None"
    assert model.labels_ is not None, "labels_ is None"
    assert len(model.labels_) == 30, f"labels length {len(model.labels_)} != 30"
    print(f"  OK: cost={model.cost_:.4f}, gap={model.relative_gap_:.4e}, status={model.status_}")
    print(f"  summary: {model.summary()}")
    params = model.get_params()
    assert "n_clusters" in params and "opt_gap" in params, "get_params missing keys"
    print(f"  get_params: {len(params)} keys")
except Exception as e:
    errors.append(f"Test 1: {e}")
    print(f"  FAIL: {e}")

# ---- 2. OrdinaryKMeans set_params ----
print("\n=== Test 2: set_params / get_params ===")
try:
    m = OrdinaryKMeans(n_clusters=3)
    m.set_params(time_limit=100.0, opt_gap=0.01, verbose=0)
    assert m.time_limit == 100.0, f"time_limit={m.time_limit}"
    assert m.opt_gap == 0.01, f"opt_gap={m.opt_gap}"
    m.set_params(output_dir="C:/tmp", output_prefix="test")
    assert m.output_dir == "C:/tmp"
    assert m.output_prefix == "test"
    try:
        m.set_params(bad_param=5)
        errors.append("Test 2: should have raised ValueError")
    except ValueError:
        pass  # expected
    print("  OK")
except Exception as e:
    errors.append(f"Test 2: {e}")
    print(f"  FAIL: {e}")

# ---- 3. FairKMeans (tau, 1D labels) ----
print("\n=== Test 3: FairKMeans tau (1D labels) ===")
try:
    groups = np.array([0, 0, 1, 0, 1, 1, 0, 1] * 4)[:30]
    model3 = FairKMeans(n_clusters=2, groups=groups, fairness_type='tau',
                         fairness_param=0.9, verbose=0, save_output=False,
                         heuristic_only=True)
    model3.fit(X)
    assert model3.cost_ is not None
    assert model3.labels_ is not None
    print(f"  OK: cost={model3.cost_:.4f}, gap={model3.relative_gap_:.4e}, status={model3.status_}")
except Exception as e:
    errors.append(f"Test 3: {e}")
    print(f"  FAIL: {e}")

# ---- 4. FairKMeans (alpha, 2D boolean) ----
print("\n=== Test 4: FairKMeans alpha (2D boolean) ===")
try:
    bool_groups = np.zeros((30, 2), dtype=np.float64)
    bool_groups[:15, 0] = 1.0
    bool_groups[15:, 1] = 1.0
    model4 = FairKMeans(n_clusters=2, groups=bool_groups, fairness_type='alpha',
                         fairness_param=0.9, verbose=0, save_output=False,
                         heuristic_only=True)
    model4.fit(X)
    assert model4.cost_ is not None
    print(f"  OK: cost={model4.cost_:.4f}, gap={model4.relative_gap_:.4e}, status={model4.status_}")
except Exception as e:
    errors.append(f"Test 4: {e}")
    print(f"  FAIL: {e}")

# ---- 5. FairKMeans validation ----
print("\n=== Test 5: FairKMeans validation ===")
try:
    # validation happens at fit() time
    m_bad1 = FairKMeans(n_clusters=2, groups=groups, fairness_type='bad')
    try:
        m_bad1.fit(X)
        errors.append("Test 5a: should have raised ValueError")
    except ValueError:
        pass  # expected — bad fairness_type caught at fit
    m_bad2 = FairKMeans(n_clusters=2, groups=groups, fairness_param=1.5)
    try:
        m_bad2.fit(X)
        errors.append("Test 5b: should have raised ValueError")
    except ValueError:
        pass  # expected — bad fairness_param caught at fit
    print("  OK")
except Exception as e:
    errors.append(f"Test 5: {e}")
    print(f"  FAIL: {e}")

# ---- 6. SpectralKMeans ----
print("\n=== Test 6: SpectralKMeans ===")
try:
    from sklearn.metrics.pairwise import rbf_kernel
    W = rbf_kernel(X, gamma=1.0)
    np.fill_diagonal(W, 0)
    D = np.diag(W.sum(axis=1))
    L_mat = D - W
    model6 = SpectralKMeans(n_clusters=2, verbose=0, save_output=False, heuristic_only=True)
    model6.fit(L_mat)
    assert model6.cost_ is not None
    print(f"  OK: cost={model6.cost_:.4f}, gap={model6.relative_gap_:.4e}, status={model6.status_}")
except Exception as e:
    errors.append(f"Test 6: {e}")
    print(f"  FAIL: {e}")

# ---- 7. Functional API ----
print("\n=== Test 7: Functional solve_kmeans ===")
try:
    result = solve_kmeans(X, n_clusters=2, verbose=0, save_output=False, heuristic_only=True)
    assert "cost" in result
    assert "labels" in result
    assert "solver_time" in result
    print(f"  OK: cost={result['cost']:.4f}, keys={sorted(result.keys())}")
except Exception as e:
    errors.append(f"Test 7: {e}")
    print(f"  FAIL: {e}")

# ---- Summary ----
print("\n" + "=" * 60)
if errors:
    print(f"FAILED: {len(errors)} test(s)")
    for e in errors:
        print(f"  - {e}")
else:
    print("ALL 7 TESTS PASSED")
