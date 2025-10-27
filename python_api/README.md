# Python Bindings for cutLPK

This directory provides a minimal Python interface that mirrors the
scikit-learn API for the ordinary k-means workflow implemented in this
repository.

## Layout

- `_cutlpk`: pybind11 extension exposing the C++ solver entry point.
- `cutlpk/ordinary_kmeans.py`: high-level estimator-style wrapper.
- `CMakeLists.txt`: build script to compile the extension and link it
  against the existing C++ core library.

## Building the extension

Activate the same environment used to build the C++ project and make
sure all solver dependencies (CUDA, cuPDLP, Gurobi, HiGHS, etc.) are
available. Then run:

```bash
cmake -S python_api -B python_api/build
cmake --build python_api/build -j
```

The compiled module will be placed next to the `cutlpk` package inside
`python_api/build/cutlpk`. Add this directory to `PYTHONPATH` (or install
it) to experiment with the API, for example:

```bash
export PYTHONPATH="$PWD/python_api/build"
python - <<'PY'
import numpy as np
from cutlpk.ordinary_kmeans import OrdinaryKMeans

X = np.random.rand(50, 3)
model = OrdinaryKMeans(n_clusters=3).fit(X)
print(model.cost_, model.relative_gap_)
PY
```

## Usage

The high-level wrapper keeps the interface close to scikit-learn:

```python
from cutlpk.ordinary_kmeans import OrdinaryKMeans

model = OrdinaryKMeans(n_clusters=4, warm_start=True, lloyd_random_starts=64)
model.fit(data)
print(model.cost_, model.relative_gap_)
```

For quick scripts you can also call the functional helper:

```python
from cutlpk.ordinary_kmeans import solve_kmeans
result = solve_kmeans(data, 4, opt_gap=1e-5)
print(result["cost"], result["relative_gap"])
```
