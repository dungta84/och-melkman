# ogon-rs (PyO3 + NumPy)

Copyright (c) 2026 Institute of Mathematical and Computational Sciences (IMACS), HCMUT, VNU-HCM. MIT License (see the top-level `LICENSE`). Authors: Phong Thi Thu Huyen, Tran Anh Dung, Phan Thanh An.

This package keeps the optimized Rust core generator and exposes it to Python via PyO3.
It also includes the core orthogonal convex hull logic from the Python reference.

## Python API

```python
import ogon_rs

poly = ogon_rs.generate_ogon(1_000_000, seed=0)
hull = ogon_rs.orthogonal_hull(poly)
hull_path = ogon_rs.orthogonal_hull_path(poly)

# `poly`, `hull`, and `hull_path` are NumPy arrays of shape (N, 2), dtype=int32.
```

The point-taking functions accept any 2D array-like object with shape `(N, 2)`,
including NumPy arrays and Python sequences such as `[(x, y), ...]`.

The returned generator polygon is closed: the last point equals the first point.

## Build and install with uv

### Option A: local dev install

```bash
uv venv
source .venv/bin/activate
uv tool install maturin
uv pip install numpy
maturin develop --release
```

### Option B: build a wheel

```bash
uv venv
source .venv/bin/activate
uv tool install maturin
uv pip install numpy
maturin build --release
uv pip install target/wheels/ogon_rs-*.whl
```

## Plot from Python

Generator only:

```bash
uv run python python/plot_ogon.py 1000000 --seed 0
```

Orthogonal hull from a coordinate file:

```bash
uv run python python/plot_ogon_hull.py ogon_1000000.txt
```

## Notes

- The generator core algorithm is unchanged: `choose c -> inflate -> cut`.
- The orthogonal hull logic mirrors the deque-based incremental Python algorithm,
  but performs deque updates in place instead of copying on every step.
